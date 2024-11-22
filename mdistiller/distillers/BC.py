import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ._base import Distiller
from ._common import ConvReg, get_feat_shapes
from ..engine.utils import count_parameters, load_checkpoint, print_by_pbar, AverageMeter, difference_sets, \
    model_convert, time_watcher
from ..models import cifar_model_dict, imagenet_model_dict
from ..models.cifar.new_convolution import low_rank_conv_scheme2, low_rank_conv_scheme1
from ..models.cifar.resnet import ResNet
from ..models.cifar.vgg import VGG
from ..models.cifar.wrn import WideResNet
from ..models.cifar.resnetv2 import ResNet as ResNetv2
from ..models.imagenet.resnet import ResNet as ResNet_imagenet
from ..distillers.tckd_pretrain import tckd_ls_loss
import pdb


def check_params_nan(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"Parameter {name} contains NaN values.")
            raise ValueError
        if torch.isinf(param).any():
            print(f"Parameter {name} contains Inf values.")
            raise ValueError


class trainingSVD(Distiller):

    def __init__(self, student, teacher, cfg):
        super(trainingSVD, self).__init__(student, teacher)
        self.cfg = cfg
        self.train_state = "scratch"  # "scratch", "compression", "finetune"

        # unchangeable
        if self.cfg.TSVD.SCHEME == 1:
            self.low_rank_conv = low_rank_conv_scheme1
        else:
            self.low_rank_conv = low_rank_conv_scheme2
        self.ce_loss_weight = cfg.TSVD.LOSS.CE_WEIGHT
        self.init_sparse_loss_weight = cfg.TSVD.LOSS.SPARSE_WEIGHT
        self.l2_alpha = cfg.SOLVER.WEIGHT_DECAY
        self.finetune_epoch = cfg.TSVD.FINETUNE_MILESTONE
        self.low_rank_scheme = cfg.TSVD.SCHEME
        self.tckd_like_loss = cfg.TSVD.LOSS.TCKD_LIKE
        # can be changed when training
        self.sparse_loss_weight = self.init_sparse_loss_weight
        self.epoch = 0
        self.iter = 0
        self.epoch_iter = int(self.cfg.DATASET.NUM_DATA / self.cfg.SOLVER.BATCH_SIZE) + 1
        # self.epoch_iter = 196
        # print("WARNING: self.epoch_iter = 196 !!!")
        self.rank_decline_ratio = 1

        self.iter_time_watcher = None
        # self.iter_time_watcher = time_watcher()

        if teacher is not None:
            # output = self.teacher(torch.rand(1, 3, 32, 32))[0]
            # assert output.shape[0] == cfg.DATASET.NUM_CLASSES,\
            #     "the pretrain model is not suitable for dataset ({})".format(cfg.DATASET.TYPE)
            self.train_state = "compression"
            self.reparameterize_normal2lowrank()
        else:
            raise ValueError
        self.distiller_dataparallel()
        del self.teacher

    def update_epoch_and_iter(self):
        self.iter += 1
        if self.epoch != int(self.iter / float(self.epoch_iter)):
            self.epoch = int(self.iter / float(self.epoch_iter))
            return True
        return False

    def reparameterize_normal2lowrank(self):
        pass
        if self.cfg.DATASET.TYPE == "imagenet":
            # import pdb
            # pdb.set_trace()
            model_teacher = imagenet_model_dict[self.cfg.DISTILLER.TEACHER](pretrained=True,
                                                                            num_classes=self.cfg.DATASET.NUM_CLASSES)
            load_ret = self.student.load_state_dict(model_teacher.state_dict(), strict=False)
            print(f"result of reparameterize pretrain model:", load_ret)
        else:
            _, pretrain_model_path = cifar_model_dict[self.cfg.DISTILLER.TEACHER]
            self.student.load_state_dict(load_checkpoint(pretrain_model_path)["model"], strict=False)
        # load first conv, bn, and fc layer
        conv2d = []
        rep_conv = []
        if (isinstance(self.teacher, ResNet) or isinstance(self.teacher, ResNetv2)
            or isinstance(self.teacher, WideResNet)) or isinstance(self.teacher, ResNet_imagenet):
            for j, i in self.teacher.named_modules():
                if j.find('layer') != -1 and j.find('conv') != -1 and i.kernel_size == (3, 3):
                    # the conv in the basicblock(in "layer") and not the downsample("conv" instead of "downsample")
                    # and the size of the conv2d is 3*3
                    assert isinstance(i, nn.Conv2d)
                    if i.bias is None:
                        conv2d.append(i)
        elif isinstance(self.teacher, VGG):
            for j, i in self.teacher.named_modules():
                if isinstance(i, nn.Conv2d):
                    if i.bias is None:
                        conv2d.append(i)
        # elif isinstance(self.teacher, WideResNet):
        #     for j, i in self.teacher.named_modules():
        #         if isinstance(i, nn.Conv2d):
        #             if i.bias is None:
        #                 conv2d.append(i)
        else:
            print("the teacher model is not supported to Low Rank Compression")
        # import pdb
        # pdb.set_trace()
        for _, i in self.student.named_modules():
            if isinstance(i, self.low_rank_conv):
                rep_conv.append(i)
        assert len(conv2d) == len(rep_conv), "the number of conv2d and rep_conv is not equal, normal={}:rep={}".format(
            len(conv2d), len(rep_conv))

        for i in range(len(conv2d)):
            print("conv2d:{}, rep_conv:{}, decoupling".format(conv2d[i].weight.shape, rep_conv[i].conv2_p.shape))
            outC, inC, H, W = conv2d[i].weight.shape
            if self.cfg.TSVD.SCHEME == 2:  # scheme 2
                U, S, V = torch.svd(conv2d[i].weight.data.permute(0, 2, 1, 3).reshape(outC * H, -1))
                # outC, inC, H, W -> H*inC, W*outC
            elif self.cfg.TSVD.SCHEME == 1:  # scheme 1
                U, S, V = torch.svd(conv2d[i].weight.data.reshape(outC, -1))
            else:
                raise ValueError

            # 对S进行归一化，由于batch norm对USV无条件地进行了缩放，所以不需要对U、V进行处理
            max_value = torch.max(S)
            assert max_value > 0
            S = S / max_value

            if rep_conv[i].conv2_p.data.shape != U.shape:
                print("{} != {}".format(U.shape, rep_conv[i].conv2_p.data.shape))
                print("{} != {}".format(V.shape, rep_conv[i].conv1_p.data.shape))
                print("{} != {}".format(S.shape, rep_conv[i].singular_p.data.shape))

            assert rep_conv[i].conv2_p.data.shape == U.shape, "{} != {}".format(
                U.shape, rep_conv[i].conv2_p.data.shape)
            assert rep_conv[i].conv1_p.data.shape == V.shape
            assert rep_conv[i].singular_p.data.shape == S.shape
            rep_conv[i].conv2_p.data = nn.Parameter(U).data  # conv2_p is U !!!!
            rep_conv[i].conv1_p.data = nn.Parameter(V).data
            rep_conv[i].singular_p.data = nn.Parameter(S).data
            # check_params_nan(rep_conv[i])

        print("Student model param after reparameterize is: {}".format(
            count_parameters(self.student, only_trainable=True)))

    def get_learnable_parameters(self):
        # set learnable just singular
        # ret = []
        # learnable_list = []
        # for name, param in self.named_parameters():
        #     for module in learnable_list:
        #         if name.find(module) != -1:
        #             ret.append(param)
        #             break
        #         else:
        #             param.requires_grad = False
        # return ret
        return super().get_learnable_parameters()

    def get_no_weightdecay_parameters(self):
        ret = []
        no_weightdecay_list = ['conv1_p', 'conv2_p', 'singular_p']
        for name, param in self.named_parameters():
            for module in no_weightdecay_list:
                if name.find(module) != -1:
                    ret.append(param)
                    break
        return ret

    def get_extra_parameters(self):
        num_p = 0
        if hasattr(self, "conv_reg"):
            for p in self.conv_reg.parameters():
                num_p += p.numel()
        return num_p

    def forward_train(self, image, target, **kwargs):
        if self.iter_time_watcher is not None:
            self.iter_time_watcher.next_in_time = time.perf_counter()
            self.iter_time_watcher.print_time()
            self.iter_time_watcher.in_time = time.perf_counter()
        # check_params_nan(self.student)
        epoch_updated = self.update_epoch_and_iter()
        if epoch_updated:
            self.report_model_rank()
        if self.epoch < self.finetune_epoch:
            if self.cfg.TSVD.WARMUP_COMPRESSION:
                self.sparse_loss_weight = self.init_sparse_loss_weight * 0.5 * (
                        np.cos(np.pi * ((self.epoch + 1) / self.finetune_epoch - 1)) + 1)
            else:
                self.sparse_loss_weight = self.init_sparse_loss_weight
        else:
            self.sparse_loss_weight = 0.0

        if self.train_state == "compression":
            # self.S_Normalize(self.student)
            pruning_iter = max(10, self.epoch_iter // 3)  # 剪枝的iter间隔，每epoch剪枝3次，最少间隔10个iter才能剪枝一次
            if self.iter % max(pruning_iter, 1) == 0:
                self.model_pruning(self.low_rank_conv, self.student)
                pass
            if self.epoch == self.finetune_epoch:
                self.sparse_loss_weight = 0.0
                self.train_state = "finetune"


        logits_student, _ = self.student(image)
        if self.iter_time_watcher is not None:
            self.iter_time_watcher.forward_time = time.perf_counter()
        # with torch.no_grad():
        #     _, feature_teacher = self.teacher_module(image)
        # losses
        if self.tckd_like_loss:
            loss_ce = tckd_ls_loss(logits_student, target, alpha=0.95)  # 采用TCKD_LS式的监督
        else:
            loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)  # 计算与真实值的交叉熵误差
        sparse_loss_list = []
        l2_loss_list = []

        for _, i in self.student.named_modules():
            if isinstance(i, self.low_rank_conv):
                V, U, rank, S = i.conv1_p, i.conv2_p, i.rank, i.singular()
                if len(i.unfreeze_rank) > 0:
                    num_unfreeze_rank = len(i.unfreeze_rank)
                else:
                    raise ValueError
                index = torch.tensor(i.unfreeze_rank, dtype=torch.int64).to(self.cfg.DEVICE)
                if not index.dtype == torch.int64:
                    raise ValueError
                else:
                    U = torch.index_select(U, dim=1, index=index)
                    V = torch.index_select(V, dim=1, index=index)
                    S = torch.index_select(S, dim=0, index=index)

                # if i.rank > 384:  # prune the last stage conv
                loss2 = torch.norm(S, 1)
                loss2 = loss2 * torch.sqrt(torch.tensor(num_unfreeze_rank))
                sparse_loss_list.append(loss2)


                # loss3 = ((U @ V.transpose(1, 0)) ** 2).sum() / 2.0
                loss3 = ((U @ torch.diag(S) @ V.transpose(1, 0)) ** 2).sum() / 2.0   #  这样是不行的，我们对S施加L1，鼓励S减小，如果对USV矩阵施加L2，就无法约束U, V的值
                # loss3 = loss3 * S.mean(dim=0).detach()  # 计算loss的时候再乘一个S的均值，梯度被阻断所以L2loss不会减小S的值
                l2_loss_list.append(loss3)

        loss_sparse = self.sparse_loss_weight * sum(sparse_loss_list)
        l2_loss = self.l2_alpha * sum(l2_loss_list)

        sharpness = self.logits_sharpness(logits_student, 1)[0]

        if self.train_state == "compression":
            losses_dict = {
                "loss_ce": loss_ce,
                "loss_sparse": loss_sparse,
                "l2_loss": l2_loss,
                "sharpness": sharpness.detach(),
            }
            # print(losses_dict)
            # sys.stdout.flush()
        elif self.train_state == "scratch" or self.train_state == "finetune":
            losses_dict = {
                "loss_ce": loss_ce,
                "loss_sparse": loss_sparse.detach(),
                "l2_loss": l2_loss,
                "sharpness": sharpness.detach(),
            }
        else:
            raise ValueError
        for key, value in losses_dict.items():
            assert not torch.isnan(value).any(), "loss {} is nan".format(key)
            # if torch.isnan(value).any():
            #     print("loss {} is nan".format(key))
            #     pdb.set_trace()

        if self.iter_time_watcher is not None:
            self.iter_time_watcher.backward_time = time.perf_counter()
            self.iter_time_watcher.out_time = time.perf_counter()
        return logits_student, losses_dict

    def report_model_rank(self):
        # For debug: print rank of eack layer
        rank_ratio = []
        name_list = []
        rank_list = []
        for name, i in self.student.named_modules():
            if isinstance(i, self.low_rank_conv):
                # print("layer: {}, rank: {}".format(name, i.singular()))
                rank_ratio.append(len(i.unfreeze_rank) / i.rank)
                rank_list.append("{}/{}".format(len(i.unfreeze_rank), i.rank))
                name_list.append(name)
        format_rank_list = ["{:.2f}".format(rank) for rank in rank_ratio]
        # print("rank_ratio: {}".format(format_rank_list))
        # print bar chart in terminal
        max_value = max(rank_ratio)
        for rank_, ratio, name_ in zip(rank_list, rank_ratio, name_list):
            # print(value, max_value)
            bar = '█' * int(ratio * 50 // max_value) + "  " + str(rank_)  # 50 是柱子的最大宽度
            print(f"{name_}: {bar}")

    def model_pruning(self, low_rank_conv, model, percentage=0.005):
        has_pruned = False
        rank_decline_ratio = AverageMeter()
        for name, i in model.named_modules():
            if isinstance(i, low_rank_conv):
                for j, rank in enumerate(i.singular()):
                    if abs(rank) <= percentage and j in i.unfreeze_rank and len(i.unfreeze_rank) > 1:
                        has_pruned = True
                        i.unfreeze_rank.remove(j)  # 将第j个秩记为freeze，从unfreeze rank中删掉，计算loss时不再计算它
                        i.rank_mask[j] = 0.0  # 对应的mask置0
                if len(i.unfreeze_rank) == 0:
                    print(name)
                    print(i.singular_p)
                    print(i.rank_mask)
                    raise ValueError
                assert i.rank > 0
                rank_decline_ratio.update(len(i.unfreeze_rank) / float(i.rank))
        if has_pruned:
            self.report_model_rank()
            sys.stdout.flush()
        #     self.rank_decline_ratio = rank_decline_ratio.avg
        #     print_by_pbar("Student model rank decline ratio is: {}".format(rank_decline_ratio.avg))

    def S_Normalize(self, model):
        for name, i in model.named_modules():
            if isinstance(i, self.low_rank_conv):
                if i.singular_p.device != self.cfg.DEVICE:
                    i.singular_p = i.singular_p.to(self.cfg.DEVICE)
                # singular_p = i.singular_p.data.detach()
                # max_value = torch.max(abs(singular_p))
                # assert max_value > 0, "singular param is {}, which is not allowed".format(singular_p)
                # i.singular_p.data = i.singular_p.data / max_value
                singular = i.singular().detach()
                max_value = torch.max(abs(singular))
                # singular 中是否有nan
                assert not torch.isnan(
                    singular).any().item(), "nan exist in singular param {}, which is not allowed".format(singular)
                assert max_value > 0, "singular param is {}, which is not allowed".format(singular)
                i.singular_p.data = i.singular_p * i.rank_mask.to(self.cfg.DEVICE)
                i.singular_p.data = i.singular_p.data / max_value

    def model_convert(self, model):  # 将低秩模型转换为正常模型
        scheme = self.low_rank_scheme
        num_conv = self.cfg.TSVD.NUM_CONV
        # num_conv = "two"
        return model_convert(model, scheme, num_conv)
