import torch
import torch.nn as nn
from collections import OrderedDict
from scipy.linalg import svd
import numpy as np
import torch.nn.functional as F
from numpy.linalg import matrix_rank

# from mdistiller.engine.utils import load_cfg


def new_conv(in_channels, out_channels, kernel_size, stride, padding, bias=False, scheme=1):
    assert bias==False or bias == None
    """3x3 convolution with new implement"""
    if scheme == 2:
        return low_rank_conv_scheme2(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
    elif scheme == 1:
        return low_rank_conv_scheme1(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
    else:
        raise ValueError



# 基类
class low_rank_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=False,
                 padding_mode="zeros", deploy=False):
        super(low_rank_conv, self).__init__()
        assert (bias is None) or (not bias)
        # config
        self.deploy = deploy
        self.inC = in_channels
        self.outC = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.groups = groups
        self.rank = self.get_rank()
        self.pruning_gate = 0.0  # 剪枝阈值，小于阈值的秩将会被剪掉
        # variables
        self.unfreeze_rank = [m for m in range(self.rank)]
        self.rank_mask = torch.ones(self.rank)
        # layers
        # self.bn = nn.BatchNorm2d(num_features=self.rank, affine=True)
        self.singular_p = nn.Parameter(torch.empty(self.rank))
        torch.nn.init.ones_(self.singular_p)

    def get_rank(self):
        return NotImplementedError

    def singular(self):
        # method 1
        # return self.singular_p
        # method 2
        # return F.relu(self.singular_p)
        # method 3
        # print(self.rank_mask.shape, self.singular_p.shape)
        return self.rank_mask.to(self.singular_p.device) * self.singular_p

    def pruning_rank(self, V, U, S):
        # 根据阈值计算保留哪些秩，没有更改mask（虽然S的值是经过mask后的）
        unfreeze_rank = [m for m in range(self.rank)]
        for i, rank in enumerate(S.detach().cpu().numpy()):
            if abs(rank) <= self.pruning_gate:
                unfreeze_rank.remove(i)
        unfreeze_rank = torch.tensor(data=unfreeze_rank, dtype=torch.int64).to(self.conv1_p.device)
        U = torch.index_select(U, dim=1, index=unfreeze_rank)
        V = torch.index_select(V, dim=0, index=unfreeze_rank)  # V has transpose, so dim is 0
        S = torch.index_select(S, dim=0, index=unfreeze_rank)
        new_rank = len(unfreeze_rank)
        return V, U, S, new_rank

    def forward(self, x):
        if self.deploy:
            assert hasattr(self, "new_conv")
            x = self.new_conv(x)
        else:
            V, U = self.matrix_to_param()
            # conv1_weight, conv2_weight = self.param_to_conv_weight(V, U, num_conv=2)
            # x = F.conv2d(
            #     x,
            #     conv1_weight,
            #     bias=None,
            #     stride=self.stride_1,
            #     padding=self.padding_1,
            #     dilation=self.dilation,
            #     groups=self.groups,
            # )
            # x = F.conv2d(
            #     x,
            #     conv2_weight,
            #     bias=None,
            #     stride=self.stride_2,
            #     padding=self.padding_2,
            #     dilation=self.dilation,
            #     groups=self.groups,
            # )
            conv_weight = self.param_to_conv_weight(V, U, num_conv=1)
            x = F.conv2d(
                x,
                conv_weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        return x


    def matrix_to_param(self):
        V = self.conv1_p.transpose(1, 0)
        U = self.conv2_p
        V = torch.diag(self.singular()) @ V
        U = U # @ torch.diag(self.singular() ** 0.5)
        return V, U


class low_rank_conv_scheme1(low_rank_conv):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=False,
                 padding_mode="zeros", deploy=False):
        super(low_rank_conv_scheme1, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            deploy=deploy,
        )
        self.stride_1 = stride
        self.stride_2 = 1
        self.padding_1 = padding
        self.padding_2 = 0

        self.conv1_p = nn.Parameter(torch.empty(in_channels * kernel_size, self.rank))  # U matrix
        self.conv2_p = nn.Parameter(torch.empty(out_channels * kernel_size, self.rank))  # V matrix
        torch.nn.init.kaiming_normal_(self.conv1_p, mode="fan_out", nonlinearity="relu")
        torch.nn.init.kaiming_normal_(self.conv2_p, mode="fan_out", nonlinearity="relu")



    def get_rank(self):
        return min(self.rank, self.inC * self.kernel_size * self.kernel_size, self.outC)

    def param_to_conv_weight(self, conv1_matrix, conv2_matrix):
        conv1_matrix = conv1_matrix.reshape(-1, self.inC, self.kernel_size, self.kernel_size)
        conv2_matrix = conv2_matrix.reshape(self.outC, -1, 1, 1)
        return conv1_matrix, conv2_matrix

    def matrix_to_compact_conv(self, num_conv="two"):
        # num_conv = "one" or "two"
        V, U = self.matrix_to_param()
        S = self.singular()
        V, U, S, new_rank = self.pruning_rank(V, U, S)
        if num_conv == "one":
            padding = [max(self.padding_1), max(self.padding_2)]
            stride = [max(self.stride_1), max(self.stride_2)]
            M = (U @ V).reshape(self.outC, self.inC, self.kernel_size, self.kernel_size)
            self.new_conv = nn.Conv2d(out_channels=self.outC, in_channels=self.inC, kernel_size=self.kernel_size,
                                      padding=padding, stride=stride, bias=False)
            self.new_conv.weight.data = M
        elif num_conv == "two":
            V, U = self.param_to_conv_weight(V, U)
            conv1 = nn.Conv2d(out_channels=new_rank, in_channels=self.inC, kernel_size=self.kernel_size,
                              padding=self.padding_1, stride=self.stride_1, bias=False)
            conv2 = nn.Conv2d(out_channels=self.outC, in_channels=new_rank, kernel_size=1,
                              padding=self.padding_2, stride=self.stride_2, bias=False)
            conv1.weight.data = V
            conv2.weight.data = U
            self.new_conv = nn.Sequential(conv1, conv2)
        else:
            raise ValueError
        delattr(self, "conv1_p")
        delattr(self, "conv2_p")
        delattr(self, "singular_p")

class low_rank_conv_scheme2(low_rank_conv):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=False,
                 padding_mode="zeros", deploy=False):
        super(low_rank_conv_scheme2, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            deploy=deploy,
        )
        self.stride = stride
        self.padding = padding

        self.stride_1 = [1, stride]
        self.stride_2 = [stride, 1]
        self.padding_1 = [0, padding]
        self.padding_2 = [padding, 0]

        self.conv1_p = nn.Parameter(torch.empty(in_channels * kernel_size, self.rank))  # U
        self.conv2_p = nn.Parameter(torch.empty(out_channels * kernel_size, self.rank))  # V
        torch.nn.init.kaiming_normal_(self.conv1_p, mode="fan_out", nonlinearity="relu")
        torch.nn.init.kaiming_normal_(self.conv2_p, mode="fan_out", nonlinearity="relu")

    def get_rank(self):
        return min(self.inC * self.kernel_size, self.outC * self.kernel_size)

    def param_to_conv_weight(self, conv1_matrix, conv2_matrix, num_conv=2):
        if num_conv == 1:
            conv_matrix = conv2_matrix @ conv1_matrix
            conv_matrix = conv_matrix.reshape(self.outC, self.kernel_size, self.inC, self.kernel_size).permute(0, 2, 1, 3)
            return conv_matrix
        elif num_conv == 2:
            conv1_weight = conv1_matrix.reshape(-1, 1, self.inC, self.kernel_size).permute(0, 2, 1, 3)
            conv2_weight = conv2_matrix.reshape(self.outC, self.kernel_size, -1, 1).permute(0, 2, 1, 3)
            # -1的位置为rank, 不能事先确定rank的具体值
            return conv1_weight, conv2_weight

    def matrix_to_compact_conv(self, num_conv="one"):
        V, U = self.matrix_to_param()
        S = self.singular()
        V, U, S, new_rank = self.pruning_rank(V, U, S)
        if num_conv == "one":
            # i = self
            # M = (i.conv2_p @ torch.diag(i.singular()) @ i.conv1_p.transpose(1, 0))  # .reshape(self.outC,
            # # self.kernel_size, self.inC, self.kernel_size).permute(0, 2, 1, 3)
            # print(M)
            # if (U @ V).shape[0] != self.outC * self.kernel_size\
            #         or (U @ V).shape[1] != self.inC * self.kernel_size:
            #     print((U @ V).shape)
            M = (U @ V).reshape(self.outC, self.kernel_size, self.inC, self.kernel_size).permute(0, 2, 1, 3)
            padding = [max(self.padding_1), max(self.padding_2)]
            stride = [max(self.stride_1), max(self.stride_2)]
            self.new_conv = nn.Conv2d(out_channels=self.outC, in_channels=self.inC, kernel_size=self.kernel_size,
                                      padding=padding, stride=stride, bias=False)
            self.new_conv.weight.data = M
        elif num_conv == "two":
            V, U = self.param_to_conv_weight(V, U)
            conv1 = nn.Conv2d(out_channels=new_rank, in_channels=self.inC, kernel_size=[1, self.kernel_size],
                              padding=self.padding_1, stride=self.stride_1, bias=False)
            conv2 = nn.Conv2d(out_channels=self.outC, in_channels=new_rank, kernel_size=[self.kernel_size, 1],
                              padding=self.padding_2, stride=self.stride_2, bias=False)
            conv1.weight.data = V
            conv2.weight.data = U
            self.new_conv = nn.Sequential(conv1, conv2)
        else:
            raise ValueError
        delattr(self, "conv1_p")
        delattr(self, "conv2_p")
        delattr(self, "singular_p")
