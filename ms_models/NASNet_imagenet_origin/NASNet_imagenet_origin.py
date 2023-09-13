import numpy as np
import mindspore
import mindspore.numpy as ms_np
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, Parameter


class Module0(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_0_kernel_size, conv2d_0_stride):
        super(Module0, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=conv2d_0_kernel_size,
                                  stride=conv2d_0_stride,
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        return opt_relu_1


class MindSporeModel(nn.Cell):

    def __init__(self):
        super(MindSporeModel, self).__init__()
        self.transpose_0 = P.Transpose()
        self.module0_0 = Module0(conv2d_0_in_channels=3,
                                 conv2d_0_out_channels=32,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(2, 2))
        self.conv2d_3 = nn.Conv2d(in_channels=32,
                                  out_channels=11,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.pad_avgpool2d_4 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_4 = nn.AvgPool2d(kernel_size=(1, 1), stride=(2, 2))
        self.conv2d_5 = nn.Conv2d(in_channels=32,
                                  out_channels=32,
                                  kernel_size=(7, 7),
                                  stride=(2, 2),
                                  padding=(3, 3, 3, 3),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=32,
                                  has_bias=False)
        self.conv2d_6 = nn.Conv2d(in_channels=32,
                                  out_channels=32,
                                  kernel_size=(5, 5),
                                  stride=(2, 2),
                                  padding=(2, 2, 2, 2),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=32,
                                  has_bias=False)
        self.pad_7 = nn.Pad(paddings=((0, 0), (0, 0), (0, 1), (0, 1)), mode="CONSTANT")
        self.conv2d_8 = nn.Conv2d(in_channels=32,
                                  out_channels=32,
                                  kernel_size=(7, 7),
                                  stride=(2, 2),
                                  padding=(3, 3, 3, 3),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=32,
                                  has_bias=False)
        self.pad_9 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT")
        self.conv2d_11 = nn.Conv2d(in_channels=32,
                                   out_channels=11,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=False)
        self.conv2d_12 = nn.Conv2d(in_channels=32,
                                   out_channels=11,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_13 = nn.Conv2d(in_channels=32,
                                   out_channels=11,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.transpose_14 = P.Transpose()
        self.conv2d_15 = nn.Conv2d(in_channels=32,
                                   out_channels=11,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_10 = nn.ReLU()
        self.pad_maxpool2d_16 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_16 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.pad_avgpool2d_17 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_17 = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.relu_19 = nn.ReLU()
        self.relu_20 = nn.ReLU()
        self.relu_22 = nn.ReLU()
        self.stridedslice_21 = P.StridedSlice()
        self.stridedslice_21_begin = (0, 1, 1, 0)
        self.stridedslice_21_end = (1, 112, 112, 32)
        self.stridedslice_21_strides = (1, 1, 1, 1)
        self.conv2d_18 = nn.Conv2d(in_channels=11,
                                   out_channels=11,
                                   kernel_size=(5, 5),
                                   stride=(2, 2),
                                   padding=(2, 2, 2, 2),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=11,
                                   has_bias=False)
        self.conv2d_24 = nn.Conv2d(in_channels=11,
                                   out_channels=11,
                                   kernel_size=(7, 7),
                                   stride=(1, 1),
                                   padding=(3, 3, 3, 3),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=11,
                                   has_bias=False)
        self.conv2d_25 = nn.Conv2d(in_channels=11,
                                   out_channels=11,
                                   kernel_size=(5, 5),
                                   stride=(1, 1),
                                   padding=(2, 2, 2, 2),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=11,
                                   has_bias=False)
        self.conv2d_27 = nn.Conv2d(in_channels=11,
                                   out_channels=11,
                                   kernel_size=(7, 7),
                                   stride=(1, 1),
                                   padding=(3, 3, 3, 3),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=11,
                                   has_bias=False)
        self.transpose_26 = P.Transpose()
        self.conv2d_23 = nn.Conv2d(in_channels=11,
                                   out_channels=11,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_29 = nn.Conv2d(in_channels=11,
                                   out_channels=11,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_30 = nn.Conv2d(in_channels=11,
                                   out_channels=11,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_32 = nn.Conv2d(in_channels=11,
                                   out_channels=11,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.pad_avgpool2d_31 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_31 = nn.AvgPool2d(kernel_size=(1, 1), stride=(2, 2))
        self.relu_28 = nn.ReLU()
        self.conv2d_36 = nn.Conv2d(in_channels=32,
                                   out_channels=11,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=False)
        self.conv2d_33 = nn.Conv2d(in_channels=11,
                                   out_channels=11,
                                   kernel_size=(5, 5),
                                   stride=(1, 1),
                                   padding=(2, 2, 2, 2),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=11,
                                   has_bias=False)
        self.concat_38 = P.Concat(axis=1)
        self.conv2d_37 = nn.Conv2d(in_channels=11,
                                   out_channels=11,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.batchnorm2d_40 = nn.BatchNorm2d(num_features=22, eps=9.999999974752427e-07, momentum=0.9997000098228455)
        self.relu_43 = nn.ReLU()
        self.pad_avgpool2d_41 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_41 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.relu_42 = nn.ReLU()
        self.conv2d_46 = nn.Conv2d(in_channels=22,
                                   out_channels=22,
                                   kernel_size=(7, 7),
                                   stride=(2, 2),
                                   padding=(2, 3, 2, 3),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=22,
                                   has_bias=False)
        self.conv2d_47 = nn.Conv2d(in_channels=22,
                                   out_channels=22,
                                   kernel_size=(5, 5),
                                   stride=(2, 2),
                                   padding=(1, 2, 1, 2),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=22,
                                   has_bias=False)
        self.conv2d_48 = nn.Conv2d(in_channels=22,
                                   out_channels=22,
                                   kernel_size=(7, 7),
                                   stride=(2, 2),
                                   padding=(2, 3, 2, 3),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=22,
                                   has_bias=False)
        self.conv2d_45 = nn.Conv2d(in_channels=11,
                                   out_channels=11,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=11,
                                   has_bias=False)
        self.conv2d_50 = nn.Conv2d(in_channels=22,
                                   out_channels=22,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_51 = nn.Conv2d(in_channels=22,
                                   out_channels=22,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_52 = nn.Conv2d(in_channels=22,
                                   out_channels=22,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_49 = nn.Conv2d(in_channels=11,
                                   out_channels=11,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_54 = nn.ReLU()
        self.relu_55 = nn.ReLU()
        self.relu_56 = nn.ReLU()
        self.relu_53 = nn.ReLU()
        self.conv2d_58 = nn.Conv2d(in_channels=22,
                                   out_channels=22,
                                   kernel_size=(7, 7),
                                   stride=(1, 1),
                                   padding=(3, 3, 3, 3),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=22,
                                   has_bias=False)
        self.conv2d_59 = nn.Conv2d(in_channels=22,
                                   out_channels=22,
                                   kernel_size=(5, 5),
                                   stride=(1, 1),
                                   padding=(2, 2, 2, 2),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=22,
                                   has_bias=False)
        self.conv2d_60 = nn.Conv2d(in_channels=22,
                                   out_channels=22,
                                   kernel_size=(7, 7),
                                   stride=(1, 1),
                                   padding=(3, 3, 3, 3),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=22,
                                   has_bias=False)
        self.conv2d_57 = nn.Conv2d(in_channels=11,
                                   out_channels=11,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=11,
                                   has_bias=False)
        self.conv2d_62 = nn.Conv2d(in_channels=22,
                                   out_channels=22,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_63 = nn.Conv2d(in_channels=22,
                                   out_channels=22,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_64 = nn.Conv2d(in_channels=22,
                                   out_channels=22,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_61 = nn.Conv2d(in_channels=11,
                                   out_channels=11,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.concat_66 = P.Concat(axis=1)
        self.relu_67 = nn.ReLU()
        self.conv2d_68 = nn.Conv2d(in_channels=44,
                                   out_channels=22,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.pad_avgpool2d_69 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_69 = nn.AvgPool2d(kernel_size=(1, 1), stride=(2, 2))
        self.pad_70 = nn.Pad(paddings=((0, 0), (0, 0), (0, 1), (0, 1)), mode="CONSTANT")
        self.pad_71 = nn.Pad(paddings=((0, 0), (0, 0), (0, 1), (0, 1)), mode="CONSTANT")
        self.conv2d_73 = nn.Conv2d(in_channels=44,
                                   out_channels=22,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=False)
        self.transpose_74 = P.Transpose()
        self.relu_72 = nn.ReLU()
        self.pad_maxpool2d_75 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_75 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.pad_avgpool2d_76 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_76 = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.stridedslice_78 = P.StridedSlice()
        self.stridedslice_78_begin = (0, 1, 1, 0)
        self.stridedslice_78_end = (1, 57, 57, 44)
        self.stridedslice_78_strides = (1, 1, 1, 1)
        self.conv2d_77 = nn.Conv2d(in_channels=22,
                                   out_channels=22,
                                   kernel_size=(5, 5),
                                   stride=(2, 2),
                                   padding=(1, 2, 1, 2),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=22,
                                   has_bias=False)
        self.transpose_82 = P.Transpose()
        self.conv2d_81 = nn.Conv2d(in_channels=22,
                                   out_channels=22,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.pad_avgpool2d_84 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_84 = nn.AvgPool2d(kernel_size=(1, 1), stride=(2, 2))
        self.relu_83 = nn.ReLU()
        self.conv2d_86 = nn.Conv2d(in_channels=44,
                                   out_channels=22,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=False)
        self.conv2d_85 = nn.Conv2d(in_channels=22,
                                   out_channels=22,
                                   kernel_size=(5, 5),
                                   stride=(1, 1),
                                   padding=(2, 2, 2, 2),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=22,
                                   has_bias=False)
        self.concat_88 = P.Concat(axis=1)
        self.conv2d_87 = nn.Conv2d(in_channels=22,
                                   out_channels=22,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.batchnorm2d_90 = nn.BatchNorm2d(num_features=44, eps=9.999999974752427e-07, momentum=0.9997000098228455)
        self.pad_avgpool2d_91 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_91 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.transpose_93 = P.Transpose()
        self.relu_92 = nn.ReLU()
        self.conv2d_97 = nn.Conv2d(in_channels=22,
                                   out_channels=22,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=22,
                                   has_bias=False)
        self.relu_98 = nn.ReLU()
        self.pad_avgpool2d_94 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_94 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.pad_avgpool2d_95 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_95 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.conv2d_100 = nn.Conv2d(in_channels=22,
                                    out_channels=22,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_101 = P.Transpose()
        self.transpose_102 = P.Transpose()
        self.transpose_103 = P.Transpose()
        self.relu_105 = nn.ReLU()
        self.conv2d_106 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_107 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_108 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_109 = nn.Conv2d(in_channels=22,
                                    out_channels=22,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=22,
                                    has_bias=False)
        self.conv2d_110 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_111 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_112 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_104 = P.Transpose()
        self.conv2d_113 = nn.Conv2d(in_channels=22,
                                    out_channels=22,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_114 = nn.ReLU()
        self.relu_115 = nn.ReLU()
        self.relu_116 = nn.ReLU()
        self.conv2d_118 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_119 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_120 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.concat_121 = P.Concat(axis=1)
        self.conv2d_122 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_123 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_124 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_125 = nn.ReLU()
        self.conv2d_127 = nn.Conv2d(in_channels=88,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_128 = nn.Conv2d(in_channels=88,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_129 = P.Transpose()
        self.transpose_130 = P.Transpose()
        self.transpose_133 = P.Transpose()
        self.relu_135 = nn.ReLU()
        self.relu_137 = nn.ReLU()
        self.pad_avgpool2d_131 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_131 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.pad_avgpool2d_132 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_132 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.transpose_139 = P.Transpose()
        self.transpose_140 = P.Transpose()
        self.transpose_141 = P.Transpose()
        self.pad_avgpool2d_134 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_134 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.transpose_143 = P.Transpose()
        self.transpose_144 = P.Transpose()
        self.conv2d_146 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_147 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_148 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.transpose_138 = P.Transpose()
        self.conv2d_149 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_150 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_151 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_152 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_153 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_154 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_155 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_142 = P.Transpose()
        self.relu_156 = nn.ReLU()
        self.relu_157 = nn.ReLU()
        self.relu_158 = nn.ReLU()
        self.relu_159 = nn.ReLU()
        self.relu_160 = nn.ReLU()
        self.conv2d_161 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_162 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_163 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_164 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_165 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_166 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_167 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_168 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_169 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_170 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_173 = P.Transpose()
        self.transpose_174 = P.Transpose()
        self.transpose_175 = P.Transpose()
        self.concat_177 = P.Concat(axis=3)
        self.relu_178 = nn.ReLU()
        self.transpose_179 = P.Transpose()
        self.transpose_180 = P.Transpose()
        self.conv2d_181 = nn.Conv2d(in_channels=264,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_182 = nn.Conv2d(in_channels=264,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_183 = P.Transpose()
        self.transpose_185 = P.Transpose()
        self.relu_188 = nn.ReLU()
        self.relu_190 = nn.ReLU()
        self.pad_avgpool2d_184 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_184 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.pad_avgpool2d_186 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_186 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.pad_avgpool2d_187 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_187 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.transpose_192 = P.Transpose()
        self.transpose_193 = P.Transpose()
        self.transpose_195 = P.Transpose()
        self.transpose_196 = P.Transpose()
        self.transpose_197 = P.Transpose()
        self.transpose_189 = P.Transpose()
        self.conv2d_199 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_200 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_201 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_202 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_203 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_204 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_205 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_206 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_207 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_208 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_198 = P.Transpose()
        self.relu_209 = nn.ReLU()
        self.relu_210 = nn.ReLU()
        self.relu_211 = nn.ReLU()
        self.relu_212 = nn.ReLU()
        self.relu_213 = nn.ReLU()
        self.conv2d_214 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_215 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_216 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_217 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_218 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_219 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_220 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_221 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_222 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_223 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_225 = P.Transpose()
        self.transpose_227 = P.Transpose()
        self.transpose_229 = P.Transpose()
        self.concat_230 = P.Concat(axis=3)
        self.relu_231 = nn.ReLU()
        self.transpose_232 = P.Transpose()
        self.transpose_233 = P.Transpose()
        self.conv2d_234 = nn.Conv2d(in_channels=264,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_235 = nn.Conv2d(in_channels=264,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_236 = P.Transpose()
        self.transpose_239 = P.Transpose()
        self.relu_241 = nn.ReLU()
        self.relu_243 = nn.ReLU()
        self.pad_avgpool2d_237 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_237 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.pad_avgpool2d_238 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_238 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.transpose_245 = P.Transpose()
        self.transpose_246 = P.Transpose()
        self.transpose_247 = P.Transpose()
        self.pad_avgpool2d_240 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_240 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.transpose_249 = P.Transpose()
        self.transpose_250 = P.Transpose()
        self.conv2d_252 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_253 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_254 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.transpose_244 = P.Transpose()
        self.conv2d_255 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_256 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_257 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_258 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_259 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_260 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_261 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_248 = P.Transpose()
        self.relu_262 = nn.ReLU()
        self.relu_263 = nn.ReLU()
        self.relu_264 = nn.ReLU()
        self.relu_265 = nn.ReLU()
        self.relu_266 = nn.ReLU()
        self.conv2d_267 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_268 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_269 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_270 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_271 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_272 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_273 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_274 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_275 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_276 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_279 = P.Transpose()
        self.transpose_280 = P.Transpose()
        self.transpose_281 = P.Transpose()
        self.concat_283 = P.Concat(axis=3)
        self.relu_284 = nn.ReLU()
        self.transpose_285 = P.Transpose()
        self.transpose_286 = P.Transpose()
        self.conv2d_287 = nn.Conv2d(in_channels=264,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module0_1 = Module0(conv2d_0_in_channels=264,
                                 conv2d_0_out_channels=88,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1))
        self.transpose_289 = P.Transpose()
        self.conv2d_294 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(7, 7),
                                    stride=(2, 2),
                                    padding=(2, 3, 2, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_295 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(5, 5),
                                    stride=(2, 2),
                                    padding=(1, 2, 1, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_296 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(7, 7),
                                    stride=(2, 2),
                                    padding=(2, 3, 2, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.relu_292 = nn.ReLU()
        self.conv2d_300 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_301 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_302 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.pad_avgpool2d_290 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_290 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.transpose_297 = P.Transpose()
        self.transpose_298 = P.Transpose()
        self.relu_305 = nn.ReLU()
        self.relu_306 = nn.ReLU()
        self.relu_307 = nn.ReLU()
        self.transpose_293 = P.Transpose()
        self.conv2d_303 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_304 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_310 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(7, 7),
                                    stride=(1, 1),
                                    padding=(3, 3, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_311 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_312 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(7, 7),
                                    stride=(1, 1),
                                    padding=(3, 3, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_308 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_309 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_315 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_316 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_317 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_313 = nn.ReLU()
        self.relu_314 = nn.ReLU()
        self.conv2d_318 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_319 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=44,
                                    has_bias=False)
        self.conv2d_320 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_321 = nn.Conv2d(in_channels=44,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_323 = P.Transpose()
        self.transpose_324 = P.Transpose()
        self.concat_326 = P.Concat(axis=3)
        self.relu_327 = nn.ReLU()
        self.transpose_328 = P.Transpose()
        self.transpose_329 = P.Transpose()
        self.transpose_330 = P.Transpose()
        self.conv2d_331 = nn.Conv2d(in_channels=264,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.pad_avgpool2d_332 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_332 = nn.AvgPool2d(kernel_size=(1, 1), stride=(2, 2))
        self.pad_333 = nn.Pad(paddings=((0, 0), (0, 0), (0, 1), (0, 1)), mode="CONSTANT")
        self.pad_334 = nn.Pad(paddings=((0, 0), (0, 0), (0, 1), (0, 1)), mode="CONSTANT")
        self.conv2d_336 = nn.Conv2d(in_channels=264,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.transpose_337 = P.Transpose()
        self.relu_335 = nn.ReLU()
        self.pad_maxpool2d_338 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_338 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.pad_avgpool2d_339 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_339 = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.stridedslice_341 = P.StridedSlice()
        self.stridedslice_341_begin = (0, 1, 1, 0)
        self.stridedslice_341_end = (1, 29, 29, 264)
        self.stridedslice_341_strides = (1, 1, 1, 1)
        self.conv2d_340 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(5, 5),
                                    stride=(2, 2),
                                    padding=(1, 2, 1, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.transpose_345 = P.Transpose()
        self.conv2d_344 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.pad_avgpool2d_347 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_347 = nn.AvgPool2d(kernel_size=(1, 1), stride=(2, 2))
        self.relu_346 = nn.ReLU()
        self.conv2d_349 = nn.Conv2d(in_channels=264,
                                    out_channels=44,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.conv2d_348 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.concat_351 = P.Concat(axis=1)
        self.conv2d_350 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.batchnorm2d_353 = nn.BatchNorm2d(num_features=88, eps=9.999999974752427e-07, momentum=0.9997000098228455)
        self.pad_avgpool2d_354 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_354 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.transpose_356 = P.Transpose()
        self.relu_355 = nn.ReLU()
        self.conv2d_360 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.relu_361 = nn.ReLU()
        self.pad_avgpool2d_357 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_357 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.pad_avgpool2d_358 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_358 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.conv2d_363 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_364 = P.Transpose()
        self.transpose_365 = P.Transpose()
        self.transpose_366 = P.Transpose()
        self.relu_368 = nn.ReLU()
        self.conv2d_369 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_370 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_371 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_372 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_373 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_374 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_375 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_367 = P.Transpose()
        self.conv2d_376 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_377 = nn.ReLU()
        self.relu_378 = nn.ReLU()
        self.relu_379 = nn.ReLU()
        self.conv2d_381 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_382 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_383 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.concat_384 = P.Concat(axis=1)
        self.conv2d_385 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_386 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_387 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_388 = nn.ReLU()
        self.conv2d_390 = nn.Conv2d(in_channels=352,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_391 = nn.Conv2d(in_channels=352,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_392 = P.Transpose()
        self.transpose_393 = P.Transpose()
        self.transpose_396 = P.Transpose()
        self.relu_398 = nn.ReLU()
        self.relu_400 = nn.ReLU()
        self.pad_avgpool2d_394 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_394 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.pad_avgpool2d_395 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_395 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.transpose_402 = P.Transpose()
        self.transpose_403 = P.Transpose()
        self.transpose_404 = P.Transpose()
        self.pad_avgpool2d_397 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_397 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.transpose_406 = P.Transpose()
        self.transpose_407 = P.Transpose()
        self.conv2d_409 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_410 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_411 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.transpose_401 = P.Transpose()
        self.conv2d_412 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_413 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_414 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_415 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_416 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_417 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_418 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_405 = P.Transpose()
        self.relu_419 = nn.ReLU()
        self.relu_420 = nn.ReLU()
        self.relu_421 = nn.ReLU()
        self.relu_422 = nn.ReLU()
        self.relu_423 = nn.ReLU()
        self.conv2d_424 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_425 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_426 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_427 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_428 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_429 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_430 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_431 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_432 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_433 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_436 = P.Transpose()
        self.transpose_437 = P.Transpose()
        self.transpose_438 = P.Transpose()
        self.concat_440 = P.Concat(axis=3)
        self.relu_441 = nn.ReLU()
        self.transpose_442 = P.Transpose()
        self.transpose_443 = P.Transpose()
        self.conv2d_444 = nn.Conv2d(in_channels=528,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_445 = nn.Conv2d(in_channels=528,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_446 = P.Transpose()
        self.transpose_448 = P.Transpose()
        self.relu_451 = nn.ReLU()
        self.relu_453 = nn.ReLU()
        self.pad_avgpool2d_447 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_447 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.pad_avgpool2d_449 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_449 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.pad_avgpool2d_450 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_450 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.transpose_455 = P.Transpose()
        self.transpose_456 = P.Transpose()
        self.transpose_458 = P.Transpose()
        self.transpose_459 = P.Transpose()
        self.transpose_460 = P.Transpose()
        self.transpose_452 = P.Transpose()
        self.conv2d_462 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_463 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_464 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_465 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_466 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_467 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_468 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_469 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_470 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_471 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_461 = P.Transpose()
        self.relu_472 = nn.ReLU()
        self.relu_473 = nn.ReLU()
        self.relu_474 = nn.ReLU()
        self.relu_475 = nn.ReLU()
        self.relu_476 = nn.ReLU()
        self.conv2d_477 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_478 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_479 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_480 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_481 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_482 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_483 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_484 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_485 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_486 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_488 = P.Transpose()
        self.transpose_490 = P.Transpose()
        self.transpose_492 = P.Transpose()
        self.concat_493 = P.Concat(axis=3)
        self.relu_494 = nn.ReLU()
        self.transpose_495 = P.Transpose()
        self.transpose_496 = P.Transpose()
        self.conv2d_497 = nn.Conv2d(in_channels=528,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_498 = nn.Conv2d(in_channels=528,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_499 = P.Transpose()
        self.transpose_502 = P.Transpose()
        self.relu_504 = nn.ReLU()
        self.relu_506 = nn.ReLU()
        self.pad_avgpool2d_500 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_500 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.pad_avgpool2d_501 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_501 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.transpose_508 = P.Transpose()
        self.transpose_509 = P.Transpose()
        self.transpose_510 = P.Transpose()
        self.pad_avgpool2d_503 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_503 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.transpose_512 = P.Transpose()
        self.transpose_513 = P.Transpose()
        self.conv2d_515 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_516 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_517 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.transpose_507 = P.Transpose()
        self.conv2d_518 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_519 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_520 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_521 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_522 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_523 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_524 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_511 = P.Transpose()
        self.relu_525 = nn.ReLU()
        self.relu_526 = nn.ReLU()
        self.relu_527 = nn.ReLU()
        self.relu_528 = nn.ReLU()
        self.relu_529 = nn.ReLU()
        self.conv2d_530 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_531 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_532 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_533 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_534 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_535 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_536 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_537 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_538 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_539 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_542 = P.Transpose()
        self.transpose_543 = P.Transpose()
        self.transpose_544 = P.Transpose()
        self.concat_546 = P.Concat(axis=3)
        self.relu_547 = nn.ReLU()
        self.transpose_548 = P.Transpose()
        self.transpose_549 = P.Transpose()
        self.conv2d_550 = nn.Conv2d(in_channels=528,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module0_2 = Module0(conv2d_0_in_channels=528,
                                 conv2d_0_out_channels=176,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_stride=(1, 1))
        self.transpose_552 = P.Transpose()
        self.conv2d_557 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(7, 7),
                                    stride=(2, 2),
                                    padding=(2, 3, 2, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_558 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(5, 5),
                                    stride=(2, 2),
                                    padding=(1, 2, 1, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_559 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(7, 7),
                                    stride=(2, 2),
                                    padding=(2, 3, 2, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.relu_555 = nn.ReLU()
        self.conv2d_563 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_564 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_565 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.pad_avgpool2d_553 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_553 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.transpose_560 = P.Transpose()
        self.transpose_561 = P.Transpose()
        self.relu_568 = nn.ReLU()
        self.relu_569 = nn.ReLU()
        self.relu_570 = nn.ReLU()
        self.transpose_556 = P.Transpose()
        self.conv2d_566 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_567 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_573 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(7, 7),
                                    stride=(1, 1),
                                    padding=(3, 3, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_574 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_575 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(7, 7),
                                    stride=(1, 1),
                                    padding=(3, 3, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_571 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_572 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_578 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_579 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_580 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_576 = nn.ReLU()
        self.relu_577 = nn.ReLU()
        self.conv2d_581 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_582 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=88,
                                    has_bias=False)
        self.conv2d_583 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_584 = nn.Conv2d(in_channels=88,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_586 = P.Transpose()
        self.transpose_587 = P.Transpose()
        self.concat_589 = P.Concat(axis=3)
        self.relu_590 = nn.ReLU()
        self.transpose_591 = P.Transpose()
        self.transpose_592 = P.Transpose()
        self.transpose_593 = P.Transpose()
        self.conv2d_594 = nn.Conv2d(in_channels=528,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.pad_avgpool2d_595 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_595 = nn.AvgPool2d(kernel_size=(1, 1), stride=(2, 2))
        self.pad_596 = nn.Pad(paddings=((0, 0), (0, 0), (0, 1), (0, 1)), mode="CONSTANT")
        self.pad_597 = nn.Pad(paddings=((0, 0), (0, 0), (0, 1), (0, 1)), mode="CONSTANT")
        self.conv2d_599 = nn.Conv2d(in_channels=528,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.transpose_600 = P.Transpose()
        self.relu_598 = nn.ReLU()
        self.pad_maxpool2d_601 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_601 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.pad_avgpool2d_602 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_602 = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.stridedslice_604 = P.StridedSlice()
        self.stridedslice_604_begin = (0, 1, 1, 0)
        self.stridedslice_604_end = (1, 15, 15, 528)
        self.stridedslice_604_strides = (1, 1, 1, 1)
        self.conv2d_603 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(5, 5),
                                    stride=(2, 2),
                                    padding=(1, 2, 1, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.transpose_608 = P.Transpose()
        self.conv2d_607 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.pad_avgpool2d_610 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_610 = nn.AvgPool2d(kernel_size=(1, 1), stride=(2, 2))
        self.relu_609 = nn.ReLU()
        self.conv2d_612 = nn.Conv2d(in_channels=528,
                                    out_channels=88,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=False)
        self.conv2d_611 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.concat_614 = P.Concat(axis=1)
        self.conv2d_613 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.batchnorm2d_616 = nn.BatchNorm2d(num_features=176, eps=9.999999974752427e-07, momentum=0.9997000098228455)
        self.pad_avgpool2d_617 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_617 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.transpose_619 = P.Transpose()
        self.relu_618 = nn.ReLU()
        self.conv2d_623 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.relu_624 = nn.ReLU()
        self.pad_avgpool2d_620 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_620 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.pad_avgpool2d_621 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_621 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.conv2d_626 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_627 = P.Transpose()
        self.transpose_628 = P.Transpose()
        self.transpose_629 = P.Transpose()
        self.relu_631 = nn.ReLU()
        self.conv2d_632 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_633 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_634 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_635 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_636 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_637 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_638 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_630 = P.Transpose()
        self.conv2d_639 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_640 = nn.ReLU()
        self.relu_641 = nn.ReLU()
        self.relu_642 = nn.ReLU()
        self.conv2d_644 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_645 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_646 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.concat_647 = P.Concat(axis=1)
        self.conv2d_648 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_649 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_650 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_651 = nn.ReLU()
        self.conv2d_653 = nn.Conv2d(in_channels=704,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_654 = nn.Conv2d(in_channels=704,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_655 = P.Transpose()
        self.transpose_656 = P.Transpose()
        self.transpose_659 = P.Transpose()
        self.relu_661 = nn.ReLU()
        self.relu_663 = nn.ReLU()
        self.pad_avgpool2d_657 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_657 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.pad_avgpool2d_658 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_658 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.transpose_665 = P.Transpose()
        self.transpose_666 = P.Transpose()
        self.transpose_667 = P.Transpose()
        self.pad_avgpool2d_660 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_660 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.transpose_669 = P.Transpose()
        self.transpose_670 = P.Transpose()
        self.conv2d_672 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_673 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_674 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.transpose_664 = P.Transpose()
        self.conv2d_675 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_676 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_677 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_678 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_679 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_680 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_681 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_668 = P.Transpose()
        self.relu_682 = nn.ReLU()
        self.relu_683 = nn.ReLU()
        self.relu_684 = nn.ReLU()
        self.relu_685 = nn.ReLU()
        self.relu_686 = nn.ReLU()
        self.conv2d_687 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_688 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_689 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_690 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_691 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_692 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_693 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_694 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_695 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_696 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_699 = P.Transpose()
        self.transpose_700 = P.Transpose()
        self.transpose_701 = P.Transpose()
        self.concat_703 = P.Concat(axis=3)
        self.relu_704 = nn.ReLU()
        self.transpose_705 = P.Transpose()
        self.transpose_706 = P.Transpose()
        self.conv2d_707 = nn.Conv2d(in_channels=1056,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_708 = nn.Conv2d(in_channels=1056,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_709 = P.Transpose()
        self.transpose_711 = P.Transpose()
        self.relu_714 = nn.ReLU()
        self.relu_716 = nn.ReLU()
        self.pad_avgpool2d_710 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_710 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.pad_avgpool2d_712 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_712 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.pad_avgpool2d_713 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_713 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.transpose_718 = P.Transpose()
        self.transpose_719 = P.Transpose()
        self.transpose_721 = P.Transpose()
        self.transpose_722 = P.Transpose()
        self.transpose_723 = P.Transpose()
        self.transpose_715 = P.Transpose()
        self.conv2d_725 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_726 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_727 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_728 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_729 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_730 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_731 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_732 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_733 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_734 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_724 = P.Transpose()
        self.relu_735 = nn.ReLU()
        self.relu_736 = nn.ReLU()
        self.relu_737 = nn.ReLU()
        self.relu_738 = nn.ReLU()
        self.relu_739 = nn.ReLU()
        self.conv2d_740 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_741 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_742 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_743 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_744 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_745 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_746 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_747 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_748 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_749 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_751 = P.Transpose()
        self.transpose_753 = P.Transpose()
        self.transpose_755 = P.Transpose()
        self.concat_756 = P.Concat(axis=3)
        self.relu_757 = nn.ReLU()
        self.transpose_758 = P.Transpose()
        self.transpose_759 = P.Transpose()
        self.conv2d_760 = nn.Conv2d(in_channels=1056,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_761 = nn.Conv2d(in_channels=1056,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_762 = P.Transpose()
        self.transpose_765 = P.Transpose()
        self.relu_767 = nn.ReLU()
        self.relu_769 = nn.ReLU()
        self.pad_avgpool2d_763 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_763 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.pad_avgpool2d_764 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_764 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.transpose_771 = P.Transpose()
        self.transpose_772 = P.Transpose()
        self.transpose_773 = P.Transpose()
        self.pad_avgpool2d_766 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_766 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.transpose_775 = P.Transpose()
        self.transpose_776 = P.Transpose()
        self.conv2d_778 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_779 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_780 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.transpose_770 = P.Transpose()
        self.conv2d_781 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_782 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_783 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_784 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_785 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_786 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_787 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_774 = P.Transpose()
        self.relu_788 = nn.ReLU()
        self.relu_789 = nn.ReLU()
        self.relu_790 = nn.ReLU()
        self.relu_791 = nn.ReLU()
        self.relu_792 = nn.ReLU()
        self.conv2d_793 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_794 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_795 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_796 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_797 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_798 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_799 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_800 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_801 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_802 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_805 = P.Transpose()
        self.transpose_806 = P.Transpose()
        self.transpose_807 = P.Transpose()
        self.concat_809 = P.Concat(axis=3)
        self.relu_810 = nn.ReLU()
        self.transpose_811 = P.Transpose()
        self.conv2d_812 = nn.Conv2d(in_channels=1056,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_813 = P.Transpose()
        self.relu_815 = nn.ReLU()
        self.pad_avgpool2d_814 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_814 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.transpose_817 = P.Transpose()
        self.transpose_818 = P.Transpose()
        self.transpose_816 = P.Transpose()
        self.conv2d_820 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_821 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_822 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_823 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_824 = nn.ReLU()
        self.relu_825 = nn.ReLU()
        self.conv2d_826 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding=(2, 2, 2, 2),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_827 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=176,
                                    has_bias=False)
        self.conv2d_828 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_829 = nn.Conv2d(in_channels=176,
                                    out_channels=176,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.transpose_831 = P.Transpose()
        self.transpose_832 = P.Transpose()
        self.concat_834 = P.Concat(axis=3)
        self.relu_835 = nn.ReLU()
        self.transpose_836 = P.Transpose()
        self.avgpool2d_837 = nn.AvgPool2d(kernel_size=(7, 7))
        self.transpose_838 = P.Transpose()
        self.reshape_839 = P.Reshape()
        self.reshape_839_shape = tuple([1, 1056])
        self.matmul_840_w = Parameter(Tensor(np.random.uniform(0, 1, (1056, 1000)).astype(np.float32)), name=None)
        self.add_841_bias = Parameter(Tensor(np.random.uniform(0, 1, (1000, )).astype(np.float32)), name=None)
        self.softmax_842 = nn.Softmax(axis=-1)

    def construct(self, input_2):
        opt_transpose_0 = self.transpose_0(input_2, (0, 3, 1, 2))
        module0_0_opt = self.module0_0(opt_transpose_0)
        opt_conv2d_3 = self.conv2d_3(module0_0_opt)
        opt_avgpool2d_4 = self.pad_avgpool2d_4(module0_0_opt)
        opt_avgpool2d_4 = self.avgpool2d_4(opt_avgpool2d_4)
        opt_conv2d_5 = self.conv2d_5(module0_0_opt)
        opt_conv2d_6 = self.conv2d_6(module0_0_opt)
        opt_pad_7 = self.pad_7(module0_0_opt)
        opt_conv2d_8 = self.conv2d_8(module0_0_opt)
        opt_pad_9 = self.pad_9(opt_conv2d_3)
        opt_conv2d_11 = self.conv2d_11(opt_avgpool2d_4)
        opt_conv2d_12 = self.conv2d_12(opt_conv2d_5)
        opt_conv2d_13 = self.conv2d_13(opt_conv2d_6)
        opt_transpose_14 = self.transpose_14(opt_pad_7, (0, 2, 3, 1))
        opt_conv2d_15 = self.conv2d_15(opt_conv2d_8)
        opt_relu_10 = self.relu_10(opt_conv2d_3)
        opt_maxpool2d_16 = self.pad_maxpool2d_16(opt_pad_9)
        opt_maxpool2d_16 = self.maxpool2d_16(opt_maxpool2d_16)
        opt_avgpool2d_17 = self.pad_avgpool2d_17(opt_pad_9)
        opt_avgpool2d_17 = self.avgpool2d_17(opt_avgpool2d_17)
        opt_relu_19 = self.relu_19(opt_conv2d_12)
        opt_relu_20 = self.relu_20(opt_conv2d_13)
        opt_relu_22 = self.relu_22(opt_conv2d_15)
        opt_stridedslice_21 = self.stridedslice_21(opt_transpose_14, self.stridedslice_21_begin,
                                                   self.stridedslice_21_end, self.stridedslice_21_strides)
        opt_conv2d_18 = self.conv2d_18(opt_relu_10)
        opt_conv2d_24 = self.conv2d_24(opt_relu_19)
        opt_conv2d_25 = self.conv2d_25(opt_relu_20)
        opt_conv2d_27 = self.conv2d_27(opt_relu_22)
        opt_transpose_26 = self.transpose_26(opt_stridedslice_21, (0, 3, 1, 2))
        opt_conv2d_23 = self.conv2d_23(opt_conv2d_18)
        opt_conv2d_29 = self.conv2d_29(opt_conv2d_24)
        opt_conv2d_30 = self.conv2d_30(opt_conv2d_25)
        opt_conv2d_32 = self.conv2d_32(opt_conv2d_27)
        opt_avgpool2d_31 = self.pad_avgpool2d_31(opt_transpose_26)
        opt_avgpool2d_31 = self.avgpool2d_31(opt_avgpool2d_31)
        opt_relu_28 = self.relu_28(opt_conv2d_23)
        opt_add_34 = P.Add()(opt_maxpool2d_16, opt_conv2d_29)
        opt_add_35 = P.Add()(opt_avgpool2d_17, opt_conv2d_30)
        opt_conv2d_36 = self.conv2d_36(opt_avgpool2d_31)
        opt_conv2d_33 = self.conv2d_33(opt_relu_28)
        opt_concat_38 = self.concat_38((opt_conv2d_11, opt_conv2d_36, ))
        opt_conv2d_37 = self.conv2d_37(opt_conv2d_33)
        opt_batchnorm2d_40 = self.batchnorm2d_40(opt_concat_38)
        opt_add_39 = P.Add()(opt_conv2d_37, opt_conv2d_32)
        opt_relu_43 = self.relu_43(opt_batchnorm2d_40)
        opt_avgpool2d_41 = self.pad_avgpool2d_41(opt_add_39)
        opt_avgpool2d_41 = self.avgpool2d_41(opt_avgpool2d_41)
        opt_relu_42 = self.relu_42(opt_add_39)
        opt_conv2d_46 = self.conv2d_46(opt_relu_43)
        opt_conv2d_47 = self.conv2d_47(opt_relu_43)
        opt_add_44 = P.Add()(opt_add_34, opt_avgpool2d_41)
        opt_conv2d_48 = self.conv2d_48(opt_relu_43)
        opt_conv2d_45 = self.conv2d_45(opt_relu_42)
        opt_conv2d_50 = self.conv2d_50(opt_conv2d_46)
        opt_conv2d_51 = self.conv2d_51(opt_conv2d_47)
        opt_conv2d_52 = self.conv2d_52(opt_conv2d_48)
        opt_conv2d_49 = self.conv2d_49(opt_conv2d_45)
        opt_relu_54 = self.relu_54(opt_conv2d_50)
        opt_relu_55 = self.relu_55(opt_conv2d_51)
        opt_relu_56 = self.relu_56(opt_conv2d_52)
        opt_relu_53 = self.relu_53(opt_conv2d_49)
        opt_conv2d_58 = self.conv2d_58(opt_relu_54)
        opt_conv2d_59 = self.conv2d_59(opt_relu_55)
        opt_conv2d_60 = self.conv2d_60(opt_relu_56)
        opt_conv2d_57 = self.conv2d_57(opt_relu_53)
        opt_conv2d_62 = self.conv2d_62(opt_conv2d_58)
        opt_conv2d_63 = self.conv2d_63(opt_conv2d_59)
        opt_conv2d_64 = self.conv2d_64(opt_conv2d_60)
        opt_conv2d_61 = self.conv2d_61(opt_conv2d_57)
        opt_add_65 = P.Add()(opt_conv2d_61, opt_maxpool2d_16)
        opt_concat_66 = self.concat_66((opt_add_34, opt_add_35, opt_add_44, opt_add_65, ))
        opt_relu_67 = self.relu_67(opt_concat_66)
        opt_conv2d_68 = self.conv2d_68(opt_relu_67)
        opt_avgpool2d_69 = self.pad_avgpool2d_69(opt_relu_67)
        opt_avgpool2d_69 = self.avgpool2d_69(opt_avgpool2d_69)
        opt_pad_70 = self.pad_70(opt_relu_67)
        opt_pad_71 = self.pad_71(opt_conv2d_68)
        opt_conv2d_73 = self.conv2d_73(opt_avgpool2d_69)
        opt_transpose_74 = self.transpose_74(opt_pad_70, (0, 2, 3, 1))
        opt_relu_72 = self.relu_72(opt_conv2d_68)
        opt_maxpool2d_75 = self.pad_maxpool2d_75(opt_pad_71)
        opt_maxpool2d_75 = self.maxpool2d_75(opt_maxpool2d_75)
        opt_avgpool2d_76 = self.pad_avgpool2d_76(opt_pad_71)
        opt_avgpool2d_76 = self.avgpool2d_76(opt_avgpool2d_76)
        opt_stridedslice_78 = self.stridedslice_78(opt_transpose_74, self.stridedslice_78_begin,
                                                   self.stridedslice_78_end, self.stridedslice_78_strides)
        opt_conv2d_77 = self.conv2d_77(opt_relu_72)
        opt_add_79 = P.Add()(opt_maxpool2d_75, opt_conv2d_62)
        opt_add_80 = P.Add()(opt_avgpool2d_76, opt_conv2d_63)
        opt_transpose_82 = self.transpose_82(opt_stridedslice_78, (0, 3, 1, 2))
        opt_conv2d_81 = self.conv2d_81(opt_conv2d_77)
        opt_avgpool2d_84 = self.pad_avgpool2d_84(opt_transpose_82)
        opt_avgpool2d_84 = self.avgpool2d_84(opt_avgpool2d_84)
        opt_relu_83 = self.relu_83(opt_conv2d_81)
        opt_conv2d_86 = self.conv2d_86(opt_avgpool2d_84)
        opt_conv2d_85 = self.conv2d_85(opt_relu_83)
        opt_concat_88 = self.concat_88((opt_conv2d_73, opt_conv2d_86, ))
        opt_conv2d_87 = self.conv2d_87(opt_conv2d_85)
        opt_batchnorm2d_90 = self.batchnorm2d_90(opt_concat_88)
        opt_add_89 = P.Add()(opt_conv2d_87, opt_conv2d_64)
        opt_avgpool2d_91 = self.pad_avgpool2d_91(opt_add_89)
        opt_avgpool2d_91 = self.avgpool2d_91(opt_avgpool2d_91)
        opt_transpose_93 = self.transpose_93(opt_batchnorm2d_90, (0, 2, 3, 1))
        opt_relu_92 = self.relu_92(opt_add_89)
        opt_add_96 = P.Add()(opt_add_79, opt_avgpool2d_91)
        opt_conv2d_97 = self.conv2d_97(opt_relu_92)
        opt_relu_98 = self.relu_98(opt_transpose_93)
        opt_avgpool2d_94 = self.pad_avgpool2d_94(opt_batchnorm2d_90)
        opt_avgpool2d_94 = self.avgpool2d_94(opt_avgpool2d_94)
        opt_avgpool2d_95 = self.pad_avgpool2d_95(opt_batchnorm2d_90)
        opt_avgpool2d_95 = self.avgpool2d_95(opt_avgpool2d_95)
        opt_conv2d_100 = self.conv2d_100(opt_conv2d_97)
        opt_transpose_101 = self.transpose_101(opt_relu_98, (0, 3, 1, 2))
        opt_transpose_102 = self.transpose_102(opt_relu_98, (0, 3, 1, 2))
        opt_transpose_103 = self.transpose_103(opt_relu_98, (0, 3, 1, 2))
        opt_add_99 = P.Add()(opt_avgpool2d_94, opt_avgpool2d_95)
        opt_relu_105 = self.relu_105(opt_conv2d_100)
        opt_conv2d_106 = self.conv2d_106(opt_transpose_101)
        opt_conv2d_107 = self.conv2d_107(opt_transpose_102)
        opt_conv2d_108 = self.conv2d_108(opt_transpose_103)
        opt_conv2d_109 = self.conv2d_109(opt_relu_105)
        opt_conv2d_110 = self.conv2d_110(opt_conv2d_106)
        opt_conv2d_111 = self.conv2d_111(opt_conv2d_107)
        opt_conv2d_112 = self.conv2d_112(opt_conv2d_108)
        opt_transpose_104 = self.transpose_104(opt_add_99, (0, 2, 3, 1))
        opt_conv2d_113 = self.conv2d_113(opt_conv2d_109)
        opt_relu_114 = self.relu_114(opt_conv2d_110)
        opt_relu_115 = self.relu_115(opt_conv2d_111)
        opt_relu_116 = self.relu_116(opt_conv2d_112)
        opt_add_117 = P.Add()(opt_conv2d_113, opt_maxpool2d_75)
        opt_conv2d_118 = self.conv2d_118(opt_relu_114)
        opt_conv2d_119 = self.conv2d_119(opt_relu_115)
        opt_conv2d_120 = self.conv2d_120(opt_relu_116)
        opt_concat_121 = self.concat_121((opt_add_79, opt_add_80, opt_add_96, opt_add_117, ))
        opt_conv2d_122 = self.conv2d_122(opt_conv2d_118)
        opt_conv2d_123 = self.conv2d_123(opt_conv2d_119)
        opt_conv2d_124 = self.conv2d_124(opt_conv2d_120)
        opt_relu_125 = self.relu_125(opt_concat_121)
        opt_add_126 = P.Add()(opt_conv2d_123, opt_conv2d_124)
        opt_conv2d_127 = self.conv2d_127(opt_relu_125)
        opt_conv2d_128 = self.conv2d_128(opt_relu_125)
        opt_transpose_129 = self.transpose_129(opt_add_126, (0, 2, 3, 1))
        opt_transpose_130 = self.transpose_130(opt_conv2d_127, (0, 2, 3, 1))
        opt_transpose_133 = self.transpose_133(opt_conv2d_128, (0, 2, 3, 1))
        opt_relu_135 = self.relu_135(opt_transpose_130)
        opt_relu_137 = self.relu_137(opt_transpose_133)
        opt_avgpool2d_131 = self.pad_avgpool2d_131(opt_conv2d_127)
        opt_avgpool2d_131 = self.avgpool2d_131(opt_avgpool2d_131)
        opt_avgpool2d_132 = self.pad_avgpool2d_132(opt_conv2d_127)
        opt_avgpool2d_132 = self.avgpool2d_132(opt_avgpool2d_132)
        opt_transpose_139 = self.transpose_139(opt_relu_135, (0, 3, 1, 2))
        opt_transpose_140 = self.transpose_140(opt_relu_135, (0, 3, 1, 2))
        opt_transpose_141 = self.transpose_141(opt_relu_135, (0, 3, 1, 2))
        opt_avgpool2d_134 = self.pad_avgpool2d_134(opt_conv2d_128)
        opt_avgpool2d_134 = self.avgpool2d_134(opt_avgpool2d_134)
        opt_transpose_143 = self.transpose_143(opt_relu_137, (0, 3, 1, 2))
        opt_transpose_144 = self.transpose_144(opt_relu_137, (0, 3, 1, 2))
        opt_add_136 = P.Add()(opt_avgpool2d_131, opt_avgpool2d_132)
        opt_conv2d_146 = self.conv2d_146(opt_transpose_139)
        opt_conv2d_147 = self.conv2d_147(opt_transpose_140)
        opt_conv2d_148 = self.conv2d_148(opt_transpose_141)
        opt_transpose_138 = self.transpose_138(opt_avgpool2d_134, (0, 2, 3, 1))
        opt_conv2d_149 = self.conv2d_149(opt_transpose_143)
        opt_conv2d_150 = self.conv2d_150(opt_transpose_144)
        opt_conv2d_151 = self.conv2d_151(opt_conv2d_146)
        opt_conv2d_152 = self.conv2d_152(opt_conv2d_147)
        opt_conv2d_153 = self.conv2d_153(opt_conv2d_148)
        opt_add_145 = P.Add()(opt_transpose_138, opt_transpose_93)
        opt_conv2d_154 = self.conv2d_154(opt_conv2d_149)
        opt_conv2d_155 = self.conv2d_155(opt_conv2d_150)
        opt_transpose_142 = self.transpose_142(opt_add_136, (0, 2, 3, 1))
        opt_relu_156 = self.relu_156(opt_conv2d_151)
        opt_relu_157 = self.relu_157(opt_conv2d_152)
        opt_relu_158 = self.relu_158(opt_conv2d_153)
        opt_relu_159 = self.relu_159(opt_conv2d_154)
        opt_relu_160 = self.relu_160(opt_conv2d_155)
        opt_conv2d_161 = self.conv2d_161(opt_relu_156)
        opt_conv2d_162 = self.conv2d_162(opt_relu_157)
        opt_conv2d_163 = self.conv2d_163(opt_relu_158)
        opt_conv2d_164 = self.conv2d_164(opt_relu_159)
        opt_conv2d_165 = self.conv2d_165(opt_relu_160)
        opt_conv2d_166 = self.conv2d_166(opt_conv2d_161)
        opt_conv2d_167 = self.conv2d_167(opt_conv2d_162)
        opt_conv2d_168 = self.conv2d_168(opt_conv2d_163)
        opt_conv2d_169 = self.conv2d_169(opt_conv2d_164)
        opt_conv2d_170 = self.conv2d_170(opt_conv2d_165)
        opt_add_171 = P.Add()(opt_conv2d_167, opt_conv2d_168)
        opt_add_172 = P.Add()(opt_conv2d_169, opt_conv2d_122)
        opt_transpose_173 = self.transpose_173(opt_conv2d_170, (0, 2, 3, 1))
        opt_add_176 = P.Add()(opt_transpose_173, opt_transpose_133)
        opt_transpose_174 = self.transpose_174(opt_add_171, (0, 2, 3, 1))
        opt_transpose_175 = self.transpose_175(opt_add_172, (0, 2, 3, 1))
        opt_concat_177 = self.concat_177(
            (opt_transpose_93, opt_transpose_175, opt_transpose_129, opt_add_145, opt_transpose_104, opt_add_176,
             ))
        opt_relu_178 = self.relu_178(opt_concat_177)
        opt_transpose_179 = self.transpose_179(opt_relu_178, (0, 3, 1, 2))
        opt_transpose_180 = self.transpose_180(opt_relu_178, (0, 3, 1, 2))
        opt_conv2d_181 = self.conv2d_181(opt_transpose_179)
        opt_conv2d_182 = self.conv2d_182(opt_transpose_180)
        opt_transpose_183 = self.transpose_183(opt_conv2d_181, (0, 2, 3, 1))
        opt_transpose_185 = self.transpose_185(opt_conv2d_182, (0, 2, 3, 1))
        opt_relu_188 = self.relu_188(opt_transpose_183)
        opt_relu_190 = self.relu_190(opt_transpose_185)
        opt_avgpool2d_184 = self.pad_avgpool2d_184(opt_conv2d_181)
        opt_avgpool2d_184 = self.avgpool2d_184(opt_avgpool2d_184)
        opt_avgpool2d_186 = self.pad_avgpool2d_186(opt_conv2d_182)
        opt_avgpool2d_186 = self.avgpool2d_186(opt_avgpool2d_186)
        opt_avgpool2d_187 = self.pad_avgpool2d_187(opt_conv2d_182)
        opt_avgpool2d_187 = self.avgpool2d_187(opt_avgpool2d_187)
        opt_transpose_192 = self.transpose_192(opt_relu_188, (0, 3, 1, 2))
        opt_transpose_193 = self.transpose_193(opt_relu_188, (0, 3, 1, 2))
        opt_transpose_195 = self.transpose_195(opt_relu_190, (0, 3, 1, 2))
        opt_transpose_196 = self.transpose_196(opt_relu_190, (0, 3, 1, 2))
        opt_transpose_197 = self.transpose_197(opt_relu_190, (0, 3, 1, 2))
        opt_transpose_189 = self.transpose_189(opt_avgpool2d_184, (0, 2, 3, 1))
        opt_add_191 = P.Add()(opt_avgpool2d_186, opt_avgpool2d_187)
        opt_conv2d_199 = self.conv2d_199(opt_transpose_192)
        opt_conv2d_200 = self.conv2d_200(opt_transpose_193)
        opt_conv2d_201 = self.conv2d_201(opt_transpose_195)
        opt_conv2d_202 = self.conv2d_202(opt_transpose_196)
        opt_conv2d_203 = self.conv2d_203(opt_transpose_197)
        opt_add_194 = P.Add()(opt_transpose_189, opt_transpose_130)
        opt_conv2d_204 = self.conv2d_204(opt_conv2d_199)
        opt_conv2d_205 = self.conv2d_205(opt_conv2d_200)
        opt_conv2d_206 = self.conv2d_206(opt_conv2d_201)
        opt_conv2d_207 = self.conv2d_207(opt_conv2d_202)
        opt_conv2d_208 = self.conv2d_208(opt_conv2d_203)
        opt_transpose_198 = self.transpose_198(opt_add_191, (0, 2, 3, 1))
        opt_relu_209 = self.relu_209(opt_conv2d_204)
        opt_relu_210 = self.relu_210(opt_conv2d_205)
        opt_relu_211 = self.relu_211(opt_conv2d_206)
        opt_relu_212 = self.relu_212(opt_conv2d_207)
        opt_relu_213 = self.relu_213(opt_conv2d_208)
        opt_conv2d_214 = self.conv2d_214(opt_relu_209)
        opt_conv2d_215 = self.conv2d_215(opt_relu_210)
        opt_conv2d_216 = self.conv2d_216(opt_relu_211)
        opt_conv2d_217 = self.conv2d_217(opt_relu_212)
        opt_conv2d_218 = self.conv2d_218(opt_relu_213)
        opt_conv2d_219 = self.conv2d_219(opt_conv2d_214)
        opt_conv2d_220 = self.conv2d_220(opt_conv2d_215)
        opt_conv2d_221 = self.conv2d_221(opt_conv2d_216)
        opt_conv2d_222 = self.conv2d_222(opt_conv2d_217)
        opt_conv2d_223 = self.conv2d_223(opt_conv2d_218)
        opt_add_224 = P.Add()(opt_conv2d_219, opt_conv2d_166)
        opt_transpose_225 = self.transpose_225(opt_conv2d_220, (0, 2, 3, 1))
        opt_add_226 = P.Add()(opt_conv2d_222, opt_conv2d_223)
        opt_add_228 = P.Add()(opt_transpose_225, opt_transpose_183)
        opt_transpose_227 = self.transpose_227(opt_add_224, (0, 2, 3, 1))
        opt_transpose_229 = self.transpose_229(opt_add_226, (0, 2, 3, 1))
        opt_concat_230 = self.concat_230(
            (opt_transpose_130, opt_transpose_227, opt_transpose_174, opt_add_194, opt_transpose_142, opt_add_228,
             ))
        opt_relu_231 = self.relu_231(opt_concat_230)
        opt_transpose_232 = self.transpose_232(opt_relu_231, (0, 3, 1, 2))
        opt_transpose_233 = self.transpose_233(opt_relu_231, (0, 3, 1, 2))
        opt_conv2d_234 = self.conv2d_234(opt_transpose_232)
        opt_conv2d_235 = self.conv2d_235(opt_transpose_233)
        opt_transpose_236 = self.transpose_236(opt_conv2d_234, (0, 2, 3, 1))
        opt_transpose_239 = self.transpose_239(opt_conv2d_235, (0, 2, 3, 1))
        opt_relu_241 = self.relu_241(opt_transpose_236)
        opt_relu_243 = self.relu_243(opt_transpose_239)
        opt_avgpool2d_237 = self.pad_avgpool2d_237(opt_conv2d_234)
        opt_avgpool2d_237 = self.avgpool2d_237(opt_avgpool2d_237)
        opt_avgpool2d_238 = self.pad_avgpool2d_238(opt_conv2d_234)
        opt_avgpool2d_238 = self.avgpool2d_238(opt_avgpool2d_238)
        opt_transpose_245 = self.transpose_245(opt_relu_241, (0, 3, 1, 2))
        opt_transpose_246 = self.transpose_246(opt_relu_241, (0, 3, 1, 2))
        opt_transpose_247 = self.transpose_247(opt_relu_241, (0, 3, 1, 2))
        opt_avgpool2d_240 = self.pad_avgpool2d_240(opt_conv2d_235)
        opt_avgpool2d_240 = self.avgpool2d_240(opt_avgpool2d_240)
        opt_transpose_249 = self.transpose_249(opt_relu_243, (0, 3, 1, 2))
        opt_transpose_250 = self.transpose_250(opt_relu_243, (0, 3, 1, 2))
        opt_add_242 = P.Add()(opt_avgpool2d_237, opt_avgpool2d_238)
        opt_conv2d_252 = self.conv2d_252(opt_transpose_245)
        opt_conv2d_253 = self.conv2d_253(opt_transpose_246)
        opt_conv2d_254 = self.conv2d_254(opt_transpose_247)
        opt_transpose_244 = self.transpose_244(opt_avgpool2d_240, (0, 2, 3, 1))
        opt_conv2d_255 = self.conv2d_255(opt_transpose_249)
        opt_conv2d_256 = self.conv2d_256(opt_transpose_250)
        opt_conv2d_257 = self.conv2d_257(opt_conv2d_252)
        opt_conv2d_258 = self.conv2d_258(opt_conv2d_253)
        opt_conv2d_259 = self.conv2d_259(opt_conv2d_254)
        opt_add_251 = P.Add()(opt_transpose_244, opt_transpose_185)
        opt_conv2d_260 = self.conv2d_260(opt_conv2d_255)
        opt_conv2d_261 = self.conv2d_261(opt_conv2d_256)
        opt_transpose_248 = self.transpose_248(opt_add_242, (0, 2, 3, 1))
        opt_relu_262 = self.relu_262(opt_conv2d_257)
        opt_relu_263 = self.relu_263(opt_conv2d_258)
        opt_relu_264 = self.relu_264(opt_conv2d_259)
        opt_relu_265 = self.relu_265(opt_conv2d_260)
        opt_relu_266 = self.relu_266(opt_conv2d_261)
        opt_conv2d_267 = self.conv2d_267(opt_relu_262)
        opt_conv2d_268 = self.conv2d_268(opt_relu_263)
        opt_conv2d_269 = self.conv2d_269(opt_relu_264)
        opt_conv2d_270 = self.conv2d_270(opt_relu_265)
        opt_conv2d_271 = self.conv2d_271(opt_relu_266)
        opt_conv2d_272 = self.conv2d_272(opt_conv2d_267)
        opt_conv2d_273 = self.conv2d_273(opt_conv2d_268)
        opt_conv2d_274 = self.conv2d_274(opt_conv2d_269)
        opt_conv2d_275 = self.conv2d_275(opt_conv2d_270)
        opt_conv2d_276 = self.conv2d_276(opt_conv2d_271)
        opt_add_277 = P.Add()(opt_conv2d_273, opt_conv2d_274)
        opt_add_278 = P.Add()(opt_conv2d_275, opt_conv2d_221)
        opt_transpose_279 = self.transpose_279(opt_conv2d_276, (0, 2, 3, 1))
        opt_add_282 = P.Add()(opt_transpose_279, opt_transpose_239)
        opt_transpose_280 = self.transpose_280(opt_add_277, (0, 2, 3, 1))
        opt_transpose_281 = self.transpose_281(opt_add_278, (0, 2, 3, 1))
        opt_concat_283 = self.concat_283(
            (opt_transpose_185, opt_transpose_281, opt_transpose_229, opt_add_251, opt_transpose_198, opt_add_282,
             ))
        opt_relu_284 = self.relu_284(opt_concat_283)
        opt_transpose_285 = self.transpose_285(opt_relu_284, (0, 3, 1, 2))
        opt_transpose_286 = self.transpose_286(opt_relu_284, (0, 3, 1, 2))
        opt_conv2d_287 = self.conv2d_287(opt_transpose_285)
        module0_1_opt = self.module0_1(opt_transpose_286)
        opt_transpose_289 = self.transpose_289(opt_conv2d_287, (0, 2, 3, 1))
        opt_conv2d_294 = self.conv2d_294(module0_1_opt)
        opt_conv2d_295 = self.conv2d_295(module0_1_opt)
        opt_conv2d_296 = self.conv2d_296(module0_1_opt)
        opt_relu_292 = self.relu_292(opt_transpose_289)
        opt_conv2d_300 = self.conv2d_300(opt_conv2d_294)
        opt_conv2d_301 = self.conv2d_301(opt_conv2d_295)
        opt_conv2d_302 = self.conv2d_302(opt_conv2d_296)
        opt_avgpool2d_290 = self.pad_avgpool2d_290(opt_conv2d_287)
        opt_avgpool2d_290 = self.avgpool2d_290(opt_avgpool2d_290)
        opt_transpose_297 = self.transpose_297(opt_relu_292, (0, 3, 1, 2))
        opt_transpose_298 = self.transpose_298(opt_relu_292, (0, 3, 1, 2))
        opt_relu_305 = self.relu_305(opt_conv2d_300)
        opt_relu_306 = self.relu_306(opt_conv2d_301)
        opt_relu_307 = self.relu_307(opt_conv2d_302)
        opt_transpose_293 = self.transpose_293(opt_avgpool2d_290, (0, 2, 3, 1))
        opt_conv2d_303 = self.conv2d_303(opt_transpose_297)
        opt_conv2d_304 = self.conv2d_304(opt_transpose_298)
        opt_conv2d_310 = self.conv2d_310(opt_relu_305)
        opt_conv2d_311 = self.conv2d_311(opt_relu_306)
        opt_conv2d_312 = self.conv2d_312(opt_relu_307)
        opt_add_299 = P.Add()(opt_transpose_293, opt_transpose_236)
        opt_conv2d_308 = self.conv2d_308(opt_conv2d_303)
        opt_conv2d_309 = self.conv2d_309(opt_conv2d_304)
        opt_conv2d_315 = self.conv2d_315(opt_conv2d_310)
        opt_conv2d_316 = self.conv2d_316(opt_conv2d_311)
        opt_conv2d_317 = self.conv2d_317(opt_conv2d_312)
        opt_relu_313 = self.relu_313(opt_conv2d_308)
        opt_relu_314 = self.relu_314(opt_conv2d_309)
        opt_conv2d_318 = self.conv2d_318(opt_relu_313)
        opt_conv2d_319 = self.conv2d_319(opt_relu_314)
        opt_conv2d_320 = self.conv2d_320(opt_conv2d_318)
        opt_conv2d_321 = self.conv2d_321(opt_conv2d_319)
        opt_add_322 = P.Add()(opt_conv2d_320, opt_conv2d_272)
        opt_transpose_323 = self.transpose_323(opt_conv2d_321, (0, 2, 3, 1))
        opt_add_325 = P.Add()(opt_transpose_323, opt_transpose_289)
        opt_transpose_324 = self.transpose_324(opt_add_322, (0, 2, 3, 1))
        opt_concat_326 = self.concat_326(
            (opt_transpose_236, opt_transpose_324, opt_transpose_280, opt_add_299, opt_transpose_248, opt_add_325,
             ))
        opt_relu_327 = self.relu_327(opt_concat_326)
        opt_transpose_328 = self.transpose_328(opt_relu_327, (0, 3, 1, 2))
        opt_transpose_329 = self.transpose_329(opt_relu_327, (0, 3, 1, 2))
        opt_transpose_330 = self.transpose_330(opt_relu_327, (0, 3, 1, 2))
        opt_conv2d_331 = self.conv2d_331(opt_transpose_328)
        opt_avgpool2d_332 = self.pad_avgpool2d_332(opt_transpose_329)
        opt_avgpool2d_332 = self.avgpool2d_332(opt_avgpool2d_332)
        opt_pad_333 = self.pad_333(opt_transpose_330)
        opt_pad_334 = self.pad_334(opt_conv2d_331)
        opt_conv2d_336 = self.conv2d_336(opt_avgpool2d_332)
        opt_transpose_337 = self.transpose_337(opt_pad_333, (0, 2, 3, 1))
        opt_relu_335 = self.relu_335(opt_conv2d_331)
        opt_maxpool2d_338 = self.pad_maxpool2d_338(opt_pad_334)
        opt_maxpool2d_338 = self.maxpool2d_338(opt_maxpool2d_338)
        opt_avgpool2d_339 = self.pad_avgpool2d_339(opt_pad_334)
        opt_avgpool2d_339 = self.avgpool2d_339(opt_avgpool2d_339)
        opt_stridedslice_341 = self.stridedslice_341(opt_transpose_337, self.stridedslice_341_begin,
                                                     self.stridedslice_341_end, self.stridedslice_341_strides)
        opt_conv2d_340 = self.conv2d_340(opt_relu_335)
        opt_add_342 = P.Add()(opt_maxpool2d_338, opt_conv2d_315)
        opt_add_343 = P.Add()(opt_avgpool2d_339, opt_conv2d_316)
        opt_transpose_345 = self.transpose_345(opt_stridedslice_341, (0, 3, 1, 2))
        opt_conv2d_344 = self.conv2d_344(opt_conv2d_340)
        opt_avgpool2d_347 = self.pad_avgpool2d_347(opt_transpose_345)
        opt_avgpool2d_347 = self.avgpool2d_347(opt_avgpool2d_347)
        opt_relu_346 = self.relu_346(opt_conv2d_344)
        opt_conv2d_349 = self.conv2d_349(opt_avgpool2d_347)
        opt_conv2d_348 = self.conv2d_348(opt_relu_346)
        opt_concat_351 = self.concat_351((opt_conv2d_336, opt_conv2d_349, ))
        opt_conv2d_350 = self.conv2d_350(opt_conv2d_348)
        opt_batchnorm2d_353 = self.batchnorm2d_353(opt_concat_351)
        opt_add_352 = P.Add()(opt_conv2d_350, opt_conv2d_317)
        opt_avgpool2d_354 = self.pad_avgpool2d_354(opt_add_352)
        opt_avgpool2d_354 = self.avgpool2d_354(opt_avgpool2d_354)
        opt_transpose_356 = self.transpose_356(opt_batchnorm2d_353, (0, 2, 3, 1))
        opt_relu_355 = self.relu_355(opt_add_352)
        opt_add_359 = P.Add()(opt_add_342, opt_avgpool2d_354)
        opt_conv2d_360 = self.conv2d_360(opt_relu_355)
        opt_relu_361 = self.relu_361(opt_transpose_356)
        opt_avgpool2d_357 = self.pad_avgpool2d_357(opt_batchnorm2d_353)
        opt_avgpool2d_357 = self.avgpool2d_357(opt_avgpool2d_357)
        opt_avgpool2d_358 = self.pad_avgpool2d_358(opt_batchnorm2d_353)
        opt_avgpool2d_358 = self.avgpool2d_358(opt_avgpool2d_358)
        opt_conv2d_363 = self.conv2d_363(opt_conv2d_360)
        opt_transpose_364 = self.transpose_364(opt_relu_361, (0, 3, 1, 2))
        opt_transpose_365 = self.transpose_365(opt_relu_361, (0, 3, 1, 2))
        opt_transpose_366 = self.transpose_366(opt_relu_361, (0, 3, 1, 2))
        opt_add_362 = P.Add()(opt_avgpool2d_357, opt_avgpool2d_358)
        opt_relu_368 = self.relu_368(opt_conv2d_363)
        opt_conv2d_369 = self.conv2d_369(opt_transpose_364)
        opt_conv2d_370 = self.conv2d_370(opt_transpose_365)
        opt_conv2d_371 = self.conv2d_371(opt_transpose_366)
        opt_conv2d_372 = self.conv2d_372(opt_relu_368)
        opt_conv2d_373 = self.conv2d_373(opt_conv2d_369)
        opt_conv2d_374 = self.conv2d_374(opt_conv2d_370)
        opt_conv2d_375 = self.conv2d_375(opt_conv2d_371)
        opt_transpose_367 = self.transpose_367(opt_add_362, (0, 2, 3, 1))
        opt_conv2d_376 = self.conv2d_376(opt_conv2d_372)
        opt_relu_377 = self.relu_377(opt_conv2d_373)
        opt_relu_378 = self.relu_378(opt_conv2d_374)
        opt_relu_379 = self.relu_379(opt_conv2d_375)
        opt_add_380 = P.Add()(opt_conv2d_376, opt_maxpool2d_338)
        opt_conv2d_381 = self.conv2d_381(opt_relu_377)
        opt_conv2d_382 = self.conv2d_382(opt_relu_378)
        opt_conv2d_383 = self.conv2d_383(opt_relu_379)
        opt_concat_384 = self.concat_384((opt_add_342, opt_add_343, opt_add_359, opt_add_380, ))
        opt_conv2d_385 = self.conv2d_385(opt_conv2d_381)
        opt_conv2d_386 = self.conv2d_386(opt_conv2d_382)
        opt_conv2d_387 = self.conv2d_387(opt_conv2d_383)
        opt_relu_388 = self.relu_388(opt_concat_384)
        opt_add_389 = P.Add()(opt_conv2d_386, opt_conv2d_387)
        opt_conv2d_390 = self.conv2d_390(opt_relu_388)
        opt_conv2d_391 = self.conv2d_391(opt_relu_388)
        opt_transpose_392 = self.transpose_392(opt_add_389, (0, 2, 3, 1))
        opt_transpose_393 = self.transpose_393(opt_conv2d_390, (0, 2, 3, 1))
        opt_transpose_396 = self.transpose_396(opt_conv2d_391, (0, 2, 3, 1))
        opt_relu_398 = self.relu_398(opt_transpose_393)
        opt_relu_400 = self.relu_400(opt_transpose_396)
        opt_avgpool2d_394 = self.pad_avgpool2d_394(opt_conv2d_390)
        opt_avgpool2d_394 = self.avgpool2d_394(opt_avgpool2d_394)
        opt_avgpool2d_395 = self.pad_avgpool2d_395(opt_conv2d_390)
        opt_avgpool2d_395 = self.avgpool2d_395(opt_avgpool2d_395)
        opt_transpose_402 = self.transpose_402(opt_relu_398, (0, 3, 1, 2))
        opt_transpose_403 = self.transpose_403(opt_relu_398, (0, 3, 1, 2))
        opt_transpose_404 = self.transpose_404(opt_relu_398, (0, 3, 1, 2))
        opt_avgpool2d_397 = self.pad_avgpool2d_397(opt_conv2d_391)
        opt_avgpool2d_397 = self.avgpool2d_397(opt_avgpool2d_397)
        opt_transpose_406 = self.transpose_406(opt_relu_400, (0, 3, 1, 2))
        opt_transpose_407 = self.transpose_407(opt_relu_400, (0, 3, 1, 2))
        opt_add_399 = P.Add()(opt_avgpool2d_394, opt_avgpool2d_395)
        opt_conv2d_409 = self.conv2d_409(opt_transpose_402)
        opt_conv2d_410 = self.conv2d_410(opt_transpose_403)
        opt_conv2d_411 = self.conv2d_411(opt_transpose_404)
        opt_transpose_401 = self.transpose_401(opt_avgpool2d_397, (0, 2, 3, 1))
        opt_conv2d_412 = self.conv2d_412(opt_transpose_406)
        opt_conv2d_413 = self.conv2d_413(opt_transpose_407)
        opt_conv2d_414 = self.conv2d_414(opt_conv2d_409)
        opt_conv2d_415 = self.conv2d_415(opt_conv2d_410)
        opt_conv2d_416 = self.conv2d_416(opt_conv2d_411)
        opt_add_408 = P.Add()(opt_transpose_401, opt_transpose_356)
        opt_conv2d_417 = self.conv2d_417(opt_conv2d_412)
        opt_conv2d_418 = self.conv2d_418(opt_conv2d_413)
        opt_transpose_405 = self.transpose_405(opt_add_399, (0, 2, 3, 1))
        opt_relu_419 = self.relu_419(opt_conv2d_414)
        opt_relu_420 = self.relu_420(opt_conv2d_415)
        opt_relu_421 = self.relu_421(opt_conv2d_416)
        opt_relu_422 = self.relu_422(opt_conv2d_417)
        opt_relu_423 = self.relu_423(opt_conv2d_418)
        opt_conv2d_424 = self.conv2d_424(opt_relu_419)
        opt_conv2d_425 = self.conv2d_425(opt_relu_420)
        opt_conv2d_426 = self.conv2d_426(opt_relu_421)
        opt_conv2d_427 = self.conv2d_427(opt_relu_422)
        opt_conv2d_428 = self.conv2d_428(opt_relu_423)
        opt_conv2d_429 = self.conv2d_429(opt_conv2d_424)
        opt_conv2d_430 = self.conv2d_430(opt_conv2d_425)
        opt_conv2d_431 = self.conv2d_431(opt_conv2d_426)
        opt_conv2d_432 = self.conv2d_432(opt_conv2d_427)
        opt_conv2d_433 = self.conv2d_433(opt_conv2d_428)
        opt_add_434 = P.Add()(opt_conv2d_430, opt_conv2d_431)
        opt_add_435 = P.Add()(opt_conv2d_432, opt_conv2d_385)
        opt_transpose_436 = self.transpose_436(opt_conv2d_433, (0, 2, 3, 1))
        opt_add_439 = P.Add()(opt_transpose_436, opt_transpose_396)
        opt_transpose_437 = self.transpose_437(opt_add_434, (0, 2, 3, 1))
        opt_transpose_438 = self.transpose_438(opt_add_435, (0, 2, 3, 1))
        opt_concat_440 = self.concat_440(
            (opt_transpose_356, opt_transpose_438, opt_transpose_392, opt_add_408, opt_transpose_367, opt_add_439,
             ))
        opt_relu_441 = self.relu_441(opt_concat_440)
        opt_transpose_442 = self.transpose_442(opt_relu_441, (0, 3, 1, 2))
        opt_transpose_443 = self.transpose_443(opt_relu_441, (0, 3, 1, 2))
        opt_conv2d_444 = self.conv2d_444(opt_transpose_442)
        opt_conv2d_445 = self.conv2d_445(opt_transpose_443)
        opt_transpose_446 = self.transpose_446(opt_conv2d_444, (0, 2, 3, 1))
        opt_transpose_448 = self.transpose_448(opt_conv2d_445, (0, 2, 3, 1))
        opt_relu_451 = self.relu_451(opt_transpose_446)
        opt_relu_453 = self.relu_453(opt_transpose_448)
        opt_avgpool2d_447 = self.pad_avgpool2d_447(opt_conv2d_444)
        opt_avgpool2d_447 = self.avgpool2d_447(opt_avgpool2d_447)
        opt_avgpool2d_449 = self.pad_avgpool2d_449(opt_conv2d_445)
        opt_avgpool2d_449 = self.avgpool2d_449(opt_avgpool2d_449)
        opt_avgpool2d_450 = self.pad_avgpool2d_450(opt_conv2d_445)
        opt_avgpool2d_450 = self.avgpool2d_450(opt_avgpool2d_450)
        opt_transpose_455 = self.transpose_455(opt_relu_451, (0, 3, 1, 2))
        opt_transpose_456 = self.transpose_456(opt_relu_451, (0, 3, 1, 2))
        opt_transpose_458 = self.transpose_458(opt_relu_453, (0, 3, 1, 2))
        opt_transpose_459 = self.transpose_459(opt_relu_453, (0, 3, 1, 2))
        opt_transpose_460 = self.transpose_460(opt_relu_453, (0, 3, 1, 2))
        opt_transpose_452 = self.transpose_452(opt_avgpool2d_447, (0, 2, 3, 1))
        opt_add_454 = P.Add()(opt_avgpool2d_449, opt_avgpool2d_450)
        opt_conv2d_462 = self.conv2d_462(opt_transpose_455)
        opt_conv2d_463 = self.conv2d_463(opt_transpose_456)
        opt_conv2d_464 = self.conv2d_464(opt_transpose_458)
        opt_conv2d_465 = self.conv2d_465(opt_transpose_459)
        opt_conv2d_466 = self.conv2d_466(opt_transpose_460)
        opt_add_457 = P.Add()(opt_transpose_452, opt_transpose_393)
        opt_conv2d_467 = self.conv2d_467(opt_conv2d_462)
        opt_conv2d_468 = self.conv2d_468(opt_conv2d_463)
        opt_conv2d_469 = self.conv2d_469(opt_conv2d_464)
        opt_conv2d_470 = self.conv2d_470(opt_conv2d_465)
        opt_conv2d_471 = self.conv2d_471(opt_conv2d_466)
        opt_transpose_461 = self.transpose_461(opt_add_454, (0, 2, 3, 1))
        opt_relu_472 = self.relu_472(opt_conv2d_467)
        opt_relu_473 = self.relu_473(opt_conv2d_468)
        opt_relu_474 = self.relu_474(opt_conv2d_469)
        opt_relu_475 = self.relu_475(opt_conv2d_470)
        opt_relu_476 = self.relu_476(opt_conv2d_471)
        opt_conv2d_477 = self.conv2d_477(opt_relu_472)
        opt_conv2d_478 = self.conv2d_478(opt_relu_473)
        opt_conv2d_479 = self.conv2d_479(opt_relu_474)
        opt_conv2d_480 = self.conv2d_480(opt_relu_475)
        opt_conv2d_481 = self.conv2d_481(opt_relu_476)
        opt_conv2d_482 = self.conv2d_482(opt_conv2d_477)
        opt_conv2d_483 = self.conv2d_483(opt_conv2d_478)
        opt_conv2d_484 = self.conv2d_484(opt_conv2d_479)
        opt_conv2d_485 = self.conv2d_485(opt_conv2d_480)
        opt_conv2d_486 = self.conv2d_486(opt_conv2d_481)
        opt_add_487 = P.Add()(opt_conv2d_482, opt_conv2d_429)
        opt_transpose_488 = self.transpose_488(opt_conv2d_483, (0, 2, 3, 1))
        opt_add_489 = P.Add()(opt_conv2d_485, opt_conv2d_486)
        opt_add_491 = P.Add()(opt_transpose_488, opt_transpose_446)
        opt_transpose_490 = self.transpose_490(opt_add_487, (0, 2, 3, 1))
        opt_transpose_492 = self.transpose_492(opt_add_489, (0, 2, 3, 1))
        opt_concat_493 = self.concat_493(
            (opt_transpose_393, opt_transpose_490, opt_transpose_437, opt_add_457, opt_transpose_405, opt_add_491,
             ))
        opt_relu_494 = self.relu_494(opt_concat_493)
        opt_transpose_495 = self.transpose_495(opt_relu_494, (0, 3, 1, 2))
        opt_transpose_496 = self.transpose_496(opt_relu_494, (0, 3, 1, 2))
        opt_conv2d_497 = self.conv2d_497(opt_transpose_495)
        opt_conv2d_498 = self.conv2d_498(opt_transpose_496)
        opt_transpose_499 = self.transpose_499(opt_conv2d_497, (0, 2, 3, 1))
        opt_transpose_502 = self.transpose_502(opt_conv2d_498, (0, 2, 3, 1))
        opt_relu_504 = self.relu_504(opt_transpose_499)
        opt_relu_506 = self.relu_506(opt_transpose_502)
        opt_avgpool2d_500 = self.pad_avgpool2d_500(opt_conv2d_497)
        opt_avgpool2d_500 = self.avgpool2d_500(opt_avgpool2d_500)
        opt_avgpool2d_501 = self.pad_avgpool2d_501(opt_conv2d_497)
        opt_avgpool2d_501 = self.avgpool2d_501(opt_avgpool2d_501)
        opt_transpose_508 = self.transpose_508(opt_relu_504, (0, 3, 1, 2))
        opt_transpose_509 = self.transpose_509(opt_relu_504, (0, 3, 1, 2))
        opt_transpose_510 = self.transpose_510(opt_relu_504, (0, 3, 1, 2))
        opt_avgpool2d_503 = self.pad_avgpool2d_503(opt_conv2d_498)
        opt_avgpool2d_503 = self.avgpool2d_503(opt_avgpool2d_503)
        opt_transpose_512 = self.transpose_512(opt_relu_506, (0, 3, 1, 2))
        opt_transpose_513 = self.transpose_513(opt_relu_506, (0, 3, 1, 2))
        opt_add_505 = P.Add()(opt_avgpool2d_500, opt_avgpool2d_501)
        opt_conv2d_515 = self.conv2d_515(opt_transpose_508)
        opt_conv2d_516 = self.conv2d_516(opt_transpose_509)
        opt_conv2d_517 = self.conv2d_517(opt_transpose_510)
        opt_transpose_507 = self.transpose_507(opt_avgpool2d_503, (0, 2, 3, 1))
        opt_conv2d_518 = self.conv2d_518(opt_transpose_512)
        opt_conv2d_519 = self.conv2d_519(opt_transpose_513)
        opt_conv2d_520 = self.conv2d_520(opt_conv2d_515)
        opt_conv2d_521 = self.conv2d_521(opt_conv2d_516)
        opt_conv2d_522 = self.conv2d_522(opt_conv2d_517)
        opt_add_514 = P.Add()(opt_transpose_507, opt_transpose_448)
        opt_conv2d_523 = self.conv2d_523(opt_conv2d_518)
        opt_conv2d_524 = self.conv2d_524(opt_conv2d_519)
        opt_transpose_511 = self.transpose_511(opt_add_505, (0, 2, 3, 1))
        opt_relu_525 = self.relu_525(opt_conv2d_520)
        opt_relu_526 = self.relu_526(opt_conv2d_521)
        opt_relu_527 = self.relu_527(opt_conv2d_522)
        opt_relu_528 = self.relu_528(opt_conv2d_523)
        opt_relu_529 = self.relu_529(opt_conv2d_524)
        opt_conv2d_530 = self.conv2d_530(opt_relu_525)
        opt_conv2d_531 = self.conv2d_531(opt_relu_526)
        opt_conv2d_532 = self.conv2d_532(opt_relu_527)
        opt_conv2d_533 = self.conv2d_533(opt_relu_528)
        opt_conv2d_534 = self.conv2d_534(opt_relu_529)
        opt_conv2d_535 = self.conv2d_535(opt_conv2d_530)
        opt_conv2d_536 = self.conv2d_536(opt_conv2d_531)
        opt_conv2d_537 = self.conv2d_537(opt_conv2d_532)
        opt_conv2d_538 = self.conv2d_538(opt_conv2d_533)
        opt_conv2d_539 = self.conv2d_539(opt_conv2d_534)
        opt_add_540 = P.Add()(opt_conv2d_536, opt_conv2d_537)
        opt_add_541 = P.Add()(opt_conv2d_538, opt_conv2d_484)
        opt_transpose_542 = self.transpose_542(opt_conv2d_539, (0, 2, 3, 1))
        opt_add_545 = P.Add()(opt_transpose_542, opt_transpose_502)
        opt_transpose_543 = self.transpose_543(opt_add_540, (0, 2, 3, 1))
        opt_transpose_544 = self.transpose_544(opt_add_541, (0, 2, 3, 1))
        opt_concat_546 = self.concat_546(
            (opt_transpose_448, opt_transpose_544, opt_transpose_492, opt_add_514, opt_transpose_461, opt_add_545,
             ))
        opt_relu_547 = self.relu_547(opt_concat_546)
        opt_transpose_548 = self.transpose_548(opt_relu_547, (0, 3, 1, 2))
        opt_transpose_549 = self.transpose_549(opt_relu_547, (0, 3, 1, 2))
        opt_conv2d_550 = self.conv2d_550(opt_transpose_548)
        module0_2_opt = self.module0_2(opt_transpose_549)
        opt_transpose_552 = self.transpose_552(opt_conv2d_550, (0, 2, 3, 1))
        opt_conv2d_557 = self.conv2d_557(module0_2_opt)
        opt_conv2d_558 = self.conv2d_558(module0_2_opt)
        opt_conv2d_559 = self.conv2d_559(module0_2_opt)
        opt_relu_555 = self.relu_555(opt_transpose_552)
        opt_conv2d_563 = self.conv2d_563(opt_conv2d_557)
        opt_conv2d_564 = self.conv2d_564(opt_conv2d_558)
        opt_conv2d_565 = self.conv2d_565(opt_conv2d_559)
        opt_avgpool2d_553 = self.pad_avgpool2d_553(opt_conv2d_550)
        opt_avgpool2d_553 = self.avgpool2d_553(opt_avgpool2d_553)
        opt_transpose_560 = self.transpose_560(opt_relu_555, (0, 3, 1, 2))
        opt_transpose_561 = self.transpose_561(opt_relu_555, (0, 3, 1, 2))
        opt_relu_568 = self.relu_568(opt_conv2d_563)
        opt_relu_569 = self.relu_569(opt_conv2d_564)
        opt_relu_570 = self.relu_570(opt_conv2d_565)
        opt_transpose_556 = self.transpose_556(opt_avgpool2d_553, (0, 2, 3, 1))
        opt_conv2d_566 = self.conv2d_566(opt_transpose_560)
        opt_conv2d_567 = self.conv2d_567(opt_transpose_561)
        opt_conv2d_573 = self.conv2d_573(opt_relu_568)
        opt_conv2d_574 = self.conv2d_574(opt_relu_569)
        opt_conv2d_575 = self.conv2d_575(opt_relu_570)
        opt_add_562 = P.Add()(opt_transpose_556, opt_transpose_499)
        opt_conv2d_571 = self.conv2d_571(opt_conv2d_566)
        opt_conv2d_572 = self.conv2d_572(opt_conv2d_567)
        opt_conv2d_578 = self.conv2d_578(opt_conv2d_573)
        opt_conv2d_579 = self.conv2d_579(opt_conv2d_574)
        opt_conv2d_580 = self.conv2d_580(opt_conv2d_575)
        opt_relu_576 = self.relu_576(opt_conv2d_571)
        opt_relu_577 = self.relu_577(opt_conv2d_572)
        opt_conv2d_581 = self.conv2d_581(opt_relu_576)
        opt_conv2d_582 = self.conv2d_582(opt_relu_577)
        opt_conv2d_583 = self.conv2d_583(opt_conv2d_581)
        opt_conv2d_584 = self.conv2d_584(opt_conv2d_582)
        opt_add_585 = P.Add()(opt_conv2d_583, opt_conv2d_535)
        opt_transpose_586 = self.transpose_586(opt_conv2d_584, (0, 2, 3, 1))
        opt_add_588 = P.Add()(opt_transpose_586, opt_transpose_552)
        opt_transpose_587 = self.transpose_587(opt_add_585, (0, 2, 3, 1))
        opt_concat_589 = self.concat_589(
            (opt_transpose_499, opt_transpose_587, opt_transpose_543, opt_add_562, opt_transpose_511, opt_add_588,
             ))
        opt_relu_590 = self.relu_590(opt_concat_589)
        opt_transpose_591 = self.transpose_591(opt_relu_590, (0, 3, 1, 2))
        opt_transpose_592 = self.transpose_592(opt_relu_590, (0, 3, 1, 2))
        opt_transpose_593 = self.transpose_593(opt_relu_590, (0, 3, 1, 2))
        opt_conv2d_594 = self.conv2d_594(opt_transpose_591)
        opt_avgpool2d_595 = self.pad_avgpool2d_595(opt_transpose_592)
        opt_avgpool2d_595 = self.avgpool2d_595(opt_avgpool2d_595)
        opt_pad_596 = self.pad_596(opt_transpose_593)
        opt_pad_597 = self.pad_597(opt_conv2d_594)
        opt_conv2d_599 = self.conv2d_599(opt_avgpool2d_595)
        opt_transpose_600 = self.transpose_600(opt_pad_596, (0, 2, 3, 1))
        opt_relu_598 = self.relu_598(opt_conv2d_594)
        opt_maxpool2d_601 = self.pad_maxpool2d_601(opt_pad_597)
        opt_maxpool2d_601 = self.maxpool2d_601(opt_maxpool2d_601)
        opt_avgpool2d_602 = self.pad_avgpool2d_602(opt_pad_597)
        opt_avgpool2d_602 = self.avgpool2d_602(opt_avgpool2d_602)
        opt_stridedslice_604 = self.stridedslice_604(opt_transpose_600, self.stridedslice_604_begin,
                                                     self.stridedslice_604_end, self.stridedslice_604_strides)
        opt_conv2d_603 = self.conv2d_603(opt_relu_598)
        opt_add_605 = P.Add()(opt_maxpool2d_601, opt_conv2d_578)
        opt_add_606 = P.Add()(opt_avgpool2d_602, opt_conv2d_579)
        opt_transpose_608 = self.transpose_608(opt_stridedslice_604, (0, 3, 1, 2))
        opt_conv2d_607 = self.conv2d_607(opt_conv2d_603)
        opt_avgpool2d_610 = self.pad_avgpool2d_610(opt_transpose_608)
        opt_avgpool2d_610 = self.avgpool2d_610(opt_avgpool2d_610)
        opt_relu_609 = self.relu_609(opt_conv2d_607)
        opt_conv2d_612 = self.conv2d_612(opt_avgpool2d_610)
        opt_conv2d_611 = self.conv2d_611(opt_relu_609)
        opt_concat_614 = self.concat_614((opt_conv2d_599, opt_conv2d_612, ))
        opt_conv2d_613 = self.conv2d_613(opt_conv2d_611)
        opt_batchnorm2d_616 = self.batchnorm2d_616(opt_concat_614)
        opt_add_615 = P.Add()(opt_conv2d_613, opt_conv2d_580)
        opt_avgpool2d_617 = self.pad_avgpool2d_617(opt_add_615)
        opt_avgpool2d_617 = self.avgpool2d_617(opt_avgpool2d_617)
        opt_transpose_619 = self.transpose_619(opt_batchnorm2d_616, (0, 2, 3, 1))
        opt_relu_618 = self.relu_618(opt_add_615)
        opt_add_622 = P.Add()(opt_add_605, opt_avgpool2d_617)
        opt_conv2d_623 = self.conv2d_623(opt_relu_618)
        opt_relu_624 = self.relu_624(opt_transpose_619)
        opt_avgpool2d_620 = self.pad_avgpool2d_620(opt_batchnorm2d_616)
        opt_avgpool2d_620 = self.avgpool2d_620(opt_avgpool2d_620)
        opt_avgpool2d_621 = self.pad_avgpool2d_621(opt_batchnorm2d_616)
        opt_avgpool2d_621 = self.avgpool2d_621(opt_avgpool2d_621)
        opt_conv2d_626 = self.conv2d_626(opt_conv2d_623)
        opt_transpose_627 = self.transpose_627(opt_relu_624, (0, 3, 1, 2))
        opt_transpose_628 = self.transpose_628(opt_relu_624, (0, 3, 1, 2))
        opt_transpose_629 = self.transpose_629(opt_relu_624, (0, 3, 1, 2))
        opt_add_625 = P.Add()(opt_avgpool2d_620, opt_avgpool2d_621)
        opt_relu_631 = self.relu_631(opt_conv2d_626)
        opt_conv2d_632 = self.conv2d_632(opt_transpose_627)
        opt_conv2d_633 = self.conv2d_633(opt_transpose_628)
        opt_conv2d_634 = self.conv2d_634(opt_transpose_629)
        opt_conv2d_635 = self.conv2d_635(opt_relu_631)
        opt_conv2d_636 = self.conv2d_636(opt_conv2d_632)
        opt_conv2d_637 = self.conv2d_637(opt_conv2d_633)
        opt_conv2d_638 = self.conv2d_638(opt_conv2d_634)
        opt_transpose_630 = self.transpose_630(opt_add_625, (0, 2, 3, 1))
        opt_conv2d_639 = self.conv2d_639(opt_conv2d_635)
        opt_relu_640 = self.relu_640(opt_conv2d_636)
        opt_relu_641 = self.relu_641(opt_conv2d_637)
        opt_relu_642 = self.relu_642(opt_conv2d_638)
        opt_add_643 = P.Add()(opt_conv2d_639, opt_maxpool2d_601)
        opt_conv2d_644 = self.conv2d_644(opt_relu_640)
        opt_conv2d_645 = self.conv2d_645(opt_relu_641)
        opt_conv2d_646 = self.conv2d_646(opt_relu_642)
        opt_concat_647 = self.concat_647((opt_add_605, opt_add_606, opt_add_622, opt_add_643, ))
        opt_conv2d_648 = self.conv2d_648(opt_conv2d_644)
        opt_conv2d_649 = self.conv2d_649(opt_conv2d_645)
        opt_conv2d_650 = self.conv2d_650(opt_conv2d_646)
        opt_relu_651 = self.relu_651(opt_concat_647)
        opt_add_652 = P.Add()(opt_conv2d_649, opt_conv2d_650)
        opt_conv2d_653 = self.conv2d_653(opt_relu_651)
        opt_conv2d_654 = self.conv2d_654(opt_relu_651)
        opt_transpose_655 = self.transpose_655(opt_add_652, (0, 2, 3, 1))
        opt_transpose_656 = self.transpose_656(opt_conv2d_653, (0, 2, 3, 1))
        opt_transpose_659 = self.transpose_659(opt_conv2d_654, (0, 2, 3, 1))
        opt_relu_661 = self.relu_661(opt_transpose_656)
        opt_relu_663 = self.relu_663(opt_transpose_659)
        opt_avgpool2d_657 = self.pad_avgpool2d_657(opt_conv2d_653)
        opt_avgpool2d_657 = self.avgpool2d_657(opt_avgpool2d_657)
        opt_avgpool2d_658 = self.pad_avgpool2d_658(opt_conv2d_653)
        opt_avgpool2d_658 = self.avgpool2d_658(opt_avgpool2d_658)
        opt_transpose_665 = self.transpose_665(opt_relu_661, (0, 3, 1, 2))
        opt_transpose_666 = self.transpose_666(opt_relu_661, (0, 3, 1, 2))
        opt_transpose_667 = self.transpose_667(opt_relu_661, (0, 3, 1, 2))
        opt_avgpool2d_660 = self.pad_avgpool2d_660(opt_conv2d_654)
        opt_avgpool2d_660 = self.avgpool2d_660(opt_avgpool2d_660)
        opt_transpose_669 = self.transpose_669(opt_relu_663, (0, 3, 1, 2))
        opt_transpose_670 = self.transpose_670(opt_relu_663, (0, 3, 1, 2))
        opt_add_662 = P.Add()(opt_avgpool2d_657, opt_avgpool2d_658)
        opt_conv2d_672 = self.conv2d_672(opt_transpose_665)
        opt_conv2d_673 = self.conv2d_673(opt_transpose_666)
        opt_conv2d_674 = self.conv2d_674(opt_transpose_667)
        opt_transpose_664 = self.transpose_664(opt_avgpool2d_660, (0, 2, 3, 1))
        opt_conv2d_675 = self.conv2d_675(opt_transpose_669)
        opt_conv2d_676 = self.conv2d_676(opt_transpose_670)
        opt_conv2d_677 = self.conv2d_677(opt_conv2d_672)
        opt_conv2d_678 = self.conv2d_678(opt_conv2d_673)
        opt_conv2d_679 = self.conv2d_679(opt_conv2d_674)
        opt_add_671 = P.Add()(opt_transpose_664, opt_transpose_619)
        opt_conv2d_680 = self.conv2d_680(opt_conv2d_675)
        opt_conv2d_681 = self.conv2d_681(opt_conv2d_676)
        opt_transpose_668 = self.transpose_668(opt_add_662, (0, 2, 3, 1))
        opt_relu_682 = self.relu_682(opt_conv2d_677)
        opt_relu_683 = self.relu_683(opt_conv2d_678)
        opt_relu_684 = self.relu_684(opt_conv2d_679)
        opt_relu_685 = self.relu_685(opt_conv2d_680)
        opt_relu_686 = self.relu_686(opt_conv2d_681)
        opt_conv2d_687 = self.conv2d_687(opt_relu_682)
        opt_conv2d_688 = self.conv2d_688(opt_relu_683)
        opt_conv2d_689 = self.conv2d_689(opt_relu_684)
        opt_conv2d_690 = self.conv2d_690(opt_relu_685)
        opt_conv2d_691 = self.conv2d_691(opt_relu_686)
        opt_conv2d_692 = self.conv2d_692(opt_conv2d_687)
        opt_conv2d_693 = self.conv2d_693(opt_conv2d_688)
        opt_conv2d_694 = self.conv2d_694(opt_conv2d_689)
        opt_conv2d_695 = self.conv2d_695(opt_conv2d_690)
        opt_conv2d_696 = self.conv2d_696(opt_conv2d_691)
        opt_add_697 = P.Add()(opt_conv2d_693, opt_conv2d_694)
        opt_add_698 = P.Add()(opt_conv2d_695, opt_conv2d_648)
        opt_transpose_699 = self.transpose_699(opt_conv2d_696, (0, 2, 3, 1))
        opt_add_702 = P.Add()(opt_transpose_699, opt_transpose_659)
        opt_transpose_700 = self.transpose_700(opt_add_697, (0, 2, 3, 1))
        opt_transpose_701 = self.transpose_701(opt_add_698, (0, 2, 3, 1))
        opt_concat_703 = self.concat_703(
            (opt_transpose_619, opt_transpose_701, opt_transpose_655, opt_add_671, opt_transpose_630, opt_add_702,
             ))
        opt_relu_704 = self.relu_704(opt_concat_703)
        opt_transpose_705 = self.transpose_705(opt_relu_704, (0, 3, 1, 2))
        opt_transpose_706 = self.transpose_706(opt_relu_704, (0, 3, 1, 2))
        opt_conv2d_707 = self.conv2d_707(opt_transpose_705)
        opt_conv2d_708 = self.conv2d_708(opt_transpose_706)
        opt_transpose_709 = self.transpose_709(opt_conv2d_707, (0, 2, 3, 1))
        opt_transpose_711 = self.transpose_711(opt_conv2d_708, (0, 2, 3, 1))
        opt_relu_714 = self.relu_714(opt_transpose_709)
        opt_relu_716 = self.relu_716(opt_transpose_711)
        opt_avgpool2d_710 = self.pad_avgpool2d_710(opt_conv2d_707)
        opt_avgpool2d_710 = self.avgpool2d_710(opt_avgpool2d_710)
        opt_avgpool2d_712 = self.pad_avgpool2d_712(opt_conv2d_708)
        opt_avgpool2d_712 = self.avgpool2d_712(opt_avgpool2d_712)
        opt_avgpool2d_713 = self.pad_avgpool2d_713(opt_conv2d_708)
        opt_avgpool2d_713 = self.avgpool2d_713(opt_avgpool2d_713)
        opt_transpose_718 = self.transpose_718(opt_relu_714, (0, 3, 1, 2))
        opt_transpose_719 = self.transpose_719(opt_relu_714, (0, 3, 1, 2))
        opt_transpose_721 = self.transpose_721(opt_relu_716, (0, 3, 1, 2))
        opt_transpose_722 = self.transpose_722(opt_relu_716, (0, 3, 1, 2))
        opt_transpose_723 = self.transpose_723(opt_relu_716, (0, 3, 1, 2))
        opt_transpose_715 = self.transpose_715(opt_avgpool2d_710, (0, 2, 3, 1))
        opt_add_717 = P.Add()(opt_avgpool2d_712, opt_avgpool2d_713)
        opt_conv2d_725 = self.conv2d_725(opt_transpose_718)
        opt_conv2d_726 = self.conv2d_726(opt_transpose_719)
        opt_conv2d_727 = self.conv2d_727(opt_transpose_721)
        opt_conv2d_728 = self.conv2d_728(opt_transpose_722)
        opt_conv2d_729 = self.conv2d_729(opt_transpose_723)
        opt_add_720 = P.Add()(opt_transpose_715, opt_transpose_656)
        opt_conv2d_730 = self.conv2d_730(opt_conv2d_725)
        opt_conv2d_731 = self.conv2d_731(opt_conv2d_726)
        opt_conv2d_732 = self.conv2d_732(opt_conv2d_727)
        opt_conv2d_733 = self.conv2d_733(opt_conv2d_728)
        opt_conv2d_734 = self.conv2d_734(opt_conv2d_729)
        opt_transpose_724 = self.transpose_724(opt_add_717, (0, 2, 3, 1))
        opt_relu_735 = self.relu_735(opt_conv2d_730)
        opt_relu_736 = self.relu_736(opt_conv2d_731)
        opt_relu_737 = self.relu_737(opt_conv2d_732)
        opt_relu_738 = self.relu_738(opt_conv2d_733)
        opt_relu_739 = self.relu_739(opt_conv2d_734)
        opt_conv2d_740 = self.conv2d_740(opt_relu_735)
        opt_conv2d_741 = self.conv2d_741(opt_relu_736)
        opt_conv2d_742 = self.conv2d_742(opt_relu_737)
        opt_conv2d_743 = self.conv2d_743(opt_relu_738)
        opt_conv2d_744 = self.conv2d_744(opt_relu_739)
        opt_conv2d_745 = self.conv2d_745(opt_conv2d_740)
        opt_conv2d_746 = self.conv2d_746(opt_conv2d_741)
        opt_conv2d_747 = self.conv2d_747(opt_conv2d_742)
        opt_conv2d_748 = self.conv2d_748(opt_conv2d_743)
        opt_conv2d_749 = self.conv2d_749(opt_conv2d_744)
        opt_add_750 = P.Add()(opt_conv2d_745, opt_conv2d_692)
        opt_transpose_751 = self.transpose_751(opt_conv2d_746, (0, 2, 3, 1))
        opt_add_752 = P.Add()(opt_conv2d_748, opt_conv2d_749)
        opt_add_754 = P.Add()(opt_transpose_751, opt_transpose_709)
        opt_transpose_753 = self.transpose_753(opt_add_750, (0, 2, 3, 1))
        opt_transpose_755 = self.transpose_755(opt_add_752, (0, 2, 3, 1))
        opt_concat_756 = self.concat_756(
            (opt_transpose_656, opt_transpose_753, opt_transpose_700, opt_add_720, opt_transpose_668, opt_add_754,
             ))
        opt_relu_757 = self.relu_757(opt_concat_756)
        opt_transpose_758 = self.transpose_758(opt_relu_757, (0, 3, 1, 2))
        opt_transpose_759 = self.transpose_759(opt_relu_757, (0, 3, 1, 2))
        opt_conv2d_760 = self.conv2d_760(opt_transpose_758)
        opt_conv2d_761 = self.conv2d_761(opt_transpose_759)
        opt_transpose_762 = self.transpose_762(opt_conv2d_760, (0, 2, 3, 1))
        opt_transpose_765 = self.transpose_765(opt_conv2d_761, (0, 2, 3, 1))
        opt_relu_767 = self.relu_767(opt_transpose_762)
        opt_relu_769 = self.relu_769(opt_transpose_765)
        opt_avgpool2d_763 = self.pad_avgpool2d_763(opt_conv2d_760)
        opt_avgpool2d_763 = self.avgpool2d_763(opt_avgpool2d_763)
        opt_avgpool2d_764 = self.pad_avgpool2d_764(opt_conv2d_760)
        opt_avgpool2d_764 = self.avgpool2d_764(opt_avgpool2d_764)
        opt_transpose_771 = self.transpose_771(opt_relu_767, (0, 3, 1, 2))
        opt_transpose_772 = self.transpose_772(opt_relu_767, (0, 3, 1, 2))
        opt_transpose_773 = self.transpose_773(opt_relu_767, (0, 3, 1, 2))
        opt_avgpool2d_766 = self.pad_avgpool2d_766(opt_conv2d_761)
        opt_avgpool2d_766 = self.avgpool2d_766(opt_avgpool2d_766)
        opt_transpose_775 = self.transpose_775(opt_relu_769, (0, 3, 1, 2))
        opt_transpose_776 = self.transpose_776(opt_relu_769, (0, 3, 1, 2))
        opt_add_768 = P.Add()(opt_avgpool2d_763, opt_avgpool2d_764)
        opt_conv2d_778 = self.conv2d_778(opt_transpose_771)
        opt_conv2d_779 = self.conv2d_779(opt_transpose_772)
        opt_conv2d_780 = self.conv2d_780(opt_transpose_773)
        opt_transpose_770 = self.transpose_770(opt_avgpool2d_766, (0, 2, 3, 1))
        opt_conv2d_781 = self.conv2d_781(opt_transpose_775)
        opt_conv2d_782 = self.conv2d_782(opt_transpose_776)
        opt_conv2d_783 = self.conv2d_783(opt_conv2d_778)
        opt_conv2d_784 = self.conv2d_784(opt_conv2d_779)
        opt_conv2d_785 = self.conv2d_785(opt_conv2d_780)
        opt_add_777 = P.Add()(opt_transpose_770, opt_transpose_711)
        opt_conv2d_786 = self.conv2d_786(opt_conv2d_781)
        opt_conv2d_787 = self.conv2d_787(opt_conv2d_782)
        opt_transpose_774 = self.transpose_774(opt_add_768, (0, 2, 3, 1))
        opt_relu_788 = self.relu_788(opt_conv2d_783)
        opt_relu_789 = self.relu_789(opt_conv2d_784)
        opt_relu_790 = self.relu_790(opt_conv2d_785)
        opt_relu_791 = self.relu_791(opt_conv2d_786)
        opt_relu_792 = self.relu_792(opt_conv2d_787)
        opt_conv2d_793 = self.conv2d_793(opt_relu_788)
        opt_conv2d_794 = self.conv2d_794(opt_relu_789)
        opt_conv2d_795 = self.conv2d_795(opt_relu_790)
        opt_conv2d_796 = self.conv2d_796(opt_relu_791)
        opt_conv2d_797 = self.conv2d_797(opt_relu_792)
        opt_conv2d_798 = self.conv2d_798(opt_conv2d_793)
        opt_conv2d_799 = self.conv2d_799(opt_conv2d_794)
        opt_conv2d_800 = self.conv2d_800(opt_conv2d_795)
        opt_conv2d_801 = self.conv2d_801(opt_conv2d_796)
        opt_conv2d_802 = self.conv2d_802(opt_conv2d_797)
        opt_add_803 = P.Add()(opt_conv2d_799, opt_conv2d_800)
        opt_add_804 = P.Add()(opt_conv2d_801, opt_conv2d_747)
        opt_transpose_805 = self.transpose_805(opt_conv2d_802, (0, 2, 3, 1))
        opt_add_808 = P.Add()(opt_transpose_805, opt_transpose_765)
        opt_transpose_806 = self.transpose_806(opt_add_803, (0, 2, 3, 1))
        opt_transpose_807 = self.transpose_807(opt_add_804, (0, 2, 3, 1))
        opt_concat_809 = self.concat_809(
            (opt_transpose_711, opt_transpose_807, opt_transpose_755, opt_add_777, opt_transpose_724, opt_add_808,
             ))
        opt_relu_810 = self.relu_810(opt_concat_809)
        opt_transpose_811 = self.transpose_811(opt_relu_810, (0, 3, 1, 2))
        opt_conv2d_812 = self.conv2d_812(opt_transpose_811)
        opt_transpose_813 = self.transpose_813(opt_conv2d_812, (0, 2, 3, 1))
        opt_relu_815 = self.relu_815(opt_transpose_813)
        opt_avgpool2d_814 = self.pad_avgpool2d_814(opt_conv2d_812)
        opt_avgpool2d_814 = self.avgpool2d_814(opt_avgpool2d_814)
        opt_transpose_817 = self.transpose_817(opt_relu_815, (0, 3, 1, 2))
        opt_transpose_818 = self.transpose_818(opt_relu_815, (0, 3, 1, 2))
        opt_transpose_816 = self.transpose_816(opt_avgpool2d_814, (0, 2, 3, 1))
        opt_conv2d_820 = self.conv2d_820(opt_transpose_817)
        opt_conv2d_821 = self.conv2d_821(opt_transpose_818)
        opt_add_819 = P.Add()(opt_transpose_816, opt_transpose_762)
        opt_conv2d_822 = self.conv2d_822(opt_conv2d_820)
        opt_conv2d_823 = self.conv2d_823(opt_conv2d_821)
        opt_relu_824 = self.relu_824(opt_conv2d_822)
        opt_relu_825 = self.relu_825(opt_conv2d_823)
        opt_conv2d_826 = self.conv2d_826(opt_relu_824)
        opt_conv2d_827 = self.conv2d_827(opt_relu_825)
        opt_conv2d_828 = self.conv2d_828(opt_conv2d_826)
        opt_conv2d_829 = self.conv2d_829(opt_conv2d_827)
        opt_add_830 = P.Add()(opt_conv2d_828, opt_conv2d_798)
        opt_transpose_831 = self.transpose_831(opt_conv2d_829, (0, 2, 3, 1))
        opt_add_833 = P.Add()(opt_transpose_831, opt_transpose_813)
        opt_transpose_832 = self.transpose_832(opt_add_830, (0, 2, 3, 1))
        opt_concat_834 = self.concat_834(
            (opt_transpose_762, opt_transpose_832, opt_transpose_806, opt_add_819, opt_transpose_774, opt_add_833,
             ))
        opt_relu_835 = self.relu_835(opt_concat_834)
        opt_transpose_836 = self.transpose_836(opt_relu_835, (0, 3, 1, 2))
        opt_avgpool2d_837 = self.avgpool2d_837(opt_transpose_836)
        opt_transpose_838 = self.transpose_838(opt_avgpool2d_837, (0, 2, 3, 1))
        opt_reshape_839 = self.reshape_839(opt_transpose_838, self.reshape_839_shape)
        opt_matmul_840 = P.matmul(opt_reshape_839, self.matmul_840_w)
        opt_add_841 = opt_matmul_840 + self.add_841_bias
        opt_softmax_842 = self.softmax_842(opt_add_841)
        return opt_softmax_842
