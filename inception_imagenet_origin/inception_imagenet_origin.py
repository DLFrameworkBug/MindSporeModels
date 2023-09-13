import numpy as np
import mindspore
import mindspore.numpy as ms_np
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, Parameter


class MindSporeModel(nn.Cell):

    def __init__(self):
        super(MindSporeModel, self).__init__()
        self.transpose_0 = P.Transpose()
        self.conv2d_1 = nn.Conv2d(in_channels=3,
                                  out_channels=32,
                                  kernel_size=(3, 3),
                                  stride=(2, 2),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_2 = nn.ReLU()
        self.conv2d_3 = nn.Conv2d(in_channels=32,
                                  out_channels=32,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_4 = nn.ReLU()
        self.conv2d_5 = nn.Conv2d(in_channels=32,
                                  out_channels=64,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_6 = nn.ReLU()
        self.pad_maxpool2d_7 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_7 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv2d_8 = nn.Conv2d(in_channels=64,
                                  out_channels=80,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_9 = nn.ReLU()
        self.conv2d_10 = nn.Conv2d(in_channels=80,
                                   out_channels=192,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_11 = nn.ReLU()
        self.pad_maxpool2d_12 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_12 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv2d_13 = nn.Conv2d(in_channels=192,
                                   out_channels=64,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.pad_avgpool2d_14 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_14 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.conv2d_15 = nn.Conv2d(in_channels=192,
                                   out_channels=48,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_16 = nn.Conv2d(in_channels=192,
                                   out_channels=64,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_17 = nn.ReLU()
        self.conv2d_18 = nn.Conv2d(in_channels=192,
                                   out_channels=32,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_19 = nn.ReLU()
        self.relu_20 = nn.ReLU()
        self.relu_21 = nn.ReLU()
        self.conv2d_22 = nn.Conv2d(in_channels=48,
                                   out_channels=64,
                                   kernel_size=(5, 5),
                                   stride=(1, 1),
                                   padding=(2, 2, 2, 2),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_23 = nn.Conv2d(in_channels=64,
                                   out_channels=96,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_24 = nn.ReLU()
        self.relu_25 = nn.ReLU()
        self.conv2d_26 = nn.Conv2d(in_channels=96,
                                   out_channels=96,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_27 = nn.ReLU()
        self.concat_28 = P.Concat(axis=1)
        self.conv2d_29 = nn.Conv2d(in_channels=256,
                                   out_channels=64,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.pad_avgpool2d_30 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_30 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.conv2d_31 = nn.Conv2d(in_channels=256,
                                   out_channels=48,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_32 = nn.Conv2d(in_channels=256,
                                   out_channels=64,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_33 = nn.ReLU()
        self.conv2d_34 = nn.Conv2d(in_channels=256,
                                   out_channels=64,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_35 = nn.ReLU()
        self.relu_36 = nn.ReLU()
        self.relu_37 = nn.ReLU()
        self.conv2d_38 = nn.Conv2d(in_channels=48,
                                   out_channels=64,
                                   kernel_size=(5, 5),
                                   stride=(1, 1),
                                   padding=(2, 2, 2, 2),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_39 = nn.Conv2d(in_channels=64,
                                   out_channels=96,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_40 = nn.ReLU()
        self.relu_41 = nn.ReLU()
        self.conv2d_42 = nn.Conv2d(in_channels=96,
                                   out_channels=96,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_43 = nn.ReLU()
        self.concat_44 = P.Concat(axis=1)
        self.conv2d_45 = nn.Conv2d(in_channels=288,
                                   out_channels=64,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.pad_avgpool2d_46 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_46 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.conv2d_47 = nn.Conv2d(in_channels=288,
                                   out_channels=48,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_48 = nn.Conv2d(in_channels=288,
                                   out_channels=64,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_49 = nn.ReLU()
        self.conv2d_50 = nn.Conv2d(in_channels=288,
                                   out_channels=64,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_51 = nn.ReLU()
        self.relu_52 = nn.ReLU()
        self.relu_53 = nn.ReLU()
        self.conv2d_54 = nn.Conv2d(in_channels=48,
                                   out_channels=64,
                                   kernel_size=(5, 5),
                                   stride=(1, 1),
                                   padding=(2, 2, 2, 2),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_55 = nn.Conv2d(in_channels=64,
                                   out_channels=96,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_56 = nn.ReLU()
        self.relu_57 = nn.ReLU()
        self.conv2d_58 = nn.Conv2d(in_channels=96,
                                   out_channels=96,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_59 = nn.ReLU()
        self.concat_60 = P.Concat(axis=1)
        self.pad_maxpool2d_61 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_61 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv2d_62 = nn.Conv2d(in_channels=288,
                                   out_channels=384,
                                   kernel_size=(3, 3),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_63 = nn.Conv2d(in_channels=288,
                                   out_channels=64,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_64 = nn.ReLU()
        self.relu_65 = nn.ReLU()
        self.conv2d_66 = nn.Conv2d(in_channels=64,
                                   out_channels=96,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_67 = nn.ReLU()
        self.conv2d_68 = nn.Conv2d(in_channels=96,
                                   out_channels=96,
                                   kernel_size=(3, 3),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_69 = nn.ReLU()
        self.concat_70 = P.Concat(axis=1)
        self.conv2d_71 = nn.Conv2d(in_channels=768,
                                   out_channels=192,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.pad_avgpool2d_72 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_72 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.conv2d_73 = nn.Conv2d(in_channels=768,
                                   out_channels=128,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_74 = nn.Conv2d(in_channels=768,
                                   out_channels=128,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_75 = nn.ReLU()
        self.conv2d_76 = nn.Conv2d(in_channels=768,
                                   out_channels=192,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_77 = nn.ReLU()
        self.relu_78 = nn.ReLU()
        self.relu_79 = nn.ReLU()
        self.conv2d_80 = nn.Conv2d(in_channels=128,
                                   out_channels=128,
                                   kernel_size=(1, 7),
                                   stride=(1, 1),
                                   padding=(0, 0, 3, 3),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_81 = nn.Conv2d(in_channels=128,
                                   out_channels=128,
                                   kernel_size=(7, 1),
                                   stride=(1, 1),
                                   padding=(3, 3, 0, 0),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_82 = nn.ReLU()
        self.relu_83 = nn.ReLU()
        self.conv2d_84 = nn.Conv2d(in_channels=128,
                                   out_channels=192,
                                   kernel_size=(7, 1),
                                   stride=(1, 1),
                                   padding=(3, 3, 0, 0),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_85 = nn.Conv2d(in_channels=128,
                                   out_channels=128,
                                   kernel_size=(1, 7),
                                   stride=(1, 1),
                                   padding=(0, 0, 3, 3),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_86 = nn.ReLU()
        self.relu_87 = nn.ReLU()
        self.conv2d_88 = nn.Conv2d(in_channels=128,
                                   out_channels=128,
                                   kernel_size=(7, 1),
                                   stride=(1, 1),
                                   padding=(3, 3, 0, 0),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_89 = nn.ReLU()
        self.conv2d_90 = nn.Conv2d(in_channels=128,
                                   out_channels=192,
                                   kernel_size=(1, 7),
                                   stride=(1, 1),
                                   padding=(0, 0, 3, 3),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_91 = nn.ReLU()
        self.concat_92 = P.Concat(axis=1)
        self.conv2d_93 = nn.Conv2d(in_channels=768,
                                   out_channels=192,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.pad_avgpool2d_94 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_94 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.conv2d_95 = nn.Conv2d(in_channels=768,
                                   out_channels=160,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_96 = nn.Conv2d(in_channels=768,
                                   out_channels=160,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_97 = nn.ReLU()
        self.conv2d_98 = nn.Conv2d(in_channels=768,
                                   out_channels=192,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_99 = nn.ReLU()
        self.relu_100 = nn.ReLU()
        self.relu_101 = nn.ReLU()
        self.conv2d_102 = nn.Conv2d(in_channels=160,
                                    out_channels=160,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_103 = nn.Conv2d(in_channels=160,
                                    out_channels=160,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_104 = nn.ReLU()
        self.relu_105 = nn.ReLU()
        self.conv2d_106 = nn.Conv2d(in_channels=160,
                                    out_channels=192,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_107 = nn.Conv2d(in_channels=160,
                                    out_channels=160,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_108 = nn.ReLU()
        self.relu_109 = nn.ReLU()
        self.conv2d_110 = nn.Conv2d(in_channels=160,
                                    out_channels=160,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_111 = nn.ReLU()
        self.conv2d_112 = nn.Conv2d(in_channels=160,
                                    out_channels=192,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_113 = nn.ReLU()
        self.concat_114 = P.Concat(axis=1)
        self.conv2d_115 = nn.Conv2d(in_channels=768,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.pad_avgpool2d_116 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_116 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.conv2d_117 = nn.Conv2d(in_channels=768,
                                    out_channels=160,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_118 = nn.Conv2d(in_channels=768,
                                    out_channels=160,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_119 = nn.ReLU()
        self.conv2d_120 = nn.Conv2d(in_channels=768,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_121 = nn.ReLU()
        self.relu_122 = nn.ReLU()
        self.relu_123 = nn.ReLU()
        self.conv2d_124 = nn.Conv2d(in_channels=160,
                                    out_channels=160,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_125 = nn.Conv2d(in_channels=160,
                                    out_channels=160,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_126 = nn.ReLU()
        self.relu_127 = nn.ReLU()
        self.conv2d_128 = nn.Conv2d(in_channels=160,
                                    out_channels=192,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_129 = nn.Conv2d(in_channels=160,
                                    out_channels=160,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_130 = nn.ReLU()
        self.relu_131 = nn.ReLU()
        self.conv2d_132 = nn.Conv2d(in_channels=160,
                                    out_channels=160,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_133 = nn.ReLU()
        self.conv2d_134 = nn.Conv2d(in_channels=160,
                                    out_channels=192,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_135 = nn.ReLU()
        self.concat_136 = P.Concat(axis=1)
        self.conv2d_137 = nn.Conv2d(in_channels=768,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.pad_avgpool2d_138 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_138 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.conv2d_139 = nn.Conv2d(in_channels=768,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_140 = nn.Conv2d(in_channels=768,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_141 = nn.ReLU()
        self.conv2d_142 = nn.Conv2d(in_channels=768,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_143 = nn.ReLU()
        self.relu_144 = nn.ReLU()
        self.relu_145 = nn.ReLU()
        self.conv2d_146 = nn.Conv2d(in_channels=192,
                                    out_channels=192,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_147 = nn.Conv2d(in_channels=192,
                                    out_channels=192,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_148 = nn.ReLU()
        self.relu_149 = nn.ReLU()
        self.conv2d_150 = nn.Conv2d(in_channels=192,
                                    out_channels=192,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_151 = nn.Conv2d(in_channels=192,
                                    out_channels=192,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_152 = nn.ReLU()
        self.relu_153 = nn.ReLU()
        self.conv2d_154 = nn.Conv2d(in_channels=192,
                                    out_channels=192,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_155 = nn.ReLU()
        self.conv2d_156 = nn.Conv2d(in_channels=192,
                                    out_channels=192,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_157 = nn.ReLU()
        self.concat_158 = P.Concat(axis=1)
        self.pad_maxpool2d_159 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_159 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv2d_160 = nn.Conv2d(in_channels=768,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_161 = nn.Conv2d(in_channels=768,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_162 = nn.ReLU()
        self.relu_163 = nn.ReLU()
        self.conv2d_164 = nn.Conv2d(in_channels=192,
                                    out_channels=320,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_165 = nn.Conv2d(in_channels=192,
                                    out_channels=192,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_166 = nn.ReLU()
        self.relu_167 = nn.ReLU()
        self.conv2d_168 = nn.Conv2d(in_channels=192,
                                    out_channels=192,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_169 = nn.ReLU()
        self.conv2d_170 = nn.Conv2d(in_channels=192,
                                    out_channels=192,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_171 = nn.ReLU()
        self.concat_172 = P.Concat(axis=1)
        self.conv2d_173 = nn.Conv2d(in_channels=1280,
                                    out_channels=320,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.pad_avgpool2d_174 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_174 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.conv2d_175 = nn.Conv2d(in_channels=1280,
                                    out_channels=384,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_176 = nn.Conv2d(in_channels=1280,
                                    out_channels=448,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_177 = nn.ReLU()
        self.conv2d_178 = nn.Conv2d(in_channels=1280,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_179 = nn.ReLU()
        self.relu_180 = nn.ReLU()
        self.relu_181 = nn.ReLU()
        self.conv2d_182 = nn.Conv2d(in_channels=384,
                                    out_channels=384,
                                    kernel_size=(1, 3),
                                    stride=(1, 1),
                                    padding=(0, 0, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_183 = nn.Conv2d(in_channels=384,
                                    out_channels=384,
                                    kernel_size=(3, 1),
                                    stride=(1, 1),
                                    padding=(1, 1, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_184 = nn.Conv2d(in_channels=448,
                                    out_channels=384,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_185 = nn.ReLU()
        self.relu_186 = nn.ReLU()
        self.relu_187 = nn.ReLU()
        self.conv2d_188 = nn.Conv2d(in_channels=384,
                                    out_channels=384,
                                    kernel_size=(1, 3),
                                    stride=(1, 1),
                                    padding=(0, 0, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_189 = nn.Conv2d(in_channels=384,
                                    out_channels=384,
                                    kernel_size=(3, 1),
                                    stride=(1, 1),
                                    padding=(1, 1, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_190 = nn.ReLU()
        self.relu_191 = nn.ReLU()
        self.concat_192 = P.Concat(axis=1)
        self.conv2d_193 = nn.Conv2d(in_channels=2048,
                                    out_channels=320,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.pad_avgpool2d_194 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_194 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.conv2d_195 = nn.Conv2d(in_channels=2048,
                                    out_channels=384,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_196 = nn.Conv2d(in_channels=2048,
                                    out_channels=448,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_197 = nn.ReLU()
        self.conv2d_198 = nn.Conv2d(in_channels=2048,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_199 = nn.ReLU()
        self.relu_200 = nn.ReLU()
        self.relu_201 = nn.ReLU()
        self.conv2d_202 = nn.Conv2d(in_channels=384,
                                    out_channels=384,
                                    kernel_size=(1, 3),
                                    stride=(1, 1),
                                    padding=(0, 0, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_203 = nn.Conv2d(in_channels=384,
                                    out_channels=384,
                                    kernel_size=(3, 1),
                                    stride=(1, 1),
                                    padding=(1, 1, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_204 = nn.Conv2d(in_channels=448,
                                    out_channels=384,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_205 = nn.ReLU()
        self.relu_206 = nn.ReLU()
        self.relu_207 = nn.ReLU()
        self.conv2d_208 = nn.Conv2d(in_channels=384,
                                    out_channels=384,
                                    kernel_size=(1, 3),
                                    stride=(1, 1),
                                    padding=(0, 0, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_209 = nn.Conv2d(in_channels=384,
                                    out_channels=384,
                                    kernel_size=(3, 1),
                                    stride=(1, 1),
                                    padding=(1, 1, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_210 = nn.ReLU()
        self.relu_211 = nn.ReLU()
        self.concat_212 = P.Concat(axis=1)
        self.avgpool2d_213 = nn.AvgPool2d(kernel_size=(8, 8))
        self.transpose_214 = P.Transpose()
        self.reshape_215 = P.Reshape()
        self.reshape_215_shape = tuple([1, 2048])
        self.matmul_216_w = Parameter(Tensor(np.random.uniform(0, 1, (2048, 1000)).astype(np.float32)), name=None)
        self.add_217_bias = Parameter(Tensor(np.random.uniform(0, 1, (1000, )).astype(np.float32)), name=None)
        self.softmax_218 = nn.Softmax(axis=-1)

    def construct(self, input_6):
        opt_transpose_0 = self.transpose_0(input_6, (0, 3, 1, 2))
        opt_conv2d_1 = self.conv2d_1(opt_transpose_0)
        opt_relu_2 = self.relu_2(opt_conv2d_1)
        opt_conv2d_3 = self.conv2d_3(opt_relu_2)
        opt_relu_4 = self.relu_4(opt_conv2d_3)
        opt_conv2d_5 = self.conv2d_5(opt_relu_4)
        opt_relu_6 = self.relu_6(opt_conv2d_5)
        opt_maxpool2d_7 = self.pad_maxpool2d_7(opt_relu_6)
        opt_maxpool2d_7 = self.maxpool2d_7(opt_maxpool2d_7)
        opt_conv2d_8 = self.conv2d_8(opt_maxpool2d_7)
        opt_relu_9 = self.relu_9(opt_conv2d_8)
        opt_conv2d_10 = self.conv2d_10(opt_relu_9)
        opt_relu_11 = self.relu_11(opt_conv2d_10)
        opt_maxpool2d_12 = self.pad_maxpool2d_12(opt_relu_11)
        opt_maxpool2d_12 = self.maxpool2d_12(opt_maxpool2d_12)
        opt_conv2d_13 = self.conv2d_13(opt_maxpool2d_12)
        opt_avgpool2d_14 = self.pad_avgpool2d_14(opt_maxpool2d_12)
        opt_avgpool2d_14 = self.avgpool2d_14(opt_avgpool2d_14)
        opt_conv2d_15 = self.conv2d_15(opt_maxpool2d_12)
        opt_conv2d_16 = self.conv2d_16(opt_maxpool2d_12)
        opt_relu_17 = self.relu_17(opt_conv2d_13)
        opt_conv2d_18 = self.conv2d_18(opt_avgpool2d_14)
        opt_relu_19 = self.relu_19(opt_conv2d_15)
        opt_relu_20 = self.relu_20(opt_conv2d_16)
        opt_relu_21 = self.relu_21(opt_conv2d_18)
        opt_conv2d_22 = self.conv2d_22(opt_relu_19)
        opt_conv2d_23 = self.conv2d_23(opt_relu_20)
        opt_relu_24 = self.relu_24(opt_conv2d_22)
        opt_relu_25 = self.relu_25(opt_conv2d_23)
        opt_conv2d_26 = self.conv2d_26(opt_relu_25)
        opt_relu_27 = self.relu_27(opt_conv2d_26)
        opt_concat_28 = self.concat_28((opt_relu_17, opt_relu_24, opt_relu_27, opt_relu_21, ))
        opt_conv2d_29 = self.conv2d_29(opt_concat_28)
        opt_avgpool2d_30 = self.pad_avgpool2d_30(opt_concat_28)
        opt_avgpool2d_30 = self.avgpool2d_30(opt_avgpool2d_30)
        opt_conv2d_31 = self.conv2d_31(opt_concat_28)
        opt_conv2d_32 = self.conv2d_32(opt_concat_28)
        opt_relu_33 = self.relu_33(opt_conv2d_29)
        opt_conv2d_34 = self.conv2d_34(opt_avgpool2d_30)
        opt_relu_35 = self.relu_35(opt_conv2d_31)
        opt_relu_36 = self.relu_36(opt_conv2d_32)
        opt_relu_37 = self.relu_37(opt_conv2d_34)
        opt_conv2d_38 = self.conv2d_38(opt_relu_35)
        opt_conv2d_39 = self.conv2d_39(opt_relu_36)
        opt_relu_40 = self.relu_40(opt_conv2d_38)
        opt_relu_41 = self.relu_41(opt_conv2d_39)
        opt_conv2d_42 = self.conv2d_42(opt_relu_41)
        opt_relu_43 = self.relu_43(opt_conv2d_42)
        opt_concat_44 = self.concat_44((opt_relu_33, opt_relu_40, opt_relu_43, opt_relu_37, ))
        opt_conv2d_45 = self.conv2d_45(opt_concat_44)
        opt_avgpool2d_46 = self.pad_avgpool2d_46(opt_concat_44)
        opt_avgpool2d_46 = self.avgpool2d_46(opt_avgpool2d_46)
        opt_conv2d_47 = self.conv2d_47(opt_concat_44)
        opt_conv2d_48 = self.conv2d_48(opt_concat_44)
        opt_relu_49 = self.relu_49(opt_conv2d_45)
        opt_conv2d_50 = self.conv2d_50(opt_avgpool2d_46)
        opt_relu_51 = self.relu_51(opt_conv2d_47)
        opt_relu_52 = self.relu_52(opt_conv2d_48)
        opt_relu_53 = self.relu_53(opt_conv2d_50)
        opt_conv2d_54 = self.conv2d_54(opt_relu_51)
        opt_conv2d_55 = self.conv2d_55(opt_relu_52)
        opt_relu_56 = self.relu_56(opt_conv2d_54)
        opt_relu_57 = self.relu_57(opt_conv2d_55)
        opt_conv2d_58 = self.conv2d_58(opt_relu_57)
        opt_relu_59 = self.relu_59(opt_conv2d_58)
        opt_concat_60 = self.concat_60((opt_relu_49, opt_relu_56, opt_relu_59, opt_relu_53, ))
        opt_maxpool2d_61 = self.pad_maxpool2d_61(opt_concat_60)
        opt_maxpool2d_61 = self.maxpool2d_61(opt_maxpool2d_61)
        opt_conv2d_62 = self.conv2d_62(opt_concat_60)
        opt_conv2d_63 = self.conv2d_63(opt_concat_60)
        opt_relu_64 = self.relu_64(opt_conv2d_62)
        opt_relu_65 = self.relu_65(opt_conv2d_63)
        opt_conv2d_66 = self.conv2d_66(opt_relu_65)
        opt_relu_67 = self.relu_67(opt_conv2d_66)
        opt_conv2d_68 = self.conv2d_68(opt_relu_67)
        opt_relu_69 = self.relu_69(opt_conv2d_68)
        opt_concat_70 = self.concat_70((opt_relu_64, opt_relu_69, opt_maxpool2d_61, ))
        opt_conv2d_71 = self.conv2d_71(opt_concat_70)
        opt_avgpool2d_72 = self.pad_avgpool2d_72(opt_concat_70)
        opt_avgpool2d_72 = self.avgpool2d_72(opt_avgpool2d_72)
        opt_conv2d_73 = self.conv2d_73(opt_concat_70)
        opt_conv2d_74 = self.conv2d_74(opt_concat_70)
        opt_relu_75 = self.relu_75(opt_conv2d_71)
        opt_conv2d_76 = self.conv2d_76(opt_avgpool2d_72)
        opt_relu_77 = self.relu_77(opt_conv2d_73)
        opt_relu_78 = self.relu_78(opt_conv2d_74)
        opt_relu_79 = self.relu_79(opt_conv2d_76)
        opt_conv2d_80 = self.conv2d_80(opt_relu_77)
        opt_conv2d_81 = self.conv2d_81(opt_relu_78)
        opt_relu_82 = self.relu_82(opt_conv2d_80)
        opt_relu_83 = self.relu_83(opt_conv2d_81)
        opt_conv2d_84 = self.conv2d_84(opt_relu_82)
        opt_conv2d_85 = self.conv2d_85(opt_relu_83)
        opt_relu_86 = self.relu_86(opt_conv2d_84)
        opt_relu_87 = self.relu_87(opt_conv2d_85)
        opt_conv2d_88 = self.conv2d_88(opt_relu_87)
        opt_relu_89 = self.relu_89(opt_conv2d_88)
        opt_conv2d_90 = self.conv2d_90(opt_relu_89)
        opt_relu_91 = self.relu_91(opt_conv2d_90)
        opt_concat_92 = self.concat_92((opt_relu_75, opt_relu_86, opt_relu_91, opt_relu_79, ))
        opt_conv2d_93 = self.conv2d_93(opt_concat_92)
        opt_avgpool2d_94 = self.pad_avgpool2d_94(opt_concat_92)
        opt_avgpool2d_94 = self.avgpool2d_94(opt_avgpool2d_94)
        opt_conv2d_95 = self.conv2d_95(opt_concat_92)
        opt_conv2d_96 = self.conv2d_96(opt_concat_92)
        opt_relu_97 = self.relu_97(opt_conv2d_93)
        opt_conv2d_98 = self.conv2d_98(opt_avgpool2d_94)
        opt_relu_99 = self.relu_99(opt_conv2d_95)
        opt_relu_100 = self.relu_100(opt_conv2d_96)
        opt_relu_101 = self.relu_101(opt_conv2d_98)
        opt_conv2d_102 = self.conv2d_102(opt_relu_99)
        opt_conv2d_103 = self.conv2d_103(opt_relu_100)
        opt_relu_104 = self.relu_104(opt_conv2d_102)
        opt_relu_105 = self.relu_105(opt_conv2d_103)
        opt_conv2d_106 = self.conv2d_106(opt_relu_104)
        opt_conv2d_107 = self.conv2d_107(opt_relu_105)
        opt_relu_108 = self.relu_108(opt_conv2d_106)
        opt_relu_109 = self.relu_109(opt_conv2d_107)
        opt_conv2d_110 = self.conv2d_110(opt_relu_109)
        opt_relu_111 = self.relu_111(opt_conv2d_110)
        opt_conv2d_112 = self.conv2d_112(opt_relu_111)
        opt_relu_113 = self.relu_113(opt_conv2d_112)
        opt_concat_114 = self.concat_114((opt_relu_97, opt_relu_108, opt_relu_113, opt_relu_101, ))
        opt_conv2d_115 = self.conv2d_115(opt_concat_114)
        opt_avgpool2d_116 = self.pad_avgpool2d_116(opt_concat_114)
        opt_avgpool2d_116 = self.avgpool2d_116(opt_avgpool2d_116)
        opt_conv2d_117 = self.conv2d_117(opt_concat_114)
        opt_conv2d_118 = self.conv2d_118(opt_concat_114)
        opt_relu_119 = self.relu_119(opt_conv2d_115)
        opt_conv2d_120 = self.conv2d_120(opt_avgpool2d_116)
        opt_relu_121 = self.relu_121(opt_conv2d_117)
        opt_relu_122 = self.relu_122(opt_conv2d_118)
        opt_relu_123 = self.relu_123(opt_conv2d_120)
        opt_conv2d_124 = self.conv2d_124(opt_relu_121)
        opt_conv2d_125 = self.conv2d_125(opt_relu_122)
        opt_relu_126 = self.relu_126(opt_conv2d_124)
        opt_relu_127 = self.relu_127(opt_conv2d_125)
        opt_conv2d_128 = self.conv2d_128(opt_relu_126)
        opt_conv2d_129 = self.conv2d_129(opt_relu_127)
        opt_relu_130 = self.relu_130(opt_conv2d_128)
        opt_relu_131 = self.relu_131(opt_conv2d_129)
        opt_conv2d_132 = self.conv2d_132(opt_relu_131)
        opt_relu_133 = self.relu_133(opt_conv2d_132)
        opt_conv2d_134 = self.conv2d_134(opt_relu_133)
        opt_relu_135 = self.relu_135(opt_conv2d_134)
        opt_concat_136 = self.concat_136((opt_relu_119, opt_relu_130, opt_relu_135, opt_relu_123, ))
        opt_conv2d_137 = self.conv2d_137(opt_concat_136)
        opt_avgpool2d_138 = self.pad_avgpool2d_138(opt_concat_136)
        opt_avgpool2d_138 = self.avgpool2d_138(opt_avgpool2d_138)
        opt_conv2d_139 = self.conv2d_139(opt_concat_136)
        opt_conv2d_140 = self.conv2d_140(opt_concat_136)
        opt_relu_141 = self.relu_141(opt_conv2d_137)
        opt_conv2d_142 = self.conv2d_142(opt_avgpool2d_138)
        opt_relu_143 = self.relu_143(opt_conv2d_139)
        opt_relu_144 = self.relu_144(opt_conv2d_140)
        opt_relu_145 = self.relu_145(opt_conv2d_142)
        opt_conv2d_146 = self.conv2d_146(opt_relu_143)
        opt_conv2d_147 = self.conv2d_147(opt_relu_144)
        opt_relu_148 = self.relu_148(opt_conv2d_146)
        opt_relu_149 = self.relu_149(opt_conv2d_147)
        opt_conv2d_150 = self.conv2d_150(opt_relu_148)
        opt_conv2d_151 = self.conv2d_151(opt_relu_149)
        opt_relu_152 = self.relu_152(opt_conv2d_150)
        opt_relu_153 = self.relu_153(opt_conv2d_151)
        opt_conv2d_154 = self.conv2d_154(opt_relu_153)
        opt_relu_155 = self.relu_155(opt_conv2d_154)
        opt_conv2d_156 = self.conv2d_156(opt_relu_155)
        opt_relu_157 = self.relu_157(opt_conv2d_156)
        opt_concat_158 = self.concat_158((opt_relu_141, opt_relu_152, opt_relu_157, opt_relu_145, ))
        opt_maxpool2d_159 = self.pad_maxpool2d_159(opt_concat_158)
        opt_maxpool2d_159 = self.maxpool2d_159(opt_maxpool2d_159)
        opt_conv2d_160 = self.conv2d_160(opt_concat_158)
        opt_conv2d_161 = self.conv2d_161(opt_concat_158)
        opt_relu_162 = self.relu_162(opt_conv2d_160)
        opt_relu_163 = self.relu_163(opt_conv2d_161)
        opt_conv2d_164 = self.conv2d_164(opt_relu_162)
        opt_conv2d_165 = self.conv2d_165(opt_relu_163)
        opt_relu_166 = self.relu_166(opt_conv2d_164)
        opt_relu_167 = self.relu_167(opt_conv2d_165)
        opt_conv2d_168 = self.conv2d_168(opt_relu_167)
        opt_relu_169 = self.relu_169(opt_conv2d_168)
        opt_conv2d_170 = self.conv2d_170(opt_relu_169)
        opt_relu_171 = self.relu_171(opt_conv2d_170)
        opt_concat_172 = self.concat_172((opt_relu_166, opt_relu_171, opt_maxpool2d_159, ))
        opt_conv2d_173 = self.conv2d_173(opt_concat_172)
        opt_avgpool2d_174 = self.pad_avgpool2d_174(opt_concat_172)
        opt_avgpool2d_174 = self.avgpool2d_174(opt_avgpool2d_174)
        opt_conv2d_175 = self.conv2d_175(opt_concat_172)
        opt_conv2d_176 = self.conv2d_176(opt_concat_172)
        opt_relu_177 = self.relu_177(opt_conv2d_173)
        opt_conv2d_178 = self.conv2d_178(opt_avgpool2d_174)
        opt_relu_179 = self.relu_179(opt_conv2d_175)
        opt_relu_180 = self.relu_180(opt_conv2d_176)
        opt_relu_181 = self.relu_181(opt_conv2d_178)
        opt_conv2d_182 = self.conv2d_182(opt_relu_179)
        opt_conv2d_183 = self.conv2d_183(opt_relu_179)
        opt_conv2d_184 = self.conv2d_184(opt_relu_180)
        opt_relu_185 = self.relu_185(opt_conv2d_182)
        opt_relu_186 = self.relu_186(opt_conv2d_183)
        opt_relu_187 = self.relu_187(opt_conv2d_184)
        opt_conv2d_188 = self.conv2d_188(opt_relu_187)
        opt_conv2d_189 = self.conv2d_189(opt_relu_187)
        opt_relu_190 = self.relu_190(opt_conv2d_188)
        opt_relu_191 = self.relu_191(opt_conv2d_189)
        opt_concat_192 = self.concat_192(
            (opt_relu_177, opt_relu_185, opt_relu_186, opt_relu_190, opt_relu_191, opt_relu_181,
             ))
        opt_conv2d_193 = self.conv2d_193(opt_concat_192)
        opt_avgpool2d_194 = self.pad_avgpool2d_194(opt_concat_192)
        opt_avgpool2d_194 = self.avgpool2d_194(opt_avgpool2d_194)
        opt_conv2d_195 = self.conv2d_195(opt_concat_192)
        opt_conv2d_196 = self.conv2d_196(opt_concat_192)
        opt_relu_197 = self.relu_197(opt_conv2d_193)
        opt_conv2d_198 = self.conv2d_198(opt_avgpool2d_194)
        opt_relu_199 = self.relu_199(opt_conv2d_195)
        opt_relu_200 = self.relu_200(opt_conv2d_196)
        opt_relu_201 = self.relu_201(opt_conv2d_198)
        opt_conv2d_202 = self.conv2d_202(opt_relu_199)
        opt_conv2d_203 = self.conv2d_203(opt_relu_199)
        opt_conv2d_204 = self.conv2d_204(opt_relu_200)
        opt_relu_205 = self.relu_205(opt_conv2d_202)
        opt_relu_206 = self.relu_206(opt_conv2d_203)
        opt_relu_207 = self.relu_207(opt_conv2d_204)
        opt_conv2d_208 = self.conv2d_208(opt_relu_207)
        opt_conv2d_209 = self.conv2d_209(opt_relu_207)
        opt_relu_210 = self.relu_210(opt_conv2d_208)
        opt_relu_211 = self.relu_211(opt_conv2d_209)
        opt_concat_212 = self.concat_212(
            (opt_relu_197, opt_relu_205, opt_relu_206, opt_relu_210, opt_relu_211, opt_relu_201,
             ))
        opt_avgpool2d_213 = self.avgpool2d_213(opt_concat_212)
        opt_transpose_214 = self.transpose_214(opt_avgpool2d_213, (0, 2, 3, 1))
        opt_reshape_215 = self.reshape_215(opt_transpose_214, self.reshape_215_shape)
        opt_matmul_216 = P.matmul(opt_reshape_215, self.matmul_216_w)
        opt_add_217 = opt_matmul_216 + self.add_217_bias
        opt_softmax_218 = self.softmax_218(opt_add_217)
        return opt_softmax_218
