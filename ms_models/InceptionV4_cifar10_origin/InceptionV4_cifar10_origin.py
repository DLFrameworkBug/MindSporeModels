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
        self.pad_maxpool2d_7 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.maxpool2d_7 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv2d_8 = nn.Conv2d(in_channels=64,
                                  out_channels=96,
                                  kernel_size=(3, 3),
                                  stride=(2, 2),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_9 = nn.ReLU()
        self.concat_10 = P.Concat(axis=1)
        self.conv2d_11 = nn.Conv2d(in_channels=160,
                                   out_channels=64,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_12 = nn.Conv2d(in_channels=160,
                                   out_channels=64,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_13 = nn.ReLU()
        self.relu_14 = nn.ReLU()
        self.conv2d_15 = nn.Conv2d(in_channels=64,
                                   out_channels=96,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_16 = nn.Conv2d(in_channels=64,
                                   out_channels=64,
                                   kernel_size=(7, 1),
                                   stride=(1, 1),
                                   padding=(3, 3, 0, 0),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_17 = nn.ReLU()
        self.relu_18 = nn.ReLU()
        self.conv2d_19 = nn.Conv2d(in_channels=64,
                                   out_channels=64,
                                   kernel_size=(1, 7),
                                   stride=(1, 1),
                                   padding=(0, 0, 3, 3),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_20 = nn.ReLU()
        self.conv2d_21 = nn.Conv2d(in_channels=64,
                                   out_channels=96,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_22 = nn.ReLU()
        self.concat_23 = P.Concat(axis=1)
        self.pad_maxpool2d_24 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.maxpool2d_24 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv2d_25 = nn.Conv2d(in_channels=192,
                                   out_channels=192,
                                   kernel_size=(3, 3),
                                   stride=(2, 2),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_26 = nn.ReLU()
        self.concat_27 = P.Concat(axis=1)
        self.relu_28 = nn.ReLU()
        self.conv2d_29 = nn.Conv2d(in_channels=384,
                                   out_channels=96,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.pad_avgpool2d_30 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_30 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.conv2d_31 = nn.Conv2d(in_channels=384,
                                   out_channels=64,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_32 = nn.Conv2d(in_channels=384,
                                   out_channels=64,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_33 = nn.ReLU()
        self.conv2d_34 = nn.Conv2d(in_channels=384,
                                   out_channels=96,
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
        self.conv2d_38 = nn.Conv2d(in_channels=64,
                                   out_channels=96,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
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
        self.batchnorm2d_45 = nn.BatchNorm2d(num_features=384, eps=9.999999974752427e-07, momentum=0.9900000095367432)
        self.relu_46 = nn.ReLU()
        self.conv2d_47 = nn.Conv2d(in_channels=384,
                                   out_channels=96,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.pad_avgpool2d_48 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_48 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.conv2d_49 = nn.Conv2d(in_channels=384,
                                   out_channels=64,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_50 = nn.Conv2d(in_channels=384,
                                   out_channels=64,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_51 = nn.ReLU()
        self.conv2d_52 = nn.Conv2d(in_channels=384,
                                   out_channels=96,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_53 = nn.ReLU()
        self.relu_54 = nn.ReLU()
        self.relu_55 = nn.ReLU()
        self.conv2d_56 = nn.Conv2d(in_channels=64,
                                   out_channels=96,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_57 = nn.Conv2d(in_channels=64,
                                   out_channels=96,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_58 = nn.ReLU()
        self.relu_59 = nn.ReLU()
        self.conv2d_60 = nn.Conv2d(in_channels=96,
                                   out_channels=96,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_61 = nn.ReLU()
        self.concat_62 = P.Concat(axis=1)
        self.batchnorm2d_63 = nn.BatchNorm2d(num_features=384, eps=9.999999974752427e-07, momentum=0.9900000095367432)
        self.relu_64 = nn.ReLU()
        self.conv2d_65 = nn.Conv2d(in_channels=384,
                                   out_channels=96,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.pad_avgpool2d_66 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_66 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.conv2d_67 = nn.Conv2d(in_channels=384,
                                   out_channels=64,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_68 = nn.Conv2d(in_channels=384,
                                   out_channels=64,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_69 = nn.ReLU()
        self.conv2d_70 = nn.Conv2d(in_channels=384,
                                   out_channels=96,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_71 = nn.ReLU()
        self.relu_72 = nn.ReLU()
        self.relu_73 = nn.ReLU()
        self.conv2d_74 = nn.Conv2d(in_channels=64,
                                   out_channels=96,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_75 = nn.Conv2d(in_channels=64,
                                   out_channels=96,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_76 = nn.ReLU()
        self.relu_77 = nn.ReLU()
        self.conv2d_78 = nn.Conv2d(in_channels=96,
                                   out_channels=96,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_79 = nn.ReLU()
        self.concat_80 = P.Concat(axis=1)
        self.batchnorm2d_81 = nn.BatchNorm2d(num_features=384, eps=9.999999974752427e-07, momentum=0.9900000095367432)
        self.relu_82 = nn.ReLU()
        self.conv2d_83 = nn.Conv2d(in_channels=384,
                                   out_channels=96,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.pad_avgpool2d_84 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_84 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.conv2d_85 = nn.Conv2d(in_channels=384,
                                   out_channels=64,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_86 = nn.Conv2d(in_channels=384,
                                   out_channels=64,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_87 = nn.ReLU()
        self.conv2d_88 = nn.Conv2d(in_channels=384,
                                   out_channels=96,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_89 = nn.ReLU()
        self.relu_90 = nn.ReLU()
        self.relu_91 = nn.ReLU()
        self.conv2d_92 = nn.Conv2d(in_channels=64,
                                   out_channels=96,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_93 = nn.Conv2d(in_channels=64,
                                   out_channels=96,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_94 = nn.ReLU()
        self.relu_95 = nn.ReLU()
        self.conv2d_96 = nn.Conv2d(in_channels=96,
                                   out_channels=96,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_97 = nn.ReLU()
        self.concat_98 = P.Concat(axis=1)
        self.batchnorm2d_99 = nn.BatchNorm2d(num_features=384, eps=9.999999974752427e-07, momentum=0.9900000095367432)
        self.relu_100 = nn.ReLU()
        self.pad_maxpool2d_101 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_101 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv2d_102 = nn.Conv2d(in_channels=384,
                                    out_channels=384,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_103 = nn.Conv2d(in_channels=384,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_104 = nn.ReLU()
        self.relu_105 = nn.ReLU()
        self.conv2d_106 = nn.Conv2d(in_channels=192,
                                    out_channels=224,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_107 = nn.ReLU()
        self.conv2d_108 = nn.Conv2d(in_channels=224,
                                    out_channels=256,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_109 = nn.ReLU()
        self.concat_110 = P.Concat(axis=1)
        self.batchnorm2d_111 = nn.BatchNorm2d(num_features=1024, eps=9.999999974752427e-07, momentum=0.9900000095367432)
        self.relu_112 = nn.ReLU()
        self.conv2d_113 = nn.Conv2d(in_channels=1024,
                                    out_channels=384,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.pad_avgpool2d_114 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_114 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.conv2d_115 = nn.Conv2d(in_channels=1024,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_116 = nn.Conv2d(in_channels=1024,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_117 = nn.ReLU()
        self.conv2d_118 = nn.Conv2d(in_channels=1024,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_119 = nn.ReLU()
        self.relu_120 = nn.ReLU()
        self.relu_121 = nn.ReLU()
        self.conv2d_122 = nn.Conv2d(in_channels=192,
                                    out_channels=224,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_123 = nn.Conv2d(in_channels=192,
                                    out_channels=192,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_124 = nn.ReLU()
        self.relu_125 = nn.ReLU()
        self.conv2d_126 = nn.Conv2d(in_channels=224,
                                    out_channels=256,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_127 = nn.Conv2d(in_channels=192,
                                    out_channels=224,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_128 = nn.ReLU()
        self.relu_129 = nn.ReLU()
        self.conv2d_130 = nn.Conv2d(in_channels=224,
                                    out_channels=224,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_131 = nn.ReLU()
        self.conv2d_132 = nn.Conv2d(in_channels=224,
                                    out_channels=256,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_133 = nn.ReLU()
        self.concat_134 = P.Concat(axis=1)
        self.batchnorm2d_135 = nn.BatchNorm2d(num_features=1024, eps=9.999999974752427e-07, momentum=0.9900000095367432)
        self.relu_136 = nn.ReLU()
        self.conv2d_137 = nn.Conv2d(in_channels=1024,
                                    out_channels=384,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.pad_avgpool2d_138 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_138 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.conv2d_139 = nn.Conv2d(in_channels=1024,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_140 = nn.Conv2d(in_channels=1024,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_141 = nn.ReLU()
        self.conv2d_142 = nn.Conv2d(in_channels=1024,
                                    out_channels=128,
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
                                    out_channels=224,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_147 = nn.Conv2d(in_channels=192,
                                    out_channels=192,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_148 = nn.ReLU()
        self.relu_149 = nn.ReLU()
        self.conv2d_150 = nn.Conv2d(in_channels=224,
                                    out_channels=256,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_151 = nn.Conv2d(in_channels=192,
                                    out_channels=224,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_152 = nn.ReLU()
        self.relu_153 = nn.ReLU()
        self.conv2d_154 = nn.Conv2d(in_channels=224,
                                    out_channels=224,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_155 = nn.ReLU()
        self.conv2d_156 = nn.Conv2d(in_channels=224,
                                    out_channels=256,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_157 = nn.ReLU()
        self.concat_158 = P.Concat(axis=1)
        self.batchnorm2d_159 = nn.BatchNorm2d(num_features=1024, eps=9.999999974752427e-07, momentum=0.9900000095367432)
        self.relu_160 = nn.ReLU()
        self.conv2d_161 = nn.Conv2d(in_channels=1024,
                                    out_channels=384,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.pad_avgpool2d_162 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_162 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.conv2d_163 = nn.Conv2d(in_channels=1024,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_164 = nn.Conv2d(in_channels=1024,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_165 = nn.ReLU()
        self.conv2d_166 = nn.Conv2d(in_channels=1024,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_167 = nn.ReLU()
        self.relu_168 = nn.ReLU()
        self.relu_169 = nn.ReLU()
        self.conv2d_170 = nn.Conv2d(in_channels=192,
                                    out_channels=224,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_171 = nn.Conv2d(in_channels=192,
                                    out_channels=192,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_172 = nn.ReLU()
        self.relu_173 = nn.ReLU()
        self.conv2d_174 = nn.Conv2d(in_channels=224,
                                    out_channels=256,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_175 = nn.Conv2d(in_channels=192,
                                    out_channels=224,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_176 = nn.ReLU()
        self.relu_177 = nn.ReLU()
        self.conv2d_178 = nn.Conv2d(in_channels=224,
                                    out_channels=224,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_179 = nn.ReLU()
        self.conv2d_180 = nn.Conv2d(in_channels=224,
                                    out_channels=256,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_181 = nn.ReLU()
        self.concat_182 = P.Concat(axis=1)
        self.batchnorm2d_183 = nn.BatchNorm2d(num_features=1024, eps=9.999999974752427e-07, momentum=0.9900000095367432)
        self.relu_184 = nn.ReLU()
        self.conv2d_185 = nn.Conv2d(in_channels=1024,
                                    out_channels=384,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.pad_avgpool2d_186 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_186 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.conv2d_187 = nn.Conv2d(in_channels=1024,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_188 = nn.Conv2d(in_channels=1024,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_189 = nn.ReLU()
        self.conv2d_190 = nn.Conv2d(in_channels=1024,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_191 = nn.ReLU()
        self.relu_192 = nn.ReLU()
        self.relu_193 = nn.ReLU()
        self.conv2d_194 = nn.Conv2d(in_channels=192,
                                    out_channels=224,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_195 = nn.Conv2d(in_channels=192,
                                    out_channels=192,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_196 = nn.ReLU()
        self.relu_197 = nn.ReLU()
        self.conv2d_198 = nn.Conv2d(in_channels=224,
                                    out_channels=256,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_199 = nn.Conv2d(in_channels=192,
                                    out_channels=224,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_200 = nn.ReLU()
        self.relu_201 = nn.ReLU()
        self.conv2d_202 = nn.Conv2d(in_channels=224,
                                    out_channels=224,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_203 = nn.ReLU()
        self.conv2d_204 = nn.Conv2d(in_channels=224,
                                    out_channels=256,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_205 = nn.ReLU()
        self.concat_206 = P.Concat(axis=1)
        self.batchnorm2d_207 = nn.BatchNorm2d(num_features=1024, eps=9.999999974752427e-07, momentum=0.9900000095367432)
        self.relu_208 = nn.ReLU()
        self.conv2d_209 = nn.Conv2d(in_channels=1024,
                                    out_channels=384,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.pad_avgpool2d_210 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_210 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.conv2d_211 = nn.Conv2d(in_channels=1024,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_212 = nn.Conv2d(in_channels=1024,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_213 = nn.ReLU()
        self.conv2d_214 = nn.Conv2d(in_channels=1024,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_215 = nn.ReLU()
        self.relu_216 = nn.ReLU()
        self.relu_217 = nn.ReLU()
        self.conv2d_218 = nn.Conv2d(in_channels=192,
                                    out_channels=224,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_219 = nn.Conv2d(in_channels=192,
                                    out_channels=192,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_220 = nn.ReLU()
        self.relu_221 = nn.ReLU()
        self.conv2d_222 = nn.Conv2d(in_channels=224,
                                    out_channels=256,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_223 = nn.Conv2d(in_channels=192,
                                    out_channels=224,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_224 = nn.ReLU()
        self.relu_225 = nn.ReLU()
        self.conv2d_226 = nn.Conv2d(in_channels=224,
                                    out_channels=224,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_227 = nn.ReLU()
        self.conv2d_228 = nn.Conv2d(in_channels=224,
                                    out_channels=256,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_229 = nn.ReLU()
        self.concat_230 = P.Concat(axis=1)
        self.batchnorm2d_231 = nn.BatchNorm2d(num_features=1024, eps=9.999999974752427e-07, momentum=0.9900000095367432)
        self.relu_232 = nn.ReLU()
        self.conv2d_233 = nn.Conv2d(in_channels=1024,
                                    out_channels=384,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.pad_avgpool2d_234 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_234 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.conv2d_235 = nn.Conv2d(in_channels=1024,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_236 = nn.Conv2d(in_channels=1024,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_237 = nn.ReLU()
        self.conv2d_238 = nn.Conv2d(in_channels=1024,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_239 = nn.ReLU()
        self.relu_240 = nn.ReLU()
        self.relu_241 = nn.ReLU()
        self.conv2d_242 = nn.Conv2d(in_channels=192,
                                    out_channels=224,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_243 = nn.Conv2d(in_channels=192,
                                    out_channels=192,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_244 = nn.ReLU()
        self.relu_245 = nn.ReLU()
        self.conv2d_246 = nn.Conv2d(in_channels=224,
                                    out_channels=256,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_247 = nn.Conv2d(in_channels=192,
                                    out_channels=224,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_248 = nn.ReLU()
        self.relu_249 = nn.ReLU()
        self.conv2d_250 = nn.Conv2d(in_channels=224,
                                    out_channels=224,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_251 = nn.ReLU()
        self.conv2d_252 = nn.Conv2d(in_channels=224,
                                    out_channels=256,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_253 = nn.ReLU()
        self.concat_254 = P.Concat(axis=1)
        self.batchnorm2d_255 = nn.BatchNorm2d(num_features=1024, eps=9.999999974752427e-07, momentum=0.9900000095367432)
        self.relu_256 = nn.ReLU()
        self.conv2d_257 = nn.Conv2d(in_channels=1024,
                                    out_channels=384,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.pad_avgpool2d_258 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_258 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.conv2d_259 = nn.Conv2d(in_channels=1024,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_260 = nn.Conv2d(in_channels=1024,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_261 = nn.ReLU()
        self.conv2d_262 = nn.Conv2d(in_channels=1024,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_263 = nn.ReLU()
        self.relu_264 = nn.ReLU()
        self.relu_265 = nn.ReLU()
        self.conv2d_266 = nn.Conv2d(in_channels=192,
                                    out_channels=224,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_267 = nn.Conv2d(in_channels=192,
                                    out_channels=192,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_268 = nn.ReLU()
        self.relu_269 = nn.ReLU()
        self.conv2d_270 = nn.Conv2d(in_channels=224,
                                    out_channels=256,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_271 = nn.Conv2d(in_channels=192,
                                    out_channels=224,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_272 = nn.ReLU()
        self.relu_273 = nn.ReLU()
        self.conv2d_274 = nn.Conv2d(in_channels=224,
                                    out_channels=224,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_275 = nn.ReLU()
        self.conv2d_276 = nn.Conv2d(in_channels=224,
                                    out_channels=256,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_277 = nn.ReLU()
        self.concat_278 = P.Concat(axis=1)
        self.batchnorm2d_279 = nn.BatchNorm2d(num_features=1024, eps=9.999999974752427e-07, momentum=0.9900000095367432)
        self.relu_280 = nn.ReLU()
        self.pad_maxpool2d_281 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.maxpool2d_281 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv2d_282 = nn.Conv2d(in_channels=1024,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_283 = nn.Conv2d(in_channels=1024,
                                    out_channels=256,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_284 = nn.ReLU()
        self.relu_285 = nn.ReLU()
        self.conv2d_286 = nn.Conv2d(in_channels=192,
                                    out_channels=192,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_287 = nn.Conv2d(in_channels=256,
                                    out_channels=256,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_288 = nn.ReLU()
        self.relu_289 = nn.ReLU()
        self.conv2d_290 = nn.Conv2d(in_channels=256,
                                    out_channels=320,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_291 = nn.ReLU()
        self.conv2d_292 = nn.Conv2d(in_channels=320,
                                    out_channels=320,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_293 = nn.ReLU()
        self.concat_294 = P.Concat(axis=1)
        self.batchnorm2d_295 = nn.BatchNorm2d(num_features=1536, eps=9.999999974752427e-07, momentum=0.9900000095367432)
        self.relu_296 = nn.ReLU()
        self.conv2d_297 = nn.Conv2d(in_channels=1536,
                                    out_channels=256,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.pad_avgpool2d_298 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_298 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.conv2d_299 = nn.Conv2d(in_channels=1536,
                                    out_channels=384,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_300 = nn.Conv2d(in_channels=1536,
                                    out_channels=384,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_301 = nn.ReLU()
        self.conv2d_302 = nn.Conv2d(in_channels=1536,
                                    out_channels=256,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_303 = nn.ReLU()
        self.relu_304 = nn.ReLU()
        self.relu_305 = nn.ReLU()
        self.conv2d_306 = nn.Conv2d(in_channels=384,
                                    out_channels=256,
                                    kernel_size=(1, 3),
                                    stride=(1, 1),
                                    padding=(0, 0, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_307 = nn.Conv2d(in_channels=384,
                                    out_channels=256,
                                    kernel_size=(3, 1),
                                    stride=(1, 1),
                                    padding=(1, 1, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_308 = nn.Conv2d(in_channels=384,
                                    out_channels=448,
                                    kernel_size=(1, 3),
                                    stride=(1, 1),
                                    padding=(0, 0, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_309 = nn.ReLU()
        self.relu_310 = nn.ReLU()
        self.relu_311 = nn.ReLU()
        self.conv2d_312 = nn.Conv2d(in_channels=448,
                                    out_channels=512,
                                    kernel_size=(3, 1),
                                    stride=(1, 1),
                                    padding=(1, 1, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_313 = nn.ReLU()
        self.conv2d_314 = nn.Conv2d(in_channels=512,
                                    out_channels=256,
                                    kernel_size=(3, 1),
                                    stride=(1, 1),
                                    padding=(1, 1, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_315 = nn.Conv2d(in_channels=512,
                                    out_channels=256,
                                    kernel_size=(1, 3),
                                    stride=(1, 1),
                                    padding=(0, 0, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_316 = nn.ReLU()
        self.relu_317 = nn.ReLU()
        self.concat_318 = P.Concat(axis=1)
        self.batchnorm2d_319 = nn.BatchNorm2d(num_features=1536, eps=9.999999974752427e-07, momentum=0.9900000095367432)
        self.relu_320 = nn.ReLU()
        self.conv2d_321 = nn.Conv2d(in_channels=1536,
                                    out_channels=256,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.pad_avgpool2d_322 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_322 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.conv2d_323 = nn.Conv2d(in_channels=1536,
                                    out_channels=384,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_324 = nn.Conv2d(in_channels=1536,
                                    out_channels=384,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_325 = nn.ReLU()
        self.conv2d_326 = nn.Conv2d(in_channels=1536,
                                    out_channels=256,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_327 = nn.ReLU()
        self.relu_328 = nn.ReLU()
        self.relu_329 = nn.ReLU()
        self.conv2d_330 = nn.Conv2d(in_channels=384,
                                    out_channels=256,
                                    kernel_size=(1, 3),
                                    stride=(1, 1),
                                    padding=(0, 0, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_331 = nn.Conv2d(in_channels=384,
                                    out_channels=256,
                                    kernel_size=(3, 1),
                                    stride=(1, 1),
                                    padding=(1, 1, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_332 = nn.Conv2d(in_channels=384,
                                    out_channels=448,
                                    kernel_size=(1, 3),
                                    stride=(1, 1),
                                    padding=(0, 0, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_333 = nn.ReLU()
        self.relu_334 = nn.ReLU()
        self.relu_335 = nn.ReLU()
        self.conv2d_336 = nn.Conv2d(in_channels=448,
                                    out_channels=512,
                                    kernel_size=(3, 1),
                                    stride=(1, 1),
                                    padding=(1, 1, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_337 = nn.ReLU()
        self.conv2d_338 = nn.Conv2d(in_channels=512,
                                    out_channels=256,
                                    kernel_size=(3, 1),
                                    stride=(1, 1),
                                    padding=(1, 1, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_339 = nn.Conv2d(in_channels=512,
                                    out_channels=256,
                                    kernel_size=(1, 3),
                                    stride=(1, 1),
                                    padding=(0, 0, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_340 = nn.ReLU()
        self.relu_341 = nn.ReLU()
        self.concat_342 = P.Concat(axis=1)
        self.batchnorm2d_343 = nn.BatchNorm2d(num_features=1536, eps=9.999999974752427e-07, momentum=0.9900000095367432)
        self.relu_344 = nn.ReLU()
        self.conv2d_345 = nn.Conv2d(in_channels=1536,
                                    out_channels=256,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.pad_avgpool2d_346 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
        self.avgpool2d_346 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.conv2d_347 = nn.Conv2d(in_channels=1536,
                                    out_channels=384,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_348 = nn.Conv2d(in_channels=1536,
                                    out_channels=384,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_349 = nn.ReLU()
        self.conv2d_350 = nn.Conv2d(in_channels=1536,
                                    out_channels=256,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_351 = nn.ReLU()
        self.relu_352 = nn.ReLU()
        self.relu_353 = nn.ReLU()
        self.conv2d_354 = nn.Conv2d(in_channels=384,
                                    out_channels=256,
                                    kernel_size=(1, 3),
                                    stride=(1, 1),
                                    padding=(0, 0, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_355 = nn.Conv2d(in_channels=384,
                                    out_channels=256,
                                    kernel_size=(3, 1),
                                    stride=(1, 1),
                                    padding=(1, 1, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_356 = nn.Conv2d(in_channels=384,
                                    out_channels=448,
                                    kernel_size=(1, 3),
                                    stride=(1, 1),
                                    padding=(0, 0, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_357 = nn.ReLU()
        self.relu_358 = nn.ReLU()
        self.relu_359 = nn.ReLU()
        self.conv2d_360 = nn.Conv2d(in_channels=448,
                                    out_channels=512,
                                    kernel_size=(3, 1),
                                    stride=(1, 1),
                                    padding=(1, 1, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_361 = nn.ReLU()
        self.conv2d_362 = nn.Conv2d(in_channels=512,
                                    out_channels=256,
                                    kernel_size=(3, 1),
                                    stride=(1, 1),
                                    padding=(1, 1, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_363 = nn.Conv2d(in_channels=512,
                                    out_channels=256,
                                    kernel_size=(1, 3),
                                    stride=(1, 1),
                                    padding=(0, 0, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_364 = nn.ReLU()
        self.relu_365 = nn.ReLU()
        self.concat_366 = P.Concat(axis=1)
        self.batchnorm2d_367 = nn.BatchNorm2d(num_features=1536, eps=9.999999974752427e-07, momentum=0.9900000095367432)
        self.relu_368 = nn.ReLU()
        self.pad_avgpool2d_369 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_369 = nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1))
        self.transpose_370 = P.Transpose()
        self.flatten_371 = nn.Flatten()
        self.matmul_372_w = Parameter(Tensor(np.random.uniform(0, 1, (1536, 10)).astype(np.float32)), name=None)
        self.add_373_bias = Parameter(Tensor(np.random.uniform(0, 1, (10, )).astype(np.float32)), name=None)
        self.softmax_374 = nn.Softmax(axis=-1)

    def construct(self, input_5):
        opt_transpose_0 = self.transpose_0(input_5, (0, 3, 1, 2))
        opt_conv2d_1 = self.conv2d_1(opt_transpose_0)
        opt_relu_2 = self.relu_2(opt_conv2d_1)
        opt_conv2d_3 = self.conv2d_3(opt_relu_2)
        opt_relu_4 = self.relu_4(opt_conv2d_3)
        opt_conv2d_5 = self.conv2d_5(opt_relu_4)
        opt_relu_6 = self.relu_6(opt_conv2d_5)
        opt_maxpool2d_7 = self.pad_maxpool2d_7(opt_relu_6)
        opt_maxpool2d_7 = self.maxpool2d_7(opt_maxpool2d_7)
        opt_conv2d_8 = self.conv2d_8(opt_relu_6)
        opt_relu_9 = self.relu_9(opt_conv2d_8)
        opt_concat_10 = self.concat_10((opt_maxpool2d_7, opt_relu_9, ))
        opt_conv2d_11 = self.conv2d_11(opt_concat_10)
        opt_conv2d_12 = self.conv2d_12(opt_concat_10)
        opt_relu_13 = self.relu_13(opt_conv2d_11)
        opt_relu_14 = self.relu_14(opt_conv2d_12)
        opt_conv2d_15 = self.conv2d_15(opt_relu_13)
        opt_conv2d_16 = self.conv2d_16(opt_relu_14)
        opt_relu_17 = self.relu_17(opt_conv2d_15)
        opt_relu_18 = self.relu_18(opt_conv2d_16)
        opt_conv2d_19 = self.conv2d_19(opt_relu_18)
        opt_relu_20 = self.relu_20(opt_conv2d_19)
        opt_conv2d_21 = self.conv2d_21(opt_relu_20)
        opt_relu_22 = self.relu_22(opt_conv2d_21)
        opt_concat_23 = self.concat_23((opt_relu_17, opt_relu_22, ))
        opt_maxpool2d_24 = self.pad_maxpool2d_24(opt_concat_23)
        opt_maxpool2d_24 = self.maxpool2d_24(opt_maxpool2d_24)
        opt_conv2d_25 = self.conv2d_25(opt_concat_23)
        opt_relu_26 = self.relu_26(opt_conv2d_25)
        opt_concat_27 = self.concat_27((opt_relu_26, opt_maxpool2d_24, ))
        opt_relu_28 = self.relu_28(opt_concat_27)
        opt_conv2d_29 = self.conv2d_29(opt_relu_28)
        opt_avgpool2d_30 = self.pad_avgpool2d_30(opt_relu_28)
        opt_avgpool2d_30 = self.avgpool2d_30(opt_avgpool2d_30)
        opt_conv2d_31 = self.conv2d_31(opt_relu_28)
        opt_conv2d_32 = self.conv2d_32(opt_relu_28)
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
        opt_concat_44 = self.concat_44((opt_relu_37, opt_relu_33, opt_relu_40, opt_relu_43, ))
        opt_batchnorm2d_45 = self.batchnorm2d_45(opt_concat_44)
        opt_relu_46 = self.relu_46(opt_batchnorm2d_45)
        opt_conv2d_47 = self.conv2d_47(opt_relu_46)
        opt_avgpool2d_48 = self.pad_avgpool2d_48(opt_relu_46)
        opt_avgpool2d_48 = self.avgpool2d_48(opt_avgpool2d_48)
        opt_conv2d_49 = self.conv2d_49(opt_relu_46)
        opt_conv2d_50 = self.conv2d_50(opt_relu_46)
        opt_relu_51 = self.relu_51(opt_conv2d_47)
        opt_conv2d_52 = self.conv2d_52(opt_avgpool2d_48)
        opt_relu_53 = self.relu_53(opt_conv2d_49)
        opt_relu_54 = self.relu_54(opt_conv2d_50)
        opt_relu_55 = self.relu_55(opt_conv2d_52)
        opt_conv2d_56 = self.conv2d_56(opt_relu_53)
        opt_conv2d_57 = self.conv2d_57(opt_relu_54)
        opt_relu_58 = self.relu_58(opt_conv2d_56)
        opt_relu_59 = self.relu_59(opt_conv2d_57)
        opt_conv2d_60 = self.conv2d_60(opt_relu_59)
        opt_relu_61 = self.relu_61(opt_conv2d_60)
        opt_concat_62 = self.concat_62((opt_relu_55, opt_relu_51, opt_relu_58, opt_relu_61, ))
        opt_batchnorm2d_63 = self.batchnorm2d_63(opt_concat_62)
        opt_relu_64 = self.relu_64(opt_batchnorm2d_63)
        opt_conv2d_65 = self.conv2d_65(opt_relu_64)
        opt_avgpool2d_66 = self.pad_avgpool2d_66(opt_relu_64)
        opt_avgpool2d_66 = self.avgpool2d_66(opt_avgpool2d_66)
        opt_conv2d_67 = self.conv2d_67(opt_relu_64)
        opt_conv2d_68 = self.conv2d_68(opt_relu_64)
        opt_relu_69 = self.relu_69(opt_conv2d_65)
        opt_conv2d_70 = self.conv2d_70(opt_avgpool2d_66)
        opt_relu_71 = self.relu_71(opt_conv2d_67)
        opt_relu_72 = self.relu_72(opt_conv2d_68)
        opt_relu_73 = self.relu_73(opt_conv2d_70)
        opt_conv2d_74 = self.conv2d_74(opt_relu_71)
        opt_conv2d_75 = self.conv2d_75(opt_relu_72)
        opt_relu_76 = self.relu_76(opt_conv2d_74)
        opt_relu_77 = self.relu_77(opt_conv2d_75)
        opt_conv2d_78 = self.conv2d_78(opt_relu_77)
        opt_relu_79 = self.relu_79(opt_conv2d_78)
        opt_concat_80 = self.concat_80((opt_relu_73, opt_relu_69, opt_relu_76, opt_relu_79, ))
        opt_batchnorm2d_81 = self.batchnorm2d_81(opt_concat_80)
        opt_relu_82 = self.relu_82(opt_batchnorm2d_81)
        opt_conv2d_83 = self.conv2d_83(opt_relu_82)
        opt_avgpool2d_84 = self.pad_avgpool2d_84(opt_relu_82)
        opt_avgpool2d_84 = self.avgpool2d_84(opt_avgpool2d_84)
        opt_conv2d_85 = self.conv2d_85(opt_relu_82)
        opt_conv2d_86 = self.conv2d_86(opt_relu_82)
        opt_relu_87 = self.relu_87(opt_conv2d_83)
        opt_conv2d_88 = self.conv2d_88(opt_avgpool2d_84)
        opt_relu_89 = self.relu_89(opt_conv2d_85)
        opt_relu_90 = self.relu_90(opt_conv2d_86)
        opt_relu_91 = self.relu_91(opt_conv2d_88)
        opt_conv2d_92 = self.conv2d_92(opt_relu_89)
        opt_conv2d_93 = self.conv2d_93(opt_relu_90)
        opt_relu_94 = self.relu_94(opt_conv2d_92)
        opt_relu_95 = self.relu_95(opt_conv2d_93)
        opt_conv2d_96 = self.conv2d_96(opt_relu_95)
        opt_relu_97 = self.relu_97(opt_conv2d_96)
        opt_concat_98 = self.concat_98((opt_relu_91, opt_relu_87, opt_relu_94, opt_relu_97, ))
        opt_batchnorm2d_99 = self.batchnorm2d_99(opt_concat_98)
        opt_relu_100 = self.relu_100(opt_batchnorm2d_99)
        opt_maxpool2d_101 = self.pad_maxpool2d_101(opt_relu_100)
        opt_maxpool2d_101 = self.maxpool2d_101(opt_maxpool2d_101)
        opt_conv2d_102 = self.conv2d_102(opt_relu_100)
        opt_conv2d_103 = self.conv2d_103(opt_relu_100)
        opt_relu_104 = self.relu_104(opt_conv2d_102)
        opt_relu_105 = self.relu_105(opt_conv2d_103)
        opt_conv2d_106 = self.conv2d_106(opt_relu_105)
        opt_relu_107 = self.relu_107(opt_conv2d_106)
        opt_conv2d_108 = self.conv2d_108(opt_relu_107)
        opt_relu_109 = self.relu_109(opt_conv2d_108)
        opt_concat_110 = self.concat_110((opt_maxpool2d_101, opt_relu_104, opt_relu_109, ))
        opt_batchnorm2d_111 = self.batchnorm2d_111(opt_concat_110)
        opt_relu_112 = self.relu_112(opt_batchnorm2d_111)
        opt_conv2d_113 = self.conv2d_113(opt_relu_112)
        opt_avgpool2d_114 = self.pad_avgpool2d_114(opt_relu_112)
        opt_avgpool2d_114 = self.avgpool2d_114(opt_avgpool2d_114)
        opt_conv2d_115 = self.conv2d_115(opt_relu_112)
        opt_conv2d_116 = self.conv2d_116(opt_relu_112)
        opt_relu_117 = self.relu_117(opt_conv2d_113)
        opt_conv2d_118 = self.conv2d_118(opt_avgpool2d_114)
        opt_relu_119 = self.relu_119(opt_conv2d_115)
        opt_relu_120 = self.relu_120(opt_conv2d_116)
        opt_relu_121 = self.relu_121(opt_conv2d_118)
        opt_conv2d_122 = self.conv2d_122(opt_relu_119)
        opt_conv2d_123 = self.conv2d_123(opt_relu_120)
        opt_relu_124 = self.relu_124(opt_conv2d_122)
        opt_relu_125 = self.relu_125(opt_conv2d_123)
        opt_conv2d_126 = self.conv2d_126(opt_relu_124)
        opt_conv2d_127 = self.conv2d_127(opt_relu_125)
        opt_relu_128 = self.relu_128(opt_conv2d_126)
        opt_relu_129 = self.relu_129(opt_conv2d_127)
        opt_conv2d_130 = self.conv2d_130(opt_relu_129)
        opt_relu_131 = self.relu_131(opt_conv2d_130)
        opt_conv2d_132 = self.conv2d_132(opt_relu_131)
        opt_relu_133 = self.relu_133(opt_conv2d_132)
        opt_concat_134 = self.concat_134((opt_relu_121, opt_relu_117, opt_relu_128, opt_relu_133, ))
        opt_batchnorm2d_135 = self.batchnorm2d_135(opt_concat_134)
        opt_relu_136 = self.relu_136(opt_batchnorm2d_135)
        opt_conv2d_137 = self.conv2d_137(opt_relu_136)
        opt_avgpool2d_138 = self.pad_avgpool2d_138(opt_relu_136)
        opt_avgpool2d_138 = self.avgpool2d_138(opt_avgpool2d_138)
        opt_conv2d_139 = self.conv2d_139(opt_relu_136)
        opt_conv2d_140 = self.conv2d_140(opt_relu_136)
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
        opt_concat_158 = self.concat_158((opt_relu_145, opt_relu_141, opt_relu_152, opt_relu_157, ))
        opt_batchnorm2d_159 = self.batchnorm2d_159(opt_concat_158)
        opt_relu_160 = self.relu_160(opt_batchnorm2d_159)
        opt_conv2d_161 = self.conv2d_161(opt_relu_160)
        opt_avgpool2d_162 = self.pad_avgpool2d_162(opt_relu_160)
        opt_avgpool2d_162 = self.avgpool2d_162(opt_avgpool2d_162)
        opt_conv2d_163 = self.conv2d_163(opt_relu_160)
        opt_conv2d_164 = self.conv2d_164(opt_relu_160)
        opt_relu_165 = self.relu_165(opt_conv2d_161)
        opt_conv2d_166 = self.conv2d_166(opt_avgpool2d_162)
        opt_relu_167 = self.relu_167(opt_conv2d_163)
        opt_relu_168 = self.relu_168(opt_conv2d_164)
        opt_relu_169 = self.relu_169(opt_conv2d_166)
        opt_conv2d_170 = self.conv2d_170(opt_relu_167)
        opt_conv2d_171 = self.conv2d_171(opt_relu_168)
        opt_relu_172 = self.relu_172(opt_conv2d_170)
        opt_relu_173 = self.relu_173(opt_conv2d_171)
        opt_conv2d_174 = self.conv2d_174(opt_relu_172)
        opt_conv2d_175 = self.conv2d_175(opt_relu_173)
        opt_relu_176 = self.relu_176(opt_conv2d_174)
        opt_relu_177 = self.relu_177(opt_conv2d_175)
        opt_conv2d_178 = self.conv2d_178(opt_relu_177)
        opt_relu_179 = self.relu_179(opt_conv2d_178)
        opt_conv2d_180 = self.conv2d_180(opt_relu_179)
        opt_relu_181 = self.relu_181(opt_conv2d_180)
        opt_concat_182 = self.concat_182((opt_relu_169, opt_relu_165, opt_relu_176, opt_relu_181, ))
        opt_batchnorm2d_183 = self.batchnorm2d_183(opt_concat_182)
        opt_relu_184 = self.relu_184(opt_batchnorm2d_183)
        opt_conv2d_185 = self.conv2d_185(opt_relu_184)
        opt_avgpool2d_186 = self.pad_avgpool2d_186(opt_relu_184)
        opt_avgpool2d_186 = self.avgpool2d_186(opt_avgpool2d_186)
        opt_conv2d_187 = self.conv2d_187(opt_relu_184)
        opt_conv2d_188 = self.conv2d_188(opt_relu_184)
        opt_relu_189 = self.relu_189(opt_conv2d_185)
        opt_conv2d_190 = self.conv2d_190(opt_avgpool2d_186)
        opt_relu_191 = self.relu_191(opt_conv2d_187)
        opt_relu_192 = self.relu_192(opt_conv2d_188)
        opt_relu_193 = self.relu_193(opt_conv2d_190)
        opt_conv2d_194 = self.conv2d_194(opt_relu_191)
        opt_conv2d_195 = self.conv2d_195(opt_relu_192)
        opt_relu_196 = self.relu_196(opt_conv2d_194)
        opt_relu_197 = self.relu_197(opt_conv2d_195)
        opt_conv2d_198 = self.conv2d_198(opt_relu_196)
        opt_conv2d_199 = self.conv2d_199(opt_relu_197)
        opt_relu_200 = self.relu_200(opt_conv2d_198)
        opt_relu_201 = self.relu_201(opt_conv2d_199)
        opt_conv2d_202 = self.conv2d_202(opt_relu_201)
        opt_relu_203 = self.relu_203(opt_conv2d_202)
        opt_conv2d_204 = self.conv2d_204(opt_relu_203)
        opt_relu_205 = self.relu_205(opt_conv2d_204)
        opt_concat_206 = self.concat_206((opt_relu_193, opt_relu_189, opt_relu_200, opt_relu_205, ))
        opt_batchnorm2d_207 = self.batchnorm2d_207(opt_concat_206)
        opt_relu_208 = self.relu_208(opt_batchnorm2d_207)
        opt_conv2d_209 = self.conv2d_209(opt_relu_208)
        opt_avgpool2d_210 = self.pad_avgpool2d_210(opt_relu_208)
        opt_avgpool2d_210 = self.avgpool2d_210(opt_avgpool2d_210)
        opt_conv2d_211 = self.conv2d_211(opt_relu_208)
        opt_conv2d_212 = self.conv2d_212(opt_relu_208)
        opt_relu_213 = self.relu_213(opt_conv2d_209)
        opt_conv2d_214 = self.conv2d_214(opt_avgpool2d_210)
        opt_relu_215 = self.relu_215(opt_conv2d_211)
        opt_relu_216 = self.relu_216(opt_conv2d_212)
        opt_relu_217 = self.relu_217(opt_conv2d_214)
        opt_conv2d_218 = self.conv2d_218(opt_relu_215)
        opt_conv2d_219 = self.conv2d_219(opt_relu_216)
        opt_relu_220 = self.relu_220(opt_conv2d_218)
        opt_relu_221 = self.relu_221(opt_conv2d_219)
        opt_conv2d_222 = self.conv2d_222(opt_relu_220)
        opt_conv2d_223 = self.conv2d_223(opt_relu_221)
        opt_relu_224 = self.relu_224(opt_conv2d_222)
        opt_relu_225 = self.relu_225(opt_conv2d_223)
        opt_conv2d_226 = self.conv2d_226(opt_relu_225)
        opt_relu_227 = self.relu_227(opt_conv2d_226)
        opt_conv2d_228 = self.conv2d_228(opt_relu_227)
        opt_relu_229 = self.relu_229(opt_conv2d_228)
        opt_concat_230 = self.concat_230((opt_relu_217, opt_relu_213, opt_relu_224, opt_relu_229, ))
        opt_batchnorm2d_231 = self.batchnorm2d_231(opt_concat_230)
        opt_relu_232 = self.relu_232(opt_batchnorm2d_231)
        opt_conv2d_233 = self.conv2d_233(opt_relu_232)
        opt_avgpool2d_234 = self.pad_avgpool2d_234(opt_relu_232)
        opt_avgpool2d_234 = self.avgpool2d_234(opt_avgpool2d_234)
        opt_conv2d_235 = self.conv2d_235(opt_relu_232)
        opt_conv2d_236 = self.conv2d_236(opt_relu_232)
        opt_relu_237 = self.relu_237(opt_conv2d_233)
        opt_conv2d_238 = self.conv2d_238(opt_avgpool2d_234)
        opt_relu_239 = self.relu_239(opt_conv2d_235)
        opt_relu_240 = self.relu_240(opt_conv2d_236)
        opt_relu_241 = self.relu_241(opt_conv2d_238)
        opt_conv2d_242 = self.conv2d_242(opt_relu_239)
        opt_conv2d_243 = self.conv2d_243(opt_relu_240)
        opt_relu_244 = self.relu_244(opt_conv2d_242)
        opt_relu_245 = self.relu_245(opt_conv2d_243)
        opt_conv2d_246 = self.conv2d_246(opt_relu_244)
        opt_conv2d_247 = self.conv2d_247(opt_relu_245)
        opt_relu_248 = self.relu_248(opt_conv2d_246)
        opt_relu_249 = self.relu_249(opt_conv2d_247)
        opt_conv2d_250 = self.conv2d_250(opt_relu_249)
        opt_relu_251 = self.relu_251(opt_conv2d_250)
        opt_conv2d_252 = self.conv2d_252(opt_relu_251)
        opt_relu_253 = self.relu_253(opt_conv2d_252)
        opt_concat_254 = self.concat_254((opt_relu_241, opt_relu_237, opt_relu_248, opt_relu_253, ))
        opt_batchnorm2d_255 = self.batchnorm2d_255(opt_concat_254)
        opt_relu_256 = self.relu_256(opt_batchnorm2d_255)
        opt_conv2d_257 = self.conv2d_257(opt_relu_256)
        opt_avgpool2d_258 = self.pad_avgpool2d_258(opt_relu_256)
        opt_avgpool2d_258 = self.avgpool2d_258(opt_avgpool2d_258)
        opt_conv2d_259 = self.conv2d_259(opt_relu_256)
        opt_conv2d_260 = self.conv2d_260(opt_relu_256)
        opt_relu_261 = self.relu_261(opt_conv2d_257)
        opt_conv2d_262 = self.conv2d_262(opt_avgpool2d_258)
        opt_relu_263 = self.relu_263(opt_conv2d_259)
        opt_relu_264 = self.relu_264(opt_conv2d_260)
        opt_relu_265 = self.relu_265(opt_conv2d_262)
        opt_conv2d_266 = self.conv2d_266(opt_relu_263)
        opt_conv2d_267 = self.conv2d_267(opt_relu_264)
        opt_relu_268 = self.relu_268(opt_conv2d_266)
        opt_relu_269 = self.relu_269(opt_conv2d_267)
        opt_conv2d_270 = self.conv2d_270(opt_relu_268)
        opt_conv2d_271 = self.conv2d_271(opt_relu_269)
        opt_relu_272 = self.relu_272(opt_conv2d_270)
        opt_relu_273 = self.relu_273(opt_conv2d_271)
        opt_conv2d_274 = self.conv2d_274(opt_relu_273)
        opt_relu_275 = self.relu_275(opt_conv2d_274)
        opt_conv2d_276 = self.conv2d_276(opt_relu_275)
        opt_relu_277 = self.relu_277(opt_conv2d_276)
        opt_concat_278 = self.concat_278((opt_relu_265, opt_relu_261, opt_relu_272, opt_relu_277, ))
        opt_batchnorm2d_279 = self.batchnorm2d_279(opt_concat_278)
        opt_relu_280 = self.relu_280(opt_batchnorm2d_279)
        opt_maxpool2d_281 = self.pad_maxpool2d_281(opt_relu_280)
        opt_maxpool2d_281 = self.maxpool2d_281(opt_maxpool2d_281)
        opt_conv2d_282 = self.conv2d_282(opt_relu_280)
        opt_conv2d_283 = self.conv2d_283(opt_relu_280)
        opt_relu_284 = self.relu_284(opt_conv2d_282)
        opt_relu_285 = self.relu_285(opt_conv2d_283)
        opt_conv2d_286 = self.conv2d_286(opt_relu_284)
        opt_conv2d_287 = self.conv2d_287(opt_relu_285)
        opt_relu_288 = self.relu_288(opt_conv2d_286)
        opt_relu_289 = self.relu_289(opt_conv2d_287)
        opt_conv2d_290 = self.conv2d_290(opt_relu_289)
        opt_relu_291 = self.relu_291(opt_conv2d_290)
        opt_conv2d_292 = self.conv2d_292(opt_relu_291)
        opt_relu_293 = self.relu_293(opt_conv2d_292)
        opt_concat_294 = self.concat_294((opt_maxpool2d_281, opt_relu_288, opt_relu_293, ))
        opt_batchnorm2d_295 = self.batchnorm2d_295(opt_concat_294)
        opt_relu_296 = self.relu_296(opt_batchnorm2d_295)
        opt_conv2d_297 = self.conv2d_297(opt_relu_296)
        opt_avgpool2d_298 = self.pad_avgpool2d_298(opt_relu_296)
        opt_avgpool2d_298 = self.avgpool2d_298(opt_avgpool2d_298)
        opt_conv2d_299 = self.conv2d_299(opt_relu_296)
        opt_conv2d_300 = self.conv2d_300(opt_relu_296)
        opt_relu_301 = self.relu_301(opt_conv2d_297)
        opt_conv2d_302 = self.conv2d_302(opt_avgpool2d_298)
        opt_relu_303 = self.relu_303(opt_conv2d_299)
        opt_relu_304 = self.relu_304(opt_conv2d_300)
        opt_relu_305 = self.relu_305(opt_conv2d_302)
        opt_conv2d_306 = self.conv2d_306(opt_relu_303)
        opt_conv2d_307 = self.conv2d_307(opt_relu_303)
        opt_conv2d_308 = self.conv2d_308(opt_relu_304)
        opt_relu_309 = self.relu_309(opt_conv2d_306)
        opt_relu_310 = self.relu_310(opt_conv2d_307)
        opt_relu_311 = self.relu_311(opt_conv2d_308)
        opt_conv2d_312 = self.conv2d_312(opt_relu_311)
        opt_relu_313 = self.relu_313(opt_conv2d_312)
        opt_conv2d_314 = self.conv2d_314(opt_relu_313)
        opt_conv2d_315 = self.conv2d_315(opt_relu_313)
        opt_relu_316 = self.relu_316(opt_conv2d_314)
        opt_relu_317 = self.relu_317(opt_conv2d_315)
        opt_concat_318 = self.concat_318(
            (opt_relu_305, opt_relu_301, opt_relu_309, opt_relu_310, opt_relu_316, opt_relu_317,
             ))
        opt_batchnorm2d_319 = self.batchnorm2d_319(opt_concat_318)
        opt_relu_320 = self.relu_320(opt_batchnorm2d_319)
        opt_conv2d_321 = self.conv2d_321(opt_relu_320)
        opt_avgpool2d_322 = self.pad_avgpool2d_322(opt_relu_320)
        opt_avgpool2d_322 = self.avgpool2d_322(opt_avgpool2d_322)
        opt_conv2d_323 = self.conv2d_323(opt_relu_320)
        opt_conv2d_324 = self.conv2d_324(opt_relu_320)
        opt_relu_325 = self.relu_325(opt_conv2d_321)
        opt_conv2d_326 = self.conv2d_326(opt_avgpool2d_322)
        opt_relu_327 = self.relu_327(opt_conv2d_323)
        opt_relu_328 = self.relu_328(opt_conv2d_324)
        opt_relu_329 = self.relu_329(opt_conv2d_326)
        opt_conv2d_330 = self.conv2d_330(opt_relu_327)
        opt_conv2d_331 = self.conv2d_331(opt_relu_327)
        opt_conv2d_332 = self.conv2d_332(opt_relu_328)
        opt_relu_333 = self.relu_333(opt_conv2d_330)
        opt_relu_334 = self.relu_334(opt_conv2d_331)
        opt_relu_335 = self.relu_335(opt_conv2d_332)
        opt_conv2d_336 = self.conv2d_336(opt_relu_335)
        opt_relu_337 = self.relu_337(opt_conv2d_336)
        opt_conv2d_338 = self.conv2d_338(opt_relu_337)
        opt_conv2d_339 = self.conv2d_339(opt_relu_337)
        opt_relu_340 = self.relu_340(opt_conv2d_338)
        opt_relu_341 = self.relu_341(opt_conv2d_339)
        opt_concat_342 = self.concat_342(
            (opt_relu_329, opt_relu_325, opt_relu_333, opt_relu_334, opt_relu_340, opt_relu_341,
             ))
        opt_batchnorm2d_343 = self.batchnorm2d_343(opt_concat_342)
        opt_relu_344 = self.relu_344(opt_batchnorm2d_343)
        opt_conv2d_345 = self.conv2d_345(opt_relu_344)
        opt_avgpool2d_346 = self.pad_avgpool2d_346(opt_relu_344)
        opt_avgpool2d_346 = self.avgpool2d_346(opt_avgpool2d_346)
        opt_conv2d_347 = self.conv2d_347(opt_relu_344)
        opt_conv2d_348 = self.conv2d_348(opt_relu_344)
        opt_relu_349 = self.relu_349(opt_conv2d_345)
        opt_conv2d_350 = self.conv2d_350(opt_avgpool2d_346)
        opt_relu_351 = self.relu_351(opt_conv2d_347)
        opt_relu_352 = self.relu_352(opt_conv2d_348)
        opt_relu_353 = self.relu_353(opt_conv2d_350)
        opt_conv2d_354 = self.conv2d_354(opt_relu_351)
        opt_conv2d_355 = self.conv2d_355(opt_relu_351)
        opt_conv2d_356 = self.conv2d_356(opt_relu_352)
        opt_relu_357 = self.relu_357(opt_conv2d_354)
        opt_relu_358 = self.relu_358(opt_conv2d_355)
        opt_relu_359 = self.relu_359(opt_conv2d_356)
        opt_conv2d_360 = self.conv2d_360(opt_relu_359)
        opt_relu_361 = self.relu_361(opt_conv2d_360)
        opt_conv2d_362 = self.conv2d_362(opt_relu_361)
        opt_conv2d_363 = self.conv2d_363(opt_relu_361)
        opt_relu_364 = self.relu_364(opt_conv2d_362)
        opt_relu_365 = self.relu_365(opt_conv2d_363)
        opt_concat_366 = self.concat_366(
            (opt_relu_353, opt_relu_349, opt_relu_357, opt_relu_358, opt_relu_364, opt_relu_365,
             ))
        opt_batchnorm2d_367 = self.batchnorm2d_367(opt_concat_366)
        opt_relu_368 = self.relu_368(opt_batchnorm2d_367)
        opt_avgpool2d_369 = self.pad_avgpool2d_369(opt_relu_368)
        opt_avgpool2d_369 = self.avgpool2d_369(opt_avgpool2d_369)
        opt_transpose_370 = self.transpose_370(opt_avgpool2d_369, (0, 2, 3, 1))
        opt_flatten_371 = self.flatten_371(opt_transpose_370)
        opt_matmul_372 = P.matmul(opt_flatten_371, self.matmul_372_w)
        opt_add_373 = opt_matmul_372 + self.add_373_bias
        opt_softmax_374 = self.softmax_374(opt_add_373)
        return opt_softmax_374
