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
                                   out_channels=96,
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
                                   out_channels=64,
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
        self.conv2d_29 = nn.Conv2d(in_channels=320,
                                   out_channels=32,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_30 = nn.Conv2d(in_channels=320,
                                   out_channels=32,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_31 = nn.Conv2d(in_channels=320,
                                   out_channels=32,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_32 = nn.ReLU()
        self.relu_33 = nn.ReLU()
        self.relu_34 = nn.ReLU()
        self.conv2d_35 = nn.Conv2d(in_channels=32,
                                   out_channels=32,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_36 = nn.Conv2d(in_channels=32,
                                   out_channels=48,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_37 = nn.ReLU()
        self.relu_38 = nn.ReLU()
        self.conv2d_39 = nn.Conv2d(in_channels=48,
                                   out_channels=64,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_40 = nn.ReLU()
        self.concat_41 = P.Concat(axis=1)
        self.conv2d_42 = nn.Conv2d(in_channels=128,
                                   out_channels=320,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.mul_43_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_45 = nn.ReLU()
        self.conv2d_46 = nn.Conv2d(in_channels=320,
                                   out_channels=32,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_47 = nn.Conv2d(in_channels=320,
                                   out_channels=32,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_48 = nn.Conv2d(in_channels=320,
                                   out_channels=32,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_49 = nn.ReLU()
        self.relu_50 = nn.ReLU()
        self.relu_51 = nn.ReLU()
        self.conv2d_52 = nn.Conv2d(in_channels=32,
                                   out_channels=32,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_53 = nn.Conv2d(in_channels=32,
                                   out_channels=48,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_54 = nn.ReLU()
        self.relu_55 = nn.ReLU()
        self.conv2d_56 = nn.Conv2d(in_channels=48,
                                   out_channels=64,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_57 = nn.ReLU()
        self.concat_58 = P.Concat(axis=1)
        self.conv2d_59 = nn.Conv2d(in_channels=128,
                                   out_channels=320,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.mul_60_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_62 = nn.ReLU()
        self.conv2d_63 = nn.Conv2d(in_channels=320,
                                   out_channels=32,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_64 = nn.Conv2d(in_channels=320,
                                   out_channels=32,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_65 = nn.Conv2d(in_channels=320,
                                   out_channels=32,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_66 = nn.ReLU()
        self.relu_67 = nn.ReLU()
        self.relu_68 = nn.ReLU()
        self.conv2d_69 = nn.Conv2d(in_channels=32,
                                   out_channels=32,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_70 = nn.Conv2d(in_channels=32,
                                   out_channels=48,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_71 = nn.ReLU()
        self.relu_72 = nn.ReLU()
        self.conv2d_73 = nn.Conv2d(in_channels=48,
                                   out_channels=64,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_74 = nn.ReLU()
        self.concat_75 = P.Concat(axis=1)
        self.conv2d_76 = nn.Conv2d(in_channels=128,
                                   out_channels=320,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.mul_77_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_79 = nn.ReLU()
        self.conv2d_80 = nn.Conv2d(in_channels=320,
                                   out_channels=32,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_81 = nn.Conv2d(in_channels=320,
                                   out_channels=32,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_82 = nn.Conv2d(in_channels=320,
                                   out_channels=32,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_83 = nn.ReLU()
        self.relu_84 = nn.ReLU()
        self.relu_85 = nn.ReLU()
        self.conv2d_86 = nn.Conv2d(in_channels=32,
                                   out_channels=32,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_87 = nn.Conv2d(in_channels=32,
                                   out_channels=48,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_88 = nn.ReLU()
        self.relu_89 = nn.ReLU()
        self.conv2d_90 = nn.Conv2d(in_channels=48,
                                   out_channels=64,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_91 = nn.ReLU()
        self.concat_92 = P.Concat(axis=1)
        self.conv2d_93 = nn.Conv2d(in_channels=128,
                                   out_channels=320,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.mul_94_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_96 = nn.ReLU()
        self.conv2d_97 = nn.Conv2d(in_channels=320,
                                   out_channels=32,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_98 = nn.Conv2d(in_channels=320,
                                   out_channels=32,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_99 = nn.Conv2d(in_channels=320,
                                   out_channels=32,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_100 = nn.ReLU()
        self.relu_101 = nn.ReLU()
        self.relu_102 = nn.ReLU()
        self.conv2d_103 = nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_104 = nn.Conv2d(in_channels=32,
                                    out_channels=48,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_105 = nn.ReLU()
        self.relu_106 = nn.ReLU()
        self.conv2d_107 = nn.Conv2d(in_channels=48,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_108 = nn.ReLU()
        self.concat_109 = P.Concat(axis=1)
        self.conv2d_110 = nn.Conv2d(in_channels=128,
                                    out_channels=320,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_111_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_113 = nn.ReLU()
        self.conv2d_114 = nn.Conv2d(in_channels=320,
                                    out_channels=32,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_115 = nn.Conv2d(in_channels=320,
                                    out_channels=32,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_116 = nn.Conv2d(in_channels=320,
                                    out_channels=32,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_117 = nn.ReLU()
        self.relu_118 = nn.ReLU()
        self.relu_119 = nn.ReLU()
        self.conv2d_120 = nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_121 = nn.Conv2d(in_channels=32,
                                    out_channels=48,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_122 = nn.ReLU()
        self.relu_123 = nn.ReLU()
        self.conv2d_124 = nn.Conv2d(in_channels=48,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_125 = nn.ReLU()
        self.concat_126 = P.Concat(axis=1)
        self.conv2d_127 = nn.Conv2d(in_channels=128,
                                    out_channels=320,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_128_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_130 = nn.ReLU()
        self.conv2d_131 = nn.Conv2d(in_channels=320,
                                    out_channels=32,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_132 = nn.Conv2d(in_channels=320,
                                    out_channels=32,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_133 = nn.Conv2d(in_channels=320,
                                    out_channels=32,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_134 = nn.ReLU()
        self.relu_135 = nn.ReLU()
        self.relu_136 = nn.ReLU()
        self.conv2d_137 = nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_138 = nn.Conv2d(in_channels=32,
                                    out_channels=48,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_139 = nn.ReLU()
        self.relu_140 = nn.ReLU()
        self.conv2d_141 = nn.Conv2d(in_channels=48,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_142 = nn.ReLU()
        self.concat_143 = P.Concat(axis=1)
        self.conv2d_144 = nn.Conv2d(in_channels=128,
                                    out_channels=320,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_145_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_147 = nn.ReLU()
        self.conv2d_148 = nn.Conv2d(in_channels=320,
                                    out_channels=32,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_149 = nn.Conv2d(in_channels=320,
                                    out_channels=32,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_150 = nn.Conv2d(in_channels=320,
                                    out_channels=32,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_151 = nn.ReLU()
        self.relu_152 = nn.ReLU()
        self.relu_153 = nn.ReLU()
        self.conv2d_154 = nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_155 = nn.Conv2d(in_channels=32,
                                    out_channels=48,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_156 = nn.ReLU()
        self.relu_157 = nn.ReLU()
        self.conv2d_158 = nn.Conv2d(in_channels=48,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_159 = nn.ReLU()
        self.concat_160 = P.Concat(axis=1)
        self.conv2d_161 = nn.Conv2d(in_channels=128,
                                    out_channels=320,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_162_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_164 = nn.ReLU()
        self.conv2d_165 = nn.Conv2d(in_channels=320,
                                    out_channels=32,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_166 = nn.Conv2d(in_channels=320,
                                    out_channels=32,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_167 = nn.Conv2d(in_channels=320,
                                    out_channels=32,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_168 = nn.ReLU()
        self.relu_169 = nn.ReLU()
        self.relu_170 = nn.ReLU()
        self.conv2d_171 = nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_172 = nn.Conv2d(in_channels=32,
                                    out_channels=48,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_173 = nn.ReLU()
        self.relu_174 = nn.ReLU()
        self.conv2d_175 = nn.Conv2d(in_channels=48,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_176 = nn.ReLU()
        self.concat_177 = P.Concat(axis=1)
        self.conv2d_178 = nn.Conv2d(in_channels=128,
                                    out_channels=320,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_179_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_181 = nn.ReLU()
        self.conv2d_182 = nn.Conv2d(in_channels=320,
                                    out_channels=32,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_183 = nn.Conv2d(in_channels=320,
                                    out_channels=32,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_184 = nn.Conv2d(in_channels=320,
                                    out_channels=32,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_185 = nn.ReLU()
        self.relu_186 = nn.ReLU()
        self.relu_187 = nn.ReLU()
        self.conv2d_188 = nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_189 = nn.Conv2d(in_channels=32,
                                    out_channels=48,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_190 = nn.ReLU()
        self.relu_191 = nn.ReLU()
        self.conv2d_192 = nn.Conv2d(in_channels=48,
                                    out_channels=64,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_193 = nn.ReLU()
        self.concat_194 = P.Concat(axis=1)
        self.conv2d_195 = nn.Conv2d(in_channels=128,
                                    out_channels=320,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_196_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_198 = nn.ReLU()
        self.pad_maxpool2d_199 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_199 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv2d_200 = nn.Conv2d(in_channels=320,
                                    out_channels=384,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_201 = nn.Conv2d(in_channels=320,
                                    out_channels=256,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_202 = nn.ReLU()
        self.relu_203 = nn.ReLU()
        self.conv2d_204 = nn.Conv2d(in_channels=256,
                                    out_channels=256,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_205 = nn.ReLU()
        self.conv2d_206 = nn.Conv2d(in_channels=256,
                                    out_channels=384,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_207 = nn.ReLU()
        self.concat_208 = P.Concat(axis=1)
        self.conv2d_209 = nn.Conv2d(in_channels=1088,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_210 = nn.Conv2d(in_channels=1088,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_211 = nn.ReLU()
        self.relu_212 = nn.ReLU()
        self.conv2d_213 = nn.Conv2d(in_channels=128,
                                    out_channels=160,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_214 = nn.ReLU()
        self.conv2d_215 = nn.Conv2d(in_channels=160,
                                    out_channels=192,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_216 = nn.ReLU()
        self.concat_217 = P.Concat(axis=1)
        self.conv2d_218 = nn.Conv2d(in_channels=384,
                                    out_channels=1088,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_219_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_221 = nn.ReLU()
        self.conv2d_222 = nn.Conv2d(in_channels=1088,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_223 = nn.Conv2d(in_channels=1088,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_224 = nn.ReLU()
        self.relu_225 = nn.ReLU()
        self.conv2d_226 = nn.Conv2d(in_channels=128,
                                    out_channels=160,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_227 = nn.ReLU()
        self.conv2d_228 = nn.Conv2d(in_channels=160,
                                    out_channels=192,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_229 = nn.ReLU()
        self.concat_230 = P.Concat(axis=1)
        self.conv2d_231 = nn.Conv2d(in_channels=384,
                                    out_channels=1088,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_232_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_234 = nn.ReLU()
        self.conv2d_235 = nn.Conv2d(in_channels=1088,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_236 = nn.Conv2d(in_channels=1088,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_237 = nn.ReLU()
        self.relu_238 = nn.ReLU()
        self.conv2d_239 = nn.Conv2d(in_channels=128,
                                    out_channels=160,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_240 = nn.ReLU()
        self.conv2d_241 = nn.Conv2d(in_channels=160,
                                    out_channels=192,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_242 = nn.ReLU()
        self.concat_243 = P.Concat(axis=1)
        self.conv2d_244 = nn.Conv2d(in_channels=384,
                                    out_channels=1088,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_245_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_247 = nn.ReLU()
        self.conv2d_248 = nn.Conv2d(in_channels=1088,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_249 = nn.Conv2d(in_channels=1088,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_250 = nn.ReLU()
        self.relu_251 = nn.ReLU()
        self.conv2d_252 = nn.Conv2d(in_channels=128,
                                    out_channels=160,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_253 = nn.ReLU()
        self.conv2d_254 = nn.Conv2d(in_channels=160,
                                    out_channels=192,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_255 = nn.ReLU()
        self.concat_256 = P.Concat(axis=1)
        self.conv2d_257 = nn.Conv2d(in_channels=384,
                                    out_channels=1088,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_258_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_260 = nn.ReLU()
        self.conv2d_261 = nn.Conv2d(in_channels=1088,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_262 = nn.Conv2d(in_channels=1088,
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
        self.conv2d_265 = nn.Conv2d(in_channels=128,
                                    out_channels=160,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_266 = nn.ReLU()
        self.conv2d_267 = nn.Conv2d(in_channels=160,
                                    out_channels=192,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_268 = nn.ReLU()
        self.concat_269 = P.Concat(axis=1)
        self.conv2d_270 = nn.Conv2d(in_channels=384,
                                    out_channels=1088,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_271_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_273 = nn.ReLU()
        self.conv2d_274 = nn.Conv2d(in_channels=1088,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_275 = nn.Conv2d(in_channels=1088,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_276 = nn.ReLU()
        self.relu_277 = nn.ReLU()
        self.conv2d_278 = nn.Conv2d(in_channels=128,
                                    out_channels=160,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_279 = nn.ReLU()
        self.conv2d_280 = nn.Conv2d(in_channels=160,
                                    out_channels=192,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_281 = nn.ReLU()
        self.concat_282 = P.Concat(axis=1)
        self.conv2d_283 = nn.Conv2d(in_channels=384,
                                    out_channels=1088,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_284_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_286 = nn.ReLU()
        self.conv2d_287 = nn.Conv2d(in_channels=1088,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_288 = nn.Conv2d(in_channels=1088,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_289 = nn.ReLU()
        self.relu_290 = nn.ReLU()
        self.conv2d_291 = nn.Conv2d(in_channels=128,
                                    out_channels=160,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_292 = nn.ReLU()
        self.conv2d_293 = nn.Conv2d(in_channels=160,
                                    out_channels=192,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_294 = nn.ReLU()
        self.concat_295 = P.Concat(axis=1)
        self.conv2d_296 = nn.Conv2d(in_channels=384,
                                    out_channels=1088,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_297_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_299 = nn.ReLU()
        self.conv2d_300 = nn.Conv2d(in_channels=1088,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_301 = nn.Conv2d(in_channels=1088,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_302 = nn.ReLU()
        self.relu_303 = nn.ReLU()
        self.conv2d_304 = nn.Conv2d(in_channels=128,
                                    out_channels=160,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_305 = nn.ReLU()
        self.conv2d_306 = nn.Conv2d(in_channels=160,
                                    out_channels=192,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_307 = nn.ReLU()
        self.concat_308 = P.Concat(axis=1)
        self.conv2d_309 = nn.Conv2d(in_channels=384,
                                    out_channels=1088,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_310_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_312 = nn.ReLU()
        self.conv2d_313 = nn.Conv2d(in_channels=1088,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_314 = nn.Conv2d(in_channels=1088,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_315 = nn.ReLU()
        self.relu_316 = nn.ReLU()
        self.conv2d_317 = nn.Conv2d(in_channels=128,
                                    out_channels=160,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_318 = nn.ReLU()
        self.conv2d_319 = nn.Conv2d(in_channels=160,
                                    out_channels=192,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_320 = nn.ReLU()
        self.concat_321 = P.Concat(axis=1)
        self.conv2d_322 = nn.Conv2d(in_channels=384,
                                    out_channels=1088,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_323_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_325 = nn.ReLU()
        self.conv2d_326 = nn.Conv2d(in_channels=1088,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_327 = nn.Conv2d(in_channels=1088,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_328 = nn.ReLU()
        self.relu_329 = nn.ReLU()
        self.conv2d_330 = nn.Conv2d(in_channels=128,
                                    out_channels=160,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_331 = nn.ReLU()
        self.conv2d_332 = nn.Conv2d(in_channels=160,
                                    out_channels=192,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_333 = nn.ReLU()
        self.concat_334 = P.Concat(axis=1)
        self.conv2d_335 = nn.Conv2d(in_channels=384,
                                    out_channels=1088,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_336_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_338 = nn.ReLU()
        self.conv2d_339 = nn.Conv2d(in_channels=1088,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_340 = nn.Conv2d(in_channels=1088,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_341 = nn.ReLU()
        self.relu_342 = nn.ReLU()
        self.conv2d_343 = nn.Conv2d(in_channels=128,
                                    out_channels=160,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_344 = nn.ReLU()
        self.conv2d_345 = nn.Conv2d(in_channels=160,
                                    out_channels=192,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_346 = nn.ReLU()
        self.concat_347 = P.Concat(axis=1)
        self.conv2d_348 = nn.Conv2d(in_channels=384,
                                    out_channels=1088,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_349_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_351 = nn.ReLU()
        self.conv2d_352 = nn.Conv2d(in_channels=1088,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_353 = nn.Conv2d(in_channels=1088,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_354 = nn.ReLU()
        self.relu_355 = nn.ReLU()
        self.conv2d_356 = nn.Conv2d(in_channels=128,
                                    out_channels=160,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_357 = nn.ReLU()
        self.conv2d_358 = nn.Conv2d(in_channels=160,
                                    out_channels=192,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_359 = nn.ReLU()
        self.concat_360 = P.Concat(axis=1)
        self.conv2d_361 = nn.Conv2d(in_channels=384,
                                    out_channels=1088,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_362_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_364 = nn.ReLU()
        self.conv2d_365 = nn.Conv2d(in_channels=1088,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_366 = nn.Conv2d(in_channels=1088,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_367 = nn.ReLU()
        self.relu_368 = nn.ReLU()
        self.conv2d_369 = nn.Conv2d(in_channels=128,
                                    out_channels=160,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_370 = nn.ReLU()
        self.conv2d_371 = nn.Conv2d(in_channels=160,
                                    out_channels=192,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_372 = nn.ReLU()
        self.concat_373 = P.Concat(axis=1)
        self.conv2d_374 = nn.Conv2d(in_channels=384,
                                    out_channels=1088,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_375_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_377 = nn.ReLU()
        self.conv2d_378 = nn.Conv2d(in_channels=1088,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_379 = nn.Conv2d(in_channels=1088,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_380 = nn.ReLU()
        self.relu_381 = nn.ReLU()
        self.conv2d_382 = nn.Conv2d(in_channels=128,
                                    out_channels=160,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_383 = nn.ReLU()
        self.conv2d_384 = nn.Conv2d(in_channels=160,
                                    out_channels=192,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_385 = nn.ReLU()
        self.concat_386 = P.Concat(axis=1)
        self.conv2d_387 = nn.Conv2d(in_channels=384,
                                    out_channels=1088,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_388_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_390 = nn.ReLU()
        self.conv2d_391 = nn.Conv2d(in_channels=1088,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_392 = nn.Conv2d(in_channels=1088,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_393 = nn.ReLU()
        self.relu_394 = nn.ReLU()
        self.conv2d_395 = nn.Conv2d(in_channels=128,
                                    out_channels=160,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_396 = nn.ReLU()
        self.conv2d_397 = nn.Conv2d(in_channels=160,
                                    out_channels=192,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_398 = nn.ReLU()
        self.concat_399 = P.Concat(axis=1)
        self.conv2d_400 = nn.Conv2d(in_channels=384,
                                    out_channels=1088,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_401_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_403 = nn.ReLU()
        self.conv2d_404 = nn.Conv2d(in_channels=1088,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_405 = nn.Conv2d(in_channels=1088,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_406 = nn.ReLU()
        self.relu_407 = nn.ReLU()
        self.conv2d_408 = nn.Conv2d(in_channels=128,
                                    out_channels=160,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_409 = nn.ReLU()
        self.conv2d_410 = nn.Conv2d(in_channels=160,
                                    out_channels=192,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_411 = nn.ReLU()
        self.concat_412 = P.Concat(axis=1)
        self.conv2d_413 = nn.Conv2d(in_channels=384,
                                    out_channels=1088,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_414_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_416 = nn.ReLU()
        self.conv2d_417 = nn.Conv2d(in_channels=1088,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_418 = nn.Conv2d(in_channels=1088,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_419 = nn.ReLU()
        self.relu_420 = nn.ReLU()
        self.conv2d_421 = nn.Conv2d(in_channels=128,
                                    out_channels=160,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_422 = nn.ReLU()
        self.conv2d_423 = nn.Conv2d(in_channels=160,
                                    out_channels=192,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_424 = nn.ReLU()
        self.concat_425 = P.Concat(axis=1)
        self.conv2d_426 = nn.Conv2d(in_channels=384,
                                    out_channels=1088,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_427_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_429 = nn.ReLU()
        self.conv2d_430 = nn.Conv2d(in_channels=1088,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_431 = nn.Conv2d(in_channels=1088,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_432 = nn.ReLU()
        self.relu_433 = nn.ReLU()
        self.conv2d_434 = nn.Conv2d(in_channels=128,
                                    out_channels=160,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_435 = nn.ReLU()
        self.conv2d_436 = nn.Conv2d(in_channels=160,
                                    out_channels=192,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_437 = nn.ReLU()
        self.concat_438 = P.Concat(axis=1)
        self.conv2d_439 = nn.Conv2d(in_channels=384,
                                    out_channels=1088,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_440_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_442 = nn.ReLU()
        self.conv2d_443 = nn.Conv2d(in_channels=1088,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_444 = nn.Conv2d(in_channels=1088,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_445 = nn.ReLU()
        self.relu_446 = nn.ReLU()
        self.conv2d_447 = nn.Conv2d(in_channels=128,
                                    out_channels=160,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_448 = nn.ReLU()
        self.conv2d_449 = nn.Conv2d(in_channels=160,
                                    out_channels=192,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_450 = nn.ReLU()
        self.concat_451 = P.Concat(axis=1)
        self.conv2d_452 = nn.Conv2d(in_channels=384,
                                    out_channels=1088,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_453_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_455 = nn.ReLU()
        self.conv2d_456 = nn.Conv2d(in_channels=1088,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_457 = nn.Conv2d(in_channels=1088,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_458 = nn.ReLU()
        self.relu_459 = nn.ReLU()
        self.conv2d_460 = nn.Conv2d(in_channels=128,
                                    out_channels=160,
                                    kernel_size=(1, 7),
                                    stride=(1, 1),
                                    padding=(0, 0, 3, 3),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_461 = nn.ReLU()
        self.conv2d_462 = nn.Conv2d(in_channels=160,
                                    out_channels=192,
                                    kernel_size=(7, 1),
                                    stride=(1, 1),
                                    padding=(3, 3, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_463 = nn.ReLU()
        self.concat_464 = P.Concat(axis=1)
        self.conv2d_465 = nn.Conv2d(in_channels=384,
                                    out_channels=1088,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_466_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_468 = nn.ReLU()
        self.pad_maxpool2d_469 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_469 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv2d_470 = nn.Conv2d(in_channels=1088,
                                    out_channels=256,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_471 = nn.Conv2d(in_channels=1088,
                                    out_channels=256,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_472 = nn.Conv2d(in_channels=1088,
                                    out_channels=256,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_473 = nn.ReLU()
        self.relu_474 = nn.ReLU()
        self.relu_475 = nn.ReLU()
        self.conv2d_476 = nn.Conv2d(in_channels=256,
                                    out_channels=384,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_477 = nn.Conv2d(in_channels=256,
                                    out_channels=288,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_478 = nn.Conv2d(in_channels=256,
                                    out_channels=288,
                                    kernel_size=(3, 3),
                                    stride=(1, 1),
                                    padding=(1, 1, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_479 = nn.ReLU()
        self.relu_480 = nn.ReLU()
        self.relu_481 = nn.ReLU()
        self.conv2d_482 = nn.Conv2d(in_channels=288,
                                    out_channels=320,
                                    kernel_size=(3, 3),
                                    stride=(2, 2),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_483 = nn.ReLU()
        self.concat_484 = P.Concat(axis=1)
        self.conv2d_485 = nn.Conv2d(in_channels=2080,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_486 = nn.Conv2d(in_channels=2080,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_487 = nn.ReLU()
        self.relu_488 = nn.ReLU()
        self.conv2d_489 = nn.Conv2d(in_channels=192,
                                    out_channels=224,
                                    kernel_size=(1, 3),
                                    stride=(1, 1),
                                    padding=(0, 0, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_490 = nn.ReLU()
        self.conv2d_491 = nn.Conv2d(in_channels=224,
                                    out_channels=256,
                                    kernel_size=(3, 1),
                                    stride=(1, 1),
                                    padding=(1, 1, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_492 = nn.ReLU()
        self.concat_493 = P.Concat(axis=1)
        self.conv2d_494 = nn.Conv2d(in_channels=448,
                                    out_channels=2080,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_495_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_497 = nn.ReLU()
        self.conv2d_498 = nn.Conv2d(in_channels=2080,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_499 = nn.Conv2d(in_channels=2080,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_500 = nn.ReLU()
        self.relu_501 = nn.ReLU()
        self.conv2d_502 = nn.Conv2d(in_channels=192,
                                    out_channels=224,
                                    kernel_size=(1, 3),
                                    stride=(1, 1),
                                    padding=(0, 0, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_503 = nn.ReLU()
        self.conv2d_504 = nn.Conv2d(in_channels=224,
                                    out_channels=256,
                                    kernel_size=(3, 1),
                                    stride=(1, 1),
                                    padding=(1, 1, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_505 = nn.ReLU()
        self.concat_506 = P.Concat(axis=1)
        self.conv2d_507 = nn.Conv2d(in_channels=448,
                                    out_channels=2080,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_508_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_510 = nn.ReLU()
        self.conv2d_511 = nn.Conv2d(in_channels=2080,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_512 = nn.Conv2d(in_channels=2080,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_513 = nn.ReLU()
        self.relu_514 = nn.ReLU()
        self.conv2d_515 = nn.Conv2d(in_channels=192,
                                    out_channels=224,
                                    kernel_size=(1, 3),
                                    stride=(1, 1),
                                    padding=(0, 0, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_516 = nn.ReLU()
        self.conv2d_517 = nn.Conv2d(in_channels=224,
                                    out_channels=256,
                                    kernel_size=(3, 1),
                                    stride=(1, 1),
                                    padding=(1, 1, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_518 = nn.ReLU()
        self.concat_519 = P.Concat(axis=1)
        self.conv2d_520 = nn.Conv2d(in_channels=448,
                                    out_channels=2080,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_521_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_523 = nn.ReLU()
        self.conv2d_524 = nn.Conv2d(in_channels=2080,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_525 = nn.Conv2d(in_channels=2080,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_526 = nn.ReLU()
        self.relu_527 = nn.ReLU()
        self.conv2d_528 = nn.Conv2d(in_channels=192,
                                    out_channels=224,
                                    kernel_size=(1, 3),
                                    stride=(1, 1),
                                    padding=(0, 0, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_529 = nn.ReLU()
        self.conv2d_530 = nn.Conv2d(in_channels=224,
                                    out_channels=256,
                                    kernel_size=(3, 1),
                                    stride=(1, 1),
                                    padding=(1, 1, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_531 = nn.ReLU()
        self.concat_532 = P.Concat(axis=1)
        self.conv2d_533 = nn.Conv2d(in_channels=448,
                                    out_channels=2080,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_534_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_536 = nn.ReLU()
        self.conv2d_537 = nn.Conv2d(in_channels=2080,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_538 = nn.Conv2d(in_channels=2080,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_539 = nn.ReLU()
        self.relu_540 = nn.ReLU()
        self.conv2d_541 = nn.Conv2d(in_channels=192,
                                    out_channels=224,
                                    kernel_size=(1, 3),
                                    stride=(1, 1),
                                    padding=(0, 0, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_542 = nn.ReLU()
        self.conv2d_543 = nn.Conv2d(in_channels=224,
                                    out_channels=256,
                                    kernel_size=(3, 1),
                                    stride=(1, 1),
                                    padding=(1, 1, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_544 = nn.ReLU()
        self.concat_545 = P.Concat(axis=1)
        self.conv2d_546 = nn.Conv2d(in_channels=448,
                                    out_channels=2080,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_547_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_549 = nn.ReLU()
        self.conv2d_550 = nn.Conv2d(in_channels=2080,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_551 = nn.Conv2d(in_channels=2080,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_552 = nn.ReLU()
        self.relu_553 = nn.ReLU()
        self.conv2d_554 = nn.Conv2d(in_channels=192,
                                    out_channels=224,
                                    kernel_size=(1, 3),
                                    stride=(1, 1),
                                    padding=(0, 0, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_555 = nn.ReLU()
        self.conv2d_556 = nn.Conv2d(in_channels=224,
                                    out_channels=256,
                                    kernel_size=(3, 1),
                                    stride=(1, 1),
                                    padding=(1, 1, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_557 = nn.ReLU()
        self.concat_558 = P.Concat(axis=1)
        self.conv2d_559 = nn.Conv2d(in_channels=448,
                                    out_channels=2080,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_560_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_562 = nn.ReLU()
        self.conv2d_563 = nn.Conv2d(in_channels=2080,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_564 = nn.Conv2d(in_channels=2080,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_565 = nn.ReLU()
        self.relu_566 = nn.ReLU()
        self.conv2d_567 = nn.Conv2d(in_channels=192,
                                    out_channels=224,
                                    kernel_size=(1, 3),
                                    stride=(1, 1),
                                    padding=(0, 0, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_568 = nn.ReLU()
        self.conv2d_569 = nn.Conv2d(in_channels=224,
                                    out_channels=256,
                                    kernel_size=(3, 1),
                                    stride=(1, 1),
                                    padding=(1, 1, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_570 = nn.ReLU()
        self.concat_571 = P.Concat(axis=1)
        self.conv2d_572 = nn.Conv2d(in_channels=448,
                                    out_channels=2080,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_573_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_575 = nn.ReLU()
        self.conv2d_576 = nn.Conv2d(in_channels=2080,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_577 = nn.Conv2d(in_channels=2080,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_578 = nn.ReLU()
        self.relu_579 = nn.ReLU()
        self.conv2d_580 = nn.Conv2d(in_channels=192,
                                    out_channels=224,
                                    kernel_size=(1, 3),
                                    stride=(1, 1),
                                    padding=(0, 0, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_581 = nn.ReLU()
        self.conv2d_582 = nn.Conv2d(in_channels=224,
                                    out_channels=256,
                                    kernel_size=(3, 1),
                                    stride=(1, 1),
                                    padding=(1, 1, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_583 = nn.ReLU()
        self.concat_584 = P.Concat(axis=1)
        self.conv2d_585 = nn.Conv2d(in_channels=448,
                                    out_channels=2080,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_586_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_588 = nn.ReLU()
        self.conv2d_589 = nn.Conv2d(in_channels=2080,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_590 = nn.Conv2d(in_channels=2080,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_591 = nn.ReLU()
        self.relu_592 = nn.ReLU()
        self.conv2d_593 = nn.Conv2d(in_channels=192,
                                    out_channels=224,
                                    kernel_size=(1, 3),
                                    stride=(1, 1),
                                    padding=(0, 0, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_594 = nn.ReLU()
        self.conv2d_595 = nn.Conv2d(in_channels=224,
                                    out_channels=256,
                                    kernel_size=(3, 1),
                                    stride=(1, 1),
                                    padding=(1, 1, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_596 = nn.ReLU()
        self.concat_597 = P.Concat(axis=1)
        self.conv2d_598 = nn.Conv2d(in_channels=448,
                                    out_channels=2080,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_599_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.relu_601 = nn.ReLU()
        self.conv2d_602 = nn.Conv2d(in_channels=2080,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.conv2d_603 = nn.Conv2d(in_channels=2080,
                                    out_channels=192,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_604 = nn.ReLU()
        self.relu_605 = nn.ReLU()
        self.conv2d_606 = nn.Conv2d(in_channels=192,
                                    out_channels=224,
                                    kernel_size=(1, 3),
                                    stride=(1, 1),
                                    padding=(0, 0, 1, 1),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_607 = nn.ReLU()
        self.conv2d_608 = nn.Conv2d(in_channels=224,
                                    out_channels=256,
                                    kernel_size=(3, 1),
                                    stride=(1, 1),
                                    padding=(1, 1, 0, 0),
                                    pad_mode="pad",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_609 = nn.ReLU()
        self.concat_610 = P.Concat(axis=1)
        self.conv2d_611 = nn.Conv2d(in_channels=448,
                                    out_channels=2080,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.mul_612_w = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1, 1)).astype(np.float32)), name=None)
        self.conv2d_614 = nn.Conv2d(in_channels=2080,
                                    out_channels=1536,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_615 = nn.ReLU()
        self.avgpool2d_616 = nn.AvgPool2d(kernel_size=(8, 8))
        self.transpose_617 = P.Transpose()
        self.reshape_618 = P.Reshape()
        self.reshape_618_shape = tuple([1, 1536])
        self.matmul_619_w = Parameter(Tensor(np.random.uniform(0, 1, (1536, 1000)).astype(np.float32)), name=None)
        self.add_620_bias = Parameter(Tensor(np.random.uniform(0, 1, (1000, )).astype(np.float32)), name=None)
        self.softmax_621 = nn.Softmax(axis=-1)

    def construct(self, input_3):
        opt_transpose_0 = self.transpose_0(input_3, (0, 3, 1, 2))
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
        opt_conv2d_30 = self.conv2d_30(opt_concat_28)
        opt_conv2d_31 = self.conv2d_31(opt_concat_28)
        opt_relu_32 = self.relu_32(opt_conv2d_29)
        opt_relu_33 = self.relu_33(opt_conv2d_30)
        opt_relu_34 = self.relu_34(opt_conv2d_31)
        opt_conv2d_35 = self.conv2d_35(opt_relu_33)
        opt_conv2d_36 = self.conv2d_36(opt_relu_34)
        opt_relu_37 = self.relu_37(opt_conv2d_35)
        opt_relu_38 = self.relu_38(opt_conv2d_36)
        opt_conv2d_39 = self.conv2d_39(opt_relu_38)
        opt_relu_40 = self.relu_40(opt_conv2d_39)
        opt_concat_41 = self.concat_41((opt_relu_32, opt_relu_37, opt_relu_40, ))
        opt_conv2d_42 = self.conv2d_42(opt_concat_41)
        opt_mul_43 = opt_conv2d_42 * self.mul_43_w
        opt_add_44 = P.Add()(opt_concat_28, opt_mul_43)
        opt_relu_45 = self.relu_45(opt_add_44)
        opt_conv2d_46 = self.conv2d_46(opt_relu_45)
        opt_conv2d_47 = self.conv2d_47(opt_relu_45)
        opt_conv2d_48 = self.conv2d_48(opt_relu_45)
        opt_relu_49 = self.relu_49(opt_conv2d_46)
        opt_relu_50 = self.relu_50(opt_conv2d_47)
        opt_relu_51 = self.relu_51(opt_conv2d_48)
        opt_conv2d_52 = self.conv2d_52(opt_relu_50)
        opt_conv2d_53 = self.conv2d_53(opt_relu_51)
        opt_relu_54 = self.relu_54(opt_conv2d_52)
        opt_relu_55 = self.relu_55(opt_conv2d_53)
        opt_conv2d_56 = self.conv2d_56(opt_relu_55)
        opt_relu_57 = self.relu_57(opt_conv2d_56)
        opt_concat_58 = self.concat_58((opt_relu_49, opt_relu_54, opt_relu_57, ))
        opt_conv2d_59 = self.conv2d_59(opt_concat_58)
        opt_mul_60 = opt_conv2d_59 * self.mul_60_w
        opt_add_61 = P.Add()(opt_relu_45, opt_mul_60)
        opt_relu_62 = self.relu_62(opt_add_61)
        opt_conv2d_63 = self.conv2d_63(opt_relu_62)
        opt_conv2d_64 = self.conv2d_64(opt_relu_62)
        opt_conv2d_65 = self.conv2d_65(opt_relu_62)
        opt_relu_66 = self.relu_66(opt_conv2d_63)
        opt_relu_67 = self.relu_67(opt_conv2d_64)
        opt_relu_68 = self.relu_68(opt_conv2d_65)
        opt_conv2d_69 = self.conv2d_69(opt_relu_67)
        opt_conv2d_70 = self.conv2d_70(opt_relu_68)
        opt_relu_71 = self.relu_71(opt_conv2d_69)
        opt_relu_72 = self.relu_72(opt_conv2d_70)
        opt_conv2d_73 = self.conv2d_73(opt_relu_72)
        opt_relu_74 = self.relu_74(opt_conv2d_73)
        opt_concat_75 = self.concat_75((opt_relu_66, opt_relu_71, opt_relu_74, ))
        opt_conv2d_76 = self.conv2d_76(opt_concat_75)
        opt_mul_77 = opt_conv2d_76 * self.mul_77_w
        opt_add_78 = P.Add()(opt_relu_62, opt_mul_77)
        opt_relu_79 = self.relu_79(opt_add_78)
        opt_conv2d_80 = self.conv2d_80(opt_relu_79)
        opt_conv2d_81 = self.conv2d_81(opt_relu_79)
        opt_conv2d_82 = self.conv2d_82(opt_relu_79)
        opt_relu_83 = self.relu_83(opt_conv2d_80)
        opt_relu_84 = self.relu_84(opt_conv2d_81)
        opt_relu_85 = self.relu_85(opt_conv2d_82)
        opt_conv2d_86 = self.conv2d_86(opt_relu_84)
        opt_conv2d_87 = self.conv2d_87(opt_relu_85)
        opt_relu_88 = self.relu_88(opt_conv2d_86)
        opt_relu_89 = self.relu_89(opt_conv2d_87)
        opt_conv2d_90 = self.conv2d_90(opt_relu_89)
        opt_relu_91 = self.relu_91(opt_conv2d_90)
        opt_concat_92 = self.concat_92((opt_relu_83, opt_relu_88, opt_relu_91, ))
        opt_conv2d_93 = self.conv2d_93(opt_concat_92)
        opt_mul_94 = opt_conv2d_93 * self.mul_94_w
        opt_add_95 = P.Add()(opt_relu_79, opt_mul_94)
        opt_relu_96 = self.relu_96(opt_add_95)
        opt_conv2d_97 = self.conv2d_97(opt_relu_96)
        opt_conv2d_98 = self.conv2d_98(opt_relu_96)
        opt_conv2d_99 = self.conv2d_99(opt_relu_96)
        opt_relu_100 = self.relu_100(opt_conv2d_97)
        opt_relu_101 = self.relu_101(opt_conv2d_98)
        opt_relu_102 = self.relu_102(opt_conv2d_99)
        opt_conv2d_103 = self.conv2d_103(opt_relu_101)
        opt_conv2d_104 = self.conv2d_104(opt_relu_102)
        opt_relu_105 = self.relu_105(opt_conv2d_103)
        opt_relu_106 = self.relu_106(opt_conv2d_104)
        opt_conv2d_107 = self.conv2d_107(opt_relu_106)
        opt_relu_108 = self.relu_108(opt_conv2d_107)
        opt_concat_109 = self.concat_109((opt_relu_100, opt_relu_105, opt_relu_108, ))
        opt_conv2d_110 = self.conv2d_110(opt_concat_109)
        opt_mul_111 = opt_conv2d_110 * self.mul_111_w
        opt_add_112 = P.Add()(opt_relu_96, opt_mul_111)
        opt_relu_113 = self.relu_113(opt_add_112)
        opt_conv2d_114 = self.conv2d_114(opt_relu_113)
        opt_conv2d_115 = self.conv2d_115(opt_relu_113)
        opt_conv2d_116 = self.conv2d_116(opt_relu_113)
        opt_relu_117 = self.relu_117(opt_conv2d_114)
        opt_relu_118 = self.relu_118(opt_conv2d_115)
        opt_relu_119 = self.relu_119(opt_conv2d_116)
        opt_conv2d_120 = self.conv2d_120(opt_relu_118)
        opt_conv2d_121 = self.conv2d_121(opt_relu_119)
        opt_relu_122 = self.relu_122(opt_conv2d_120)
        opt_relu_123 = self.relu_123(opt_conv2d_121)
        opt_conv2d_124 = self.conv2d_124(opt_relu_123)
        opt_relu_125 = self.relu_125(opt_conv2d_124)
        opt_concat_126 = self.concat_126((opt_relu_117, opt_relu_122, opt_relu_125, ))
        opt_conv2d_127 = self.conv2d_127(opt_concat_126)
        opt_mul_128 = opt_conv2d_127 * self.mul_128_w
        opt_add_129 = P.Add()(opt_relu_113, opt_mul_128)
        opt_relu_130 = self.relu_130(opt_add_129)
        opt_conv2d_131 = self.conv2d_131(opt_relu_130)
        opt_conv2d_132 = self.conv2d_132(opt_relu_130)
        opt_conv2d_133 = self.conv2d_133(opt_relu_130)
        opt_relu_134 = self.relu_134(opt_conv2d_131)
        opt_relu_135 = self.relu_135(opt_conv2d_132)
        opt_relu_136 = self.relu_136(opt_conv2d_133)
        opt_conv2d_137 = self.conv2d_137(opt_relu_135)
        opt_conv2d_138 = self.conv2d_138(opt_relu_136)
        opt_relu_139 = self.relu_139(opt_conv2d_137)
        opt_relu_140 = self.relu_140(opt_conv2d_138)
        opt_conv2d_141 = self.conv2d_141(opt_relu_140)
        opt_relu_142 = self.relu_142(opt_conv2d_141)
        opt_concat_143 = self.concat_143((opt_relu_134, opt_relu_139, opt_relu_142, ))
        opt_conv2d_144 = self.conv2d_144(opt_concat_143)
        opt_mul_145 = opt_conv2d_144 * self.mul_145_w
        opt_add_146 = P.Add()(opt_relu_130, opt_mul_145)
        opt_relu_147 = self.relu_147(opt_add_146)
        opt_conv2d_148 = self.conv2d_148(opt_relu_147)
        opt_conv2d_149 = self.conv2d_149(opt_relu_147)
        opt_conv2d_150 = self.conv2d_150(opt_relu_147)
        opt_relu_151 = self.relu_151(opt_conv2d_148)
        opt_relu_152 = self.relu_152(opt_conv2d_149)
        opt_relu_153 = self.relu_153(opt_conv2d_150)
        opt_conv2d_154 = self.conv2d_154(opt_relu_152)
        opt_conv2d_155 = self.conv2d_155(opt_relu_153)
        opt_relu_156 = self.relu_156(opt_conv2d_154)
        opt_relu_157 = self.relu_157(opt_conv2d_155)
        opt_conv2d_158 = self.conv2d_158(opt_relu_157)
        opt_relu_159 = self.relu_159(opt_conv2d_158)
        opt_concat_160 = self.concat_160((opt_relu_151, opt_relu_156, opt_relu_159, ))
        opt_conv2d_161 = self.conv2d_161(opt_concat_160)
        opt_mul_162 = opt_conv2d_161 * self.mul_162_w
        opt_add_163 = P.Add()(opt_relu_147, opt_mul_162)
        opt_relu_164 = self.relu_164(opt_add_163)
        opt_conv2d_165 = self.conv2d_165(opt_relu_164)
        opt_conv2d_166 = self.conv2d_166(opt_relu_164)
        opt_conv2d_167 = self.conv2d_167(opt_relu_164)
        opt_relu_168 = self.relu_168(opt_conv2d_165)
        opt_relu_169 = self.relu_169(opt_conv2d_166)
        opt_relu_170 = self.relu_170(opt_conv2d_167)
        opt_conv2d_171 = self.conv2d_171(opt_relu_169)
        opt_conv2d_172 = self.conv2d_172(opt_relu_170)
        opt_relu_173 = self.relu_173(opt_conv2d_171)
        opt_relu_174 = self.relu_174(opt_conv2d_172)
        opt_conv2d_175 = self.conv2d_175(opt_relu_174)
        opt_relu_176 = self.relu_176(opt_conv2d_175)
        opt_concat_177 = self.concat_177((opt_relu_168, opt_relu_173, opt_relu_176, ))
        opt_conv2d_178 = self.conv2d_178(opt_concat_177)
        opt_mul_179 = opt_conv2d_178 * self.mul_179_w
        opt_add_180 = P.Add()(opt_relu_164, opt_mul_179)
        opt_relu_181 = self.relu_181(opt_add_180)
        opt_conv2d_182 = self.conv2d_182(opt_relu_181)
        opt_conv2d_183 = self.conv2d_183(opt_relu_181)
        opt_conv2d_184 = self.conv2d_184(opt_relu_181)
        opt_relu_185 = self.relu_185(opt_conv2d_182)
        opt_relu_186 = self.relu_186(opt_conv2d_183)
        opt_relu_187 = self.relu_187(opt_conv2d_184)
        opt_conv2d_188 = self.conv2d_188(opt_relu_186)
        opt_conv2d_189 = self.conv2d_189(opt_relu_187)
        opt_relu_190 = self.relu_190(opt_conv2d_188)
        opt_relu_191 = self.relu_191(opt_conv2d_189)
        opt_conv2d_192 = self.conv2d_192(opt_relu_191)
        opt_relu_193 = self.relu_193(opt_conv2d_192)
        opt_concat_194 = self.concat_194((opt_relu_185, opt_relu_190, opt_relu_193, ))
        opt_conv2d_195 = self.conv2d_195(opt_concat_194)
        opt_mul_196 = opt_conv2d_195 * self.mul_196_w
        opt_add_197 = P.Add()(opt_relu_181, opt_mul_196)
        opt_relu_198 = self.relu_198(opt_add_197)
        opt_maxpool2d_199 = self.pad_maxpool2d_199(opt_relu_198)
        opt_maxpool2d_199 = self.maxpool2d_199(opt_maxpool2d_199)
        opt_conv2d_200 = self.conv2d_200(opt_relu_198)
        opt_conv2d_201 = self.conv2d_201(opt_relu_198)
        opt_relu_202 = self.relu_202(opt_conv2d_200)
        opt_relu_203 = self.relu_203(opt_conv2d_201)
        opt_conv2d_204 = self.conv2d_204(opt_relu_203)
        opt_relu_205 = self.relu_205(opt_conv2d_204)
        opt_conv2d_206 = self.conv2d_206(opt_relu_205)
        opt_relu_207 = self.relu_207(opt_conv2d_206)
        opt_concat_208 = self.concat_208((opt_relu_202, opt_relu_207, opt_maxpool2d_199, ))
        opt_conv2d_209 = self.conv2d_209(opt_concat_208)
        opt_conv2d_210 = self.conv2d_210(opt_concat_208)
        opt_relu_211 = self.relu_211(opt_conv2d_209)
        opt_relu_212 = self.relu_212(opt_conv2d_210)
        opt_conv2d_213 = self.conv2d_213(opt_relu_212)
        opt_relu_214 = self.relu_214(opt_conv2d_213)
        opt_conv2d_215 = self.conv2d_215(opt_relu_214)
        opt_relu_216 = self.relu_216(opt_conv2d_215)
        opt_concat_217 = self.concat_217((opt_relu_211, opt_relu_216, ))
        opt_conv2d_218 = self.conv2d_218(opt_concat_217)
        opt_mul_219 = opt_conv2d_218 * self.mul_219_w
        opt_add_220 = P.Add()(opt_concat_208, opt_mul_219)
        opt_relu_221 = self.relu_221(opt_add_220)
        opt_conv2d_222 = self.conv2d_222(opt_relu_221)
        opt_conv2d_223 = self.conv2d_223(opt_relu_221)
        opt_relu_224 = self.relu_224(opt_conv2d_222)
        opt_relu_225 = self.relu_225(opt_conv2d_223)
        opt_conv2d_226 = self.conv2d_226(opt_relu_225)
        opt_relu_227 = self.relu_227(opt_conv2d_226)
        opt_conv2d_228 = self.conv2d_228(opt_relu_227)
        opt_relu_229 = self.relu_229(opt_conv2d_228)
        opt_concat_230 = self.concat_230((opt_relu_224, opt_relu_229, ))
        opt_conv2d_231 = self.conv2d_231(opt_concat_230)
        opt_mul_232 = opt_conv2d_231 * self.mul_232_w
        opt_add_233 = P.Add()(opt_relu_221, opt_mul_232)
        opt_relu_234 = self.relu_234(opt_add_233)
        opt_conv2d_235 = self.conv2d_235(opt_relu_234)
        opt_conv2d_236 = self.conv2d_236(opt_relu_234)
        opt_relu_237 = self.relu_237(opt_conv2d_235)
        opt_relu_238 = self.relu_238(opt_conv2d_236)
        opt_conv2d_239 = self.conv2d_239(opt_relu_238)
        opt_relu_240 = self.relu_240(opt_conv2d_239)
        opt_conv2d_241 = self.conv2d_241(opt_relu_240)
        opt_relu_242 = self.relu_242(opt_conv2d_241)
        opt_concat_243 = self.concat_243((opt_relu_237, opt_relu_242, ))
        opt_conv2d_244 = self.conv2d_244(opt_concat_243)
        opt_mul_245 = opt_conv2d_244 * self.mul_245_w
        opt_add_246 = P.Add()(opt_relu_234, opt_mul_245)
        opt_relu_247 = self.relu_247(opt_add_246)
        opt_conv2d_248 = self.conv2d_248(opt_relu_247)
        opt_conv2d_249 = self.conv2d_249(opt_relu_247)
        opt_relu_250 = self.relu_250(opt_conv2d_248)
        opt_relu_251 = self.relu_251(opt_conv2d_249)
        opt_conv2d_252 = self.conv2d_252(opt_relu_251)
        opt_relu_253 = self.relu_253(opt_conv2d_252)
        opt_conv2d_254 = self.conv2d_254(opt_relu_253)
        opt_relu_255 = self.relu_255(opt_conv2d_254)
        opt_concat_256 = self.concat_256((opt_relu_250, opt_relu_255, ))
        opt_conv2d_257 = self.conv2d_257(opt_concat_256)
        opt_mul_258 = opt_conv2d_257 * self.mul_258_w
        opt_add_259 = P.Add()(opt_relu_247, opt_mul_258)
        opt_relu_260 = self.relu_260(opt_add_259)
        opt_conv2d_261 = self.conv2d_261(opt_relu_260)
        opt_conv2d_262 = self.conv2d_262(opt_relu_260)
        opt_relu_263 = self.relu_263(opt_conv2d_261)
        opt_relu_264 = self.relu_264(opt_conv2d_262)
        opt_conv2d_265 = self.conv2d_265(opt_relu_264)
        opt_relu_266 = self.relu_266(opt_conv2d_265)
        opt_conv2d_267 = self.conv2d_267(opt_relu_266)
        opt_relu_268 = self.relu_268(opt_conv2d_267)
        opt_concat_269 = self.concat_269((opt_relu_263, opt_relu_268, ))
        opt_conv2d_270 = self.conv2d_270(opt_concat_269)
        opt_mul_271 = opt_conv2d_270 * self.mul_271_w
        opt_add_272 = P.Add()(opt_relu_260, opt_mul_271)
        opt_relu_273 = self.relu_273(opt_add_272)
        opt_conv2d_274 = self.conv2d_274(opt_relu_273)
        opt_conv2d_275 = self.conv2d_275(opt_relu_273)
        opt_relu_276 = self.relu_276(opt_conv2d_274)
        opt_relu_277 = self.relu_277(opt_conv2d_275)
        opt_conv2d_278 = self.conv2d_278(opt_relu_277)
        opt_relu_279 = self.relu_279(opt_conv2d_278)
        opt_conv2d_280 = self.conv2d_280(opt_relu_279)
        opt_relu_281 = self.relu_281(opt_conv2d_280)
        opt_concat_282 = self.concat_282((opt_relu_276, opt_relu_281, ))
        opt_conv2d_283 = self.conv2d_283(opt_concat_282)
        opt_mul_284 = opt_conv2d_283 * self.mul_284_w
        opt_add_285 = P.Add()(opt_relu_273, opt_mul_284)
        opt_relu_286 = self.relu_286(opt_add_285)
        opt_conv2d_287 = self.conv2d_287(opt_relu_286)
        opt_conv2d_288 = self.conv2d_288(opt_relu_286)
        opt_relu_289 = self.relu_289(opt_conv2d_287)
        opt_relu_290 = self.relu_290(opt_conv2d_288)
        opt_conv2d_291 = self.conv2d_291(opt_relu_290)
        opt_relu_292 = self.relu_292(opt_conv2d_291)
        opt_conv2d_293 = self.conv2d_293(opt_relu_292)
        opt_relu_294 = self.relu_294(opt_conv2d_293)
        opt_concat_295 = self.concat_295((opt_relu_289, opt_relu_294, ))
        opt_conv2d_296 = self.conv2d_296(opt_concat_295)
        opt_mul_297 = opt_conv2d_296 * self.mul_297_w
        opt_add_298 = P.Add()(opt_relu_286, opt_mul_297)
        opt_relu_299 = self.relu_299(opt_add_298)
        opt_conv2d_300 = self.conv2d_300(opt_relu_299)
        opt_conv2d_301 = self.conv2d_301(opt_relu_299)
        opt_relu_302 = self.relu_302(opt_conv2d_300)
        opt_relu_303 = self.relu_303(opt_conv2d_301)
        opt_conv2d_304 = self.conv2d_304(opt_relu_303)
        opt_relu_305 = self.relu_305(opt_conv2d_304)
        opt_conv2d_306 = self.conv2d_306(opt_relu_305)
        opt_relu_307 = self.relu_307(opt_conv2d_306)
        opt_concat_308 = self.concat_308((opt_relu_302, opt_relu_307, ))
        opt_conv2d_309 = self.conv2d_309(opt_concat_308)
        opt_mul_310 = opt_conv2d_309 * self.mul_310_w
        opt_add_311 = P.Add()(opt_relu_299, opt_mul_310)
        opt_relu_312 = self.relu_312(opt_add_311)
        opt_conv2d_313 = self.conv2d_313(opt_relu_312)
        opt_conv2d_314 = self.conv2d_314(opt_relu_312)
        opt_relu_315 = self.relu_315(opt_conv2d_313)
        opt_relu_316 = self.relu_316(opt_conv2d_314)
        opt_conv2d_317 = self.conv2d_317(opt_relu_316)
        opt_relu_318 = self.relu_318(opt_conv2d_317)
        opt_conv2d_319 = self.conv2d_319(opt_relu_318)
        opt_relu_320 = self.relu_320(opt_conv2d_319)
        opt_concat_321 = self.concat_321((opt_relu_315, opt_relu_320, ))
        opt_conv2d_322 = self.conv2d_322(opt_concat_321)
        opt_mul_323 = opt_conv2d_322 * self.mul_323_w
        opt_add_324 = P.Add()(opt_relu_312, opt_mul_323)
        opt_relu_325 = self.relu_325(opt_add_324)
        opt_conv2d_326 = self.conv2d_326(opt_relu_325)
        opt_conv2d_327 = self.conv2d_327(opt_relu_325)
        opt_relu_328 = self.relu_328(opt_conv2d_326)
        opt_relu_329 = self.relu_329(opt_conv2d_327)
        opt_conv2d_330 = self.conv2d_330(opt_relu_329)
        opt_relu_331 = self.relu_331(opt_conv2d_330)
        opt_conv2d_332 = self.conv2d_332(opt_relu_331)
        opt_relu_333 = self.relu_333(opt_conv2d_332)
        opt_concat_334 = self.concat_334((opt_relu_328, opt_relu_333, ))
        opt_conv2d_335 = self.conv2d_335(opt_concat_334)
        opt_mul_336 = opt_conv2d_335 * self.mul_336_w
        opt_add_337 = P.Add()(opt_relu_325, opt_mul_336)
        opt_relu_338 = self.relu_338(opt_add_337)
        opt_conv2d_339 = self.conv2d_339(opt_relu_338)
        opt_conv2d_340 = self.conv2d_340(opt_relu_338)
        opt_relu_341 = self.relu_341(opt_conv2d_339)
        opt_relu_342 = self.relu_342(opt_conv2d_340)
        opt_conv2d_343 = self.conv2d_343(opt_relu_342)
        opt_relu_344 = self.relu_344(opt_conv2d_343)
        opt_conv2d_345 = self.conv2d_345(opt_relu_344)
        opt_relu_346 = self.relu_346(opt_conv2d_345)
        opt_concat_347 = self.concat_347((opt_relu_341, opt_relu_346, ))
        opt_conv2d_348 = self.conv2d_348(opt_concat_347)
        opt_mul_349 = opt_conv2d_348 * self.mul_349_w
        opt_add_350 = P.Add()(opt_relu_338, opt_mul_349)
        opt_relu_351 = self.relu_351(opt_add_350)
        opt_conv2d_352 = self.conv2d_352(opt_relu_351)
        opt_conv2d_353 = self.conv2d_353(opt_relu_351)
        opt_relu_354 = self.relu_354(opt_conv2d_352)
        opt_relu_355 = self.relu_355(opt_conv2d_353)
        opt_conv2d_356 = self.conv2d_356(opt_relu_355)
        opt_relu_357 = self.relu_357(opt_conv2d_356)
        opt_conv2d_358 = self.conv2d_358(opt_relu_357)
        opt_relu_359 = self.relu_359(opt_conv2d_358)
        opt_concat_360 = self.concat_360((opt_relu_354, opt_relu_359, ))
        opt_conv2d_361 = self.conv2d_361(opt_concat_360)
        opt_mul_362 = opt_conv2d_361 * self.mul_362_w
        opt_add_363 = P.Add()(opt_relu_351, opt_mul_362)
        opt_relu_364 = self.relu_364(opt_add_363)
        opt_conv2d_365 = self.conv2d_365(opt_relu_364)
        opt_conv2d_366 = self.conv2d_366(opt_relu_364)
        opt_relu_367 = self.relu_367(opt_conv2d_365)
        opt_relu_368 = self.relu_368(opt_conv2d_366)
        opt_conv2d_369 = self.conv2d_369(opt_relu_368)
        opt_relu_370 = self.relu_370(opt_conv2d_369)
        opt_conv2d_371 = self.conv2d_371(opt_relu_370)
        opt_relu_372 = self.relu_372(opt_conv2d_371)
        opt_concat_373 = self.concat_373((opt_relu_367, opt_relu_372, ))
        opt_conv2d_374 = self.conv2d_374(opt_concat_373)
        opt_mul_375 = opt_conv2d_374 * self.mul_375_w
        opt_add_376 = P.Add()(opt_relu_364, opt_mul_375)
        opt_relu_377 = self.relu_377(opt_add_376)
        opt_conv2d_378 = self.conv2d_378(opt_relu_377)
        opt_conv2d_379 = self.conv2d_379(opt_relu_377)
        opt_relu_380 = self.relu_380(opt_conv2d_378)
        opt_relu_381 = self.relu_381(opt_conv2d_379)
        opt_conv2d_382 = self.conv2d_382(opt_relu_381)
        opt_relu_383 = self.relu_383(opt_conv2d_382)
        opt_conv2d_384 = self.conv2d_384(opt_relu_383)
        opt_relu_385 = self.relu_385(opt_conv2d_384)
        opt_concat_386 = self.concat_386((opt_relu_380, opt_relu_385, ))
        opt_conv2d_387 = self.conv2d_387(opt_concat_386)
        opt_mul_388 = opt_conv2d_387 * self.mul_388_w
        opt_add_389 = P.Add()(opt_relu_377, opt_mul_388)
        opt_relu_390 = self.relu_390(opt_add_389)
        opt_conv2d_391 = self.conv2d_391(opt_relu_390)
        opt_conv2d_392 = self.conv2d_392(opt_relu_390)
        opt_relu_393 = self.relu_393(opt_conv2d_391)
        opt_relu_394 = self.relu_394(opt_conv2d_392)
        opt_conv2d_395 = self.conv2d_395(opt_relu_394)
        opt_relu_396 = self.relu_396(opt_conv2d_395)
        opt_conv2d_397 = self.conv2d_397(opt_relu_396)
        opt_relu_398 = self.relu_398(opt_conv2d_397)
        opt_concat_399 = self.concat_399((opt_relu_393, opt_relu_398, ))
        opt_conv2d_400 = self.conv2d_400(opt_concat_399)
        opt_mul_401 = opt_conv2d_400 * self.mul_401_w
        opt_add_402 = P.Add()(opt_relu_390, opt_mul_401)
        opt_relu_403 = self.relu_403(opt_add_402)
        opt_conv2d_404 = self.conv2d_404(opt_relu_403)
        opt_conv2d_405 = self.conv2d_405(opt_relu_403)
        opt_relu_406 = self.relu_406(opt_conv2d_404)
        opt_relu_407 = self.relu_407(opt_conv2d_405)
        opt_conv2d_408 = self.conv2d_408(opt_relu_407)
        opt_relu_409 = self.relu_409(opt_conv2d_408)
        opt_conv2d_410 = self.conv2d_410(opt_relu_409)
        opt_relu_411 = self.relu_411(opt_conv2d_410)
        opt_concat_412 = self.concat_412((opt_relu_406, opt_relu_411, ))
        opt_conv2d_413 = self.conv2d_413(opt_concat_412)
        opt_mul_414 = opt_conv2d_413 * self.mul_414_w
        opt_add_415 = P.Add()(opt_relu_403, opt_mul_414)
        opt_relu_416 = self.relu_416(opt_add_415)
        opt_conv2d_417 = self.conv2d_417(opt_relu_416)
        opt_conv2d_418 = self.conv2d_418(opt_relu_416)
        opt_relu_419 = self.relu_419(opt_conv2d_417)
        opt_relu_420 = self.relu_420(opt_conv2d_418)
        opt_conv2d_421 = self.conv2d_421(opt_relu_420)
        opt_relu_422 = self.relu_422(opt_conv2d_421)
        opt_conv2d_423 = self.conv2d_423(opt_relu_422)
        opt_relu_424 = self.relu_424(opt_conv2d_423)
        opt_concat_425 = self.concat_425((opt_relu_419, opt_relu_424, ))
        opt_conv2d_426 = self.conv2d_426(opt_concat_425)
        opt_mul_427 = opt_conv2d_426 * self.mul_427_w
        opt_add_428 = P.Add()(opt_relu_416, opt_mul_427)
        opt_relu_429 = self.relu_429(opt_add_428)
        opt_conv2d_430 = self.conv2d_430(opt_relu_429)
        opt_conv2d_431 = self.conv2d_431(opt_relu_429)
        opt_relu_432 = self.relu_432(opt_conv2d_430)
        opt_relu_433 = self.relu_433(opt_conv2d_431)
        opt_conv2d_434 = self.conv2d_434(opt_relu_433)
        opt_relu_435 = self.relu_435(opt_conv2d_434)
        opt_conv2d_436 = self.conv2d_436(opt_relu_435)
        opt_relu_437 = self.relu_437(opt_conv2d_436)
        opt_concat_438 = self.concat_438((opt_relu_432, opt_relu_437, ))
        opt_conv2d_439 = self.conv2d_439(opt_concat_438)
        opt_mul_440 = opt_conv2d_439 * self.mul_440_w
        opt_add_441 = P.Add()(opt_relu_429, opt_mul_440)
        opt_relu_442 = self.relu_442(opt_add_441)
        opt_conv2d_443 = self.conv2d_443(opt_relu_442)
        opt_conv2d_444 = self.conv2d_444(opt_relu_442)
        opt_relu_445 = self.relu_445(opt_conv2d_443)
        opt_relu_446 = self.relu_446(opt_conv2d_444)
        opt_conv2d_447 = self.conv2d_447(opt_relu_446)
        opt_relu_448 = self.relu_448(opt_conv2d_447)
        opt_conv2d_449 = self.conv2d_449(opt_relu_448)
        opt_relu_450 = self.relu_450(opt_conv2d_449)
        opt_concat_451 = self.concat_451((opt_relu_445, opt_relu_450, ))
        opt_conv2d_452 = self.conv2d_452(opt_concat_451)
        opt_mul_453 = opt_conv2d_452 * self.mul_453_w
        opt_add_454 = P.Add()(opt_relu_442, opt_mul_453)
        opt_relu_455 = self.relu_455(opt_add_454)
        opt_conv2d_456 = self.conv2d_456(opt_relu_455)
        opt_conv2d_457 = self.conv2d_457(opt_relu_455)
        opt_relu_458 = self.relu_458(opt_conv2d_456)
        opt_relu_459 = self.relu_459(opt_conv2d_457)
        opt_conv2d_460 = self.conv2d_460(opt_relu_459)
        opt_relu_461 = self.relu_461(opt_conv2d_460)
        opt_conv2d_462 = self.conv2d_462(opt_relu_461)
        opt_relu_463 = self.relu_463(opt_conv2d_462)
        opt_concat_464 = self.concat_464((opt_relu_458, opt_relu_463, ))
        opt_conv2d_465 = self.conv2d_465(opt_concat_464)
        opt_mul_466 = opt_conv2d_465 * self.mul_466_w
        opt_add_467 = P.Add()(opt_relu_455, opt_mul_466)
        opt_relu_468 = self.relu_468(opt_add_467)
        opt_maxpool2d_469 = self.pad_maxpool2d_469(opt_relu_468)
        opt_maxpool2d_469 = self.maxpool2d_469(opt_maxpool2d_469)
        opt_conv2d_470 = self.conv2d_470(opt_relu_468)
        opt_conv2d_471 = self.conv2d_471(opt_relu_468)
        opt_conv2d_472 = self.conv2d_472(opt_relu_468)
        opt_relu_473 = self.relu_473(opt_conv2d_470)
        opt_relu_474 = self.relu_474(opt_conv2d_471)
        opt_relu_475 = self.relu_475(opt_conv2d_472)
        opt_conv2d_476 = self.conv2d_476(opt_relu_473)
        opt_conv2d_477 = self.conv2d_477(opt_relu_474)
        opt_conv2d_478 = self.conv2d_478(opt_relu_475)
        opt_relu_479 = self.relu_479(opt_conv2d_476)
        opt_relu_480 = self.relu_480(opt_conv2d_477)
        opt_relu_481 = self.relu_481(opt_conv2d_478)
        opt_conv2d_482 = self.conv2d_482(opt_relu_481)
        opt_relu_483 = self.relu_483(opt_conv2d_482)
        opt_concat_484 = self.concat_484((opt_relu_479, opt_relu_480, opt_relu_483, opt_maxpool2d_469, ))
        opt_conv2d_485 = self.conv2d_485(opt_concat_484)
        opt_conv2d_486 = self.conv2d_486(opt_concat_484)
        opt_relu_487 = self.relu_487(opt_conv2d_485)
        opt_relu_488 = self.relu_488(opt_conv2d_486)
        opt_conv2d_489 = self.conv2d_489(opt_relu_488)
        opt_relu_490 = self.relu_490(opt_conv2d_489)
        opt_conv2d_491 = self.conv2d_491(opt_relu_490)
        opt_relu_492 = self.relu_492(opt_conv2d_491)
        opt_concat_493 = self.concat_493((opt_relu_487, opt_relu_492, ))
        opt_conv2d_494 = self.conv2d_494(opt_concat_493)
        opt_mul_495 = opt_conv2d_494 * self.mul_495_w
        opt_add_496 = P.Add()(opt_concat_484, opt_mul_495)
        opt_relu_497 = self.relu_497(opt_add_496)
        opt_conv2d_498 = self.conv2d_498(opt_relu_497)
        opt_conv2d_499 = self.conv2d_499(opt_relu_497)
        opt_relu_500 = self.relu_500(opt_conv2d_498)
        opt_relu_501 = self.relu_501(opt_conv2d_499)
        opt_conv2d_502 = self.conv2d_502(opt_relu_501)
        opt_relu_503 = self.relu_503(opt_conv2d_502)
        opt_conv2d_504 = self.conv2d_504(opt_relu_503)
        opt_relu_505 = self.relu_505(opt_conv2d_504)
        opt_concat_506 = self.concat_506((opt_relu_500, opt_relu_505, ))
        opt_conv2d_507 = self.conv2d_507(opt_concat_506)
        opt_mul_508 = opt_conv2d_507 * self.mul_508_w
        opt_add_509 = P.Add()(opt_relu_497, opt_mul_508)
        opt_relu_510 = self.relu_510(opt_add_509)
        opt_conv2d_511 = self.conv2d_511(opt_relu_510)
        opt_conv2d_512 = self.conv2d_512(opt_relu_510)
        opt_relu_513 = self.relu_513(opt_conv2d_511)
        opt_relu_514 = self.relu_514(opt_conv2d_512)
        opt_conv2d_515 = self.conv2d_515(opt_relu_514)
        opt_relu_516 = self.relu_516(opt_conv2d_515)
        opt_conv2d_517 = self.conv2d_517(opt_relu_516)
        opt_relu_518 = self.relu_518(opt_conv2d_517)
        opt_concat_519 = self.concat_519((opt_relu_513, opt_relu_518, ))
        opt_conv2d_520 = self.conv2d_520(opt_concat_519)
        opt_mul_521 = opt_conv2d_520 * self.mul_521_w
        opt_add_522 = P.Add()(opt_relu_510, opt_mul_521)
        opt_relu_523 = self.relu_523(opt_add_522)
        opt_conv2d_524 = self.conv2d_524(opt_relu_523)
        opt_conv2d_525 = self.conv2d_525(opt_relu_523)
        opt_relu_526 = self.relu_526(opt_conv2d_524)
        opt_relu_527 = self.relu_527(opt_conv2d_525)
        opt_conv2d_528 = self.conv2d_528(opt_relu_527)
        opt_relu_529 = self.relu_529(opt_conv2d_528)
        opt_conv2d_530 = self.conv2d_530(opt_relu_529)
        opt_relu_531 = self.relu_531(opt_conv2d_530)
        opt_concat_532 = self.concat_532((opt_relu_526, opt_relu_531, ))
        opt_conv2d_533 = self.conv2d_533(opt_concat_532)
        opt_mul_534 = opt_conv2d_533 * self.mul_534_w
        opt_add_535 = P.Add()(opt_relu_523, opt_mul_534)
        opt_relu_536 = self.relu_536(opt_add_535)
        opt_conv2d_537 = self.conv2d_537(opt_relu_536)
        opt_conv2d_538 = self.conv2d_538(opt_relu_536)
        opt_relu_539 = self.relu_539(opt_conv2d_537)
        opt_relu_540 = self.relu_540(opt_conv2d_538)
        opt_conv2d_541 = self.conv2d_541(opt_relu_540)
        opt_relu_542 = self.relu_542(opt_conv2d_541)
        opt_conv2d_543 = self.conv2d_543(opt_relu_542)
        opt_relu_544 = self.relu_544(opt_conv2d_543)
        opt_concat_545 = self.concat_545((opt_relu_539, opt_relu_544, ))
        opt_conv2d_546 = self.conv2d_546(opt_concat_545)
        opt_mul_547 = opt_conv2d_546 * self.mul_547_w
        opt_add_548 = P.Add()(opt_relu_536, opt_mul_547)
        opt_relu_549 = self.relu_549(opt_add_548)
        opt_conv2d_550 = self.conv2d_550(opt_relu_549)
        opt_conv2d_551 = self.conv2d_551(opt_relu_549)
        opt_relu_552 = self.relu_552(opt_conv2d_550)
        opt_relu_553 = self.relu_553(opt_conv2d_551)
        opt_conv2d_554 = self.conv2d_554(opt_relu_553)
        opt_relu_555 = self.relu_555(opt_conv2d_554)
        opt_conv2d_556 = self.conv2d_556(opt_relu_555)
        opt_relu_557 = self.relu_557(opt_conv2d_556)
        opt_concat_558 = self.concat_558((opt_relu_552, opt_relu_557, ))
        opt_conv2d_559 = self.conv2d_559(opt_concat_558)
        opt_mul_560 = opt_conv2d_559 * self.mul_560_w
        opt_add_561 = P.Add()(opt_relu_549, opt_mul_560)
        opt_relu_562 = self.relu_562(opt_add_561)
        opt_conv2d_563 = self.conv2d_563(opt_relu_562)
        opt_conv2d_564 = self.conv2d_564(opt_relu_562)
        opt_relu_565 = self.relu_565(opt_conv2d_563)
        opt_relu_566 = self.relu_566(opt_conv2d_564)
        opt_conv2d_567 = self.conv2d_567(opt_relu_566)
        opt_relu_568 = self.relu_568(opt_conv2d_567)
        opt_conv2d_569 = self.conv2d_569(opt_relu_568)
        opt_relu_570 = self.relu_570(opt_conv2d_569)
        opt_concat_571 = self.concat_571((opt_relu_565, opt_relu_570, ))
        opt_conv2d_572 = self.conv2d_572(opt_concat_571)
        opt_mul_573 = opt_conv2d_572 * self.mul_573_w
        opt_add_574 = P.Add()(opt_relu_562, opt_mul_573)
        opt_relu_575 = self.relu_575(opt_add_574)
        opt_conv2d_576 = self.conv2d_576(opt_relu_575)
        opt_conv2d_577 = self.conv2d_577(opt_relu_575)
        opt_relu_578 = self.relu_578(opt_conv2d_576)
        opt_relu_579 = self.relu_579(opt_conv2d_577)
        opt_conv2d_580 = self.conv2d_580(opt_relu_579)
        opt_relu_581 = self.relu_581(opt_conv2d_580)
        opt_conv2d_582 = self.conv2d_582(opt_relu_581)
        opt_relu_583 = self.relu_583(opt_conv2d_582)
        opt_concat_584 = self.concat_584((opt_relu_578, opt_relu_583, ))
        opt_conv2d_585 = self.conv2d_585(opt_concat_584)
        opt_mul_586 = opt_conv2d_585 * self.mul_586_w
        opt_add_587 = P.Add()(opt_relu_575, opt_mul_586)
        opt_relu_588 = self.relu_588(opt_add_587)
        opt_conv2d_589 = self.conv2d_589(opt_relu_588)
        opt_conv2d_590 = self.conv2d_590(opt_relu_588)
        opt_relu_591 = self.relu_591(opt_conv2d_589)
        opt_relu_592 = self.relu_592(opt_conv2d_590)
        opt_conv2d_593 = self.conv2d_593(opt_relu_592)
        opt_relu_594 = self.relu_594(opt_conv2d_593)
        opt_conv2d_595 = self.conv2d_595(opt_relu_594)
        opt_relu_596 = self.relu_596(opt_conv2d_595)
        opt_concat_597 = self.concat_597((opt_relu_591, opt_relu_596, ))
        opt_conv2d_598 = self.conv2d_598(opt_concat_597)
        opt_mul_599 = opt_conv2d_598 * self.mul_599_w
        opt_add_600 = P.Add()(opt_relu_588, opt_mul_599)
        opt_relu_601 = self.relu_601(opt_add_600)
        opt_conv2d_602 = self.conv2d_602(opt_relu_601)
        opt_conv2d_603 = self.conv2d_603(opt_relu_601)
        opt_relu_604 = self.relu_604(opt_conv2d_602)
        opt_relu_605 = self.relu_605(opt_conv2d_603)
        opt_conv2d_606 = self.conv2d_606(opt_relu_605)
        opt_relu_607 = self.relu_607(opt_conv2d_606)
        opt_conv2d_608 = self.conv2d_608(opt_relu_607)
        opt_relu_609 = self.relu_609(opt_conv2d_608)
        opt_concat_610 = self.concat_610((opt_relu_604, opt_relu_609, ))
        opt_conv2d_611 = self.conv2d_611(opt_concat_610)
        opt_mul_612 = opt_conv2d_611 * self.mul_612_w
        opt_add_613 = P.Add()(opt_relu_601, opt_mul_612)
        opt_conv2d_614 = self.conv2d_614(opt_add_613)
        opt_relu_615 = self.relu_615(opt_conv2d_614)
        opt_avgpool2d_616 = self.avgpool2d_616(opt_relu_615)
        opt_transpose_617 = self.transpose_617(opt_avgpool2d_616, (0, 2, 3, 1))
        opt_reshape_618 = self.reshape_618(opt_transpose_617, self.reshape_618_shape)
        opt_matmul_619 = P.matmul(opt_reshape_618, self.matmul_619_w)
        opt_add_620 = opt_matmul_619 + self.add_620_bias
        opt_softmax_621 = self.softmax_621(opt_add_620)
        return opt_softmax_621
