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
        self.transpose_1 = P.Transpose()
        self.conv2d_2 = nn.Conv2d(in_channels=3,
                                  out_channels=96,
                                  kernel_size=(5, 5),
                                  stride=(1, 1),
                                  padding=(2, 2, 2, 2),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.conv2d_3 = nn.Conv2d(in_channels=3,
                                  out_channels=96,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.leakyrelu_4 = nn.LeakyReLU(alpha=0.30000001192092896)
        self.leakyrelu_5 = nn.LeakyReLU(alpha=0.30000001192092896)
        self.conv2d_6 = nn.Conv2d(in_channels=96,
                                  out_channels=96,
                                  kernel_size=(5, 5),
                                  stride=(1, 1),
                                  padding=(2, 2, 2, 2),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.conv2d_7 = nn.Conv2d(in_channels=96,
                                  out_channels=96,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.leakyrelu_8 = nn.LeakyReLU(alpha=0.30000001192092896)
        self.leakyrelu_9 = nn.LeakyReLU(alpha=0.30000001192092896)
        self.concat_10 = P.Concat(axis=2)
        self.conv2d_11 = nn.Conv2d(in_channels=96,
                                   out_channels=96,
                                   kernel_size=(5, 5),
                                   stride=(2, 2),
                                   padding=(1, 2, 1, 2),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.leakyrelu_12 = nn.LeakyReLU(alpha=0.30000001192092896)
        self.transpose_13 = P.Transpose()
        self.batchnorm2d_14 = nn.BatchNorm2d(num_features=32, eps=9.999999974752427e-07, momentum=0.9900000095367432)
        self.transpose_15 = P.Transpose()
        self.transpose_16 = P.Transpose()
        self.conv2d_17 = nn.Conv2d(in_channels=96,
                                   out_channels=192,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_18 = nn.Conv2d(in_channels=96,
                                   out_channels=192,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.leakyrelu_19 = nn.LeakyReLU(alpha=0.30000001192092896)
        self.leakyrelu_20 = nn.LeakyReLU(alpha=0.30000001192092896)
        self.conv2d_21 = nn.Conv2d(in_channels=192,
                                   out_channels=192,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_22 = nn.Conv2d(in_channels=192,
                                   out_channels=192,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.leakyrelu_23 = nn.LeakyReLU(alpha=0.30000001192092896)
        self.leakyrelu_24 = nn.LeakyReLU(alpha=0.30000001192092896)
        self.concat_25 = P.Concat(axis=2)
        self.conv2d_26 = nn.Conv2d(in_channels=192,
                                   out_channels=192,
                                   kernel_size=(3, 3),
                                   stride=(2, 2),
                                   padding=(0, 1, 0, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.leakyrelu_27 = nn.LeakyReLU(alpha=0.30000001192092896)
        self.transpose_28 = P.Transpose()
        self.batchnorm2d_29 = nn.BatchNorm2d(num_features=32, eps=9.999999974752427e-07, momentum=0.9900000095367432)
        self.transpose_30 = P.Transpose()
        self.transpose_31 = P.Transpose()
        self.transpose_32 = P.Transpose()
        self.conv2d_33 = nn.Conv2d(in_channels=192,
                                   out_channels=256,
                                   kernel_size=(3, 3),
                                   stride=(1, 1),
                                   padding=(1, 1, 1, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_34 = nn.Conv2d(in_channels=192,
                                   out_channels=256,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.conv2d_35 = nn.Conv2d(in_channels=192,
                                   out_channels=256,
                                   kernel_size=(5, 5),
                                   stride=(1, 1),
                                   padding=(2, 2, 2, 2),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.leakyrelu_36 = nn.LeakyReLU(alpha=0.30000001192092896)
        self.leakyrelu_37 = nn.LeakyReLU(alpha=0.30000001192092896)
        self.leakyrelu_38 = nn.LeakyReLU(alpha=0.30000001192092896)
        self.concat_39 = P.Concat(axis=2)
        self.conv2d_40 = nn.Conv2d(in_channels=256,
                                   out_channels=192,
                                   kernel_size=(3, 3),
                                   stride=(2, 2),
                                   padding=(0, 1, 0, 1),
                                   pad_mode="pad",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.leakyrelu_41 = nn.LeakyReLU(alpha=0.30000001192092896)
        self.transpose_42 = P.Transpose()
        self.flatten_43 = nn.Flatten()
        self.matmul_44_w = Parameter(Tensor(np.random.uniform(0, 1, (36864, 10)).astype(np.float32)), name=None)
        self.add_45_bias = Parameter(Tensor(np.random.uniform(0, 1, (10, )).astype(np.float32)), name=None)
        self.softmax_46 = nn.Softmax(axis=-1)

    def construct(self, input_4):
        opt_transpose_0 = self.transpose_0(input_4, (0, 3, 1, 2))
        opt_transpose_1 = self.transpose_1(input_4, (0, 3, 1, 2))
        opt_conv2d_2 = self.conv2d_2(opt_transpose_0)
        opt_conv2d_3 = self.conv2d_3(opt_transpose_1)
        opt_leakyrelu_4 = self.leakyrelu_4(opt_conv2d_2)
        opt_leakyrelu_5 = self.leakyrelu_5(opt_conv2d_3)
        opt_conv2d_6 = self.conv2d_6(opt_leakyrelu_4)
        opt_conv2d_7 = self.conv2d_7(opt_leakyrelu_5)
        opt_leakyrelu_8 = self.leakyrelu_8(opt_conv2d_6)
        opt_leakyrelu_9 = self.leakyrelu_9(opt_conv2d_7)
        opt_concat_10 = self.concat_10((opt_leakyrelu_8, opt_leakyrelu_9, ))
        opt_conv2d_11 = self.conv2d_11(opt_concat_10)
        opt_leakyrelu_12 = self.leakyrelu_12(opt_conv2d_11)
        opt_transpose_13 = self.transpose_13(opt_leakyrelu_12, (0, 2, 3, 1))
        opt_batchnorm2d_14 = self.batchnorm2d_14(opt_transpose_13)
        opt_transpose_15 = self.transpose_15(opt_batchnorm2d_14, (0, 3, 1, 2))
        opt_transpose_16 = self.transpose_16(opt_batchnorm2d_14, (0, 3, 1, 2))
        opt_conv2d_17 = self.conv2d_17(opt_transpose_15)
        opt_conv2d_18 = self.conv2d_18(opt_transpose_16)
        opt_leakyrelu_19 = self.leakyrelu_19(opt_conv2d_17)
        opt_leakyrelu_20 = self.leakyrelu_20(opt_conv2d_18)
        opt_conv2d_21 = self.conv2d_21(opt_leakyrelu_19)
        opt_conv2d_22 = self.conv2d_22(opt_leakyrelu_20)
        opt_leakyrelu_23 = self.leakyrelu_23(opt_conv2d_21)
        opt_leakyrelu_24 = self.leakyrelu_24(opt_conv2d_22)
        opt_concat_25 = self.concat_25((opt_leakyrelu_23, opt_leakyrelu_24, ))
        opt_conv2d_26 = self.conv2d_26(opt_concat_25)
        opt_leakyrelu_27 = self.leakyrelu_27(opt_conv2d_26)
        opt_transpose_28 = self.transpose_28(opt_leakyrelu_27, (0, 2, 3, 1))
        opt_batchnorm2d_29 = self.batchnorm2d_29(opt_transpose_28)
        opt_transpose_30 = self.transpose_30(opt_batchnorm2d_29, (0, 3, 1, 2))
        opt_transpose_31 = self.transpose_31(opt_batchnorm2d_29, (0, 3, 1, 2))
        opt_transpose_32 = self.transpose_32(opt_batchnorm2d_29, (0, 3, 1, 2))
        opt_conv2d_33 = self.conv2d_33(opt_transpose_30)
        opt_conv2d_34 = self.conv2d_34(opt_transpose_31)
        opt_conv2d_35 = self.conv2d_35(opt_transpose_32)
        opt_leakyrelu_36 = self.leakyrelu_36(opt_conv2d_33)
        opt_leakyrelu_37 = self.leakyrelu_37(opt_conv2d_34)
        opt_leakyrelu_38 = self.leakyrelu_38(opt_conv2d_35)
        opt_concat_39 = self.concat_39((opt_leakyrelu_36, opt_leakyrelu_37, opt_leakyrelu_38, ))
        opt_conv2d_40 = self.conv2d_40(opt_concat_39)
        opt_leakyrelu_41 = self.leakyrelu_41(opt_conv2d_40)
        opt_transpose_42 = self.transpose_42(opt_leakyrelu_41, (0, 2, 3, 1))
        opt_flatten_43 = self.flatten_43(opt_transpose_42)
        opt_matmul_44 = P.matmul(opt_flatten_43, self.matmul_44_w)
        opt_add_45 = opt_matmul_44 + self.add_45_bias
        opt_softmax_46 = self.softmax_46(opt_add_45)
        return opt_softmax_46
