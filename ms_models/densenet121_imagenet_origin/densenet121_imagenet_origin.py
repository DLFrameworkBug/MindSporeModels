import numpy as np
import mindspore
import mindspore.numpy as ms_np
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, Parameter


class Module0(nn.Cell):

    def __init__(self, batchnorm2d_0_num_features, conv2d_2_in_channels):
        super(Module0, self).__init__()
        self.batchnorm2d_0 = nn.BatchNorm2d(num_features=batchnorm2d_0_num_features,
                                            eps=1.0009999940052694e-08,
                                            momentum=0.9900000095367432)
        self.relu_1 = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(in_channels=conv2d_2_in_channels,
                                  out_channels=128,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_3 = nn.ReLU()
        self.conv2d_4 = nn.Conv2d(in_channels=128,
                                  out_channels=32,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=False)
        self.concat_5 = P.Concat(axis=1)

    def construct(self, x):
        opt_batchnorm2d_0 = self.batchnorm2d_0(x)
        opt_relu_1 = self.relu_1(opt_batchnorm2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        opt_relu_3 = self.relu_3(opt_conv2d_2)
        opt_conv2d_4 = self.conv2d_4(opt_relu_3)
        opt_concat_5 = self.concat_5((x, opt_conv2d_4, ))
        return opt_concat_5


class Module13(nn.Cell):

    def __init__(self, module0_0_batchnorm2d_0_num_features, module0_0_conv2d_2_in_channels,
                 module0_1_batchnorm2d_0_num_features, module0_1_conv2d_2_in_channels,
                 module0_2_batchnorm2d_0_num_features, module0_2_conv2d_2_in_channels,
                 module0_3_batchnorm2d_0_num_features, module0_3_conv2d_2_in_channels):
        super(Module13, self).__init__()
        self.module0_0 = Module0(batchnorm2d_0_num_features=module0_0_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_0_conv2d_2_in_channels)
        self.module0_1 = Module0(batchnorm2d_0_num_features=module0_1_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_1_conv2d_2_in_channels)
        self.module0_2 = Module0(batchnorm2d_0_num_features=module0_2_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_2_conv2d_2_in_channels)
        self.module0_3 = Module0(batchnorm2d_0_num_features=module0_3_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_3_conv2d_2_in_channels)

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        module0_1_opt = self.module0_1(module0_0_opt)
        module0_2_opt = self.module0_2(module0_1_opt)
        module0_3_opt = self.module0_3(module0_2_opt)
        return module0_3_opt


class Module11(nn.Cell):

    def __init__(self):
        super(Module11, self).__init__()
        self.module0_0 = Module0(batchnorm2d_0_num_features=192, conv2d_2_in_channels=192)
        self.module0_1 = Module0(batchnorm2d_0_num_features=224, conv2d_2_in_channels=224)
        self.batchnorm2d_0 = nn.BatchNorm2d(num_features=256, eps=1.0009999940052694e-08, momentum=0.9900000095367432)
        self.relu_1 = nn.ReLU()

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        module0_1_opt = self.module0_1(module0_0_opt)
        opt_batchnorm2d_0 = self.batchnorm2d_0(module0_1_opt)
        opt_relu_1 = self.relu_1(opt_batchnorm2d_0)
        return opt_relu_1


class Module8(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels):
        super(Module8, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=False)
        self.pad_avgpool2d_1 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_1 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_avgpool2d_1 = self.pad_avgpool2d_1(opt_conv2d_0)
        opt_avgpool2d_1 = self.avgpool2d_1(opt_avgpool2d_1)
        return opt_avgpool2d_1


class Module92(nn.Cell):

    def __init__(
            self, batchnorm2d_0_num_features, module0_0_batchnorm2d_0_num_features, module0_0_conv2d_2_in_channels,
            module0_1_batchnorm2d_0_num_features, module0_1_conv2d_2_in_channels, module0_2_batchnorm2d_0_num_features,
            module0_2_conv2d_2_in_channels, module0_3_batchnorm2d_0_num_features, module0_3_conv2d_2_in_channels,
            module0_4_batchnorm2d_0_num_features, module0_4_conv2d_2_in_channels, module0_5_batchnorm2d_0_num_features,
            module0_5_conv2d_2_in_channels, module0_6_batchnorm2d_0_num_features, module0_6_conv2d_2_in_channels,
            module0_7_batchnorm2d_0_num_features, module0_7_conv2d_2_in_channels, module0_8_batchnorm2d_0_num_features,
            module0_8_conv2d_2_in_channels, module0_9_batchnorm2d_0_num_features, module0_9_conv2d_2_in_channels,
            module0_10_batchnorm2d_0_num_features, module0_10_conv2d_2_in_channels,
            module0_11_batchnorm2d_0_num_features, module0_11_conv2d_2_in_channels, module8_0_conv2d_0_in_channels,
            module8_0_conv2d_0_out_channels, module0_12_batchnorm2d_0_num_features, module0_12_conv2d_2_in_channels,
            module0_13_batchnorm2d_0_num_features, module0_13_conv2d_2_in_channels,
            module0_14_batchnorm2d_0_num_features, module0_14_conv2d_2_in_channels,
            module0_15_batchnorm2d_0_num_features, module0_15_conv2d_2_in_channels):
        super(Module92, self).__init__()
        self.module0_0 = Module0(batchnorm2d_0_num_features=module0_0_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_0_conv2d_2_in_channels)
        self.module0_1 = Module0(batchnorm2d_0_num_features=module0_1_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_1_conv2d_2_in_channels)
        self.module0_2 = Module0(batchnorm2d_0_num_features=module0_2_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_2_conv2d_2_in_channels)
        self.module0_3 = Module0(batchnorm2d_0_num_features=module0_3_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_3_conv2d_2_in_channels)
        self.module0_4 = Module0(batchnorm2d_0_num_features=module0_4_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_4_conv2d_2_in_channels)
        self.module0_5 = Module0(batchnorm2d_0_num_features=module0_5_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_5_conv2d_2_in_channels)
        self.module0_6 = Module0(batchnorm2d_0_num_features=module0_6_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_6_conv2d_2_in_channels)
        self.module0_7 = Module0(batchnorm2d_0_num_features=module0_7_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_7_conv2d_2_in_channels)
        self.module0_8 = Module0(batchnorm2d_0_num_features=module0_8_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_8_conv2d_2_in_channels)
        self.module0_9 = Module0(batchnorm2d_0_num_features=module0_9_batchnorm2d_0_num_features,
                                 conv2d_2_in_channels=module0_9_conv2d_2_in_channels)
        self.module0_10 = Module0(batchnorm2d_0_num_features=module0_10_batchnorm2d_0_num_features,
                                  conv2d_2_in_channels=module0_10_conv2d_2_in_channels)
        self.module0_11 = Module0(batchnorm2d_0_num_features=module0_11_batchnorm2d_0_num_features,
                                  conv2d_2_in_channels=module0_11_conv2d_2_in_channels)
        self.batchnorm2d_0 = nn.BatchNorm2d(num_features=batchnorm2d_0_num_features,
                                            eps=1.0009999940052694e-08,
                                            momentum=0.9900000095367432)
        self.relu_1 = nn.ReLU()
        self.module8_0 = Module8(conv2d_0_in_channels=module8_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module8_0_conv2d_0_out_channels)
        self.module0_12 = Module0(batchnorm2d_0_num_features=module0_12_batchnorm2d_0_num_features,
                                  conv2d_2_in_channels=module0_12_conv2d_2_in_channels)
        self.module0_13 = Module0(batchnorm2d_0_num_features=module0_13_batchnorm2d_0_num_features,
                                  conv2d_2_in_channels=module0_13_conv2d_2_in_channels)
        self.module0_14 = Module0(batchnorm2d_0_num_features=module0_14_batchnorm2d_0_num_features,
                                  conv2d_2_in_channels=module0_14_conv2d_2_in_channels)
        self.module0_15 = Module0(batchnorm2d_0_num_features=module0_15_batchnorm2d_0_num_features,
                                  conv2d_2_in_channels=module0_15_conv2d_2_in_channels)

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        module0_1_opt = self.module0_1(module0_0_opt)
        module0_2_opt = self.module0_2(module0_1_opt)
        module0_3_opt = self.module0_3(module0_2_opt)
        module0_4_opt = self.module0_4(module0_3_opt)
        module0_5_opt = self.module0_5(module0_4_opt)
        module0_6_opt = self.module0_6(module0_5_opt)
        module0_7_opt = self.module0_7(module0_6_opt)
        module0_8_opt = self.module0_8(module0_7_opt)
        module0_9_opt = self.module0_9(module0_8_opt)
        module0_10_opt = self.module0_10(module0_9_opt)
        module0_11_opt = self.module0_11(module0_10_opt)
        opt_batchnorm2d_0 = self.batchnorm2d_0(module0_11_opt)
        opt_relu_1 = self.relu_1(opt_batchnorm2d_0)
        module8_0_opt = self.module8_0(opt_relu_1)
        module0_12_opt = self.module0_12(module8_0_opt)
        module0_13_opt = self.module0_13(module0_12_opt)
        module0_14_opt = self.module0_14(module0_13_opt)
        module0_15_opt = self.module0_15(module0_14_opt)
        return module0_15_opt


class Module87(nn.Cell):

    def __init__(self):
        super(Module87, self).__init__()
        self.module0_0 = Module0(batchnorm2d_0_num_features=640, conv2d_2_in_channels=640)
        self.module0_1 = Module0(batchnorm2d_0_num_features=672, conv2d_2_in_channels=672)
        self.module0_2 = Module0(batchnorm2d_0_num_features=704, conv2d_2_in_channels=704)
        self.module0_3 = Module0(batchnorm2d_0_num_features=736, conv2d_2_in_channels=736)
        self.module0_4 = Module0(batchnorm2d_0_num_features=768, conv2d_2_in_channels=768)
        self.module0_5 = Module0(batchnorm2d_0_num_features=800, conv2d_2_in_channels=800)
        self.module0_6 = Module0(batchnorm2d_0_num_features=832, conv2d_2_in_channels=832)
        self.module0_7 = Module0(batchnorm2d_0_num_features=864, conv2d_2_in_channels=864)
        self.module0_8 = Module0(batchnorm2d_0_num_features=896, conv2d_2_in_channels=896)
        self.module0_9 = Module0(batchnorm2d_0_num_features=928, conv2d_2_in_channels=928)
        self.module0_10 = Module0(batchnorm2d_0_num_features=960, conv2d_2_in_channels=960)
        self.module0_11 = Module0(batchnorm2d_0_num_features=992, conv2d_2_in_channels=992)
        self.batchnorm2d_0 = nn.BatchNorm2d(num_features=1024, eps=1.0009999940052694e-08, momentum=0.9900000095367432)
        self.relu_1 = nn.ReLU()

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        module0_1_opt = self.module0_1(module0_0_opt)
        module0_2_opt = self.module0_2(module0_1_opt)
        module0_3_opt = self.module0_3(module0_2_opt)
        module0_4_opt = self.module0_4(module0_3_opt)
        module0_5_opt = self.module0_5(module0_4_opt)
        module0_6_opt = self.module0_6(module0_5_opt)
        module0_7_opt = self.module0_7(module0_6_opt)
        module0_8_opt = self.module0_8(module0_7_opt)
        module0_9_opt = self.module0_9(module0_8_opt)
        module0_10_opt = self.module0_10(module0_9_opt)
        module0_11_opt = self.module0_11(module0_10_opt)
        opt_batchnorm2d_0 = self.batchnorm2d_0(module0_11_opt)
        opt_relu_1 = self.relu_1(opt_batchnorm2d_0)
        return opt_relu_1


class MindSporeModel(nn.Cell):

    def __init__(self):
        super(MindSporeModel, self).__init__()
        self.transpose_0 = P.Transpose()
        self.conv2d_1 = nn.Conv2d(in_channels=3,
                                  out_channels=64,
                                  kernel_size=(7, 7),
                                  stride=(2, 2),
                                  padding=(3, 3, 3, 3),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_2 = nn.ReLU()
        self.pad_3 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT")
        self.pad_maxpool2d_4 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_4 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.module13_0 = Module13(module0_0_batchnorm2d_0_num_features=64,
                                   module0_0_conv2d_2_in_channels=64,
                                   module0_1_batchnorm2d_0_num_features=96,
                                   module0_1_conv2d_2_in_channels=96,
                                   module0_2_batchnorm2d_0_num_features=128,
                                   module0_2_conv2d_2_in_channels=128,
                                   module0_3_batchnorm2d_0_num_features=160,
                                   module0_3_conv2d_2_in_channels=160)
        self.module11_0 = Module11()
        self.module8_0 = Module8(conv2d_0_in_channels=256, conv2d_0_out_channels=128)
        self.module92_0 = Module92(batchnorm2d_0_num_features=512,
                                   module0_0_batchnorm2d_0_num_features=128,
                                   module0_0_conv2d_2_in_channels=128,
                                   module0_1_batchnorm2d_0_num_features=160,
                                   module0_1_conv2d_2_in_channels=160,
                                   module0_2_batchnorm2d_0_num_features=192,
                                   module0_2_conv2d_2_in_channels=192,
                                   module0_3_batchnorm2d_0_num_features=224,
                                   module0_3_conv2d_2_in_channels=224,
                                   module0_4_batchnorm2d_0_num_features=256,
                                   module0_4_conv2d_2_in_channels=256,
                                   module0_5_batchnorm2d_0_num_features=288,
                                   module0_5_conv2d_2_in_channels=288,
                                   module0_6_batchnorm2d_0_num_features=320,
                                   module0_6_conv2d_2_in_channels=320,
                                   module0_7_batchnorm2d_0_num_features=352,
                                   module0_7_conv2d_2_in_channels=352,
                                   module0_8_batchnorm2d_0_num_features=384,
                                   module0_8_conv2d_2_in_channels=384,
                                   module0_9_batchnorm2d_0_num_features=416,
                                   module0_9_conv2d_2_in_channels=416,
                                   module0_10_batchnorm2d_0_num_features=448,
                                   module0_10_conv2d_2_in_channels=448,
                                   module0_11_batchnorm2d_0_num_features=480,
                                   module0_11_conv2d_2_in_channels=480,
                                   module8_0_conv2d_0_in_channels=512,
                                   module8_0_conv2d_0_out_channels=256,
                                   module0_12_batchnorm2d_0_num_features=256,
                                   module0_12_conv2d_2_in_channels=256,
                                   module0_13_batchnorm2d_0_num_features=288,
                                   module0_13_conv2d_2_in_channels=288,
                                   module0_14_batchnorm2d_0_num_features=320,
                                   module0_14_conv2d_2_in_channels=320,
                                   module0_15_batchnorm2d_0_num_features=352,
                                   module0_15_conv2d_2_in_channels=352)
        self.module13_1 = Module13(module0_0_batchnorm2d_0_num_features=384,
                                   module0_0_conv2d_2_in_channels=384,
                                   module0_1_batchnorm2d_0_num_features=416,
                                   module0_1_conv2d_2_in_channels=416,
                                   module0_2_batchnorm2d_0_num_features=448,
                                   module0_2_conv2d_2_in_channels=448,
                                   module0_3_batchnorm2d_0_num_features=480,
                                   module0_3_conv2d_2_in_channels=480)
        self.module13_2 = Module13(module0_0_batchnorm2d_0_num_features=512,
                                   module0_0_conv2d_2_in_channels=512,
                                   module0_1_batchnorm2d_0_num_features=544,
                                   module0_1_conv2d_2_in_channels=544,
                                   module0_2_batchnorm2d_0_num_features=576,
                                   module0_2_conv2d_2_in_channels=576,
                                   module0_3_batchnorm2d_0_num_features=608,
                                   module0_3_conv2d_2_in_channels=608)
        self.module92_1 = Module92(batchnorm2d_0_num_features=1024,
                                   module0_0_batchnorm2d_0_num_features=640,
                                   module0_0_conv2d_2_in_channels=640,
                                   module0_1_batchnorm2d_0_num_features=672,
                                   module0_1_conv2d_2_in_channels=672,
                                   module0_2_batchnorm2d_0_num_features=704,
                                   module0_2_conv2d_2_in_channels=704,
                                   module0_3_batchnorm2d_0_num_features=736,
                                   module0_3_conv2d_2_in_channels=736,
                                   module0_4_batchnorm2d_0_num_features=768,
                                   module0_4_conv2d_2_in_channels=768,
                                   module0_5_batchnorm2d_0_num_features=800,
                                   module0_5_conv2d_2_in_channels=800,
                                   module0_6_batchnorm2d_0_num_features=832,
                                   module0_6_conv2d_2_in_channels=832,
                                   module0_7_batchnorm2d_0_num_features=864,
                                   module0_7_conv2d_2_in_channels=864,
                                   module0_8_batchnorm2d_0_num_features=896,
                                   module0_8_conv2d_2_in_channels=896,
                                   module0_9_batchnorm2d_0_num_features=928,
                                   module0_9_conv2d_2_in_channels=928,
                                   module0_10_batchnorm2d_0_num_features=960,
                                   module0_10_conv2d_2_in_channels=960,
                                   module0_11_batchnorm2d_0_num_features=992,
                                   module0_11_conv2d_2_in_channels=992,
                                   module8_0_conv2d_0_in_channels=1024,
                                   module8_0_conv2d_0_out_channels=512,
                                   module0_12_batchnorm2d_0_num_features=512,
                                   module0_12_conv2d_2_in_channels=512,
                                   module0_13_batchnorm2d_0_num_features=544,
                                   module0_13_conv2d_2_in_channels=544,
                                   module0_14_batchnorm2d_0_num_features=576,
                                   module0_14_conv2d_2_in_channels=576,
                                   module0_15_batchnorm2d_0_num_features=608,
                                   module0_15_conv2d_2_in_channels=608)
        self.module87_0 = Module87()
        self.avgpool2d_367 = nn.AvgPool2d(kernel_size=(7, 7))
        self.transpose_368 = P.Transpose()
        self.reshape_369 = P.Reshape()
        self.reshape_369_shape = tuple([1, 1024])
        self.matmul_370_w = Parameter(Tensor(np.random.uniform(0, 1, (1024, 1000)).astype(np.float32)), name=None)
        self.add_371_bias = Parameter(Tensor(np.random.uniform(0, 1, (1000, )).astype(np.float32)), name=None)
        self.softmax_372 = nn.Softmax(axis=-1)

    def construct(self, input_2):
        opt_transpose_0 = self.transpose_0(input_2, (0, 3, 1, 2))
        opt_conv2d_1 = self.conv2d_1(opt_transpose_0)
        opt_relu_2 = self.relu_2(opt_conv2d_1)
        opt_pad_3 = self.pad_3(opt_relu_2)
        opt_maxpool2d_4 = self.pad_maxpool2d_4(opt_pad_3)
        opt_maxpool2d_4 = self.maxpool2d_4(opt_maxpool2d_4)
        module13_0_opt = self.module13_0(opt_maxpool2d_4)
        module11_0_opt = self.module11_0(module13_0_opt)
        module8_0_opt = self.module8_0(module11_0_opt)
        module92_0_opt = self.module92_0(module8_0_opt)
        module13_1_opt = self.module13_1(module92_0_opt)
        module13_2_opt = self.module13_2(module13_1_opt)
        module92_1_opt = self.module92_1(module13_2_opt)
        module87_0_opt = self.module87_0(module92_1_opt)
        opt_avgpool2d_367 = self.avgpool2d_367(module87_0_opt)
        opt_transpose_368 = self.transpose_368(opt_avgpool2d_367, (0, 2, 3, 1))
        opt_reshape_369 = self.reshape_369(opt_transpose_368, self.reshape_369_shape)
        opt_matmul_370 = P.matmul(opt_reshape_369, self.matmul_370_w)
        opt_add_371 = opt_matmul_370 + self.add_371_bias
        opt_softmax_372 = self.softmax_372(opt_add_371)
        return opt_softmax_372
