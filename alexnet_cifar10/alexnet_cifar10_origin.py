import numpy as np
import mindspore
import mindspore.numpy as ms_np
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, Parameter


class Module1(nn.Cell):

    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_0_kernel_size, conv2d_0_stride,
                 conv2d_0_padding):
        super(Module1, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=conv2d_0_kernel_size,
                                  stride=conv2d_0_stride,
                                  padding=conv2d_0_padding,
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        return opt_relu_1


class Module7(nn.Cell):

    def __init__(self, maxpool2d_0_kernel_size, batchnorm2d_1_num_features, module1_0_conv2d_0_in_channels,
                 module1_0_conv2d_0_out_channels, module1_0_conv2d_0_kernel_size, module1_0_conv2d_0_stride,
                 module1_0_conv2d_0_padding):
        super(Module7, self).__init__()
        self.module1_0 = Module1(conv2d_0_in_channels=module1_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module1_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module1_0_conv2d_0_kernel_size,
                                 conv2d_0_stride=module1_0_conv2d_0_stride,
                                 conv2d_0_padding=module1_0_conv2d_0_padding)
        self.pad_maxpool2d_0 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.maxpool2d_0 = nn.MaxPool2d(kernel_size=maxpool2d_0_kernel_size, stride=(2, 2))
        self.batchnorm2d_1 = nn.BatchNorm2d(num_features=batchnorm2d_1_num_features,
                                            eps=9.999999974752427e-07,
                                            momentum=0.9900000095367432)

    def construct(self, x):
        module1_0_opt = self.module1_0(x)
        opt_maxpool2d_0 = self.pad_maxpool2d_0(module1_0_opt)
        opt_maxpool2d_0 = self.maxpool2d_0(opt_maxpool2d_0)
        opt_batchnorm2d_1 = self.batchnorm2d_1(opt_maxpool2d_0)
        return opt_batchnorm2d_1


class Linear(nn.Cell):

    def __init__(self, matmul_0_w_shape, add_1_bias_shape):
        super(Linear, self).__init__()
        self.matmul_0_w = Parameter(Tensor(np.random.uniform(0, 1, matmul_0_w_shape).astype(np.float32)), name=None)
        self.add_1_bias = Parameter(Tensor(np.random.uniform(0, 1, add_1_bias_shape).astype(np.float32)), name=None)

    def construct(self, x):
        opt_matmul_0 = P.matmul(x, self.matmul_0_w)
        opt_add_1 = opt_matmul_0 + self.add_1_bias
        return opt_add_1


class MindSporeModel(nn.Cell):

    def __init__(self):
        super(MindSporeModel, self).__init__()
        self.transpose_0 = P.Transpose()
        self.module7_0 = Module7(maxpool2d_0_kernel_size=(2, 2),
                                 batchnorm2d_1_num_features=96,
                                 module1_0_conv2d_0_in_channels=3,
                                 module1_0_conv2d_0_out_channels=96,
                                 module1_0_conv2d_0_kernel_size=(3, 3),
                                 module1_0_conv2d_0_stride=(2, 2),
                                 module1_0_conv2d_0_padding=(0, 1, 0, 1))
        self.module7_1 = Module7(maxpool2d_0_kernel_size=(3, 3),
                                 batchnorm2d_1_num_features=256,
                                 module1_0_conv2d_0_in_channels=96,
                                 module1_0_conv2d_0_out_channels=256,
                                 module1_0_conv2d_0_kernel_size=(5, 5),
                                 module1_0_conv2d_0_stride=(1, 1),
                                 module1_0_conv2d_0_padding=(2, 2, 2, 2))
        self.module1_0 = Module1(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=384,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=(1, 1, 1, 1))
        self.module1_1 = Module1(conv2d_0_in_channels=384,
                                 conv2d_0_out_channels=384,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_stride=(1, 1),
                                 conv2d_0_padding=(1, 1, 1, 1))
        self.module7_2 = Module7(maxpool2d_0_kernel_size=(3, 3),
                                 batchnorm2d_1_num_features=256,
                                 module1_0_conv2d_0_in_channels=384,
                                 module1_0_conv2d_0_out_channels=256,
                                 module1_0_conv2d_0_kernel_size=(3, 3),
                                 module1_0_conv2d_0_stride=(1, 1),
                                 module1_0_conv2d_0_padding=(1, 1, 1, 1))
        self.transpose_17 = P.Transpose()
        self.flatten_18 = nn.Flatten()
        self.linear_0 = Linear(matmul_0_w_shape=(256, 4096), add_1_bias_shape=(4096, ))
        self.relu_21 = nn.ReLU()
        self.linear_1 = Linear(matmul_0_w_shape=(4096, 4096), add_1_bias_shape=(4096, ))
        self.relu_24 = nn.ReLU()
        self.linear_2 = Linear(matmul_0_w_shape=(4096, 10), add_1_bias_shape=(10, ))
        self.softmax_27 = nn.Softmax(axis=-1)

    def construct(self, conv2d_1_input):
        opt_transpose_0 = self.transpose_0(conv2d_1_input, (0, 3, 1, 2))
        module7_0_opt = self.module7_0(opt_transpose_0)
        module7_1_opt = self.module7_1(module7_0_opt)
        module1_0_opt = self.module1_0(module7_1_opt)
        module1_1_opt = self.module1_1(module1_0_opt)
        module7_2_opt = self.module7_2(module1_1_opt)
        opt_transpose_17 = self.transpose_17(module7_2_opt, (0, 2, 3, 1))
        opt_flatten_18 = self.flatten_18(opt_transpose_17)
        linear_0_opt = self.linear_0(opt_flatten_18)
        opt_relu_21 = self.relu_21(linear_0_opt)
        linear_1_opt = self.linear_1(opt_relu_21)
        opt_relu_24 = self.relu_24(linear_1_opt)
        linear_2_opt = self.linear_2(opt_relu_24)
        opt_softmax_27 = self.softmax_27(linear_2_opt)
        return opt_softmax_27
