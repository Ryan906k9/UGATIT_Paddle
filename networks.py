import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, Linear
from paddle.fluid.layers import elementwise_div, unsqueeze, expand_as, elementwise_sub
import numpy as np


class ResnetGenerator(fluid.dygraph.Layer):
    def __init__(self, name_scope, input_nc, output_nc, ngf=64, n_blocks=4, img_size=256, light=False):
        super(ResnetGenerator, self).__init__(name_scope)
        name_scope = self.full_name()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        # DownBlock = []
        # DownBlock += [nn.ReflectionPad2d(3),
        #              nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
        #              nn.InstanceNorm2d(ngf),
        #              nn.ReLU(True)]

        self.conv1 = Conv2D(num_channels=3, num_filters=64, filter_size=7, stride=1, padding=0)
        self.conv2 = Conv2D(num_channels=64, num_filters=128, filter_size=3, stride=2, padding=0)
        self.conv3 = Conv2D(num_channels=128, num_filters=256, filter_size=3, stride=2, padding=0)

        # Down-Sampling Bottleneck
        n_downsampling = 2
        mult = 2 ** n_downsampling
        # for i in range(n_blocks):
        #    DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]
        self.resnet1 = ResnetBlock('resnet1',ngf * mult, use_bias=False)
        self.resnet2 = ResnetBlock('resnet2',ngf * mult, use_bias=False)
        self.resnet3 = ResnetBlock('resnet3',ngf * mult, use_bias=False)
        self.resnet4 = ResnetBlock('resnet4',ngf * mult, use_bias=False)

        # Class Activation Map
        # self.gap_fc = nn.Linear(ngf * mult, 1, bias=False)
        # self.gmp_fc = nn.Linear(ngf * mult, 1, bias=False)
        # self.conv1x1 = nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=1, stride=1, bias=True)
        # self.relu = nn.ReLU(True)
        self.gap_fc = Linear(ngf * mult, 1)
        self.gmp_fc = Linear(ngf * mult, 1)
        self.conv1x1 = Conv2D(num_channels=ngf * mult * 2, num_filters=ngf * mult, filter_size=1, stride=1, padding=0,
                              bias_attr=True)

        # Gamma, Beta block
        if self.light:
            # FC = [nn.Linear(ngf * mult, ngf * mult, bias=False),
            #      nn.ReLU(True),
            #      nn.Linear(ngf * mult, ngf * mult, bias=False),
            #      nn.ReLU(True)]
            self.fc1 = paddle.fluid.dygraph.Linear(input_dim=ngf * mult, output_dim=ngf * mult, act='relu')
            self.fc2 = paddle.fluid.dygraph.Linear(input_dim=ngf * mult, output_dim=ngf * mult, act='relu')
        else:
            # FC = [nn.Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias=False),
            #      nn.ReLU(True),
            #      nn.Linear(ngf * mult, ngf * mult, bias=False),
            #      nn.ReLU(True)]
            self.fc1 = paddle.fluid.dygraph.Linear(input_dim=img_size // mult * img_size // mult * ngf * mult,
                                              output_dim=ngf * mult, act='relu')
            self.fc2 = paddle.fluid.dygraph.Linear(input_dim=ngf * mult, output_dim=ngf * mult, act='relu')
        # self.gamma = nn.Linear(ngf * mult, ngf * mult, bias=False)
        self.gamma = paddle.fluid.dygraph.Linear(input_dim=ngf * mult, output_dim=ngf * mult)
        # self.beta = nn.Linear(ngf * mult, ngf * mult, bias=False)
        self.beta = paddle.fluid.dygraph.Linear(input_dim=ngf * mult, output_dim=ngf * mult)

        # Up-Sampling Bottleneck
        #for i in range(n_blocks):
        #    setattr(self, 'UpBlock1_' + str(i + 1), ResnetAdaILNBlock('ResnetAdaILNBlock',ngf * mult, use_bias=False))
        self.UpBlock1_1 = ResnetAdaILNBlock('ResnetAdaILNBlock', ngf * mult, use_bias=False)
        self.UpBlock1_2 = ResnetAdaILNBlock('ResnetAdaILNBlock', ngf * mult, use_bias=False)
        self.UpBlock1_3 = ResnetAdaILNBlock('ResnetAdaILNBlock', ngf * mult, use_bias=False)
        self.UpBlock1_4 = ResnetAdaILNBlock('ResnetAdaILNBlock', ngf * mult, use_bias=False)

        # Up-Sampling
        # UpBlock2 = []
        # for i in range(n_downsampling):
        #    mult = 2**(n_downsampling - i)
        #    UpBlock2 += [nn.Upsample(scale_factor=2, mode='nearest'),
        #                 nn.ReflectionPad2d(1),
        #                 nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
        #                 ILN(int(ngf * mult / 2)),
        #                 nn.ReLU(True)]
        mult = 2 ** (n_downsampling - 0)
        self.conv4 = Conv2D(num_channels=ngf * mult, num_filters=int(ngf * mult / 2), filter_size=3, stride=1,
                            padding=0)
        self.iln4 = ILN('ILN1',int(ngf * mult / 2))

        mult = 2 ** (n_downsampling - 1)
        self.conv5 = Conv2D(num_channels=ngf * mult, num_filters=int(ngf * mult / 2), filter_size=3, stride=1,
                            padding=0)
        self.iln5 = ILN('ILN2',int(ngf * mult / 2))

        # UpBlock2 += [nn.ReflectionPad2d(3),
        #             nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
        #             nn.Tanh()]
        self.conv6 = Conv2D(num_channels=ngf, num_filters=output_nc, filter_size=7, stride=1, padding=0)

    def forward(self, input):
        # 检查维度
        #print('input shape =',input.shape)
        x = fluid.layers.pad2d(input, paddings=[ 3, 3, 3, 3 ], mode='reflect')
        x = self.conv1(x)
        x = paddle.fluid.layers.instance_norm(x)
        x = paddle.fluid.layers.relu(x)
        # 第 1 次卷积后，w 和 h 不变，((256-7+3*2)/1)+1 = 256

        x = fluid.layers.pad2d(x, paddings=[ 1, 1, 1, 1 ], mode='reflect')
        x = self.conv2(x)
        x = paddle.fluid.layers.instance_norm(x)
        x = paddle.fluid.layers.relu(x)
        # 第 2 次卷积后，w 和 h 变为一半，((256-3+1*2)/2)+1 = 128.5 = 128

        x = fluid.layers.pad2d(x, paddings=[ 1, 1, 1, 1 ], mode='reflect')
        x = self.conv3(x)
        x = paddle.fluid.layers.instance_norm(x)
        x = paddle.fluid.layers.relu(x)
        # 第 3 次卷积后，w 和 h 变为 1/4，((128-3+1*2)/2)+1 = 64.5 = 64
        # 检查维度
        #print('after 3 conv shape =', x.shape)

        x = self.resnet1(x)
        x = self.resnet2(x)
        x = self.resnet3(x)
        x = self.resnet4(x)
        # 检查维度
        #print('after 4 resnet shape =', x.shape)

        # gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap = paddle.fluid.layers.adaptive_pool2d(x, 1, pool_type='avg')
        #print('gap shape =', gap.shape)

        # gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_logit = self.gap_fc(paddle.fluid.layers.reshape(x=gap, shape=[ x.shape[ 0 ], -1 ]))
        #print('gap shape =', gap.shape)
        #print('gap_logit shape =', gap_logit.shape)

        # gap_weight = list(self.gap_fc.parameters())[0]
        gap_weight = list(self.gap_fc.parameters())[ 0 ]
        gap_weight = paddle.fluid.layers.reshape(gap_weight, shape=[-1, gap.shape[1]])
        #print('gap_weight shape =', gap_weight.shape)

        # gap = x * gap_weight.unsqueeze(2).unsqueeze(3)
        gap_weight = paddle.fluid.layers.unsqueeze(gap_weight, axes=[ 2, 3 ])
        gap = x * gap_weight
        #print('gap_weight shape =', gap_weight.shape)
        #print('gap shape =', gap.shape)

        # gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp = paddle.fluid.layers.adaptive_pool2d(x, 1, pool_type='max')
        #print('gmp shape =', gmp.shape)

        # gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_logit = self.gmp_fc(paddle.fluid.layers.reshape(x=gmp, shape=[ x.shape[ 0 ], -1 ]))
        #print('gmp shape =', gmp.shape)
        #print('gmp_logit shape =', gmp_logit.shape)

        # gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp_weight = list(self.gmp_fc.parameters())[ 0 ]
        gmp_weight = paddle.fluid.layers.reshape(gmp_weight, shape=[ -1, gap.shape[ 1 ] ])
        #print('gmp_weight shape =', gmp_weight.shape)

        # gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)
        gmp_weight = paddle.fluid.layers.unsqueeze(gmp_weight, axes=[ 2, 3 ])
        gmp = x * gmp_weight
        #print('gmp_weight shape =', gmp_weight.shape)
        #print('gmp shape =', gmp.shape)

        # cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        cam_logit = paddle.fluid.layers.concat([ gap_logit, gmp_logit ], 1)
        #print('cam_logit shape =', cam_logit.shape)
        # x = torch.cat([gap, gmp], 1)
        x = paddle.fluid.layers.concat([ gap, gmp ], 1)
        #print('x =', x.shape)
        # x = self.relu(self.conv1x1(x))
        x = paddle.fluid.layers.relu(self.conv1x1(x))
        # 检查维度
        #print('after cam shape =', x.shape)
        # heatmap = torch.sum(x, dim=1, keepdim=True)
        heatmap = paddle.fluid.layers.reduce_sum(x, dim=1, keep_dim=True)
        #print('heatmap shape =', heatmap.shape)

        if self.light:
            # x_ = torch.nn.functional.adaptive_avg_pool2d(x, 1)
            x_ = paddle.fluid.layers.adaptive_pool2d(x, 1, pool_type='avg')
            # x_ = self.FC(x_.view(x_.shape[0], -1))
            x_ = paddle.fluid.layers.reshape(x=x_, shape=[ x_.shape[ 0 ], -1 ])
            x_ = self.fc1(x_)
            x_ = self.fc2(x_)

        else:
            # x_ = self.FC(x.view(x.shape[0], -1))
            x_ = paddle.fluid.layers.reshape(x=x, shape=[ x.shape[ 0 ], -1 ])
            x_ = self.fc1(x_)
            x_ = self.fc2(x_)
        # gamma, beta = self.gamma(x_), self.beta(x_)

        gamma, beta = self.gamma(x_), self.beta(x_)

        # 这两句未改
        #for i in range(self.n_blocks):
        #    x = getattr(self, 'UpBlock1_' + str(i + 1))(x, gamma, beta)
        x = self.UpBlock1_1(x, gamma, beta)
        x = self.UpBlock1_2(x, gamma, beta)
        x = self.UpBlock1_3(x, gamma, beta)
        x = self.UpBlock1_4(x, gamma, beta)

        # out = self.UpBlock2(x)
        x = paddle.fluid.layers.resize_nearest(x, scale=2)
        x = fluid.layers.pad2d(x, paddings=[ 1, 1, 1, 1 ], mode='reflect')
        x = self.conv4(x)
        x = self.iln4(x)
        x = paddle.fluid.layers.relu(x)

        x = paddle.fluid.layers.resize_nearest(x, scale=2)
        x = fluid.layers.pad2d(x, paddings=[ 1, 1, 1, 1 ], mode='reflect')
        x = self.conv5(x)
        x = self.iln5(x)
        x = paddle.fluid.layers.relu(x)

        x = fluid.layers.pad2d(x, paddings=[ 3, 3, 3, 3 ], mode='reflect')
        x = self.conv6(x)
        out = paddle.fluid.layers.tanh(x)

        return out, cam_logit, heatmap


class ResnetBlock(fluid.dygraph.Layer):
    def __init__(self, name_scope, dim, use_bias):
        super(ResnetBlock, self).__init__(name_scope)
        name_scope = self.full_name()

        # conv_block = []
        # conv_block += [nn.ReflectionPad2d(1),
        #               nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
        #               nn.InstanceNorm2d(dim),
        #               nn.ReLU(True)]

        # conv_block += [nn.ReflectionPad2d(1),
        #               nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
        #               nn.InstanceNorm2d(dim)]

        # self.conv_block = nn.Sequential(*conv_block)

        self.conv1 = Conv2D(num_channels=dim, num_filters=dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.conv2 = Conv2D(num_channels=dim, num_filters=dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)

    def forward(self, x):
        # out = x + self.conv_block(x)
        y = fluid.layers.pad2d(x, paddings=[ 1, 1, 1, 1 ], mode='reflect')
        y = self.conv1(y)
        y = paddle.fluid.layers.instance_norm(y)
        y = paddle.fluid.layers.relu(y)

        y = fluid.layers.pad2d(y, paddings=[ 1, 1, 1, 1 ], mode='reflect')
        y = self.conv2(y)
        y = paddle.fluid.layers.instance_norm(y)

        out = x + y

        return out


class ResnetAdaILNBlock(fluid.dygraph.Layer):
    def __init__(self,name_scope, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__(name_scope)
        name_scope = self.full_name()

        # self.pad1 = nn.ReflectionPad2d(1)
        # self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.conv1 = Conv2D(num_channels=dim, num_filters=dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.norm1 = adaILN('adaILN1',dim)
        # self.relu1 = nn.ReLU(True)

        # self.pad2 = nn.ReflectionPad2d(1)
        # self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.conv2 = Conv2D(num_channels=dim, num_filters=dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.norm2 = adaILN('adaILN2',dim)

    def forward(self, x, gamma, beta):
        # out = self.pad1(x)
        out = fluid.layers.pad2d(x, paddings=[ 1, 1, 1, 1 ], mode='reflect')
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        # out = self.relu1(out)
        out = paddle.fluid.layers.relu(out)

        # out = self.pad2(out)
        out = fluid.layers.pad2d(out, paddings=[ 1, 1, 1, 1 ], mode='reflect')
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x


class adaILN(fluid.dygraph.Layer):
    def __init__(self, name_scope,num_features, eps=1e-5):
        super(adaILN, self).__init__(name_scope)
        name_scope = self.full_name()

        self.eps = eps
        # self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        x = np.ones([ 1, num_features, 1, 1 ], np.float32)
        self.rho = fluid.dygraph.to_variable(x, zero_copy=None)
        # self.rho.data.fill_(0.9)
        self.rho = self.rho * 0.9

    def forward(self, input, gamma, beta):
        # 检查输入的情况
        #print('input.shape =',input.shape)
        #print(type(input))
        #print(gamma.shape)
        #print(beta.shape)
        # in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        in_mean = paddle.fluid.layers.reduce_mean(input, dim=[ 2, 3 ], keep_dim=True)
        in_mean_sub = input - in_mean
        in_var = paddle.fluid.layers.reduce_mean(paddle.fluid.layers.square(in_mean_sub), dim=[ 2, 3 ], keep_dim=True)
        out_in = in_mean_sub / paddle.fluid.layers.sqrt(in_var + self.eps)
        # ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        ln_mean = paddle.fluid.layers.reduce_mean(input, dim=[ 1, 2, 3 ], keep_dim=True)
        ln_mean_sub = input - ln_mean
        ln_var = paddle.fluid.layers.reduce_mean(paddle.fluid.layers.square(ln_mean_sub), dim=[ 2, 3 ], keep_dim=True)
        out_ln = ln_mean_sub / paddle.fluid.layers.sqrt(ln_var + self.eps)
        # 检查shape
        #print('input = ', input.shape) #[256, 256, 64, 64]
        #print('rho = ', self.rho.shape) #[1, 256, 1, 1]
        #print('out_in = ', out_in.shape) #[256, 256, 64, 64]
        #print('out_ln = ', out_ln.shape) #[256, 256, 64, 64]

        # out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        #out = out_in.numpy() * paddle.fluid.layers.expand_as(self.rho, out_in).numpy() + \
        #      out_ln.numpy() * (1 - paddle.fluid.layers.expand_as(self.rho, out_ln).numpy())
        #out = fluid.dygraph.to_variable(out, zero_copy=None)
        out = paddle.fluid.layers.expand(self.rho,expand_times=[input.shape[0], 1, 1, 1]) * out_in \
              + (1 - paddle.fluid.layers.expand(self.rho,expand_times=[input.shape[0], 1, 1, 1])) * out_ln
        #print('OK1')
        #print('out = ',out.shape)

        # out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
        #out = out.numpy() * expand_as(unsqueeze(gamma, axes=[ 2, 3 ]), out).numpy() + \
        #      expand_as(unsqueeze(beta, axes=[ 2, 3 ]), out).numpy()
        #out = fluid.dygraph.to_variable(out, zero_copy=None)
        out = out * unsqueeze(gamma, axes=[ 2, 3 ]) + unsqueeze(beta, axes=[ 2, 3 ])

        return out


class ILN(fluid.dygraph.Layer):
    def __init__(self,name_scope, num_features, eps=1e-5):
        super(ILN, self).__init__(name_scope)
        name_scope = self.full_name()

        self.eps = eps
        # self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        x = np.ones([ 1, num_features, 1, 1 ], np.float32)
        self.rho = fluid.dygraph.to_variable(x, zero_copy=None)
        # self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = fluid.dygraph.to_variable(x, zero_copy=None)
        # self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = fluid.dygraph.to_variable(x, zero_copy=None)
        # self.rho.data.fill_(0.0)
        self.rho = self.rho * 0.0
        # self.gamma.data.fill_(1.0)
        self.gamma = self.gamma * 1.0
        # self.beta.data.fill_(0.0)
        self.beta = self.beta * 0.0

    def forward(self, input):
        # 检查输入的情况
        #print('input.shape =',input.shape)
        #print(type(input))
        
        # in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        in_mean = paddle.fluid.layers.reduce_mean(input, dim=[ 2, 3 ], keep_dim=True)
        in_mean_sub = input - in_mean
        in_var = paddle.fluid.layers.reduce_mean(paddle.fluid.layers.square(in_mean_sub), dim=[ 2, 3 ], keep_dim=True)

        # out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        out_in = in_mean_sub / paddle.fluid.layers.sqrt(in_var + self.eps)

        # ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        ln_mean = paddle.fluid.layers.reduce_mean(input, dim=[ 1, 2, 3 ], keep_dim=True)
        ln_mean_sub = input - ln_mean
        ln_var = paddle.fluid.layers.reduce_mean(paddle.fluid.layers.square(ln_mean_sub), dim=[ 2, 3 ], keep_dim=True)

        # out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out_ln = ln_mean_sub / paddle.fluid.layers.sqrt(ln_var + self.eps)

        # 检查维度
        #print('rho =',self.rho.shape)
        #print('gamma =',self.gamma.shape)
        #print('beta =',self.beta.shape)
        #print('out_in =',out_in.shape)
        #print('out_ln =',out_ln.shape)
        # out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = paddle.fluid.layers.expand_as(self.rho, out_in) * out_in \
              + (1 - paddle.fluid.layers.expand_as(self.rho, out_ln)) * out_ln

        # out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)
        out = out * paddle.fluid.layers.expand_as(self.gamma, out) \
              + paddle.fluid.layers.expand_as(self.beta, out)
        #print('OK_ILN')
        return out


class Spectralnorm(fluid.dygraph.Layer):

    def __init__(self,
                 layer,
                 dim=0,
                 power_iters=1,
                 eps=1e-12,
                 dtype='float32'):
        super(Spectralnorm, self).__init__()
        self.spectral_norm = fluid.dygraph.SpectralNorm(layer.weight.shape, dim, power_iters, eps, dtype)
        self.dim = dim
        self.power_iters = power_iters
        self.eps = eps
        self.layer = layer
        weight = layer._parameters[ 'weight' ]
        del layer._parameters[ 'weight' ]
        self.weight_orig = self.create_parameter(weight.shape, dtype=weight.dtype)
        self.weight_orig.set_value(weight)

    def forward(self, x):
        weight = self.spectral_norm(self.weight_orig)
        self.layer.weight = weight
        out = self.layer(x)
        return out


class Discriminator_G(fluid.dygraph.Layer):
    def __init__(self,name_scope, input_nc, ndf=64, n_layers=7):
        super(Discriminator_G, self).__init__(name_scope)
        name_scope = self.full_name()

        # model = [nn.ReflectionPad2d(1),
        #         nn.utils.spectral_norm(
        #         nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
        #         nn.LeakyReLU(0.2, True)]
        self.conv1 = Conv2D(num_channels=input_nc, num_filters=ndf, filter_size=4, stride=2, padding=0, bias_attr=True)
        self.spectral_norm_c1 = Spectralnorm(self.conv1,
                                           dim=0,
                                           power_iters=1,
                                           eps=1e-12,
                                           dtype='float32')

        # for i in range(1, n_layers - 2):
        #    mult = 2 ** (i - 1)
        #    model += [nn.ReflectionPad2d(1),
        #              nn.utils.spectral_norm(
        #              nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
        #              nn.LeakyReLU(0.2, True)]

        mult = 2 ** (1 - 1)
        self.conv2 = Conv2D(num_channels=ndf * mult, num_filters=ndf * mult * 2, filter_size=4, stride=2, padding=0,
                            bias_attr=True)
        self.spectral_norm_c2 = Spectralnorm(self.conv2,
                                           dim=0,
                                           power_iters=1,
                                           eps=1e-12,
                                           dtype='float32')

        mult = 2 ** (2 - 1)
        self.conv3 = Conv2D(num_channels=ndf * mult, num_filters=ndf * mult * 2, filter_size=4, stride=2, padding=0,
                            bias_attr=True)
        self.spectral_norm_c3 = Spectralnorm(self.conv3,
                                           dim=0,
                                           power_iters=1,
                                           eps=1e-12,
                                           dtype='float32')

        mult = 2 ** (3 - 1)
        self.conv4 = Conv2D(num_channels=ndf * mult, num_filters=ndf * mult * 2, filter_size=4, stride=2, padding=0,
                            bias_attr=True)
        self.spectral_norm_c4 = Spectralnorm(self.conv4,
                                           dim=0,
                                           power_iters=1,
                                           eps=1e-12,
                                           dtype='float32')

        mult = 2 ** (4 - 1)
        self.conv5 = Conv2D(num_channels=ndf * mult, num_filters=ndf * mult * 2, filter_size=4, stride=2, padding=0,
                            bias_attr=True)
        self.spectral_norm_c5 = Spectralnorm(self.conv5,
                                           dim=0,
                                           power_iters=1,
                                           eps=1e-12,
                                           dtype='float32')

        mult = 2 ** (n_layers - 2 - 1)
        # model += [nn.ReflectionPad2d(1),
        #          nn.utils.spectral_norm(
        #          nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
        #          nn.LeakyReLU(0.2, True)]
        self.conv6 = Conv2D(num_channels=ndf * mult, num_filters=ndf * mult * 2, filter_size=4, stride=1, padding=0,
                            bias_attr=True)
        self.spectral_norm_c6 = Spectralnorm(self.conv6,
                                           dim=0,
                                           power_iters=1,
                                           eps=1e-12,
                                           dtype='float32')







        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        # self.gap_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.gap_fc = Linear(input_dim=ndf * mult, output_dim=1, param_attr=None, bias_attr=False,
                                                  act=None, dtype='float32')
        self.spectral_norm5 = Spectralnorm(self.gap_fc,
                                           dim=0,
                                           power_iters=1,
                                           eps=1e-12,
                                           dtype='float32')

        # self.gmp_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.gmp_fc = Linear(input_dim=ndf * mult, output_dim=1, param_attr=None, bias_attr=False,
                                                  act=None, dtype='float32')
        self.spectral_norm6 = Spectralnorm(self.gmp_fc,
                                           dim=0,
                                           power_iters=1,
                                           eps=1e-12,
                                           dtype='float32')

        # self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True)
        self.conv1x1 = Conv2D(num_channels=ndf * mult * 2, num_filters=ndf * mult, filter_size=1, stride=1, padding=0,
                              bias_attr=True)

        # self.leaky_relu = nn.LeakyReLU(0.2, True)

        # self.pad = nn.ReflectionPad2d(1)

        # self.conv = nn.utils.spectral_norm(
        #    nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))
        self.conv = Conv2D(num_channels=ndf * mult, num_filters=1, filter_size=4, stride=1, padding=0, bias_attr=False)
        self.spectral_norm7 = Spectralnorm(self.conv,
                                           dim=0,
                                           power_iters=1,
                                           eps=1e-12,
                                           dtype='float32')

        # self.model = nn.Sequential(*model)

    def forward(self, input):
        # x = self.model(input)
        x = fluid.layers.pad2d(input, paddings=[ 1, 1, 1, 1 ], mode='reflect')
        x = self.spectral_norm_c1(x)
        x = fluid.layers.leaky_relu(x, alpha=0.2, name=None)

        x = fluid.layers.pad2d(x, paddings=[ 1, 1, 1, 1 ], mode='reflect')
        x = self.spectral_norm_c2(x)
        x = fluid.layers.leaky_relu(x, alpha=0.2, name=None)

        x = fluid.layers.pad2d(x, paddings=[ 1, 1, 1, 1 ], mode='reflect')
        x = self.spectral_norm_c3(x)
        x = fluid.layers.leaky_relu(x, alpha=0.2, name=None)

        x = fluid.layers.pad2d(x, paddings=[ 1, 1, 1, 1 ], mode='reflect')
        x = self.spectral_norm_c4(x)
        x = fluid.layers.leaky_relu(x, alpha=0.2, name=None)

        x = fluid.layers.pad2d(x, paddings=[ 1, 1, 1, 1 ], mode='reflect')
        x = self.spectral_norm_c5(x)
        x = fluid.layers.leaky_relu(x, alpha=0.2, name=None)

        x = fluid.layers.pad2d(x, paddings=[ 1, 1, 1, 1 ], mode='reflect')
        x = self.spectral_norm_c6(x)
        x = fluid.layers.leaky_relu(x, alpha=0.2, name=None)





        # gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap = paddle.fluid.layers.adaptive_pool2d(x, 1, pool_type='avg')
        #print('D_G_gap',gap.shape)

        # gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_logit = self.spectral_norm5(paddle.fluid.layers.reshape(x=gap, shape=[ x.shape[ 0 ], -1 ]))
        #print('gap_logit',gap_logit.shape)

        # gap_weight = list(self.gap_fc.parameters())[0]
        gap_weight = list(self.spectral_norm5.parameters())[0]
        gap_weight = paddle.fluid.layers.reshape(gap_weight, shape=[-1, gap.shape[1]])
        #print('gap_weight',gap_weight.shape)

        # gap = x * gap_weight.unsqueeze(2).unsqueeze(3)
        # 是否需要重新赋值给 gap_weight?
        # gap_weight = paddle.fluid.layers.unsqueeze(gap_weight, axes=[2,3])
        # gap = x * gap_weight
        gap = x * paddle.fluid.layers.unsqueeze(gap_weight, axes=[ 2, 3 ])
        #print('D_G_gap', gap.shape)

        # gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp = paddle.fluid.layers.adaptive_pool2d(x, 1, pool_type='max')

        # gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        #gmp_logit = self.gmp_fc(paddle.fluid.layers.reshape(x=gmp, shape=[ x.shape[ 0 ], -1 ]))
        gmp_logit = self.spectral_norm6(paddle.fluid.layers.reshape(x=gmp, shape=[ x.shape[ 0 ], -1 ]))

        # gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp_weight = list(self.spectral_norm6.parameters())[ 0 ]
        gmp_weight = paddle.fluid.layers.reshape(gap_weight, shape=[-1, gmp.shape[1]])
        #print('gmp_weight',gap_weight.shape)

        # gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)
        gmp = x * paddle.fluid.layers.unsqueeze(gmp_weight, axes=[ 2, 3 ])

        # cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        cam_logit = paddle.fluid.layers.concat([ gap_logit, gmp_logit ], 1)

        # x = torch.cat([gap, gmp], 1)
        x = paddle.fluid.layers.concat([ gap, gmp ], 1)

        # x = self.leaky_relu(self.conv1x1(x))
        x = self.conv1x1(x)
        x = paddle.fluid.layers.leaky_relu(x, alpha=0.2, name=None)

        # heatmap = torch.sum(x, dim=1, keepdim=True)
        heatmap = paddle.fluid.layers.reduce_sum(x, dim=1, keep_dim=True)

        # x = self.pad(x)
        x = fluid.layers.pad2d(x, paddings=[ 1, 1, 1, 1 ], mode='reflect')

        # out = self.conv(x)
        out = self.spectral_norm7(x)

        return out, cam_logit, heatmap




class Discriminator_L(fluid.dygraph.Layer):
    def __init__(self,name_scope, input_nc, ndf=64, n_layers=5):
        super(Discriminator_L, self).__init__(name_scope)
        name_scope = self.full_name()

        # model = [nn.ReflectionPad2d(1),
        #         nn.utils.spectral_norm(
        #         nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
        #         nn.LeakyReLU(0.2, True)]
        self.conv1 = Conv2D(num_channels=input_nc, num_filters=ndf, filter_size=4, stride=2, padding=0, bias_attr=True)
        self.spectral_norm1 = Spectralnorm(self.conv1,
                                           dim=0,
                                           power_iters=1,
                                           eps=1e-12,
                                           dtype='float32')

        # for i in range(1, n_layers - 2):
        #    mult = 2 ** (i - 1)
        #    model += [nn.ReflectionPad2d(1),
        #              nn.utils.spectral_norm(
        #              nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
        #              nn.LeakyReLU(0.2, True)]

        mult = 2 ** (1 - 1)
        self.conv2 = Conv2D(num_channels=ndf * mult, num_filters=ndf * mult * 2, filter_size=4, stride=2, padding=0,
                            bias_attr=True)
        self.spectral_norm2 = Spectralnorm(self.conv2,
                                           dim=0,
                                           power_iters=1,
                                           eps=1e-12,
                                           dtype='float32')

        mult = 2 ** (2 - 1)
        self.conv3 = Conv2D(num_channels=ndf * mult, num_filters=ndf * mult * 2, filter_size=4, stride=2, padding=0,
                            bias_attr=True)
        self.spectral_norm3 = Spectralnorm(self.conv3,
                                           dim=0,
                                           power_iters=1,
                                           eps=1e-12,
                                           dtype='float32')

        mult = 2 ** (n_layers - 2 - 1)
        # model += [nn.ReflectionPad2d(1),
        #          nn.utils.spectral_norm(
        #          nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
        #          nn.LeakyReLU(0.2, True)]
        self.conv4 = Conv2D(num_channels=ndf * mult, num_filters=ndf * mult * 2, filter_size=4, stride=1, padding=0,
                            bias_attr=True)
        self.spectral_norm4 = Spectralnorm(self.conv4,
                                           dim=0,
                                           power_iters=1,
                                           eps=1e-12,
                                           dtype='float32')

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        # self.gap_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.gap_fc = Linear(input_dim=ndf * mult, output_dim=1, param_attr=None, bias_attr=None,
                                                  act=None, dtype='float32')
        self.spectral_norm5 = Spectralnorm(self.gap_fc,
                                           dim=0,
                                           power_iters=1,
                                           eps=1e-12,
                                           dtype='float32')

        # self.gmp_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.gmp_fc = Linear(input_dim=ndf * mult, output_dim=1, param_attr=None, bias_attr=None,
                                                  act=None, dtype='float32')
        self.spectral_norm6 = Spectralnorm(self.gmp_fc,
                                           dim=0,
                                           power_iters=1,
                                           eps=1e-12,
                                           dtype='float32')

        # self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True)
        self.conv1x1 = Conv2D(num_channels=ndf * mult * 2, num_filters=ndf * mult, filter_size=1, stride=1, padding=0,
                              bias_attr=True)

        # self.leaky_relu = nn.LeakyReLU(0.2, True)

        # self.pad = nn.ReflectionPad2d(1)

        # self.conv = nn.utils.spectral_norm(
        #    nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))
        self.conv5 = Conv2D(num_channels=ndf * mult, num_filters=1, filter_size=4, stride=1, padding=0, bias_attr=False)
        self.spectral_norm7 = Spectralnorm(self.conv5,
                                           dim=0,
                                           power_iters=1,
                                           eps=1e-12,
                                           dtype='float32')

        # self.model = nn.Sequential(*model)

    def forward(self, input):
        # x = self.model(input)
        x = fluid.layers.pad2d(input, paddings=[ 1, 1, 1, 1 ], mode='reflect')
        x = self.spectral_norm1(x)
        x = fluid.layers.leaky_relu(x, alpha=0.2, name=None)

        x = fluid.layers.pad2d(x, paddings=[ 1, 1, 1, 1 ], mode='reflect')
        x = self.spectral_norm2(x)
        x = fluid.layers.leaky_relu(x, alpha=0.2, name=None)

        x = fluid.layers.pad2d(x, paddings=[ 1, 1, 1, 1 ], mode='reflect')
        x = self.spectral_norm3(x)
        x = fluid.layers.leaky_relu(x, alpha=0.2, name=None)

        x = fluid.layers.pad2d(x, paddings=[ 1, 1, 1, 1 ], mode='reflect')
        x = self.spectral_norm4(x)
        x = fluid.layers.leaky_relu(x, alpha=0.2, name=None)



        # gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap = paddle.fluid.layers.adaptive_pool2d(x, 1, pool_type='avg')
        #print('D_L_gap_1',gap.shape)

        # gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        # 是否需要重新赋值给 gap？
        # gap = paddle.fluid.layers.reshape(x=gap, shape=[x.shape[0], -1])
        # gap_logit = self.gap_fc(gap)
        #gap_logit = self.gap_fc(paddle.fluid.layers.reshape(x=gap, shape=[ x.shape[ 0 ], -1 ]))
        gap_logit = self.spectral_norm5(paddle.fluid.layers.reshape(x=gap, shape=[ x.shape[ 0 ], -1 ]))

        # gap_weight = list(self.gap_fc.parameters())[0]
        gap_weight = list(self.spectral_norm5.parameters())[ 0 ]
        gap_weight = paddle.fluid.layers.reshape(gap_weight, shape=[ -1, gap.shape[ 1 ] ])

        # gap = x * gap_weight.unsqueeze(2).unsqueeze(3)
        # 是否需要重新赋值给 gap_weight?
        # gap_weight = paddle.fluid.layers.unsqueeze(gap_weight, axes=[2,3])
        # gap = x * gap_weight
        gap = x * paddle.fluid.layers.unsqueeze(gap_weight, axes=[ 2, 3 ])
        #print('D_L_gap_2', gap.shape)

        # gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp = paddle.fluid.layers.adaptive_pool2d(x, 1, pool_type='max')
        #print('D_L_gmp_1', gmp.shape)
        # gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        #gmp_logit = self.gmp_fc(paddle.fluid.layers.reshape(x=gmp, shape=[ x.shape[ 0 ], -1 ]))
        gmp_logit = self.spectral_norm6(paddle.fluid.layers.reshape(x=gmp, shape=[ x.shape[ 0 ], -1 ]))

        # gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp_weight = list(self.spectral_norm6.parameters())[ 0 ]
        gmp_weight = paddle.fluid.layers.reshape(gmp_weight, shape=[ -1, gmp.shape[ 1 ] ])

        # gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)
        gmp = x * paddle.fluid.layers.unsqueeze(gmp_weight, axes=[ 2, 3 ])
        #print('D_L_gmp_2', gmp.shape)
        # cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        cam_logit = paddle.fluid.layers.concat([ gap_logit, gmp_logit ], 1)

        # x = torch.cat([gap, gmp], 1)
        x = paddle.fluid.layers.concat([ gap, gmp ], 1)

        # x = self.leaky_relu(self.conv1x1(x))
        x = self.conv1x1(x)
        x = paddle.fluid.layers.leaky_relu(x, alpha=0.2, name=None)

        # heatmap = torch.sum(x, dim=1, keepdim=True)
        heatmap = paddle.fluid.layers.reduce_sum(x, dim=1, keep_dim=True)

        # x = self.pad(x)
        x = fluid.layers.pad2d(x, paddings=[ 1, 1, 1, 1 ], mode='reflect')

        # out = self.conv(x)
        out = self.spectral_norm7(x)

        return out, cam_logit, heatmap

# 定义上采样模块
class Upsample(fluid.dygraph.Layer):
    def __init__(self, scale=2):
        super(Upsample, self).__init__()
        self.scale = scale

    def forward(self, inputs):
        # get dynamic upsample output shape
        shape_nchw = fluid.layers.shape(inputs)
        shape_hw = fluid.layers.slice(shape_nchw, axes=[0], starts=[2], ends=[4])
        shape_hw.stop_gradient = True
        in_shape = fluid.layers.cast(shape_hw, dtype='int32')
        out_shape = in_shape * self.scale
        out_shape.stop_gradient = True

        # reisze by actual_shape
        out = fluid.layers.resize_nearest(
            input=inputs, scale=self.scale, actual_shape=out_shape)
        return out



class RhoClipper(object):

    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):

        # if hasattr(module, 'rho'):
        #    w = module.rho.data
        #    w = w.clamp(self.clip_min, self.clip_max)
        #    module.rho.data = w
        for name, param in module.named_parameters():
            if 'rho' in name:
                param.set_value(fluid.layers.clip(param, self.clip_min, self.clip_max))

class BCEWithLogitsLoss():
    def __init__(self, weight=None, reduction='mean'):
        self.weight = weight
        self.reduction = 'mean'

    def __call__(self, x, label):
        out = paddle.fluid.layers.sigmoid_cross_entropy_with_logits(x, label)
        if self.reduction == 'sum':
            return fluid.layers.reduce_sum(out)
        elif self.reduction == 'mean':
            return fluid.layers.reduce_mean(out)
        else:
            return out
