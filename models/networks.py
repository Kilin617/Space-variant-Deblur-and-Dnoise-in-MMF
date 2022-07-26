import scipy.stats as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np


def weights_init(m):
    """Initial the weight"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum() # Make sure sum of values in gaussian kernel equals 1.


class gaussain_filter(nn.Module):
    """Create a Gaussian Conv2D Layer Which produce keep same dimension of input"""
    def __init__(self, in_channels, out_channels, kernel_size, sigma):
        super(gaussain_filter, self).__init__()
        gaussian_kernel = torch.from_numpy(gkern(kernel_size, sigma)).float()
        gaussian_kernel_d = gaussian_kernel.view(1, 1, kernel_size, kernel_size).repeat(in_channels, 1, 1, 1)

        # gaussian_kernel = np.tile(gaussian_kernel,(in_channels, 1, 1))
        # gaussian_kernel = np.expand_dims(gaussian_kernel, axis=1)

        self.gaussian_filter = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=int((kernel_size - 1) / 2), bias=False)

        self.gaussian_filter.weight.data = gaussian_kernel_d
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, x):
        return self.gaussian_filter(x)


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[], use_parallel=False,
             learn_residual=False, Add_gauss=False):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks_sr':
        netG = ResnetGeneratorSR(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
                                 gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual, upscale=4, gauss=Add_gauss)
    elif which_model_netG == 'resnet_9blocks_sr_gau':
        netG = ResnetGeneratorSR_Gauss(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
                                 gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual, upscale=4, gauss=Add_gauss)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, which_model_netD, n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[],
             use_parallel=True):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids, use_parallel=use_parallel)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids, use_parallel=use_parallel)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(
            self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
            n_blocks=6, gpu_ids=[], use_parallel=True, learn_residual=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        n_downsampling = 2

        model += [
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True)
        ]

        # Middle Resnet block
        for i in range(n_blocks):
            model += [
                ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)
            ]

        model += [
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(64),
            nn.ReLU(True),
        ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)
        if self.learn_residual:
            output = torch.clamp(input + output, min=-1, max=1)
        return output


# Define a resnet block
class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()

        padAndConv = {
            'reflect': [
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
            'replicate': [
                nn.ReplicationPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
            'zero': [
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias)]
        }

        try:
            blocks = padAndConv[padding_type] + [
                norm_layer(dim),
                nn.ReLU(True)
            ] + [nn.Dropout(0.5)] if use_dropout else [] + padAndConv[padding_type] + [
                norm_layer(dim)
            ]
        except:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        self.conv_block = nn.Sequential(*blocks)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetGeneratorSR(nn.Module):
    def __init__(
            self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
            n_blocks=6, gpu_ids=[], use_parallel=True, learn_residual=False, padding_type='reflect', upscale=4, gauss=False):
        assert (n_blocks >= 0)
        super(ResnetGeneratorSR, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        self.Add_gauss = gauss
        if self.Add_gauss:
            self.Conv_gauss = gaussain_filter(in_channels=input_nc, out_channels=output_nc, kernel_size=5, sigma=1)
        else:
            self.Conv_gauss = None

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        model += [
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True)
        ]

        for i in range(n_blocks):
            model += [
                ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)
            ]

        channel_mid = 100  # upscale^2*4
        model += [
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, channel_mid, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(channel_mid),
            nn.ReLU(True),
        ]
        # UpSampling
        model += [
            nn.ConvTranspose2d(channel_mid, int(channel_mid / upscale), kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=use_bias),
            norm_layer(int(channel_mid / upscale)),
            nn.ReLU(True),

            nn.ConvTranspose2d(int(channel_mid / upscale), int(channel_mid / upscale ** 2), kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=use_bias),
            norm_layer(int(channel_mid / upscale ** 2)),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(int(channel_mid / upscale ** 2), output_nc, kernel_size=7, padding=0),
            # nn.Conv2d(channel_mid, output_nc,kernel_size=7, padding=0),

            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            output = self.model(input)
        if self.learn_residual:
            # output = input + output
            input_up = F.interpolate(input, scale_factor=4, mode='bilinear')
            if self.Add_gauss:
                output1 = self.Conv_gauss(input_up)
            else:
                output1 = input
            output = torch.clamp(output1 + output, min=-1, max=1)
        return output


class ResnetGeneratorSR_Gauss(nn.Module):
    def __init__(
            self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
            n_blocks=6, gpu_ids=[], use_parallel=False, learn_residual=False, padding_type='reflect', upscale=4, gauss=False):
        assert (n_blocks >= 0)
        super(ResnetGeneratorSR_Gauss, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        self.Add_gauss = gauss
        if self.Add_gauss:
            self.Conv_gauss = gaussain_filter(in_channels=input_nc, out_channels=output_nc, kernel_size=5, sigma=1)
        else:
            self.Conv_gauss = None

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc*2, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        model += [
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True)
        ]

        for i in range(n_blocks):
            model += [
                ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)
            ]

        channel_mid = 100  # upscale^2*4
        model += [
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, channel_mid, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            norm_layer(channel_mid),
            nn.ReLU(True),
        ]
        # UpSampling
        model += [
            nn.ConvTranspose2d(channel_mid, int(channel_mid / upscale), kernel_size=3, stride=2, padding=1,
                               output_padding=1, bias=use_bias),
            norm_layer(int(channel_mid / upscale)),
            nn.ReLU(True),

            nn.ConvTranspose2d(int(channel_mid / upscale), int(channel_mid / upscale ** 2), kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=use_bias),
            norm_layer(int(channel_mid / upscale ** 2)),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(int(channel_mid / upscale ** 2), output_nc, kernel_size=7, padding=0),

            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            output = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            if self.Add_gauss:
                output1 = self.Conv_gauss(input)
                output2 = torch.cat((input, output1), dim=1)
            else:
                output2 = torch.cat((input, input), dim=1)

            output = self.model(output2)
        if self.learn_residual:
            input_up = F.interpolate(input, scale_factor=4, mode='nearest')
            output = torch.clamp(input_up + output, min=-1, max=1)
        return output



# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[],
                 use_parallel=True):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
