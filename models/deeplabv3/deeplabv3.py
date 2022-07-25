"""
# Code Adapted from:
# https://github.com/sthalles/deeplab_v3
#
# MIT License
#
# Copyright (c) 2018 Thalles Santos Silva
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
"""
import logging
import torch
from torch import nn
from .wider_resnet import wider_resnet38_a2
from .mynn import initialize_weights, Norm2d, Upsample


class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out



class DeepWV3Plus(nn.Module):
    """
    WideResNet38 version of DeepLabV3
    mod1
    pool2
    mod2 bot_fine
    pool3
    mod3-7
    bot_aspp

    structure: [3, 3, 6, 3, 1, 1]
    channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048),
              (1024, 2048, 4096)]
    """

    def __init__(self, num_classes, trunk='WideResnet38', load_imagenet=False):

        super(DeepWV3Plus, self).__init__()
        logging.info("Trunk: %s", trunk)

        wide_resnet = wider_resnet38_a2(classes=1000, dilation=True)
        if load_imagenet:
            try:
                checkpoint = torch.load('./pretrained_models/wider_resnet38.pth.tar', map_location='cpu')
                wide_resnet.load_state_dict(checkpoint['state_dict'])
                del checkpoint
            except:
                print("Please download the ImageNet weights of WideResNet38 in our repo to ./pretrained_models/wider_resnet38.pth.tar.")
                raise RuntimeError("=====================Could not load ImageNet weights of WideResNet38 network.=======================")

        self.mod1 = wide_resnet.mod1
        self.mod2 = wide_resnet.mod2
        self.mod3 = wide_resnet.mod3
        self.mod4 = wide_resnet.mod4
        self.mod5 = wide_resnet.mod5
        self.mod6 = wide_resnet.mod6
        self.mod7 = wide_resnet.mod7
        self.pool2 = wide_resnet.pool2
        self.pool3 = wide_resnet.pool3
        del wide_resnet

        self.aspp = _AtrousSpatialPyramidPoolingModule(4096, 256,
                                                       output_stride=8)

        self.bot_fine = nn.Conv2d(128, 48, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(1280, 256, kernel_size=1, bias=False)

        self.final = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

        initialize_weights(self.final)

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    def forward(self, inp, target_size=None):
        if target_size == None:
            target_size = inp.size()[2:4]

        inp = (inp - self.mean) / self.std

        x = self.mod1(inp)
        m2 = self.mod2(self.pool2(x))
        x = self.mod3(self.pool3(m2))
        x = self.mod4(x)
        x = self.mod5(x)
        x = self.mod6(x)
        x = self.mod7(x)
        x = self.aspp(x)
        dec0_up = self.bot_aspp(x)

        dec0_fine = self.bot_fine(m2)
        dec0_up = Upsample(dec0_up, m2.size()[2:])
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)

        dec1 = self.final(dec0)
        out = Upsample(dec1, target_size)
        return out

    def load_pretrained_weights(self):
        state = torch.load('deeplabv3_wrn_cityscapes_best.pth')['state_dict']
        params = dict()
        for k, v in state.items():
            params[k.replace('module.', '')] = v
        status = self.load_state_dict(params, strict=False)
        print(status)

    def load_pretrained_weights_cv0(self):
        state = torch.load('cityscapes_cv0_wideresnet38_nosdcaug.pth')['state_dict']
        params = dict()
        for k, v in state.items():
            params[k.replace('module.', '')] = v
        status = self.load_state_dict(params, strict=False)
        print(status)


batchnorm_momentum = 0.01
class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True,
                 bias=False, dilation=1):
        super(_BNReluConv, self).__init__()
        if batch_norm:
            self.add_module('norm', nn.BatchNorm2d(num_maps_in, momentum=batchnorm_momentum))
        self.add_module('relu', nn.ReLU(inplace=True))
        padding = k // 2
        self.add_module('conv', nn.Conv2d(num_maps_in, num_maps_out,
                                          kernel_size=k, padding=padding, bias=bias, dilation=dilation))


    def forward(self, *inputs):
        inputs = torch.cat(inputs, 1)
        return super(_BNReluConv, self).forward(inputs)

class DeepWV3PlusTH(nn.Module):
    """
    WideResNet38 version of DeepLabV3
    mod1
    pool2
    mod2 bot_fine
    pool3
    mod3-7
    bot_aspp

    structure: [3, 3, 6, 3, 1, 1]
    channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048),
              (1024, 2048, 4096)]
    """

    def __init__(self, num_classes, trunk='WideResnet38', load_imagenet=False):

        super(DeepWV3PlusTH, self).__init__()
        logging.info("Trunk: %s", trunk)

        wide_resnet = wider_resnet38_a2(classes=1000, dilation=True)
        if load_imagenet:
            try:
                checkpoint = torch.load('./pretrained_models/wider_resnet38.pth.tar', map_location='cpu')
                wide_resnet.load_state_dict(checkpoint['state_dict'])
                del checkpoint
            except:
                print("Please download the ImageNet weights of WideResNet38 in our repo to ./pretrained_models/wider_resnet38.pth.tar.")
                raise RuntimeError("=====================Could not load ImageNet weights of WideResNet38 network.=======================")

        self.mod1 = wide_resnet.mod1
        self.mod2 = wide_resnet.mod2
        self.mod3 = wide_resnet.mod3
        self.mod4 = wide_resnet.mod4
        self.mod5 = wide_resnet.mod5
        self.mod6 = wide_resnet.mod6
        self.mod7 = wide_resnet.mod7
        self.pool2 = wide_resnet.pool2
        self.pool3 = wide_resnet.pool3
        del wide_resnet

        self.aspp = _AtrousSpatialPyramidPoolingModule(4096, 256,
                                                       output_stride=8)

        self.bot_fine = nn.Conv2d(128, 48, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(1280, 256, kernel_size=1, bias=False)

        self.final = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

        initialize_weights(self.final)

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

        self.logits_ood = _BNReluConv(256 + 48, 2, k=1, bias=True)

    def forward(self, inp, target_size=None):
        if target_size == None:
            target_size = inp.size()[2:4]

        inp = (inp - self.mean) / self.std

        x = self.mod1(inp)
        m2 = self.mod2(self.pool2(x))
        x = self.mod3(self.pool3(m2))
        x = self.mod4(x)
        x = self.mod5(x)
        x = self.mod6(x)
        x = self.mod7(x)
        x = self.aspp(x)
        dec0_up = self.bot_aspp(x)

        dec0_fine = self.bot_fine(m2)
        dec0_up = Upsample(dec0_up, m2.size()[2:])
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)

        dec1 = self.final(dec0)
        out = Upsample(dec1, target_size)

        out_ood = self.logits_ood(dec0)
        out_ood = Upsample(out_ood, target_size)
        return out, out_ood

    def load_pretrained_weights(self):
        state = torch.load('deeplabv3_wrn_cityscapes_best.pth')['state_dict']
        params = dict()
        for k, v in state.items():
            params[k.replace('module.', '')] = v
        status = self.load_state_dict(params, strict=False)
        print(status)

    def load_pretrained_weights_cv0(self):
        state = torch.load('cityscapes_cv0_wideresnet38_nosdcaug.pth')['state_dict']
        params = dict()
        for k, v in state.items():
            params[k.replace('module.', '')] = v
        status = self.load_state_dict(params, strict=False)
        print(status)
