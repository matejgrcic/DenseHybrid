from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


batchnorm_momentum = 0.01

DENSE_NET_169_BLOCK_CONFIG = (6, 12, 32, 32)
DENSE_NET_121_BLOCK_CONFIG = (6, 12, 24, 16)


checkpoint = lambda func, *inputs: cp.checkpoint(func, *inputs, preserve_rng_state=True)

def _checkpoint_unit(bn1, relu1, conv1, bn2, relu2, conv2):
    def func(*x):
        x = torch.cat(x, 1)
        x = conv1(relu1(bn1(x)))
        return conv2(relu2(bn2(x)))
    return func

def _checkpoint_transition(norm, relu, conv, pool):
    def func(*x):
        x = torch.cat(x, 1)
        x = norm(x)
        x = conv(relu(x))
        return pool(x)
    return func

def _checkpoint_bnreluconv(bn, relu, conv):
    def func(*x):
        x = torch.cat(x, 1)
        x = bn(x)
        return conv(relu(x))
    return func

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, dilation, efficient=False, checkpointing=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features, momentum=batchnorm_momentum))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate,
                                                momentum=batchnorm_momentum))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1 * dilation,
                                           bias=False, dilation=dilation))

        self.checkpointing = checkpointing
        if checkpointing:
            self.conv_func = _checkpoint_unit(self.norm1, self.relu1, self.conv1, self.norm2, self.relu2, self.conv2)

    def forward(self, *inputs):
        if self.checkpointing:
            if self.training:
                return checkpoint(self.conv_func, *inputs)
            else:
                return self.conv_func(*inputs)
        else:
            inputs = torch.cat(inputs, 1)
            return super(_DenseLayer, self).forward(inputs)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 dilation=1, efficient=True, checkpointing=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate,
                                bn_size, dilation, efficient=efficient, checkpointing=checkpointing)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, x):
        x = [x]
        for layer in self.children():
            x.append(layer(*x))
        return torch.cat(x, 1)



class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, checkpointing=False):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features, momentum=batchnorm_momentum))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=3, stride=2,
                                             padding=1, ceil_mode=False, count_include_pad=False))

        self.checkpointing = checkpointing
        if checkpointing:
            self.conv_func = _checkpoint_transition(self.norm, self.relu, self.conv, self.pool)


    def forward(self, *inputs):
        if self.checkpointing:
            if self.training:
                return checkpoint(self.conv_func, *inputs)
            else:
                return self.conv_func(*inputs)
        else:
            inputs = torch.cat(inputs, 1)
            return super(_Transition, self).forward(inputs)

class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True,
                 bias=False, dilation=1, checkpointing=False):
        super(_BNReluConv, self).__init__()
        if batch_norm:
            self.add_module('norm', nn.BatchNorm2d(num_maps_in, momentum=batchnorm_momentum))
        self.add_module('relu', nn.ReLU(inplace=True))
        padding = k // 2
        self.add_module('conv', nn.Conv2d(num_maps_in, num_maps_out,
                                          kernel_size=k, padding=padding, bias=bias, dilation=dilation))

        self.checkpointing = checkpointing
        if checkpointing:
            self.conv_func = _checkpoint_bnreluconv(self.norm, self.relu, self.conv)

    def forward(self, *inputs):
        if self.checkpointing:
            if self.training:
                return checkpoint(self.conv_func, *inputs)
            else:
                return self.conv_func(*inputs)
        else:
            inputs = torch.cat(inputs, 1)
            return super(_BNReluConv, self).forward(inputs)

class _BNReluConvReAct(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True,
                 bias=False, dilation=1, store_samples=False):
        super(_BNReluConvReAct, self).__init__()
        if batch_norm:
            self.add_module('norm', nn.BatchNorm2d(num_maps_in, momentum=batchnorm_momentum))
        self.add_module('relu', nn.ReLU(inplace=True))
        padding = k // 2
        self.add_module('conv', nn.Conv2d(num_maps_in, num_maps_out,
                                          kernel_size=k, padding=padding, bias=bias, dilation=dilation))
        self.sample_buffer = None
        self.step = 0
        self.store_samples = store_samples

    def forward(self, x):
        for i, t in enumerate(self.children()):
            x = t(x)
            if i == 1:
                if self.store_samples and self.step < 10:
                    N, C, H, W = x.shape
                    self.sample_buffer = torch.zeros(10*N, C, H, W) if isinstance(self.sample_buffer, type(None)) else self.sample_buffer
                    self.sample_buffer[self.step*N: self.step*N+N] = x.cpu()
                    self.step += 1
        return x

    def forward_with_thres(self, x, tr):
        for i, t in enumerate(self.children()):
            x = t(x)
            if i == 1:
                mask = x < tr.to(x)
                x = mask * x
        return x


class _BNConvReLU(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, batch_norm=True,
                 bias=False, dilation=1, checkpointing=False):
        super(_BNConvReLU, self).__init__()
        if batch_norm:
            self.add_module('norm', nn.BatchNorm2d(num_maps_in, momentum=batchnorm_momentum))
        padding = k // 2
        self.add_module('conv', nn.Conv2d(num_maps_in, num_maps_out,
                                          kernel_size=k, padding=padding, bias=bias, dilation=dilation))
        self.add_module('relu', nn.ReLU(inplace=False))
        self.checkpointing = checkpointing
        if checkpointing:
            self.conv_func = _checkpoint_bnreluconv(self.norm, self.conv, self.relu)

    def forward(self, *inputs):
        if self.checkpointing:
            if self.training:
                return checkpoint(self.conv_func, *inputs)
            else:
                return self.conv_func(*inputs)
        else:
            inputs = torch.cat(inputs, 1)
            return super(_BNConvReLU, self).forward(inputs)

class Args:
    def __init__(self):
        self.last_block_pooling = 0

class SpatialPyramidPooling(nn.Module):
    def __init__(self, num_maps_in, num_levels, bt_size=512, level_size=128, out_size=256,
                 grids=[6, 3, 2, 1], square_grid=False, checkpointing=False):
        super(SpatialPyramidPooling, self).__init__()
        self.grids = grids
        self.square_grid = square_grid
        self.spp = nn.Sequential()
        self.spp.add_module('spp_bn', _BNReluConv(num_maps_in, bt_size, k=1, checkpointing=checkpointing))
        num_features = bt_size
        final_size = num_features
        for i in range(num_levels):
            final_size += level_size
            self.spp.add_module('spp' + str(i), _BNReluConv(num_features, level_size, k=1, checkpointing=checkpointing))
        self.spp.add_module('spp_fuse', _BNReluConv(final_size, out_size, k=1, checkpointing=checkpointing))

    def forward(self, x):
        levels = []
        target_size = x.size()[2:4]

        ar = target_size[1] / target_size[0]

        x = self.spp[0].forward(x)
        levels.append(x)
        num = len(self.spp) - 1

        for i in range(1, num):
            if not self.square_grid:
                grid_size = (self.grids[i - 1], max(1, round(ar * self.grids[i - 1])))
                x_pooled = F.adaptive_avg_pool2d(x, grid_size)
            else:
                x_pooled = F.adaptive_avg_pool2d(x, self.grids[i - 1])
            level = self.spp[i].forward(x_pooled)

            level = F.upsample(level, target_size, mode='bilinear')
            levels.append(level)
        x = torch.cat(levels, 1)
        return self.spp[-1].forward(x)


class _Upsample(nn.Module):
    def __init__(self, num_maps_in, skip_maps_in, num_maps_out, num_classes=0, checkpointing=False):
        super(_Upsample, self).__init__()
        print('Upsample layer: in =', num_maps_in, ', skip =', skip_maps_in,
              ' out =', num_maps_out)
        self.bottleneck = _BNReluConv(skip_maps_in, num_maps_in, k=1, checkpointing=checkpointing)
        self.blend_conv = _BNReluConv(num_maps_in, num_maps_out, k=3, checkpointing=checkpointing)
        self.logits_aux = _BNReluConv(num_maps_in, num_classes, k=1, bias=True, checkpointing=checkpointing)

    def forward(self, x, skip):
        skip = self.bottleneck(skip)
        skip_size = skip.size()[2:4]
        aux = self.logits_aux(x)
        x = F.interpolate(x, skip_size, mode='bilinear', align_corners=False)
        x = x + skip
        x = self.blend_conv(x)
        return x, aux


class DenseNet(nn.Module):
    def __init__(self, args, growth_rate=32, block_config=DENSE_NET_121_BLOCK_CONFIG,
                 num_init_features=64, bn_size=4, checkpointing=False):

        super(DenseNet, self).__init__()
        self.block_config = block_config
        self.growth_rate = growth_rate
        args.last_block_pooling = 2 ** 5

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features, momentum=batchnorm_momentum)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))
        self.first_block_idx = len(self.features)

        dilations = [1, 1, 1, 1]
        num_features = num_init_features
        self.skip_sizes = []

        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                dilation=dilations[i], checkpointing=checkpointing)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                self.skip_sizes.append(num_features)
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2, checkpointing=checkpointing)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.num_features = num_features

    def forward(self, x, target_size=None):
        skip_layers = []
        if target_size == None:
            target_size = x.size()[2:4]
        for i in range(self.first_block_idx + 1):
            x = self.features[i].forward(x)

        for i in range(self.first_block_idx + 1, self.first_block_idx + 6, 2):
            if len(self.features[i]) > 3 and self.features[i][3].stride > 1:
                skip_layers.append(x)
            x = self.features[i].forward(x)
            x = self.features[i + 1].forward(x)

        return x, skip_layers


class Ladder(nn.Module):
    def __init__(self, args, num_classes=19, checkpointing=False):
        super(Ladder, self).__init__()
        self.num_classes = num_classes
        self.backbone = DenseNet(args, checkpointing=checkpointing)

        self.upsample_layers = nn.Sequential()
        spp_square_grid = False
        spp_grids = [8,4,2,1]
        num_levels = 4
        args.last_block_pooling = 2**5
        up_sizes = [256, 256, 128]

        num_features = self.backbone.num_features

        self.spp_size = 512
        level_size = self.spp_size // num_levels
        bt_size = self.spp_size
        self.spp = SpatialPyramidPooling(num_features, num_levels, bt_size, level_size,
        self.spp_size, spp_grids, spp_square_grid, checkpointing=checkpointing)
        num_features = self.spp_size

        assert len(up_sizes) == len(self.backbone.skip_sizes)
        for i in range(len(self.backbone.skip_sizes)):
            upsample = _Upsample(num_features, self.backbone.skip_sizes[-1-i], up_sizes[i],
                               num_classes=self.num_classes, checkpointing=checkpointing)
            num_features = up_sizes[i]
            self.upsample_layers.add_module('upsample_'+str(i), upsample)

        self.num_features = num_features

    def forward(self, x, target_size=None):
        x, skip_layers = self.backbone.forward(x)

        x = self.spp(x)

        for i, skip in enumerate(reversed(skip_layers)):
            x, _ = self.upsample_layers[i].forward(x, skip)

        return x


class UpsampleWrapper(nn.Module):
    def __init__(self, args, num_features, skip_sizes, checkpointing=False, up_sizes=[256, 256, 128]):
        super(UpsampleWrapper, self).__init__()

        self.upsample_layers = nn.Sequential()
        args.last_block_pooling = 2 ** 5

        assert len(up_sizes) == len(skip_sizes)
        for i in range(len(skip_sizes)):
            upsample = _Upsample(num_features, skip_sizes[-1 - i], up_sizes[i],
                                 num_classes=1, checkpointing=checkpointing)
            num_features = up_sizes[i]
            self.upsample_layers.add_module('upsample_' + str(i), upsample)

        self.num_features = num_features

    def forward(self, x, skip_layers):
        for i, skip in enumerate(reversed(skip_layers)):
            x, _ = self.upsample_layers[i].forward(x, skip)
        return x


class SPPWrapper(nn.Module):
    def __init__(self, num_features, checkpointing=False):
        super(SPPWrapper, self).__init__()

        self.spp_size = 512
        spp_square_grid = False
        spp_grids = [8, 4, 2, 1]
        num_levels = 4
        level_size = self.spp_size // num_levels
        bt_size = self.spp_size
        self.spp = SpatialPyramidPooling(num_features, num_levels, bt_size, level_size,
                                         self.spp_size, spp_grids, spp_square_grid, checkpointing=checkpointing)
        self.num_features = self.spp_size

    def forward(self, x):
        return self.spp(x)


class LadderDenseNet(nn.Module):
    def __init__(self, args, num_classes=19, checkpointing=False):
        super(LadderDenseNet, self).__init__()

        self.num_classes = num_classes
        self.backbone = DenseNet(args, checkpointing=checkpointing)
        self.spp = SPPWrapper(self.backbone.num_features, checkpointing=checkpointing)
        self.upsample = UpsampleWrapper(args, self.spp.num_features, self.backbone.skip_sizes, checkpointing=checkpointing)
        self.logits = _BNReluConv(self.upsample.num_features, self.num_classes, k=1, bias=True, checkpointing=checkpointing)


    def forward(self, x, target_size=None):
        if target_size == None:
            target_size = x.size()[2:4]

        x, skip_layers = self.backbone.forward(x)
        x = self.spp(x)
        x = self.upsample(x, skip_layers)
        x = self.logits(x)
        x = F.upsample(x, target_size, mode='bilinear', align_corners=False)

        return x

class LadderDenseNetTH(nn.Module):
    def __init__(self, args, num_classes=19, checkpointing=False):
        super(LadderDenseNetTH, self).__init__()

        self.num_classes = num_classes
        self.backbone = DenseNet(args, checkpointing=checkpointing)
        self.spp = SPPWrapper(self.backbone.num_features, checkpointing=checkpointing)
        self.upsample = UpsampleWrapper(args, self.spp.num_features, self.backbone.skip_sizes, checkpointing=checkpointing)
        self.logits = _BNReluConv(self.upsample.num_features, self.num_classes, k=1, bias=True, checkpointing=checkpointing)

        self.logits_ood = _BNReluConv(self.upsample.num_features, 2, k=1, bias=True, checkpointing=checkpointing)


    def forward(self, x, target_size=None):
        if target_size == None:
            target_size = x.size()[2:4]

        x, skip_layers = self.backbone.forward(x)
        x = self.spp(x)
        x_fe = self.upsample(x, skip_layers)
        x = self.logits(x_fe)
        x = F.upsample(x, target_size, mode='bilinear', align_corners=False)

        x_ood = self.logits_ood(x_fe)
        x_ood = F.upsample(x_ood, target_size, mode='bilinear', align_corners=False)
        return x, x_ood


