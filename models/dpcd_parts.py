import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from utils.path_hyperparameter import ph
import cv2
from torchvision import transforms as T
from pathlib import Path


class Conv_BN_ReLU(nn.Module):
    """ Basic convolution."""

    def __init__(self, in_channel, out_channel, kernel, stride):
        super().__init__()
        self.conv_bn_relu = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel,
                                                    padding=kernel // 2, bias=False, stride=stride),
                                          nn.BatchNorm2d(out_channel),
                                          nn.ReLU(inplace=True),
                                          )

    def forward(self, x):
        output = self.conv_bn_relu(x)

        return output


class CGSU(nn.Module):
    """Basic convolution module."""

    def __init__(self, in_channel):
        super().__init__()

        mid_channel = in_channel // 2

        self.conv1 = nn.Sequential(nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=1,
                                             bias=False),
                                   nn.BatchNorm2d(mid_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(p=ph.dropout_p)
                                   )

    def forward(self, x):
        x1, x2 = channel_split(x)
        x1 = self.conv1(x1)
        output = torch.cat([x1, x2], dim=1)

        return output


class CGSU_DOWN(nn.Module):
    """Basic convolution module with stride=2."""

    def __init__(self, in_channel):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1,
                                             stride=2, bias=False),
                                   nn.BatchNorm2d(in_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(p=ph.dropout_p)
                                   )
        self.conv_res = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # remember the tensor should be contiguous
        output1 = self.conv1(x)

        # respath
        output2 = self.conv_res(x)

        output = torch.cat([output1, output2], dim=1)

        return output


class Changer_channel_exchange(nn.Module):
    """Exchange channels of two feature uniformly-spaced with 1:1 ratio."""

    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def forward(self, x1, x2):
        N, C, H, W = x1.shape
        exchange_mask = torch.arange(C) % self.p == 0
        exchange_mask1 = exchange_mask.cuda().int().expand((N, C)).unsqueeze(-1).unsqueeze(-1)  # b,c,1,1
        exchange_mask2 = 1 - exchange_mask1
        out_x1 = exchange_mask1 * x1 + exchange_mask2 * x2
        out_x2 = exchange_mask1 * x2 + exchange_mask2 * x1

        return out_x1, out_x2


# double pooling fuse attention
class DPFA(nn.Module):
    """Fuse two feature into one feature."""

    def __init__(self, in_channel):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.k = kernel_size(in_channel)
        self.channel_conv1 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.channel_conv2 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.spatial_conv1 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.spatial_conv2 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.softmax = nn.Softmax(0)

    def forward(self, t1, t2, log=None, module_name=None,
                img_name=None):
        # channel part
        t1_channel_avg_pool = self.avg_pool(t1)  # b,c,1,1
        t1_channel_max_pool = self.max_pool(t1)  # b,c,1,1
        t2_channel_avg_pool = self.avg_pool(t2)  # b,c,1,1
        t2_channel_max_pool = self.max_pool(t2)  # b,c,1,1

        channel_pool = torch.cat([t1_channel_avg_pool, t1_channel_max_pool,
                                  t2_channel_avg_pool, t2_channel_max_pool],
                                 dim=2).squeeze(-1).transpose(1, 2)  # b,4,c
        t1_channel_attention = self.channel_conv1(channel_pool)  # b,1,c
        t2_channel_attention = self.channel_conv2(channel_pool)  # b,1,c
        channel_stack = torch.stack([t1_channel_attention, t2_channel_attention],
                                    dim=0)  # 2,b,1,c
        channel_stack = self.softmax(channel_stack).transpose(-1, -2).unsqueeze(-1)  # 2,b,c,1,1

        # spatial part
        t1_spatial_avg_pool = torch.mean(t1, dim=1, keepdim=True)  # b,1,h,w
        t1_spatial_max_pool = torch.max(t1, dim=1, keepdim=True)[0]  # b,1,h,w
        t2_spatial_avg_pool = torch.mean(t2, dim=1, keepdim=True)  # b,1,h,w
        t2_spatial_max_pool = torch.max(t2, dim=1, keepdim=True)[0]  # b,1,h,w
        spatial_pool = torch.cat([t1_spatial_avg_pool, t1_spatial_max_pool,
                                  t2_spatial_avg_pool, t2_spatial_max_pool], dim=1)  # b,4,h,w
        t1_spatial_attention = self.spatial_conv1(spatial_pool)  # b,1,h,w
        t2_spatial_attention = self.spatial_conv2(spatial_pool)  # b,1,h,w
        spatial_stack = torch.stack([t1_spatial_attention, t2_spatial_attention], dim=0)  # 2,b,1,h,w
        spatial_stack = self.softmax(spatial_stack)  # 2,b,1,h,w

        # fusion part, add 1 means residual add
        stack_attention = channel_stack + spatial_stack + 1  # 2,b,c,h,w
        fuse = stack_attention[0] * t1 + stack_attention[1] * t2  # b,c,h,w

        if log:
            log_list = [t1, t2, t1_spatial_attention, t2_spatial_attention, fuse]
            feature_name_list = ['t1', 't2', 't1_spatial_attention', 't2_spatial_attention', 'fuse']
            log_feature(log_list=log_list, module_name=module_name,
                        feature_name_list=feature_name_list,
                        img_name=img_name, module_output=True)

        return fuse


class CBAM(nn.Module):
    """Attention module."""

    def __init__(self, in_channel):
        super().__init__()
        self.k = kernel_size(in_channel)
        self.channel_conv = nn.Conv1d(2, 1, kernel_size=self.k, padding=self.k // 2)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x, log=False, module_name=None, img_name=None):
        avg_channel = self.avg_pooling(x).squeeze(-1).transpose(1, 2)  # b,1,c
        max_channel = self.max_pooling(x).squeeze(-1).transpose(1, 2)  # b,1,c
        channel_weight = self.channel_conv(torch.cat([avg_channel, max_channel], dim=1))
        channel_weight = self.sigmoid(channel_weight).transpose(1, 2).unsqueeze(-1)  # b,c,1,1
        x = channel_weight * x

        avg_spatial = torch.mean(x, dim=1, keepdim=True)  # b,1,h,w
        max_spatial = torch.max(x, dim=1, keepdim=True)[0]  # b,1,h,w
        spatial_weight = self.spatial_conv(torch.cat([avg_spatial, max_spatial], dim=1))  # b,1,h,w
        spatial_weight = self.sigmoid(spatial_weight)
        output = spatial_weight * x

        if log:
            log_list = [spatial_weight]
            feature_name_list = ['spatial_weight']
            log_feature(log_list=log_list, module_name=module_name,
                        feature_name_list=feature_name_list,
                        img_name=img_name, module_output=True)

        return output


class Encoder_Block(nn.Module):
    """ Basic block in encoder"""

    def __init__(self, in_channel, out_channel):
        super().__init__()

        assert out_channel == in_channel * 2, 'the out_channel is not in_channel*2 in encoder block'
        self.conv1 = nn.Sequential(
            CGSU_DOWN(in_channel=in_channel),
            CGSU(in_channel=out_channel),
            CGSU(in_channel=out_channel)
        )
        self.conv3 = Conv_BN_ReLU(in_channel=out_channel, out_channel=out_channel, kernel=3, stride=1)
        self.cbam = CBAM(in_channel=out_channel)

    def forward(self, x, log=False, module_name=None, img_name=None):
        x = self.conv1(x)
        x = self.conv3(x)
        x_res = x.clone()
        if log:
            output = self.cbam(x, log=log, module_name=module_name + '-x_cbam', img_name=img_name)
        else:
            output = self.cbam(x)
        output = x_res + output

        return output


class Decoder_Block(nn.Module):
    """Basic block in decoder."""

    def __init__(self, in_channel, out_channel):
        super().__init__()

        assert out_channel == in_channel // 2, 'the out_channel is not in_channel//2 in decoder block'
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.fuse = nn.Sequential(nn.Conv2d(in_channels=in_channel + out_channel, out_channels=out_channel,
                                            kernel_size=1, padding=0, bias=False),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU(inplace=True),
                                  )

    def forward(self, de, en):
        de = self.up(de)
        output = torch.cat([de, en], dim=1)
        output = self.fuse(output)

        return output


# from ECANet, in which y and b is set default to 2 and 1
def kernel_size(in_channel):
    """Compute kernel size for one dimension convolution in eca-net"""
    k = int((math.log2(in_channel) + 1) // 2)  # parameters from ECA-net
    if k % 2 == 0:
        return k + 1
    else:
        return k


def channel_split(x):
    """Half segment one feature on channel dimension into two features, mixture them on channel dimension,
    and split them into two features."""
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]


def log_feature(log_list, module_name, feature_name_list, img_name, module_output=True):
    """ Log output feature of module and model

    Log some output features in a module. Feature in :obj:`log_list` should have corresponding name
    in :obj:`feature_name_list`.

    For module output feature, interpolate it to :math:`ph.patch_size`×:math:`ph.patch_size`,
    log it in :obj:`cv2.COLORMAP_JET` format without other change,
    and log it in :obj:`cv2.COLORMAP_JET` format with equalization.
    For model output feature, log it without any change.

    Notice that feature is log in :obj:`ph.log_path`/:obj:`module_name`/
    name in :obj:`feature_name_list`/:obj:`img_name`.jpg.

    Parameter:
        log_list(list): list of output need to log.
        module_name(str): name of module which output the feature we log,
            if :obj:`module_output` is False, :obj:`module_name` equals to `model`.
        module_output(bool): determine whether output is from module or model.
        feature_name_list(list): name of output feature.
        img_name(str): name of corresponding image to output.


    """
    for k, log in enumerate(log_list):
        log = log.clone().detach()
        b, c, h, w = log.size()
        if module_output:
            log = torch.mean(log, dim=1, keepdim=True)
            log = F.interpolate(
                log * 255, scale_factor=ph.patch_size // h,
                mode='nearest').reshape(b, ph.patch_size, ph.patch_size, 1) \
                .cpu().numpy().astype(np.uint8)
            log_dir = ph.log_path + module_name + '/' + feature_name_list[k] + '/'
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            log_equalize_dir = ph.log_path + module_name + '/' + feature_name_list[k] + '_equalize/'
            Path(log_equalize_dir).mkdir(parents=True, exist_ok=True)

            for i in range(b):
                log_i = cv2.applyColorMap(log[i], cv2.COLORMAP_JET)
                cv2.imwrite(log_dir + img_name[i] + '.jpg', log_i)

                log_i_equalize = cv2.equalizeHist(log[i])
                log_i_equalize = cv2.applyColorMap(log_i_equalize, cv2.COLORMAP_JET)
                cv2.imwrite(log_equalize_dir + img_name[i] + '.jpg', log_i_equalize)
        else:
            log_dir = ph.log_path + module_name + '/' + feature_name_list[k] + '/'
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            log = torch.round(torch.sigmoid(log))
            log = F.interpolate(log, scale_factor=ph.patch_size // h,
                                mode='nearest').cpu()
            to_pil_img = T.ToPILImage(mode=None)
            for i in range(b):
                log_i = to_pil_img(log[i])
                log_i.save(log_dir + img_name[i] + '.jpg')
