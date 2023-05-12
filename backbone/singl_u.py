from collections import OrderedDict

import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F

from torch.jit.annotations import Tuple, List, Dict



class UpSample(nn.Module):

    def __init__(self, in_channels):
        """
        反卷积，上采样，通道数将会减半，
        :param in_channels: 输入通道数
        """
        super(UpSample, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=(2, 2), stride=(2, 2)),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class Unet(nn.Module):
    """
    Module that adds a FPN from on top of a set of feature maps. This is based on
    `"Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>`_.
    The feature maps are currently supposed to be in increasing depth
    order.
    The input to the model is expected to be an OrderedDict[Tensor], containing
    the feature maps on top of which the FPN will be added.
    Arguments:
        in_channels_list (list[int]): number of channels for each feature map that
            is passed to the module
        out_channels (int): number of channels of the FPN representation
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names
    """

    def __init__(self, in_channels_list, out_channels, extra_blocks=None):
        super(Unet, self).__init__()
        # 用来调整resnet特征矩阵(layer1,2,3,4)的channel（kernel_size=1）
        self.inner_blocks = nn.ModuleList()
        # 对调整后的特征矩阵使用3x3的卷积核来得到对应的预测特征矩阵
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                continue
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        self.extra_blocks = extra_blocks

        self.up1 = up_conv(ch_in=2048,ch_out=1024)
        self.Up_conv1 = conv_block(ch_in=2048, ch_out=1024)
        self.cbam1 = CBAM(channel=1024)

        self.up2 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv2 = conv_block(ch_in=1024, ch_out=512)
        self.cbam2 = CBAM(channel=512)

        self.up3 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv3 = conv_block(ch_in=512, ch_out=256)
        self.cbam3 = CBAM(channel=256)

        self.up4 = UpSample(256)
        self.up5 = UpSample(256)
        self.post = nn.Conv2d(256, 3, 3,padding=1, bias=True)




    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.layer_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:

        names = list(x.keys())
        x = list(x.values())

        # 将resnet layer4的channel调整到指定的out_channels
        # last_inner = self.inner_blocks[-1](x[-1])
        # last_inner = self.get_result_from_inner_blocks(x[-1], -1)  #[21,42]
        #
        # c4 = self.get_result_from_inner_blocks(x[2], 2)   #[42,84]
        # c3 = self.get_result_from_inner_blocks(x[1], 1)   #[84,168]
        # c2 = self.get_result_from_inner_blocks(x[0], 0)   #[168,336]

        # result中保存着每个预测特征层
        results = []
        # 将layer4调整channel后的特征矩阵，通过3x3卷积后得到对应的预测特征矩阵
        # results.append(self.layer_blocks[-1](last_inner))
        # results.append(self.get_result_from_layer_blocks(last_inner, -1))
        c5 = x[-1]
        c4 = x[2]  #[42,84]
        c4 = self.cbam1(c4) + c4
        up1 = self.up1(c5)   #[21,42] -> [42,84]
        last_inner1 = torch.cat((up1, c4), dim=1)  #1024+1024=2048
        last_inner1 = self.Up_conv1(last_inner1)    #2048->1024

        # last_inner = self.conv1(last_inner)  #384->256
        # results.insert(0, self.get_result_from_layer_blocks(last_inner, 2))


        c3 = x[1]  #[84,168]
        c3 = self.cbam2(c3) + c3
        up2 = self.up2(last_inner1)   #[42,84] -> [84,168]  1024->512
        last_inner2 = torch.cat((up2, c3), dim=1)  #512+512=1024
        last_inner2 = self.Up_conv2(last_inner2)   #1024->512

        c2 = x[0]  # [168,336]
        c2 = self.cbam3(c2) + c2
        up3 = self.up3(last_inner2)  # [84,168] -> [168,336]  512->256
        last_inner3 = torch.cat((up3, c2), dim=1)  # 256+256=512
        last_inner3 = self.Up_conv3(last_inner3)  # 512->256

        last_inner3 = last_inner3 + x[0]
        last_inner2 = last_inner2 + x[1]
        last_inner1 = last_inner1 + x[2]


        x0 =  self.get_result_from_inner_blocks(last_inner3, 0)
        x1 = self.get_result_from_inner_blocks(last_inner2, 1)
        x2 = self.get_result_from_inner_blocks(last_inner1, 2)
        x3 = self.get_result_from_inner_blocks(c5, -1)

        results.append(self.get_result_from_layer_blocks(x0, 0))
        results.append(self.get_result_from_layer_blocks(x1, 1))
        results.append(self.get_result_from_layer_blocks(x2, 2))
        results.append(self.get_result_from_layer_blocks(x3, 3))



        feat_shape = results[2].shape[-2:]
        results[2] = F.interpolate(results[3], size=feat_shape, mode="nearest") + results[2]

        feat_shape = results[1].shape[-2:]
        results[1] = F.interpolate(results[2], size=feat_shape, mode="nearest") + results[1]

        feat_shape = results[0].shape[-2:]
        results[0] = F.interpolate(results[1], size=feat_shape, mode="nearest") + results[0]


        mes_conv = results[0]

        mes_conv = self.up4(mes_conv)
        mes_conv = self.up5(mes_conv)
        mes_conv = self.post(mes_conv)





        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out,mes_conv



class LastLevelMaxPool(torch.nn.Module):
    """
    Applies a max_pool2d on top of the last feature map
    """

    def forward(self, x: List[Tensor], y: List[Tensor], names: List[str]) -> Tuple[List[Tensor], List[str]]:
        names.append("pool")
        x.append(F.max_pool2d(x[-1], 1, 2, 0))  # input, kernel_size, stride, padding
        return x, names
