import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import torch.nn as nn
import torch.nn.functional as F

import math
import torch
import torch.nn as nn
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import torch.nn as nn
import torch.nn.functional as F

def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y





class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)

        return dct_filter

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
def find_closest_dct_size(input_size):
    """
    根据输入的尺寸，寻找最近的设置。

    Args:
        input_size (int): 输入的尺寸（长或宽）。
        size_dict (dict): 预定义的尺寸字典。

    Returns:
        int: 最近的预定义尺寸。
    """
    size_dict = {64: 56, 128: 28, 256: 14, 512: 7}
    closest_size = min(size_dict.values(), key=lambda x: abs(x - input_size))
    return closest_size

class DenoisingCNN(nn.Module):
    def __init__(self, channles=16):
        super(DenoisingCNN, self).__init__()

        # 输入预处理
        self.input_preprocess = nn.Sequential(
            nn.Conv2d(4, channles // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=False),
            DepthwiseSeparableConv2d(channles // 2, channles, kernel_size=3, padding=1,stride=1),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(channles, channles, kernel_size=1, padding=0),
            nn.LeakyReLU(inplace=False),
        )

        # 卷积层堆叠
        self.conv_layers = nn.Sequential(
            nn.Conv2d(channles, channles, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(channles, channles, kernel_size=5, padding=2),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(channles, channles, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=False),
        )

        # 反卷积层或上采样
        self.output_layer = nn.Sequential(
            nn.Conv2d(channles, channles, kernel_size=1, padding=0),
            nn.LeakyReLU(inplace=False),
            DepthwiseSeparableConv2d(channles, channles // 2, kernel_size=3, padding=1,stride=1),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(channles // 2, 3, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=False)
        )
        self.out_conv = DepthwiseSeparableConv2d(4, 3, 1, 1, 0)
        self.att = MultiSpectralAttentionLayer(channles, find_closest_dct_size(channles), find_closest_dct_size(channles), reduction=16,
                                               freq_sel_method='top16')

    def forward(self, x):
        if x.shape[1]!= 3:
            x = x[:,:3,:,:]
        x_bright, _ = torch.max(x, dim=1, keepdim=True)  # CPU
        x_in = torch.cat((x, x_bright), 1)
        # 前向传播
        # print(x_in.shape)
        T = self.input_preprocess(x_in)
        # print(T.shape)
        S = self.conv_layers(T)
        # print(S.shape)
        T = T + S

        T = self.att(T)
        # print(T.shape)

        T = self.output_layer(T)
        final_min, _ = torch.min(T, dim=1, keepdim=True)  # CPU
        final = torch.cat((T, final_min), 1)
        final = self.out_conv(final)

        T = final
        return T

from Best_module.Mamba import VMUNet


class net(nn.Module):
    def __init__(self, deep_supervision=False):
        super(net, self).__init__()
        self.mamba_unet = VMUNet(input_channels=3)
        self.denoise = DenoisingCNN(64)

    def forward(self, inputs):
        inputs_denoise = self.denoise(inputs)
        out_unet = self.mamba_unet(inputs)
        final = out_unet + inputs_denoise
        return final


if __name__ == "__main__":
    print("deep_supervision: False")
    deep_supervision = False
    device = torch.device('cpu')
    inputs = torch.randn((1, 3, 224, 224)).to(device)
    model = unet59(deep_supervision=deep_supervision).to(device)
    outputs = model(inputs)
    print(outputs.shape)

    print("deep_supervision: True")
    deep_supervision = True
    model = unet59(deep_supervision=deep_supervision).to(device)
    outputs = model(inputs)
    for out in outputs:
        print(out.shape)
