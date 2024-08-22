import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import torch.nn as nn
import torch.nn.functional as F


class se_block(nn.Module):
    def __init__(self, channels, ratio=16):
        super(se_block, self).__init__()
        # 空间信息进行压缩
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 经过两次全连接层，学习不同通道的重要性
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio, False),
            nn.ReLU(),
            nn.Linear(channels // ratio, channels, False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # 取出batch size和通道数

        # b,c,w,h->b,c,1,1->b,c 压缩与通道信息学习
        avg = self.avgpool(x).view(b, c)

        # b,c->b,c->b,c,1,1 激励操作
        y = self.fc(avg).view(b, c, 1, 1)

        return x * y.expand_as(x)


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class DenoisingCNN(nn.Module):
    def __init__(self, channles):
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
        self.se_block = se_block(channels=channles)
        self.out_conv = DepthwiseSeparableConv2d(4, 3, 1, 1, 0)

    def forward(self, x):
        x_bright, _ = torch.max(x, dim=1, keepdim=True)  # CPU
        x_in = torch.cat((x, x_bright), 1)
        # 前向传播
        T = self.input_preprocess(x_in)
        # print(T.shape)
        S = self.conv_layers(T)
        # print(S.shape)
        T = T + S

        T = self.se_block(T)

        T = self.output_layer(T)
        final_min, _ = torch.min(T, dim=1, keepdim=True)  # CPU
        final = torch.cat((T, final_min), 1)
        final = self.out_conv(final)

        T = final
        return T


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual  # 将输入加到卷积层的输出上，形成残差连接
        return F.relu(out)


from Best_module.Mamba import VMUNet


class net(nn.Module):
    def __init__(self, deep_supervision=False):
        super(net, self).__init__()
        self.mamba_unet = VMUNet(input_channels=3)

    def forward(self, inputs):
        out_unet = self.mamba_unet(inputs)
        final = out_unet + inputs
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
