import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2
import numpy as np
from CBAM import BasicBlock
from antialias import Downsample as downsamp


class IlluminationEnhancementModule(nn.Module):
    def __init__(self):
        super(IlluminationEnhancementModule, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.refine1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.refine2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))

        # 使用Sigmoid函数得到调整因子，值在0到1之间
        # 这将允许在暗部区域进行增强，同时在亮部区域进行限制
        adjust_factor = self.sigmoid(out)

        # 细化网络, 进一步处理特征, 用于恢复更多细节
        out = self.relu(self.refine1(out))
        enhanced = torch.mul(out, adjust_factor)

        out = self.relu(self.refine2(enhanced))
        out = torch.add(out, identity)
        return out
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv2d(channels, channels, kernel_size=3, padding=1,stride=1)
        self.conv2 = DepthwiseSeparableConv2d(channels, channels, kernel_size=3, padding=1,stride=1)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual  # 将输入加到卷积层的输出上，形成残差连接
        return F.relu(out)


# 定义色彩还原网络模型，包含残差块
class ColorRestorationResNet(nn.Module):
    def __init__(self, channles):
        super(ColorRestorationResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, channles, kernel_size=3, padding=1)
        self.resblock1 = ResidualBlock(channles)
        self.resblock2 = ResidualBlock(channles)
        self.resblock3 = ResidualBlock(channles)
        self.conv2 = nn.Conv2d(channles, 3, kernel_size=1, padding=0)  # 假设输出也是RGB三通道

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.conv2(x)
        return x


class ColorRestorationResNet_ori(nn.Module):
    def __init__(self):
        super(ColorRestorationResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.resblock1 = ResidualBlock(64)
        self.resblock2 = ResidualBlock(64)
        self.resblock3 = ResidualBlock(64)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)  # 假设输出也是RGB三通道

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.conv2(x)
        return x


def auto_white_balance(image):
    # 将图片从BGR转换到LAB颜色空间
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # 分离通道
    l, a, b = cv2.split(lab)

    # 计算L通道的平均亮度
    avg_l = np.average(l)

    # 缩放LAB图像以达到中间亮度值128
    scaling_factor = 128 / avg_l
    l = l * scaling_factor
    l = np.clip(l, 0, 255).astype(np.uint8)

    # 将调整后的L通道与A和B通道合并
    adjusted_lab = cv2.merge([l, a, b])

    # 将LAB图片转换回BGR颜色空间
    adjusted_image = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)

    return adjusted_image


def process_tensor_awb(tensor_batch):
    # 确保输入的Tensor是浮点型并且其值在[0, 1]的范围内
    tensor_batch = torch.clamp(tensor_batch, 0, 1)

    # 如果Tensor在CUDA上，先将其移至CPU
    if tensor_batch.is_cuda:
        tensor_batch = tensor_batch.cpu()

    # 把Tensor批数据转换为NumPy数组，数组的数据会在[0, 255]的范围
    numpy_batch = (tensor_batch.detach().numpy() * 255).astype(np.uint8)

    # 如果Tensor是CHW格式，转换为HWC格式
    if numpy_batch.shape[1] == 3:
        numpy_batch = numpy_batch.transpose((0, 2, 3, 1))

    # 白平衡处理后的图片列表
    white_balanced_images = []

    # 对批数据中的每一张图片进行处理
    for image in numpy_batch:
        # 应用白平衡处理
        wb_image = auto_white_balance(image)
        # 将处理过的图像添加到列表中
        white_balanced_images.append(wb_image)

    # 将列表转换回Tensor
    tensor_wb_images = torch.tensor(np.array(white_balanced_images), dtype=torch.float32) / 255.0

    # 如果需要CHW格式，转换回来
    tensor_wb_images = tensor_wb_images.permute(0, 3, 1, 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 将处理后的Tensor移回到原来的设备（CUDA）
    tensor_wb_images = tensor_wb_images.to(device)

    return tensor_wb_images


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, mode='bilinear'):
        super(UpSample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode=mode, align_corners=False)
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv_block(x)
        return x


class FeatureTransform(nn.Module):
    def __init__(self):
        super(FeatureTransform, self).__init__()
        self.up1 = UpSample(128, 64)
        self.up2 = UpSample(64, 32)
        self.up3 = UpSample(32, 16)
        self.conv_block = ConvBlock(16, 3)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.conv_block(x)
        x = self.sigmod(x) + 1
        return x


class FeatureTransform_Cat(nn.Module):
    def __init__(self):
        super(FeatureTransform_Cat, self).__init__()
        self.filters = [16, 32, 64, 128]
        self.up1 = UpSample_res(self.filters[3])
        self.up2 = UpSample_res(self.filters[2])
        self.up3 = UpSample_res(self.filters[1])
        self.up4 = UpSample_res(self.filters[0])

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        return x


class ResidualUpSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(ResidualUpSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
                                 nn.PReLU(),
                                 nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1,
                                                    bias=bias),
                                 nn.PReLU(),
                                 nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias),
                                 nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top + bot
        return out


class UpSample_res(nn.Module):
    def __init__(self, in_channel, scale_factor=2, stride=2, kernel_size=3):
        super(UpSample_res, self).__init__()
        self.scale_factor = scale_factor
        self.residualupsample = ResidualUpSample(in_channel)

    def forward(self, x):
        out = self.residualupsample(x)
        return out


class FusionLayer(nn.Module):
    def __init__(self, inchannel, outchannel, reduction=16):
        super(FusionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(inchannel // reduction, inchannel, bias=False),
            nn.Sigmoid()
        )
        self.fusion = ConvBlock(inchannel, inchannel, 1, 1, 0)
        self.outlayer = ConvBlock(inchannel, outchannel, 1, 1, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        avg = self.fc(avg).view(b, c, 1, 1)
        max = self.max_pool(x).view(b, c)
        max = self.fc(max).view(b, c, 1, 1)
        fusion = self.fusion(avg + max)
        fusion = x * fusion.expand_as(x)
        fusion = fusion + x
        fusion = self.outlayer(fusion)
        return fusion


class ResidualDownSample(nn.Module):
    def __init__(self, in_channel, bias=False):
        super(ResidualDownSample, self).__init__()
        self.prelu = nn.PReLU()

        self.conv1 = nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=bias)
        self.downsamp = downsamp(channels=in_channel, filt_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channel, 2 * in_channel, 1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        out = self.prelu(self.conv1(x))
        out = self.downsamp(out)
        out = self.conv2(out)
        return out


class DownSample(nn.Module):
    def __init__(self, in_channel, scale_factor=2, stride=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = scale_factor
        self.residualdownsample = ResidualDownSample(in_channel)

    def forward(self, x):
        out = self.residualdownsample(x)
        return out


import torch
import torch.nn as nn


class TransformerFeatureTransform(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(TransformerFeatureTransform, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_channels, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)

        self.up2 = TransformerUpSample(input_channels)
        self.up3 = TransformerUpSample(input_channels // 2)
        self.up4 = TransformerUpSample(input_channels // 4)

        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels // 8, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        seq_len = x.size(2) * x.size(3)  # 计算序列长度
        x_reshaped = x.permute(2, 3, 0, 1).contiguous().view(seq_len, x.size(0), x.size(1))
        x_transformed = self.transformer_encoder(x_reshaped)

        x_transformed = x_transformed.view(x.size(2), x.size(3), x.size(0), x.size(1)).permute(2, 3, 0, 1)

        x = self.up2(x_transformed)
        x = self.up3(x)
        x = self.up4(x)

        x = self.conv_block(x)

        return x


class TransformerUpSample(nn.Module):
    def __init__(self, in_channels):
        super(TransformerUpSample, self).__init__()

        self.top = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels // 2, 1)
        )

        self.bot = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, in_channels // 2, 1)
        )

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top + bot
        return out
