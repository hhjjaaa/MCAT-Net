from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math  # Ensure it's Python's built-in math library
import numpy as np
import torch.nn as nn
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from transformer_model import ViT


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class DCDTQ(nn.Module):
    def __init__(self, input_channels=1, output_channels=32):
        super(DCDTQ, self).__init__()
        self.conv1 = nn.Conv2d(input_channels * 16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(16, output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        # 确保输入x是PyTorch张量
        min_val = x.min()
        max_val = x.max()

        # 归一化到0-255
        x_scaled = ((x - min_val) / (max_val - min_val) * 255).to(torch.uint8)
        x_copy = x_scaled.clone().to(torch.float32) / 255.0

        # 提取特征图
        Dig1 = torch.where(x_scaled >= 128, torch.tensor([128], dtype=torch.uint8, device=x.device),
                           torch.tensor([0], dtype=torch.uint8, device=x.device))
        x_scaled -= Dig1
        Dig2 = torch.where(x_scaled >= 64, torch.tensor([64], dtype=torch.uint8, device=x.device),
                           torch.tensor([0], dtype=torch.uint8, device=x.device))
        x_scaled -= Dig2
        Dig3 = torch.where(x_scaled >= 32, torch.tensor([32], dtype=torch.uint8, device=x.device),
                           torch.tensor([0], dtype=torch.uint8, device=x.device))
        x_scaled -= Dig3
        Dig4 = torch.where(x_scaled >= 16, torch.tensor([16], dtype=torch.uint8, device=x.device),
                           torch.tensor([0], dtype=torch.uint8, device=x.device))

        # 拼接特征图
        channels = [Dig1, Dig1, Dig2, Dig2, Dig3, Dig3, Dig4, Dig4] + [x_scaled] * 8
        x_concatenated = torch.cat(channels, dim=1).to(torch.float32) / 255.0
        channels2 = [x_copy] * 16
        x_concatenated2 = torch.cat(channels2, dim=1)

        # 卷积和批量归一化
        x_normalized = self.conv1(x_concatenated)
        x_normalized = self.bn1(x_normalized)
        x_normalized = torch.add(x_normalized, x_concatenated2)
        x_normalized = self.relu(x_normalized)

        x_normalized = self.conv2(x_normalized)
        x_normalized = self.bn2(x_normalized)
        x_normalized = self.relu(x_normalized)
        return x_normalized


class SE_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x.size() 30,40,50,30
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 30,1,50,30
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 30,1,50,30
        return self.sigmoid(x)  # 30,1,50,30

# class ECANet(nn.Module):
#     def __init__(self, in_channels, gamma=2, b=1):
#         super(ECANet, self).__init__()
#         self.in_channels = in_channels
#         self.fgp = nn.AdaptiveAvgPool2d((1, 1))
#         # kernel_size = int(abs((math.log(self.in_channels, 2) + b) / gamma))
#         # kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
#         kernel_size=3
#         self.con1 = nn.Conv1d(1,
#                               1,
#                               kernel_size=kernel_size,
#                               padding=(kernel_size - 1) // 2,
#                               bias=False)
#         self.act1 = nn.Sigmoid()
#
#     def forward(self, x):
#         output = self.fgp(x)
#         output = output.squeeze(-1).transpose(-1, -2)
#         output = self.con1(output).transpose(-1, -2).unsqueeze(-1)
#         output = self.act1(output)
#         output = torch.multiply(x, output)
#         return output
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        """
        第一层全连接层神经元个数较少，因此需要一个比例系数ratio进行缩放
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        """
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False)
        )
        """
        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=3):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x

# without BN version
class ASPP(nn.Module):
    def __init__(self, in_channel=1024, out_channel=512):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel,out_channel, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(out_channel * 5, out_channel, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = torch.nn.functional.interpolate(image_features, size=size, mode='bilinear', align_corners=False)

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net




# class DFRup(nn.Module):
#     def __init__(self, in_channels):
#         super(DFRup, self).__init__()
#         self.in_channels = in_channels
#         self.conv1 = nn.Conv2d(in_channels, in_channels*2, kernel_size=3, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(in_channels*2)
#         self.relu1 = nn.ReLU(inplace=True)
#
#         self.conv2 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1,padding=0, bias=False)
#         self.bn2 = nn.BatchNorm2d(in_channels//2)
#         self.relu2 = nn.ReLU(inplace=True)
#
#         # self.ECA = ECANet(in_channels*2)
#         self.pixel_shuffle = nn.PixelShuffle(2)
#         self.up = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
#         self.SA = SpatialAttention()
#     def forward(self,x):
#         xx=x
#
#         x = self.conv1(x)
#         # print("Shape of x:", x.shape)
#         x = self.bn1(x)
#         # print("Shape of x:", x.shape)
#         x = self.relu1(x)
#         # print("Shape of x:", x.shape)
#         x2 = self.ECA(x)
#         # print("Shape of x2:", x2.shape)
#         x2=torch.add(x,x2)
#         # print("Shape of x2:", x2.shape)
#         x2=self.pixel_shuffle(x2)
#         # print("Shape of x2:", x2.shape)
#
#         xx=self.up(xx)
#         # print("Shape of xx:", xx.shape)
#         xx=self.conv2(xx)
#         # print("Shape of xx:", xx.shape)
#         xx=self.bn2(xx)
#         # print("Shape of xx:", xx.shape)
#         xx=self.relu2(xx)
#         # print("Shape of xx:", xx.shape)
#         xx2=self.SA(xx)
#         # print("Shape of xxx2:", xx2.shape)
#         xx2=torch.add(xx2,xx)
#         # print("Shape of xx2:", xx2.shape)
#
#         xxx=torch.matmul(x2,xx2)
#         return xxx


# 定义深度可分离卷积模块
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        # 深度可分离卷积包括深度卷积和逐点卷积两个步骤
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


#位置编码
# def generate_relative_position_matrix(height, width):
#     # 生成位置索引
#     y_position = np.tile(np.arange(height)[:, None], (1, width))
#     x_position = np.tile(np.arange(width), (height, 1))
#
#     # 将位置索引缩放到-1到1之间
#     y_position = y_position / (height - 1) * 2 - 1
#     x_position = x_position / (width - 1) * 2 - 1
#
#     # 应用正弦和余弦变换
#     rpe_y = np.sin(y_position * np.pi) + np.cos(y_position * np.pi)
#     rpe_x = np.sin(x_position * np.pi) + np.cos(x_position * np.pi)
#
#     # 将两个方向的编码结合起来（简单的方法是直接相加）
#     rpe = rpe_y[:, :, None] + rpe_x[:, None, :]
#
#     return torch.tensor(rpe, dtype=torch.float32)


class MultiHeadAttentionWithPosEncoding(nn.Module):
    def __init__(self, in_channels, num_heads, max_size):
        super().__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        self.qkv = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.softmax = nn.Softmax(dim=-1)
        # 初始化最大尺寸的位置编码
        self.max_size = max_size
        self.pos_encoding = nn.Parameter(torch.zeros(1, (self.max_size//2) * (self.max_size//2), in_channels//num_heads))

    def forward(self, x):
        B, C, H, W = x.shape

        # 根据当前输入调整位置编码的尺寸
        current_size = (H//2) * (W//2)
        if current_size != self.pos_encoding.shape[1]:
            # 重新创建适应当前尺寸的位置编码
            self.pos_encoding = nn.Parameter(torch.zeros(1, current_size, C//2).to(x.device))

        # 应用1x1卷积和池化
        x = self.qkv(x)
        x = self.pool(x)  # 减半空间维度

        # 改变形状以适应多头注意力
        qkv = x.reshape(B, 3, self.num_heads, self.head_dim, current_size)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # 重排为 (3, B, num_heads, HW, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 添加位置编码到查询
        q += self.pos_encoding

        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = self.softmax(attn)
        weighted_v = attn @ v
        weighted_v = weighted_v.transpose(2, 3).reshape(B, C, H//2, W//2)

        return weighted_v





class DoubleConv3_3(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv3_3, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        # 定义第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)




        # 定义第二个卷积层
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        # self.cbam_block =cbam_block(out_channels)
    def forward(self, x):
        # 第一个卷积层

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)


        # 第二个卷积层
        x = self.conv2(x)
        x = self.bn2(x)

        x = self.relu2(x)
        # x = self.cbam_block(x)

        return x
class DoubleConv3_3SE(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv3_3SE, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        # 定义第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # 定义第二个卷积层
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.SE = SpatialAttention(kernel_size=3)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

        # 定义第二个卷积层
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.relu4 = nn.ReLU(inplace=True)
    def forward(self, x):
        # 第一个卷积层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # 第二个卷积层
        x = self.conv2(x)
        x = self.bn2(x)

        seout= self.SE(x)
        x = x*seout
        x = self.relu2(x)
        x2=self.conv3(x)
        x2=self.bn3(x2)
        x2=self.relu3(x2)
        x2=self.conv4(x2)
        x2=self.bn4(x2)
        x2=self.relu4(x2)
        x=torch.add(x,x2)



        return x

# class DoubleConv3_3noEA(nn.Module):
#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super(DoubleConv3_3, self).__init__()
#
#         if mid_channels is None:
#             mid_channels = out_channels
#
#         # 定义第一个卷积层
#         self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(mid_channels)
#         self.relu1 = nn.ReLU(inplace=True)
#
#         # 定义ECANet层
#
#
#         # 定义第二个卷积层
#         self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.relu2 = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         # 第一个卷积层
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)
#
#         # 第二个卷积层
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#
#         return x

# class DoubleConv2_2(nn.Sequential):
#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         if mid_channels is None:
#             mid_channels = out_channels
#         super(DoubleConv2_2, self).__init__(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=2, padding=1,stride=1, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=2, padding=1,stride=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

# class DoubleConv2_2(nn.Module):
#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super(DoubleConv2_2, self).__init__()
#
#         if mid_channels is None:
#             mid_channels = out_channels
#
#         # 定义第一个卷积层
#         self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=2, padding=1, stride=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(mid_channels)
#         self.relu1 = nn.ReLU(inplace=True)
#
#         # 定义ECANet层
#         self.ECA = ECANet(mid_channels)
#
#         # 定义第二个卷积层
#         self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=2, padding=1, stride=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.relu2 = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         # 第一个卷积层
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)
#
#         # 插入ECANet层
#         x = self.ECA(x)
#
#         # 第二个卷积层
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#
#         return x


# class Down(nn.Sequential):
#     def __init__(self, in_channels, out_channels):
#         super(Down, self).__init__()
#         self.pool = nn.MaxPool2d(2, stride=2)
#         self.conv = DoubleConv3_3(out_channels, out_channels)
#         self.DC = DepthwiseSeparableConv(in_channels,in_channels)
#         self.mult = MultiHeadAttentionWithPosEncoding(in_channels,2,2304)
#         self.conv1_1 =nn.Conv2d(in_channels*2, out_channels,1,bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu1 = nn.ReLU(inplace=True)
#     def forward(self, x):
#         x1 = self.pool(x)
#         x2= self.DC(x)
#         x2= self.mult(x)
#         x=torch.cat([x1, x2], dim=1)
#         x = self.conv1_1(x)
#         x= self.bn1(x)
#         x = self.relu1(x)
#         x=self.conv(x)
#         return x

class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(0.2),
            DoubleConv3_3(in_channels, out_channels)
        )
class Down2(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down2, self).__init__(
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(0.3),
            DoubleConv3_3SE(in_channels, out_channels)
        )





# class Up(nn.Module):
#     def __init__(self, in_channels):
#         super(Up, self).__init__()
#         # if bilinear:
#         #     self.up = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
#         #     self.conv = DoubleConv3_3(in_channels, out_channels, in_channels // 2)
#         # else:
#         #     self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#         #     self.conv = DoubleConv3_3(in_channels, out_channels)
#
#         self.DFR = DFRup(in_channels)
#         self.conv =nn.Conv2d(in_channels, in_channels//2, kernel_size=1, padding=0, stride=1, bias=False)
#         self.bn = nn.BatchNorm2d(in_channels//2)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#         x1 = self.DFR(x1)
#         # [N, C, H, W]
#         diff_y = x2.size()[2] - x1.size()[2]
#         diff_x = x2.size()[3] - x1.size()[3]
#
#         # padding_left, padding_right, padding_top, padding_bottom
#         x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
#                         diff_y // 2, diff_y - diff_y // 2])
#
#         x = torch.cat([x2, x1], dim=1)
#         x = self.conv(x)
#         x =self.bn(x)
#         x = self.relu(x)
#         return x
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv3_3(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3_3(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
# class Upaspp(nn.Module):
#     def __init__(self, in_channels, out_channels, bilinear=False):
#         super(Upaspp, self).__init__()
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv3_3no(out_channels, out_channels)
#             self.aspp =ASPP(in_channels,out_channels)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv3_3no(out_channels, out_channels)
#             self.aspp = ASPP(in_channels, out_channels)
#
#     def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#         x1 = self.up(x1)
#         # [N, C, H, W]
#         diff_y = x2.size()[2] - x1.size()[2]
#         diff_x = x2.size()[3] - x1.size()[3]
#
#         # padding_left, padding_right, padding_top, padding_bottom
#         x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
#                         diff_y // 2, diff_y - diff_y // 2])
#
#         x = torch.cat([x2, x1], dim=1)
#         x=self.aspp(x)
#
#         x = self.conv(x)
#         return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )
# class TransformerBlock(nn.Module):
#     def __init__(self):
#         super(TransformerBlock, self).__init__()
#         self.transformer = ViT(img_dim=12, in_channels=256, embedding_dim=256, head_num=16, mlp_dim=1024, block_num=12, patch_dim=1)
#
#     def forward(self, x):
#         t = self.transformer(x)
#         t = rearrange(t, "b (x y) c -> b c x y", x=12, y=12)
#         t=torch.add(t,x)
#         return t
class TransformerBlock(nn.Module):
    def __init__(self):
        super(TransformerBlock, self).__init__()
        self.transformer = ViT(img_dim=12, in_channels=256, embedding_dim=768, head_num=8, mlp_dim=1024, block_num=12,
                               patch_dim=1)
        self.conv1 = nn.Conv2d(in_channels=768, out_channels=256, kernel_size=1)

    def forward(self, x):
        t = self.transformer(x)
        t = rearrange(t, "b (x y) c -> b c x y", x=12, y=12)
        t = self.conv1(t)
        t=torch.add(x,t)

        return t
class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = False,
                 base_c: int = 32):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear


        self.DCDTQ = DCDTQ(in_channels, 32)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)
        self.CA1 = CoordAtt(32, 32)
        self.CA2 = CoordAtt(64, 64)
        self.CA3 = CoordAtt(128, 128)
        self.CA4 = CoordAtt(256, 256)
        self.trans =TransformerBlock()
        # self.transformer = VisionTransformer(img_size=24, patch_size=1, in_c=64)
        # self.pipeiconv = nn.Conv2d(768,128,kernel_size=1)




        self.softmax = nn.Softmax(dim=1)


    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.DCDTQ(x)
        x2 = self.down1(x1)
        x2 = self.CA2(x2)
        x3 = self.down2(x2)
        x3 = self.CA3(x3)
        x4 = self.down3(x3)
        t = self.trans(x4)

        x4 = self.CA4(x4)

        x5 = self.down4(x4)

        x = self.up1(x5, t)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out_conv(x)
        x =self.softmax(x)

        return x

