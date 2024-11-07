import numpy as np
import torch
from functorch.einops import rearrange
from torch import nn
from einops import repeat


class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()  
        self.downsample = nn.Sequential(  
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        width = int(out_channels * (base_width / 64))  
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)  # 第一个卷积层
        self.norm1 = nn.BatchNorm2d(width)  
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=2, groups=1, padding=1, dilation=1,
                               bias=False)  
        self.norm2 = nn.BatchNorm2d(width)  
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False)  
        self.norm3 = nn.BatchNorm2d(out_channels)  
        self.relu = nn.ReLU(inplace=True)  

    def forward(self, x):
        x_down = self.downsample(x)  
        x = self.conv1(x)  
        x = self.norm1(x)  
        x = self.relu(x)  
        x = self.conv2(x)  
        x = self.norm2(x)  
        x = self.relu(x)  
        x = self.conv3(x)  
        x = self.norm3(x)  
        x = x + x_down  
        x = self.relu(x)  
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()  
        self.head_num = head_num  
        self.dk = (embedding_dim // head_num) ** (1 / 2)  
        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)  
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)  

    def forward(self, x, mask=None):
        qkv = self.qkv_layer(x)  

        query, key, value = tuple(rearrange(qkv, 'b t (d k h) -> k b h t d', k=3, h=self.head_num))  
        energy = torch.einsum("... i d , ... j d -> ... i j", query, key) * self.dk  

        if mask is not None:  
            energy = energy.masked_fill(mask, -np.inf)

        attention = torch.softmax(energy, dim=-1)  

        x = torch.einsum("... i j , ... j d -> ... i d", attention, value)  

        x = rearrange(x, "b h t d -> b t (h d)")  
        x = self.out_attention(x)  

        return x


# MLP
class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()  

        self.mlp_layers = nn.Sequential(  
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),  
            nn.Dropout(0.1),  
            nn.Linear(mlp_dim, embedding_dim),  
            nn.Dropout(0.1)  
        )

    def forward(self, x):
        x = self.mlp_layers(x)  
        return x


# Transformer
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()  
        self.multi_head_attention = MultiHeadAttention(embedding_dim, head_num)  
        self.mlp = MLP(embedding_dim, mlp_dim)  

        self.layer_norm1 = nn.LayerNorm(embedding_dim)  
        self.layer_norm2 = nn.LayerNorm(embedding_dim)  

        self.dropout = nn.Dropout(0.1)  

    def forward(self, x):
        _x = self.multi_head_attention(x)  
        _x = self.dropout(_x)  
        x = x + _x  
        x = self.layer_norm1(x)  

        _x = self.mlp(x)  
        x = x + _x  
        x = self.layer_norm2(x)  

        return x



class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12):
        super().__init__()  

        self.layer_blocks = nn.ModuleList([  
            TransformerEncoderBlock(embedding_dim, head_num, mlp_dim) for _ in range(block_num)
        ])

    def forward(self, x):
        for layer_block in self.layer_blocks:  
            x = layer_block(x)  
        return x


# ViT
class ViT(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim, block_num, patch_dim,
                 classification=False, num_classes=1):
        super().__init__()  
        self.patch_dim = patch_dim  
        self.classification = classification  
        self.num_tokens = (img_dim // patch_dim) ** 2  
        self.token_dim = in_channels * (patch_dim ** 2)  

        self.projection = nn.Linear(self.token_dim, embedding_dim)  
        self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))  
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))  

        self.dropout = nn.Dropout(0.1)  

        self.transformer = TransformerEncoder(embedding_dim, head_num, mlp_dim, block_num)  


        if self.classification:  
            self.mlp_head = nn.Linear(embedding_dim, num_classes)  

    def forward(self, x):
        img_patches = rearrange(x,  
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.patch_dim, patch_y=self.patch_dim)

        batch_size, tokens, _ = img_patches.shape  
        project = self.projection(img_patches) 
        token = repeat(self.cls_token, 'b ... -> (b batch_size) ...', batch_size=batch_size)  

        patches = torch.cat((token, project), dim=1)  
        patches += self.embedding[:tokens + 1, :]  

        x = self.dropout(patches)  
        x = self.transformer(x)  
        x = self.mlp_head(x[:, 0, :]) if self.classification else x[:, 1:, :]  

        return x
