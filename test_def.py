# 开 发 人：LYQ
# 开发时间：2023/4/26 12:28
import os

import torch
import torch.utils.data
import matplotlib


from skimage.io.tests.test_mpl_imshow import plt
from torch import nn
from torch.utils.data import Dataset
import numpy as np

from global_ import fengefu, zhibiao_path



# class myDataset(Dataset):
#     def __init__(self, image_list, label_list, transformer=None):
#         self.image_list = image_list
#         self.label_list = label_list
#         self.transformer = transformer
#
#     def __getitem__(self, index):
#         image = np.load(self.image_list[index])
#         #         print("image : ",image)
#         label = np.load(self.label_list[index])
#         # 在这个地方加一个判断看一下标签里面有没有大于1的值
#         # 遍历数组，找到不在范围 [0,1] 内的错误值并标记为0（也就是背景的标签）
#         arr = label
#         for i in range(arr.shape[0]):
#             for j in range(arr.shape[1]):
#                 for k in range(arr.shape[2]):
#                     if arr[i][j][k] < 0 or arr[i][j][k] > 1:
#                         arr[i][j][k] = 0
#         # print(arr)
#         image_array = image.astype(np.float32)
#         image_array = torch.FloatTensor(image_array).unsqueeze(0)
#
#         label_array = arr.astype(np.int16)
#         label_array = torch.LongTensor(label_array)
#         myDataset_img_path = self.image_list[index]
#         myDataset_label_path = self.label_list[index]
#         return image_array, label_array, myDataset_img_path, myDataset_label_path
#
#     def __len__(self):
#         return len(self.image_list)


class myDataset(Dataset):
    def __init__(self, image_list, label_list, transformer=None):
        self.image_list = image_list
        self.label_list = label_list
        self.transformer = transformer

    def __getitem__(self, index):
        image = np.load(self.image_list[index])
        #         print("image : ",image)
        label = np.load(self.label_list[index])
        # 在这个地方加一个判断看一下标签里面有没有大于1的值
        # 遍历数组，找到不在范围 [0,1] 内的错误值并标记为0（也就是背景的标签）
        arr = label
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if arr[i][j] < 0 or arr[i][j] > 1:
                    arr[i][j] = 0
        # print(arr)
        image_array = image.astype(np.float32)
        image_array = torch.FloatTensor(image_array).unsqueeze(0)

        label_array = arr.astype(np.int16)
        label_array = torch.LongTensor(label_array)
        myDataset_img_path = self.image_list[index]
        myDataset_label_path = self.label_list[index]
        return image_array, label_array, myDataset_img_path, myDataset_label_path

    def __len__(self):
        return len(self.image_list)

def numpy_list(numpy):
    x = []
    numpy_to_list(x, numpy)
    return x


def numpy_to_list(x, numpy):
    for i in range(len(numpy)):
        if type(numpy[i]) is np.ndarray:
            numpy_to_list(x, numpy[i])
        else:
            x.append(numpy[i])


import torch
import torch.nn as nn
import torch.nn.functional as F




# class HybridLoss(nn.Module):
#     def __init__(self, gamma=2, alpha=0.1, eps=1e-7):
#         super(HybridLoss, self).__init__()
#         self.dice_loss = dice_loss()
#         self.focal_loss = FocalLoss(gamma=gamma, eps=eps)
#         self.alpha = alpha  # 权重给交叉熵损失
#         self.beta = 1 - alpha  # 权重给Dice损失
#
#     def forward(self, inputs, targets):
#         loss_focal = self.focal_loss(inputs, targets)
#         loss_dice = self.dice_loss(inputs, targets)
#         # 组合两种损失，按照指定的权重计算最终损失
#         combined_loss = self.alpha * loss_focal + self.beta * loss_dice
#         return combined_loss
class dice_loss(nn.Module):
    def __init__(self, c_num=2):  # 格式需要
        super(dice_loss, self).__init__()

    # 前向传播函数
    def forward(self, data, label):  # 格式需要 data是模型跑出来的结果。
        # data.shape: torch.Size([4, 2, 16, 96, 96]) label.shape: torch.Size([4, 16, 96, 96])
        # n=data.size(0)=4 指 batch_size=4 的值，也就是一个批次几个
        n = data.size(0)
        # 用来放本批次（一个批次是四个数据）中的每一个图的dice值
        dice_list = []
        # 一会 算本批次的平均dice 用
        all_dice = 0.
        # 本批次内，拿一个图出来
        # 把0--（n-1）依次赋值给i
        # 2023.3.6 不明白为什么要这样做？？？
        for i in range(n):
            # my_label11为取得的对应label，也可以说是前景为结节的分割图
            my_label11 = label[i]
            # my_label1为 前景为非结节的分割图，背景为结节   1-1=0，1-0=1，这样就互换了,abs是取绝对值
            my_label1 = torch.abs(1 - my_label11)
            # my_data1为我的模型预测出的 前景为非结节的分割图，data[i].shape: torch.Size([2, 16, 96, 96])  data[i][0]表示的是预测模型里面第一个通道的图像
            my_data1 = data[i][0]
            # my_data11为我的模型预测出的 前景为结节的分割图，data[i][0]表示的是预测模型里面第一个通道的图像（前景为非结节的预测图）data[i][0]前景为结节的预测图
            my_data11 = data[i][1]
            # 把my_data1拉成一维，view是将多维张量转换为一维张量，-1 参数表示维度大小自动计算       ps：前景为非结节的分割图
            m1 = my_data1.view(-1)
            # 把my_label1拉成一维     ps：前景为非结节的label图
            m2 = my_label1.view(-1)
            # 把my_data1拉成一维     ps：前景为结节的分割图
            m11 = my_data11.view(-1)
            # 把my_label1拉成一维   ps：前景为结节的label图
            m22 = my_label11.view(-1)

            dice = 0.  # dice初始化为0
            # dice loss = 1-DSC的公式，比较的是 前景为非结节的分割图
            dice += (1 - ((2. * (m1 * m2).sum() + 1) / (m1.sum() + m2.sum() + 1)))
            # dice loss = 1-DSC的公式，比较的是 前景为结节的分割图
            dice += (1 - ((2. * (m11 * m22).sum() + 1) / (m11.sum() + m22.sum() + 1)))
            # 里面放本批次中的所有图的dice，每张图的dice为 前景结节 和 前景非结节 两图的dice loss 求和
            dice_list.append(dice)
            # print("dice_list:", len(dice_list))
        # 遍历本批次所有图
        for i in range(n):
            all_dice += dice_list[i]  # 求和
        dice_loss = all_dice / n
        # 返回本批次所有图的平均dice loss
        return dice_loss


class FocalLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        ce_loss = self.ce(input, target)
        # p = torch.exp(-logp)
        pt = torch.sigmoid(input)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


import torch
import torch.nn as nn

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, data, label):
        # 获取批次大小
        n = data.size(0)
        # 初始化变量用于存储每个图像的 Tversky index
        tversky_list = []
        # 用于计算批次的平均 Tversky index
        all_tversky = 0.

        for i in range(n):
            # 前景为结节的真实分割图和预测图
            my_label11 = label[i]
            my_data11 = data[i][1]
            # 前景为非结节的真实分割图和预测图
            my_label1 = torch.abs(1 - my_label11)
            my_data1 = data[i][0]

            # 展平前景为结节和非结节的标签和预测
            m1 = my_data1.view(-1)
            m2 = my_label1.view(-1)
            m11 = my_data11.view(-1)
            m22 = my_label11.view(-1)

            # 计算每个类别的 Tversky index
            TP1 = (m1 * m2).sum()
            FP1 = ((1 - m2) * m1).sum()
            FN1 = (m2 * (1 - m1)).sum()

            TP2 = (m11 * m22).sum()
            FP2 = ((1 - m22) * m11).sum()
            FN2 = (m22 * (1 - m11)).sum()

            # Tversky formula per class
            tversky_index1 = (TP1 + 1) / (TP1 + self.alpha * FP1 + self.beta * FN1 + 1)
            tversky_index2 = (TP2 + 1) / (TP2 + self.alpha * FP2 + self.beta * FN2 + 1)

            # 平均 Tversky index
            tversky = (1 - tversky_index1) + (1 - tversky_index2)
            tversky_list.append(tversky)

        # 计算整个批次的平均 Tversky index
        for tversky_score in tversky_list:
            all_tversky += tversky_score
        average_tversky = all_tversky / n

        return average_tversky

class HybridLoss(nn.Module):
    def __init__(self, gamma=0, alpha=0.5, beta=0.5, tversky_alpha=0.3, tversky_beta=0.7, eps=1e-7):
        super(HybridLoss, self).__init__()
        self.dice_loss = dice_loss()
        self.focal_loss = FocalLoss(gamma=gamma, eps=eps)
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.alpha = alpha  # 权重给Focal损失
        self.beta = beta  # 权重给Dice损失
        self.gamma = 1 - alpha - beta  # 权重给Tversky损失

    def forward(self, inputs, targets):
        loss_focal = self.focal_loss(inputs, targets)
        loss_dice = self.dice_loss(inputs, targets)
        loss_tversky = self.tversky_loss(inputs, targets)
        combined_loss = self.alpha * loss_focal + self.beta * loss_dice + self.gamma * loss_tversky
        return combined_loss


# --------------------损失函数----------------------------------------
# 损失函数布置到gpu或cpu上
# # 实例化dice_loss并将其放到GPU上（收敛的太慢，而且验证集的loss值动荡的太了厉害了）
# Loss = dice_loss().to(DEVICE)
# 损失函数使用Focal loss函数是根据CEL改进的（运行老是出错，至今没有解决。。。。）
# Loss = FocalLoss().to(DEVICE)
# 损失函数使用CrossEntropyLoss()交叉熵函数（运行也出错，跟上面的Focal loss函数的错误一样，感觉是这个训练脚本有问题 ）
# Loss = nn.CrossEntropyLoss().to(DEVICE)
# image, output都是tensor类型，
def use_plot_2d(image, output, name, i=0, true_label=False):
    for j in range(image.shape[0]):
        # 绘制一张画板，plt.figure()
        # plt.figure()
        # 96*96     这是归一化后的
        p = image[j, :, :] + 0.25
        p = torch.unsqueeze(p, dim=2)
        # 96*96
        q = output[j, :, :]
        q = (q * 0.2).float()
        q = torch.unsqueeze(q, dim=2)
        q = p + q
        # 将q数组中，所有大于1的元素值都赋值为1。
        q[q > 1] = 1
        r = p
        # 将名为r、q、p的三个张量在第二个维度上（即高度维度）拼接起来，组成一个新的张量cat_pic。
        cat_pic = torch.cat([r, q, p], dim=2)  # 红色为空，my_label为绿色，原图为蓝色
        plt.imshow(cat_pic)

        path = zhibiao_path  # 我真的懒得引入参数了，这个path 就是 zhibiao_path

        if true_label:
            save_label = path + fengefu + 'true_pic'
            if not os.path.exists(save_label):  # 建立subset文件夹
                os.mkdir(save_label)
            plt.savefig(save_label + fengefu + name + '_%d_%d.jpg' % (i, j))
        else:
            save_pred = path + fengefu + 'pic'
            if not os.path.exists(save_pred):  # 建立subset文件夹
                os.mkdir(save_pred)
            plt.savefig(save_pred + fengefu + name + '_%d_%d.jpg' % (i, j))
        plt.close()

