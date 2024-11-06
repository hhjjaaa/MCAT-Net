import os
import torch.utils.data
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
from global_ import  output_path



class myDataset(Dataset):
    def __init__(self, image_list, label_list, transformer=None):
        self.image_list = image_list
        self.label_list = label_list
        self.transformer = transformer

    def __getitem__(self, index):
        image = np.load(self.image_list[index])
        #         print("image : ",image)
        label = np.load(self.label_list[index])

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



import torch.nn as nn






class dice_loss(nn.Module):
    def __init__(self, c_num=2):  # 格式需要
        super(dice_loss, self).__init__()

    def forward(self, data, label):

        n = data.size(0)

        dice_list = []

        all_dice = 0.

        for i in range(n):

            my_label11 = label[i]

            my_label1 = torch.abs(1 - my_label11)

            my_data1 = data[i][0]

            my_data11 = data[i][1]

            m1 = my_data1.view(-1)

            m2 = my_label1.view(-1)

            m11 = my_data11.view(-1)

            m22 = my_label11.view(-1)

            dice = 0.

            dice += (1 - ((2. * (m1 * m2).sum() + 1) / (m1.sum() + m2.sum() + 1)))

            dice += (1 - ((2. * (m11 * m22).sum() + 1) / (m11.sum() + m22.sum() + 1)))

            dice_list.append(dice)


        for i in range(n):
            all_dice += dice_list[i]
        dice_loss = all_dice / n

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
        tversky_list = []
        all_tversky = 0.

        for i in range(n):

            my_label11 = label[i]
            my_data11 = data[i][1]

            my_label1 = torch.abs(1 - my_label11)
            my_data1 = data[i][0]


            m1 = my_data1.view(-1)
            m2 = my_label1.view(-1)
            m11 = my_data11.view(-1)
            m22 = my_label11.view(-1)


            TP1 = (m1 * m2).sum()
            FP1 = ((1 - m2) * m1).sum()
            FN1 = (m2 * (1 - m1)).sum()

            TP2 = (m11 * m22).sum()
            FP2 = ((1 - m22) * m11).sum()
            FN2 = (m22 * (1 - m11)).sum()


            tversky_index1 = (TP1 + 1) / (TP1 + self.alpha * FP1 + self.beta * FN1 + 1)
            tversky_index2 = (TP2 + 1) / (TP2 + self.alpha * FP2 + self.beta * FN2 + 1)


            tversky = (1 - tversky_index1) + (1 - tversky_index2)
            tversky_list.append(tversky)


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
        self.alpha = alpha
        self.beta = beta
        self.gamma = 1 - alpha - beta

    def forward(self, inputs, targets):
        loss_focal = self.focal_loss(inputs, targets)
        loss_dice = self.dice_loss(inputs, targets)
        loss_tversky = self.tversky_loss(inputs, targets)
        combined_loss = self.alpha * loss_focal + self.beta * loss_dice + self.gamma * loss_tversky
        return combined_loss



def use_plot_2d(image, output, name, i=0, true_label=False):
    for j in range(image.shape[0]):
        p = image[j, :, :] + 0.25
        p = torch.unsqueeze(p, dim=2)
        # 96*96
        q = output[j, :, :]
        q = (q * 0.2).float()
        q = torch.unsqueeze(q, dim=2)
        q = p + q
        q[q > 1] = 1
        r = p
        cat_pic = torch.cat([r, q, p], dim=2)
        plt.imshow(cat_pic)

        path = output_path

        if true_label:
            save_label = path + '/' + 'true_pic'
            if not os.path.exists(save_label):
                os.mkdir(save_label)
            plt.savefig(save_label + '/' + name + '_%d_%d.jpg' % (i, j))
        else:
            save_pred = path + '/' + 'pic'
            if not os.path.exists(save_pred):
                os.mkdir(save_pred)
            plt.savefig(save_pred + '/' + name + '_%d_%d.jpg' % (i, j))
        plt.close()

