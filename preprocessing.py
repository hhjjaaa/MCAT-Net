from global_ import *
import os
import numpy as np
from tqdm import tqdm

from global_annos import annos_list

Path(preprocessing_img_path).mkdir(exist_ok=True,parents=True)
Path(preprocessing_msk_path).mkdir(exist_ok=True,parents=True)


def get_img_label(data_path):  ###  list 地址下所有图片的绝对地址

    img_path = []
    for t in data_path:  ###  打开subset0，打开subset1
        data_img_list = os.listdir(t)  ## 列出图
        img_path += [os.path.join(t, j) for j in
                     data_img_list]  ##'/public/home/menjingru/dataset/sk_output/bbox_image/subset1/1.3.6.1.4.1.14519.5.2.1.6279.6001.104562737760173137525888934217.npy'
    img_path.sort()
    return img_path  ##返回的也就是图像路径 或 标签路径


def get_annos_label(img_path):
    annos_path = []  # 这里边要装图的地址，结节的中心，结节的半径    要小于96/4 # ###半径最大才12


    for u in img_path:  # 图的路径
        if xitong == "windows":
            name = '1' + u.split(r"\1")[-1].split(".np")[0]  # 拿到图的名字
        else:
            name = u.split(r"/")[-1].split(".np")[0]  # 拿到图的名字
        for one in annos_list:  # 遍历有结节的图
            if one[0] == name:  # 如果有结节的图的名字 == 输入的图的名字
                for l in range(len(one[1])):  # 数一数有几个结节
                    annos_path.append(
                        [u, [one[1][l][0], one[1][l][1], one[1][l][2]], one[1][l][3]])  # 图的地址，结节的中心，结节的半径
    return annos_path  # ###半径最大才12


def preprocessing(data_path, label_path):
    data = get_img_label(data_path)  ## 图的位置列表
    label = get_img_label(label_path)  ## 标签的位置列表

    annos_img = get_annos_label(data)  # 图的位置列表 输入进去  吐出  结节附近的图的【【图片位置，结节中心，半径】列表】
    annos_label = get_annos_label(label)  # 112

    for index in tqdm(range(len(annos_img))):
        cut_list = []  ##  切割需要用的数
        img_all = annos_img[index]
        label_all = annos_label[index]
        img = np.load(img_all[0])  # 载入的是图片地址
        label = np.load(label_all[0])
        for i in range(len(img.shape)):  ###  0,1,2   →  z,y,x
            if i == 0:
                a = img_all[1][-i - 1] - 8  ### z
                b = img_all[1][-i - 1] + 8
            else:
                a = img_all[1][-i - 1] - 128  ### z
                b = img_all[1][-i - 1] + 128  ###
            if a < 0:
                if i == 0:
                    a = 0
                    b = 256
                else:
                    a = 0
                    b = 256
            elif b > img.shape[i]:
                if i == 0:
                    a = img.shape[i] - 16
                    b = img.shape[i]
                else:
                    a = img.shape[i] - 256
                    b = img.shape[i]
            else:
                pass

            cut_list.append(a)
            cut_list.append(b)

        img = img[cut_list[0]:cut_list[1], cut_list[2]:cut_list[3], cut_list[4]:cut_list[5]]  # z,y,x
        label = label[cut_list[0]:cut_list[1], cut_list[2]:cut_list[3], cut_list[4]:cut_list[5]]  # z,y,x

        img_filename = '{}_{}.npy'.format(Path(img_all[0]).stem, index)
        img_save_path = Path(preprocessing_img_path) / img_filename
        np.save(str(img_save_path), img)

        label_filename = '{}_{}.npy'.format(Path(label_all[0]).stem, index)
        label_save_path = Path(preprocessing_msk_path) / label_filename
        np.save(str(label_save_path), label)



######       数据准备
print('数据准备开始...')
data_path = []  # 装图所在subset的绝对地址，如 [D:\datasets\sk_output\bbox_image\subset0,D:\datasets\sk_output\bbox_image\subset1,..]
label_path = []  # 装标签所在subset的绝对地址，与上一行一致，为对应关系
for i in range(0,8):#8):  # 0,1,2,3,4,5,6,7   训练集
    data_path.append(bbox_img_path+fengefu+'subset%d' % i)  # 放入对应的训练集subset的绝对地址
    label_path.append(bbox_msk_path+fengefu+'subset%d' % i)

dataset_train = preprocessing(data_path, label_path)  # 送入dataset
print("train_preprocessing_ok")



data_valid_path = []  # 装图所在subset的绝对地址
label_valid_path = []  # 装标签所在subset的绝对地址
for j in range(8,9):  # 8   验证集
    data_valid_path.append(bbox_img_path+fengefu+'subset%d' % j)  # 放入对应的验证集subset的绝对地址
    label_valid_path.append(bbox_msk_path+fengefu+'subset%d' % j)
dataset_valid = preprocessing(data_valid_path, label_valid_path)  # 送入dataset
print("valid_preprocessing_ok")

data_test_path = []  # 装图所在subset的绝对地址
label_test_path = []  # 装标签所在subset的绝对地址
for ii in range(9,10):  # 9   测试集
    data_test_path.append(bbox_img_path+fengefu+'subset%d' % ii)  # 放入对应的测试集subset的绝对地址
    label_test_path.append(bbox_msk_path+fengefu+'subset%d' % ii)
dataset_test = preprocessing(data_test_path, label_test_path)  # 送入dataset
print("Test_preprocessing_ok")

