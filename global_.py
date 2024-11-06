# 修改区域

luna_path = "/home/hutianjiao/datasets/LUNA16"
xml_file_path = "/home/hutianjiao/datasets/LIDC-IDRI/LIDC-XML-only/tcia-lidc-xml"
annos_csv = "/home/hutianjiao/datasets/LUNA16/CSVFILES/annotations.csv"
new_bbox_annos_path = "/home/hutianjiao/datasets/sk_output/bbox_annos/bbox_annos.xlsx"
mask_path = "/home/hutianjiao/datasets/LUNA16/seg-lungs-LUNA16"
output_path = "/home/hutianjiao/datasets/sk_output"


bbox_img_path3 = "/bestmodel_data/testdatas/image" # bbox_image
bbox_msk_path3 = "/bestmodel_data/testdatas/label" # bbox_mask

bbox_img_path4 = "/home/hutianjiao/datasets/sk_output/LUNBimage/"
bbox_msk_path4 ="/home/hutianjiao/datasets/sk_output/LUNBlabel/"

preprocessing_img_path = "/home/hutianjiao/datasets/sk_output/preprocessing_image"
preprocessing_msk_path = "/home/hutianjiao/datasets/sk_output/preprocessing_mask"
wrong_img_path = "/home/hutianjiao/datasets/wrong_img.xls"
zhibiao_path = "/home/hutianjiao/datasets/sk_output/zhibiao"
model_path = "/home/hutianjiao/datasets/sk_output/model"

board_logdir="/home/hutianjiao/datasets/sk_output/zhibiao2"
log_dir_path="/home/hutianjiao/datasets/sk_output/zhibiao2"
best_loss_model_path="/home/hutianjiao/datasets/sk_output/zhibiao2"
best_dice_model_path="/home/hutianjiao/datasets/sk_output/zhibiao2"
iou_dice_path="/home/hutianjiao/datasets/sk_output/zhibiao2"
point_dir="/home/hutianjiao/datasets/sk_output/zhibiao2"


from pathlib import Path
zhibiao_path1 = Path(zhibiao_path)
zhibiao_path1.mkdir(exist_ok=True,parents=True)

xitong = "linux"  # "linux"


# 训练设置
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 没gpu就用cpu
valid_epoch_each = 1  # 每几轮验证一次

if xitong == "linux":
    fengefu = r"/"
else:
    fengefu = r"\\"




