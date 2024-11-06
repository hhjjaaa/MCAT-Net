# 修改区域
from pathlib import Path

bbox_img_path = "/home/hutianjiao/Project/Test_data/testdatas/image/" # bbox_image
bbox_msk_path = "/home/hutianjiao/Project/Test_data/testdatas/label/" # bbox_mask
best_dice_path="/home/hutianjiao/Project/best_dice.pth"
output_path="/home/hutianjiao/MCAT_NET/output/"
output_path = Path(output_path)
output_path.mkdir(exist_ok=True,parents=True)








