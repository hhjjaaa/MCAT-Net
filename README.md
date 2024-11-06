# MCAT-Net
Pulmonary nodule segmentation model best_model can download from : https://pan.baidu.com/s/1tcItqTRJwl5_xO4jEi_v8w Extract code: r3ux

The best trained models and test datasets can also be obtained from Google Cloud Drive:https://drive.google.com/drive/folders/1i8K2j-6mWsKl-ZW3ZeVvna7BTMLHeU2V


Change the following address in the global.py file.
bbox_img_path = "/home/hutianjiao/Project/Test_data/testdatas/image/" # Address of the test image(image)
bbox_msk_path = "/home/hutianjiao/Project/Test_data/testdatas/label/" # Address of the test image(label)
best_dice_path="/home/hutianjiao/Project/best_dice.pth" #The address of the best models
output_path="/home/hutianjiao/MCAT_NET/output/" #The output address

Then run test.py (using GPU number ‘1’ by default)
