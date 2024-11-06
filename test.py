import pathlib
import warnings
import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from global_ import *
from test_def import *
import torch.utils.data
import pandas as pd
from tqdm import tqdm
import glob

from MCAT_net import MCAT_Net

# ————————————————————————————————————————————————————————————

BATCH_SIZE = 128
model = MCAT_Net(1, 2)


DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)

torch.cuda.empty_cache()
print(f"Using device: {DEVICE}")

class testDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.annos_img = data_path
        self.annos_label = label_path

    def __getitem__(self, index):
        img_path = self.annos_img[index]
        label_path = self.annos_label[index]
        img_filename = os.path.basename(img_path)  # Extract filename

        if not isinstance(img_path, pathlib.PosixPath):
            img_path = pathlib.Path(img_path)
        if not isinstance(label_path, pathlib.PosixPath):
            label_path = pathlib.Path(label_path)

        if not img_path.exists() or not label_path.exists():
            raise FileNotFoundError(f"Image or label file not found at {img_path} or {label_path}")

        img = np.load(img_path)
        label = np.load(label_path)
        label[label == 2] = 1

        img = np.expand_dims(img, 0)
        img = torch.tensor(img, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return img, label, img_filename  # Now also returns the filename

    def __len__(self):
        return len(self.annos_img)
def calculate_hd(set1, set2):

    coords1 = set1.nonzero(as_tuple=True)
    coords2 = set2.nonzero(as_tuple=True)


    u = np.stack([c.cpu().numpy() for c in coords1], axis=-1)
    v = np.stack([c.cpu().numpy() for c in coords2], axis=-1)


    hd1 = directed_hausdorff(u, v)[0]
    hd2 = directed_hausdorff(v, u)[0]

    return max(hd1, hd2)

# —————————————————————————————————————————————————————————————
def use_plot_2d2(image, output, batch_index=0, i=0, true_label=False,
                 path=output_path, filename='image'):
    plt.figure()

    if true_label:

        p = image + 0.25
        p = torch.unsqueeze(p, dim=2)
        q = output * 0.4
        q = torch.unsqueeze(q.float(), dim=2)
        q = p + q
        q[q > 1] = 1
        r = p
        cat_pic = torch.cat([r, q, p], dim=2)
        plt.imshow(cat_pic.numpy(), cmap='gray')
    else:

        plt.imshow(output.numpy(), cmap='gray')

    directory = 'true_pic' if true_label else 'pic'
    full_path = os.path.join(path, directory)
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    # Strip any existing extension and add '.jpg'
    filename = os.path.splitext(filename)[0] + '.jpg'

    plt.savefig(os.path.join(full_path, filename), format='jpg')
    plt.close()
def test(model, testloader):
    test_correct = 0
    test_total = 0
    test_running_loss = 0
    epoch_test_iou = []
    epoch_test_dice = []
    epoch_test_sensi = []
    epoch_test_ppv = []
    epoch_test_hd = []

    model.eval()

    with torch.no_grad():
        for x, y, path in tqdm(testloader, position=0):

            length_y = len(torch.unique(y))
            if length_y != 2:
                y[y > 1] = 0
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)  # y.size(0) == batchsize  test_total = test_total+y.size(0)
            test_running_loss += loss.item()

            intersection = torch.logical_and(y, y_pred)
            union = torch.logical_or(y, y_pred)

            batch_iou = torch.sum(intersection) / max(torch.sum(union),1e-7)
            epoch_test_iou.append(batch_iou.item())

            batch_sensi = torch.sum(intersection) / max(torch.sum(y), 1e-7)
            epoch_test_sensi.append(batch_sensi.item())

            batch_ppv = torch.sum(intersection) / max(torch.sum(y_pred),1e-7)
            epoch_test_ppv.append(batch_ppv.item())

            batch_dice = 2 * torch.sum(intersection) / max((torch.sum(intersection) + torch.sum(union)),1e-7 ) # 计算每张图片的dice值 2TP/(2TP+FP+FN)   22-1022
            epoch_test_dice.append(batch_dice.item())

            batch_test_hd = calculate_hd(y, y_pred)
            epoch_test_hd.append(batch_test_hd)



    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / (test_total * 96 * 96)

    print(
          '\ttest_loss: ', round(epoch_test_loss, 4),
          '\ttest_accuracy:', round(epoch_test_acc, 4),
          '\ttest_iou:', round(np.mean(epoch_test_iou), 4),
          '\ttest Dice:', round(np.mean(epoch_test_dice), 4),
          '\ttest sensitivity: ', round(np.mean(epoch_test_sensi), 4),
          '\tepoch_test_ppv: ', round(np.mean(epoch_test_ppv), 4),
          '\ttest HD:', round(np.mean(epoch_test_hd), 4)
          )
    with torch.no_grad():
        for batch_index, (data, labels, filenames) in enumerate(test_loader):
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            output = model(data)

            for i, (img_tensor, label_tensor, filename) in enumerate(zip(data.cpu(), labels.cpu(), filenames)):
                output_tensor = output[i].cpu()
                predicted_label_tensor = torch.gt(output_tensor[1], output_tensor[0]).long()  # 转换为长整型


                use_plot_2d2(img_tensor.squeeze(0), label_tensor, batch_index=batch_index, i=i, true_label=True,
                             filename=f"true_{filename}")
                use_plot_2d2(img_tensor.squeeze(0), predicted_label_tensor, batch_index=batch_index, i=i,
                             filename=f"pred_{filename}")
                print("A prediction image is being drawn{}".format(filename))
    return epoch_test_loss, epoch_test_acc, np.mean(epoch_test_iou), np.mean(epoch_test_dice), np.mean(epoch_test_sensi), np.mean(epoch_test_ppv) , epoch_test_hd


if __name__ == '__main__':
    loss_fn = dice_loss().to(DEVICE)
    model = MCAT_Net(1, 2)
    model.to(DEVICE)


    state_dict = torch.load(best_dice_path, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()


    data_test_path = []
    label_test_path = []
    test_data = bbox_img_path + '/'
    test_data_dirs = glob.glob(test_data + 'subset_test*')  # +'*')
    for subset in test_data_dirs:
        for filename in os.listdir(subset):
            if filename.endswith(".npy") is True:
                img_path = subset + '/' + filename
                data_test_path.append(img_path)
    print('Total number of test images :', len(data_test_path))
    data_test_path.sort()

    test_label = bbox_msk_path + '/'  # 'subset%d' % i
    test_label_dirs = glob.glob(test_label + 'subset_test*')  # +'*')
    for subset in test_label_dirs:
        for filename in os.listdir(subset):
            if filename.endswith(".npy") is True:
                label_path = subset + '/' + filename
                label_test_path.append(label_path)
    print('Total number of test images :', len(label_test_path))
    label_test_path.sort()

    dataset_test = testDataset(data_test_path, label_test_path)
    test_loader = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=64)
    print("Test_dataloader_ok")

    # ——————————————————————————————————————————————————————————————————————————————————————————————————————
    epoch_test_loss, epoch_test_acc, epoch_test_iou, epoch_test_dice, epoch_test_sensi, epoch_test_ppv , epoch_test_hd = test(model, test_loader)
    tocsv_list = [epoch_test_loss, epoch_test_acc, epoch_test_iou, epoch_test_dice, epoch_test_sensi, epoch_test_ppv , epoch_test_hd]
    data = pd.DataFrame([tocsv_list])
    data.to_csv(str(output_path) + "/process_log.csv", mode='a', header=False)

