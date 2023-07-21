import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import random
import torch
import warnings
warnings.filterwarnings("ignore")

from glob import glob
from torch.utils.data import DataLoader, TensorDataset
from MDINet2c import MDINet2c

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def set_randomSeed(SEED=11):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def load_data(index, imgs_path):
    imgs = {}
    for i in range(len(imgs_path)):
        num = str(imgs_path[i][-8:-4])
        if num in index:
            if num not in imgs:
                imgs[num] = []
            img = np.load(imgs_path[i])
            img = img / 255.0
            imgs[num].append(img)
    return imgs

def generate_data(imgs, index):
    train_imgs = []
    for key in index:
        for i in range(len(imgs[key])):
            train_imgs.append(imgs[key][i])
    train_imgs = np.array(train_imgs)
    return train_imgs

def test(val_loader, model, fold):
    set_randomSeed()
    preds_list = []

    model.load_state_dict(torch.load('./model_10_pth/' + str('fold_') + str(fold+1) + '.pth', map_location=torch.device('cpu')))

    device = torch.device('cpu')
    model.to(device)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            inputs = batch[0].to(device)
            inputs.to(device)
            model.eval()
            preds = model(inputs)

            preds = preds.data.cpu().numpy()

            preds[preds > 0.5] = 1
            preds[preds != 1] = 0
            preds_list.append(int(preds[0][0]))
    return preds_list


f = open('./predict.csv', 'w')
writer = csv.writer(f)

header = ['Nodule', 'predict 1', 'predict 2', 'predict 3', 'predict 4', 'predict 5', 
          'predict 6', 'predict 7', 'predict 8', 'predict 9', 'predict 10', 'Diagnostic result']

writer.writerow(header)

imgs_path = glob(f'./texture_images/*.npy')
imgs_path.sort()

val_index = []
for i in range(len(imgs_path)): 
    val_index.append(str(imgs_path[i][-8:-4]))

imgs = load_data(val_index, imgs_path)

pred_all = []

for pth_number in range(10):
    val_imgs = generate_data(imgs, val_index)     
    val_imgs = torch.FloatTensor(val_imgs)
    set_randomSeed()
    
    val_set = TensorDataset(val_imgs)
    val_loader = DataLoader(val_set, shuffle=False)
    pred_all.append(test(val_loader=val_loader, model=MDINet2c(2, 1), fold=pth_number))

for i in range(len(imgs_path)): 
    result = 'Benign'
    check = pred_all[0][i] + pred_all[1][i] + pred_all[2][i] + pred_all[3][i] + pred_all[4][i] + pred_all[5][i] + pred_all[6][i] + pred_all[7][i] + pred_all[8][i] + pred_all[9][i]
    if check > 5: 
        result = 'Malignant'
    elif check == 5: 
        result = 'Indeterminate'
    
    row = [val_index[i] + str('.npy'), pred_all[0][i], pred_all[1][i], pred_all[2][i], pred_all[3][i], pred_all[4][i], pred_all[5][i], 
           pred_all[6][i], pred_all[7][i], pred_all[8][i], pred_all[9][i], result]
    writer.writerow(row)

f.close()
