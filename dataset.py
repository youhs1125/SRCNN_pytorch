import os
from PIL import Image
import numpy as np
import cv2
import torch
import torch.utils.data as td

def getImageFiles(path):
    img_path = []

    for p in path:
        for (root, directories, files) in os.walk(p):
            for file in files:
                file_path = os.path.join(root,file)
                img_path.append(file_path)

    file = []
    for i in range(len(img_path)):
        img = Image.open(img_path[i])
        img = np.array(img, dtype=np.float32)
        img /= 255.
        file.append(img)

    print("images: ",len(file))
    return file

def DataAugmentation(file):
    stride = 14
    img_size = 32

    crops = []
    for i in range(len(file)):
        h_thres = int((file[i].shape[0] - img_size) / stride) + 1
        w_thres = int((file[i].shape[1] - img_size) / stride) + 1
        for h in range(h_thres):
            for w in range(w_thres):
                idxh = h * stride
                idxw = w * stride
                crop_img = file[i][idxh:(idxh + img_size), idxw:(idxw + img_size)]
                crops.append(crop_img)

    print(len(crops))
    crops = np.array(crops)

    print("Augmentation Finished size:",len(crops))
    return crops

def downsampling(file, isTest=False):
    ds = []
    for i in range(len(file)):
        img_W,img_H = file[i].shape[0], file[i].shape[1]
        temp = cv2.GaussianBlur(file[i], (0, 0), 1)
        temp = cv2.resize(temp, dsize=(img_H//2, img_W//2), interpolation=cv2.INTER_CUBIC)
        ds.append(cv2.resize(temp, dsize=(img_H, img_W), interpolation=cv2.INTER_CUBIC))

    if isTest == False:
        ds = np.array(ds)
    return ds

def changeColorChannelLocation(file1,file2):

    data = np.reshape(file1, (file1.shape[0], file1.shape[-1], file1.shape[1], file1.shape[2]))
    target = np.reshape(file2, (file2.shape[0], file2.shape[-1], file2.shape[1], file2.shape[2]))

    return data, target

def getDataset():
    path = []

    path.append("Images/T91")

    file = getImageFiles(path)
    tfile = DataAugmentation(file)
    dfile = downsampling(tfile)


    target, data = changeColorChannelLocation(tfile,dfile)

    # define dataset
    target = torch.from_numpy(target)
    data = torch.from_numpy(data)

    dataset = td.TensorDataset(data, target)

    # split train, validation

    train_val_ratio = 0.8

    train_size = int(data.shape[0] * train_val_ratio)
    val_size = data.shape[0] - train_size

    train_data, val_data = td.random_split(dataset, [train_size, val_size])

    # define dataloader

    train_dataloader = td.DataLoader(train_data, batch_size=64, shuffle=True)
    val_dataloader = td.DataLoader(val_data, batch_size=64, shuffle=False)
    print(len(train_dataloader), len(val_dataloader))

    return train_dataloader, val_dataloader

def getTestData():
    path = []

    path.append("Images/Set5")
    path.append("Images/Set14")

    data = getImageFiles(path)
    target = downsampling(data,isTest=True)

    return data, target
