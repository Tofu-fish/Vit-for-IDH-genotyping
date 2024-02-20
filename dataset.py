#-*-coding=utf-8-*-

import json
import os
import random
from glob import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from augment import distort, perspective, stretch
from aug import (blur, flip, light, noise, rotate_warpAffine,
                      rotate_warpPerspective)

def nonefunc(image, *args, **kwargs):
    return image

#funcs_1 = [distort, stretch, perspective, noise, flip, nonefunc]#blur, light,
#funcs_2 = [rotate_warpPerspective, rotate_warpAffine, nonefunc]

#ClsDict = {'codel':0, 'intact':0, 'Wildtype':1}
ClsDict = {'IDH_mut':0, 'IDH_WT':1}

class ImageDataset(Dataset):
    """
    用于加载图片
    """
    def __init__(self, rootDir, transform=None, is_train=True, is_eval=False):
        """
        Args:
            rootDir: 图片路径根目录
            transform: 图片预处理函数
            is_train: 是否处于训练模式
        """

        self.root = rootDir
        self.transform = transform

        # 获取图片名称
        if not is_eval:
            self.image_names = glob(f"{self.root}/codel/*/FLAIR/*.png")*8
            self.image_names.extend(glob(f"{self.root}/intact/*/FLAIR/*.png")*3)
            self.image_names.extend(glob(f"{self.root}/Wildtype/*/FLAIR/*.png"))
        else:
            self.image_names = glob(f"{self.root}/*/*/FLAIR/*.png")

        self.is_train = is_train
        self.is_eval = is_eval

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_name = image_name.replace('\\', '/')
        flairImg = cv2.imdecode(np.fromfile(image_name, dtype=np.uint8), cv2.IMREAD_COLOR)
        t2wImg = cv2.imdecode(np.fromfile(image_name.replace('FLAIR','T2W_m'), dtype=np.uint8), cv2.IMREAD_COLOR)
        t1wImg = cv2.imdecode(np.fromfile(image_name.replace('FLAIR','T1W_m'), dtype=np.uint8), cv2.IMREAD_COLOR)
        t1wcImg = cv2.imdecode(np.fromfile(image_name.replace('FLAIR','T1WC_m'), dtype=np.uint8), cv2.IMREAD_COLOR)

        # segImg = cv2.imdecode(np.fromfile(image_name.replace('FLAIR','seg'), dtype=np.uint8), cv2.IMREAD_COLOR)

        # if self.is_train and (not self.is_eval):
        #     comms = random.sample([flip, rotate_warpAffine, nonefunc], random.randint(1,3))
        #     for comm in comms:
        #         [flairImg, t2wImg], _ = comm([[flairImg, t2wImg], np.array([[0,0],[0,0]])], ifPoint=True, ifMutil=True)

        if self.is_train:
            target = np.array(ClsDict[image_name.split('/')[-4]])

        # temp = np.where(segImg==[255,255,255])
        # x1, x2 = min(temp[1]), max(temp[1])
        # y1, y2 = min(temp[0]), max(temp[0])
        # H, W = segImg.shape[:2]
        # x1 = max(x1 - 50, 0)
        # y1 = max(y1 - 50, 0)
        # x2 = min(x2 + 50, W)
        # y2 = max(y2 + 50, H)

        # nim = np.zeros(flairImg.shape, dtype=np.uint8)
        # 进行预处理
        if self.transform:
            # nim[y1:y2, x1:x2] = flairImg[y1:y2, x1:x2]
            flairImg = self.transform(flairImg)

            # nim[y1:y2, x1:x2] = t2wImg[y1:y2, x1:x2]
            t2wImg = self.transform(t2wImg)

            # nim[y1:y2, x1:x2] = t1wImg[y1:y2, x1:x2]
            t1wImg = self.transform(t1wImg)

            # nim[y1:y2, x1:x2] = t1wcImg[y1:y2, x1:x2]
            t1wcImg = self.transform(t1wcImg)

        # im_inp = torch.stack([flairImg, t1wImg], dim=0)

        if self.is_train:
            return [flairImg, t2wImg, t1wImg, t1wcImg], target
        else:
            return [flairImg, t2wImg, t1wImg, t1wcImg]



if __name__ == '__main__':
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    from PIL import Image

    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224), interpolation=InterpolationMode.BICUBIC),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
            )
    dataset = ImageDataset('./MAE_data/classification/train', transform=transform)
    print(len(dataset))

    data_loader_train = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    for img_inp, target in data_loader_train:
        print(len(img_inp), target.shape)
        print(img_inp[0].shape, img_inp[1].shape)

        # img_inp_sum = img_inp[0]+img_inp[1]
        # print(img_inp_sum.shape)
        transforms.ToPILImage()(img_inp[0][0]).save('test0.jpg')
        transforms.ToPILImage()(img_inp[1][0]).save('test1.jpg')
        exit()

    # import time
    # start = time.time()
    # print(len(dataset))
    # for i in tqdm(range(len(dataset))):
    #     image, target = dataset[i]
    #     print(image.shape, target)
    #     exit()
    # end = time.time()
    # print(end - start)
