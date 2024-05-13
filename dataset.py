import glob
import os.path as osp
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data


def populate_train_list(lowlight_images_path):
    image_list_lowlight = glob.glob(lowlight_images_path + "\\*.jpg")
    train_list = image_list_lowlight
    random.shuffle(train_list)

    return train_list


def opencv_to_torch(src: np.ndarray) -> torch.Tensor:
    data_norm = (src / 255.0).astype(np.float32)
    data_torch = torch.from_numpy(data_norm)
    if data_torch.dim() == 2:
        # deal with single channel image
        data_torch = data_torch.unsqueeze(-1)
    data_torch = data_torch.permute(2, 0, 1)
    return data_torch


class My_DataLoader(data.Dataset):
    def __init__(self, opt):
        self.train_list = populate_train_list(opt['image_path'])
        self.size = 256

        self.data_list = self.train_list
        print("Total training examples:", len(self.train_list))

    def __getitem__(self, index):
        img_lowlight_path = self.data_list[index]

        img_lowlight = cv2.cvtColor(cv2.imread(img_lowlight_path), cv2.COLOR_BGR2RGB)
        img_lowlight = cv2.resize(img_lowlight, (self.size, self.size), interpolation=cv2.INTER_LANCZOS4)

        img_lowlight_hsv = cv2.cvtColor(img_lowlight, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(img_lowlight_hsv)

        data_lowlight = opencv_to_torch(img_lowlight)
        data_h = opencv_to_torch(h)
        data_s = opencv_to_torch(s)
        data_v = opencv_to_torch(v)
        data_v_bar = opencv_to_torch(255 - v)

        return {"input": data_lowlight,
                "input_h": data_h,
                "input_s": data_s,
                "input_v": data_v,
                "input_v_bar": data_v_bar
                }

    def __len__(self):
        return len(self.data_list)