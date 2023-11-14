import random

import cv2
import torch
from torch.utils.data import Dataset

from datasets.srm import setup_srm_layer
from utils.util import read_json


class FFpp(Dataset):
    def __init__(self, json_path, transforms, srm_prob=0.0):
        super(FFpp, self).__init__()
        # item=(img_path, label, multi_label, video_id, frame_id)
        self.items = read_json(json_path)['data']
        self.transforms = transforms
        self.srm_prob = None
        if srm_prob != 0:
            self.srm_conv = setup_srm_layer(input_channels=3)
            self.srm_prob = srm_prob

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        img = cv2.cvtColor(cv2.imread(item['img_path']), cv2.COLOR_BGR2RGB)
        img = self.transforms(image=img)['image']
        if self.srm_prob is not None and random.random() < self.srm_prob:
            img = img.view(1, img.shape[0], img.shape[1], img.shape[2])
            img = self.srm_conv(img)
            img = img.squeeze()
        return img, item['label']



def get_dataloader(dataset, batch_size):
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=12,
                                             pin_memory=True,
                                             drop_last=True,
                                             )
    return dataloader


class ConFFpp(Dataset):
    def __init__(self, json_path, transforms, srm_prob=0.):
        super(ConFFpp, self).__init__()
        # item=(img_path, label, multi_label, video_id, frame_id)
        self.items = read_json(json_path)['data']
        self.transforms = transforms
        self.srm_prob = None
        if srm_prob != 0:
            self.srm_conv = setup_srm_layer(input_channels=3)
            self.srm_prob = srm_prob

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        img = cv2.cvtColor(cv2.imread(item['img_path']), cv2.COLOR_BGR2RGB)
        x1, x2 = self.transforms(img)
        if self.srm_prob is not None and random.random() < self.srm_prob:
            x1 = x1.view(1, x1.shape[0], x1.shape[1], x1.shape[2])
            x1 = self.srm_conv(x1)
            x1 = x1.squeeze()
            x2 = x2.view(1, x2.shape[0], x2.shape[1], x2.shape[2])
            x2 = self.srm_conv(x2)
            x2 = x2.squeeze()
        return x1, x2, item['label']
