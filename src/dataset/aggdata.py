import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset,DataLoader
from skimage import io,transform
from torchvision.datasets import VisionDataset
from .utils import *

# Ignores warnings
import warnings
warnings.filterwarnings("ignore")


class AGGData(VisionDataset):
    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, data_path, train=True, transform=None, target_transform=None,
            download=False):
        super(AGGData, self).__init__(data_path,transform=transform,target_transform=target_transform)
        self.train = train

        # data and target
        self.data = None
        self.targets = None

        if download:
            self.download()
        if self.train:
            self.data, self.targets = load_agg_data(os.path.join(self.data_path, "train"))
        else:
            self.data, self.targets = load_test_data(os.path.join(self.data_path, "val"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
