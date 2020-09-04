import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from skimage import io, transform

# Ignores warnings
import warnings
warnings.filterwarnings("ignore")


class MPRData(Dataset):
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

    def __init__(self, datax, targetsx, shape, train=True,
                 transform=None, target_transform=None):
        super(MPRData, self).__init__()
        self.train = train

        self.transform = transform
        self.target_transform = transform
        # data and target
        self.data = torch.reshape(datax, shape)
        self.targets = targetsx

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

        return img, target
