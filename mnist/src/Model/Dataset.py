# general
import numpy as np

# sklearn
from sklearn.datasets import fetch_openml

# pytorch
import torch
from torch.utils import data

class MnistDataset(data.Dataset):
    def __init__(self, transform = None):
        self.mnist = fetch_openml('mnist_784', version=1)
        self.data = self.mnist.data.reshape(-1, 28, 28).astype('uint8')
        self.target = self.mnist.target.astype('int')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        target = torch.from_numpy(np.array(self.target[idx]))

        if self.transform :
            data = self.transform(data)

        return (data, target)