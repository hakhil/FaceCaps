import os
import torch
from torch.utils import data
from keras.preprocessing.image import img_to_array, load_img

# Leverages Pytorch Dataset to use data loader
class DataLoad(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, src, list_IDs, labels):
        'Initialization'
        self.src = src
        self.list_IDs = list_IDs
        self.labels = labels

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        img = load_img(self.src + ID)
        X = torch.Tensor(img_to_array(img)).view(3, 250, 250)
        y = self.labels[index]

        return X, y