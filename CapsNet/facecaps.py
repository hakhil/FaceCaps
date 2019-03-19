import numpy as np

import torch
import os
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.image as mpimg
from torch.utils import data
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
# from data_loader import DataLoad


def cnn_out_size(input_dim, padding, kernel_size, stride):
    return ((input_dim + 2 * padding - kernel_size) // stride) + 1

USE_CUDA = False
input_dim = 56
kernel_size = 9
stride = 2
capsule_input = cnn_out_size(cnn_out_size(input_dim, 0, kernel_size, 1), 0, kernel_size, stride)
num_classes=25
capsule_dimension = 64

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
        img = mpimg.imread(self.src + ID)
#         plt.imshow(img)
#         plt.show()

#         X = torch.Tensor(img).view(3, img_dimension, img_dimension)
        X = torch.Tensor(reshape_image_tensor_channels_first(img))
        y = self.labels[index]

        return X, y


class LFW:
    def __init__(self, batch_size):
        src = './Dataset/data/lfw/'
        images, targets = self.load_data(src)
        dataset_size = len(images)
        validation_split = 0.2
        split = int(np.floor(validation_split * dataset_size))

        indices = list(range(dataset_size))
        train_indices, test_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        dataset = DataLoad(src, images, targets)

        # Dataloader parameters
        params = {
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': 6,
            'drop_last': True
        }

        self.train_loader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, **params)
        self.test_loader = torch.utils.data.DataLoader(dataset, sampler=test_sampler, **params)


    def load_data(self, src):

        images = list()
        targets = list()
        classes = os.listdir(src)

        self.n_classes = len(classes)

        # Mac-specific
        if '.DS_Store' in classes:
            classes.remove('.DS_Store')

        for label in classes:
            path = src + label + '/'
            image_paths = os.listdir(path)
            for image in image_paths:
                images.append(label + '/' + image)
                targets.append(classes.index(label))

        return images, targets

class ConvLayer(nn.Module):
    def __init__(self, in_channels=3, out_channels=256, kernel_size=9):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=1
                             )

    def forward(self, x):
        return F.relu(self.conv(x))


class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9):
        super(PrimaryCaps, self).__init__()

        self.out_channels = out_channels
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0)
            for _ in range(num_capsules)])

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), self.out_channels * capsule_input * capsule_input, -1)
        return self.squash(u)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class FaceCaps(nn.Module):
    def __init__(self, num_capsules=num_classes, num_routes=32 * capsule_input * capsule_input, in_channels=8,
                 out_channels=capsule_dimension):
        super(FaceCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        print(self.W.shape)

        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if USE_CUDA:
            b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)

            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.reconstraction_layers = nn.Sequential(
            nn.Linear(capsule_dimension * num_classes, 512 * 3),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True),
            nn.Linear(512 * 3, 256 * 3),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True),
            nn.Linear(256 * 3, input_dim * input_dim * 3),
            nn.Sigmoid()
        )

    def forward(self, x, data):
        classes = torch.sqrt((x ** 2).sum(2))
        classes = F.softmax(classes)

        _, max_length_indices = classes.max(dim=1)
        masked = Variable(torch.sparse.torch.eye(num_classes))
        if USE_CUDA:
            masked = masked.cuda()
        masked = masked.index_select(dim=0, index=max_length_indices.squeeze(1).data)

        reconstructions = self.reconstraction_layers((x * masked[:, :, None, None]).view(x.size(0), -1))
        reconstructions = reconstructions.view(-1, 3, input_dim, input_dim)

        return reconstructions, masked

class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps()
        self.digit_capsules = FaceCaps()
        self.decoder = Decoder()

        self.mse_loss = nn.MSELoss()

    def forward(self, data):
        output = self.digit_capsules(self.primary_capsules(self.conv_layer(data)))
        reconstructions, masked = self.decoder(output, data)
        return output, reconstructions, masked

    def loss(self, data, x, target, reconstructions):
        return self.margin_loss(x, target) + self.reconstruction_loss(data, reconstructions)

    def margin_loss(self, x, labels, size_average=True):
        batch_size = x.size(0)

        v_c = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))

        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)

        loss = labels * left + 0.5 * (1.0 - labels) * right
        loss = loss.sum(dim=1).mean()

        return loss

    def reconstruction_loss(self, data, reconstructions):
        loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1), data.view(reconstructions.size(0), -1))
        return loss * 0.0005

def reshape_image_tensor(img):
  reshaped_img = np.zeros((input_dim,input_dim,3))
  for channel in range(3):
    for i in range(input_dim):
      for j in range(input_dim):
        reshaped_img[i, j, channel] = img[channel, i, j]
  return reshaped_img

def reshape_image_tensor_channels_first(img):
  reshaped_img = np.zeros((3,input_dim,input_dim))
  for i in range(input_dim):
    for j in range(input_dim):
      for channel in range(3):
        reshaped_img[channel, i, j] = img[i, j, channel]
  return reshaped_img


try:
  capsule_net
except NameError:
  capsule_net = CapsNet()
  print("well, it WASN'T defined after all!")
else:
  print("CapsNet is already defined.")

batch_size = 8
lfw = LFW(batch_size)

if USE_CUDA:
    capsule_net = capsule_net.cuda()
optimizer = Adam(capsule_net.parameters())

n_epochs = 3

for epoch in range(n_epochs):
    capsule_net.train()
    train_loss = 0
    for batch_id, (data, target) in enumerate(lfw.train_loader):
        img = reshape_image_tensor(data[0, :, :, :].data.cpu().numpy())
        #         plt.imshow(img)
        #         plt.axis('off')
        #         plt.show()

        target = torch.sparse.torch.eye(lfw.n_classes).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        if USE_CUDA:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        #         try:
        output, reconstructions, masked = capsule_net(data)
        loss = capsule_net.loss(data, output, target, reconstructions)
        loss.backward()
        optimizer.step()
        #         except RuntimeError as e:
        #           if 'out of memory' in str(e):
        # #             print('| WARNING: ran out of memory, retrying batch')
        #             for p in capsule_net.parameters():
        #                 if p.grad is not None:
        #                     del p.grad  # free some memory
        #             torch.cuda.empty_cache()

        train_loss += loss.item()

        if batch_id % 100:
            print("Train accuracy:", sum(np.argmax(masked.data.cpu().numpy(), 1) ==
                                         np.argmax(target.data.cpu().numpy(), 1)) / float(batch_size))

    print("Train loss", train_loss / len(lfw.train_loader))

    capsule_net.eval()
    test_loss = 0
    for batch_id, (data, target) in enumerate(lfw.test_loader):
        target = torch.sparse.torch.eye(lfw.n_classes).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        if USE_CUDA:
            data, target = data.cuda(), target.cuda()

        output, reconstructions, masked = capsule_net(data)
        loss = capsule_net.loss(data, output, target, reconstructions)

        test_loss += loss.item()

        if batch_id % 100:
            print("Test accuracy:", sum(np.argmax(masked.data.cpu().numpy(), 1) ==
                                        np.argmax(target.data.cpu().numpy(), 1)) / float(batch_size))
    print("Test loss", test_loss / len(lfw.test_loader))
    print("------------------------")