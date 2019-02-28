import os
from torch.utils import data
from data_loader import DataLoad

src = './Dataset/data/'

images = list()
targets = list()
classes = os.listdir(src)

# Mac-specific
if '.DS_Store' in classes:
    classes.remove('.DS_Store')

for label in classes:
    path = src + label + '/'
    image_paths = os.listdir(path)
    for image in image_paths:
        images.append(label + '/' + image)
        targets.append(classes.index(label))

# DataLoader params
params = {
    'batch_size': 64,
    'shuffle': True,
    'num_workers': 6
}

dataset = DataLoad(src, images, targets)
train_generator = data.DataLoader(dataset, **params)
#
for batch, labels in train_generator:
    print(batch.shape, labels.shape)
