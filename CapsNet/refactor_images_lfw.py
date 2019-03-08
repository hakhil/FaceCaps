import os
from shutil import copyfile
from PIL import Image
from resizeimage import resizeimage

classes = set()
d = dict()

src = './Dataset/filtered/lfw/'
dst = './Dataset/data/lfw/'

image_dimension = 28

try:
    os.makedirs(dst)
except FileExistsError:
    # directory already exists
    pass

for g in os.listdir(src):
    classes.add(g)
    for f in os.listdir(src + '/' + g):
        if f.endswith('.jpg'):
            try:
                os.makedirs(dst + '/' + g)
            except FileExistsError:
                pass

            print(src + '/' + g + '/' + f, "->", dst + g + '/' + f)
            # copyfile(src + '/' + g + '/' + f, dst + g + '/' + f)

            with open(src + '/' + g + '/' + f, 'r+b') as fp:
                with Image.open(fp) as image:
                    cover = resizeimage.resize_cover(image, [image_dimension, image_dimension])
                    cover.save(dst + '/' + g + '/' + f, image.format)

print('# of classes: ', len(classes))