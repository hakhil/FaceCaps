import os
from shutil import copyfile
from PIL import Image
from resizeimage import resizeimage

classes = set()
d = dict()

src = './Dataset/cplfw/images/'
dst = './Dataset/data/cplfw/'

image_dimension = 28

try:
    os.makedirs(dst)
except FileExistsError:
    # directory already exists
    pass

for f in os.listdir(src):
    if f.endswith('.jpg'):
        c = f.rsplit('_', 1)[0]
        if c in d:
            d[c] += 1
        else:
            d[c] = 1
        classes.add(c)

        try:
            os.makedirs(dst + c + '/')
        except FileExistsError:
            # directory already exists
            pass

        print(src + f, ";", dst + c + '/' + f)

        with open(src + f, 'r+b') as fp:
            with Image.open(fp) as image:
                cover = resizeimage.resize_cover(image, [image_dimension, image_dimension])
                cover.save(dst + c + '/' + f, image.format)

# copyfile(src + f, dst + c + '/' + f)
print('# of classes: ', len(classes))