import os
from PIL import Image
from resizeimage import resizeimage
from shutil import copyfile


filter_top_classes = 25

def get_cplfw_classes():
    classes = set()
    for f in os.listdir(cplfw_src):
        if f.endswith('.jpg'):
            c = f.rsplit('_', 1)[0]
            classes.add(c)
    return classes

def get_common_classes():
    return lfw_classes.intersection(cplfw_classes)

cplfw_src = './Dataset/cplfw/images/'
lfw_src = './Dataset/cleaned_data/lfw/'

cplfw_classes = get_cplfw_classes()
lfw_classes = set([f for f in os.listdir(lfw_src)])
common_classes = get_common_classes()

print("CPLFW # of classes: %d" %(len(cplfw_classes)))
print("LFW # of classes: %d" %(len(lfw_classes)))
print("Common classes in CPLFW and LFW: %d" %(len(common_classes)))

# Fetch top results for most
image_count = dict()

for g in os.listdir(lfw_src):
    if g in cplfw_classes:
        image_count[g] = len(os.listdir(lfw_src + '/' + g))

common_classes = sorted(common_classes, key=lambda k: image_count[k], reverse=True)
top_classes = common_classes[:filter_top_classes]

print("Common classes sorted in decreasing order of number of images in the class:")
print([(k, image_count[k]) for k in common_classes][:filter_top_classes][:filter_top_classes])

# Move top classes to another directory
lfw_dst = './Dataset/data/lfw'
cplfw_dst = './Dataset/data/cplfw'

cplfw_src = './Dataset/cleaned_data/cplfw'
# LFW
# for g in os.listdir(lfw_src):
#     if g in top_classes:
#         for f in os.listdir(lfw_src + '/' + g):
#             if f.endswith('.jpg'):
#                 try:
#                     os.makedirs(lfw_dst + '/' + g)
#                 except FileExistsError:
#                     pass
#                 copyfile(lfw_src + '/' + g + '/' + f, lfw_dst + '/' + g + '/' + f)

max_per_class = 50
image_dimension = 144
for g in os.listdir(lfw_src):
    if g in top_classes:
        c = 0
        for f in os.listdir(lfw_src + '/' + g):
            if f.endswith('.jpg'):
                c += 1
                try:
                    os.makedirs(lfw_dst + '/' + g)
                except FileExistsError:
                    pass
                with open(lfw_src + '/' + g + '/' + f, 'r+b') as fp:
                    with Image.open(fp) as image:
                        cover = resizeimage.resize_cover(image, [image_dimension, image_dimension])
                        cover.save(lfw_dst + '/' + g + '/' + f, image.format)
                if c == max_per_class:
                    break

# CPLFW
# image_dimension = 56
for g in os.listdir(cplfw_src):
    if g in top_classes:
        c = 0
        for f in os.listdir(cplfw_src + '/' + g):
            if f.endswith('.jpg'):
                c += 1
                try:
                    os.makedirs(cplfw_dst + '/' + g)
                except FileExistsError:
                    pass
                with open(cplfw_src + '/' + g + '/' + f, 'r+b') as fp:
                    with Image.open(fp) as image:
                        cover = resizeimage.resize_cover(image, [image_dimension, image_dimension])
                        cover.save(cplfw_dst + '/' + g + '/' + f, image.format)
                if c == max_per_class:
                    break
                # copyfile(cplfw_src + '/' + g + '/' + f, cplfw_dst + '/' + g + '/' + f)