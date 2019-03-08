import os
from shutil import copyfile

filter_top_classes = 350

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
lfw_src = './Dataset/lfw/'

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
print([(k, image_count[k]) for k in common_classes][:filter_top_classes])

# Move top classes to another directory
dst = './Dataset/filtered/lfw'

# LFW
for g in os.listdir(lfw_src):
    if g in top_classes:
        for f in os.listdir(lfw_src + '/' + g):
            if f.endswith('.jpg'):
                try:
                    os.makedirs(dst + '/' + g)
                except FileExistsError:
                    pass
                copyfile(lfw_src + '/' + g + '/' + f, dst + '/' + g + '/' + f)