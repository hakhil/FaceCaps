import os
from shutil import copyfile

classes = set()
d = dict()

src = './Dataset/cplfw/images/'
dst = './Dataset/data/'

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
        copyfile(src + f, dst + c + '/' + f)

print('# of classes: ', len(classes))