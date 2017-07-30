import numpy as np
from PIL import Image
import os

def read_img(img_path):
    with open(img_path) as fs:
        img = Image.open(fs.read())
    img_array = np.asarray(img)
    img_array.flags.writables = True

    return img_array

def read_imgs(img_dir):
    imgs = []
    for file_name in os.listdir(img_dir):
        imgs.append(read_img(img_dir+file_name))
    return np.array(imgs)



