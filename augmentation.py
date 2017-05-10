import os
import numpy as np
import scipy.ndimage.interpolation as interp
from matplotlib import pyplot
import PIL, PIL.Image


def normalize(img):
    return (img - np.mean(img)) / np.std(img)


def create_vec(fname):
    if np.array(PIL.Image.open(fname)).shape != (24, 24):
        print('{} no és de 24x24'.format(fname))
        quit()
    return normalize(np.array(PIL.Image.open(fname))).flatten()


def load_data(dir):
    return np.stack([create_vec('{}/{}'.format(dir, name))
                     for name in os.listdir(dir)])


def rotate_and_scale(img, a):
    rot = interp.rotate(img, a, reshape=False)
    a_rad = (a if a >= 0 else -a) / 180 * np.pi
    zoom = np.sin(a_rad + .25*np.pi) / np.sin(.25*np.pi)
    zoo = interp.zoom(rot, zoom)
    x = (zoo.shape[0] - 24) // 2
    final = zoo[x:x+24, x:x+24]
    return final


def augment_img(img_arr):
    img = img_arr.reshape(24, 24)
    rotate_and_scale(img, 10)
    return (normalize(np.array(rotate_and_scale(img, a)).flatten())
            for a in range(-2, 3, 2))


def augment_data(data):
    augmented_imgs = (augment_img(img) for img in data if np.std(img) != 0)
    return np.array([augm for img in augmented_imgs for augm in img])
