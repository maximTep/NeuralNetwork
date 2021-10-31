import numpy as np
import pygame
from mnist import MNIST


def load_mnist_train():
    mndata = MNIST('MNIST')
    mndata.gz = True
    return mndata.load_training()

def pixel_reader(image_: pygame.Surface, H, W, scale):
    grid_ = [[image_.get_at([y * scale, x * scale])[0] / 255 for x in range(W // scale)] for y in range(H // scale)]
    return grid_


def matrix_to_array(matrix: np.ndarray):
    n_ = len(matrix)
    m_ = len(matrix[0])
    lst = [matrix[i][j] for j in range(m_) for i in range(n_)]
    arr = np.ndarray(shape=n_ * m_)
    for i in range(n_ * m_):
        arr[i] = lst[i]
    return arr


def shrink_img_array(img_arr: np.ndarray):
    return [pix/255 for pix in img_arr]
