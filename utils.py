import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import csv
import math
import pandas
from PIL import Image
from functools import reduce
from operator import mul

from keras.preprocessing.image import ImageDataGenerator
from functools import wraps
import time

def get_args():
    import argparse
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('--dataset', default=0, type=int,
                        help='0=mnist, 1=fashion_mnist, 2=SVHN, 3=cifar10')
    parser.add_argument('--train_num', default=50000, type=int,
                        help='how many train data use')
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result/')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--l1', default=0.0, type=float,
                        help="coeff l1 regularization")
    parser.add_argument('--retrain', action='store_true',
                        help="Retrain and make weights sparse")
    parser.add_argument('--retrain_coeff', default=0.0001, type=float,
                        help="閾値")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)
    return args


def affine(img):
    rotation_deg = 40 * np.random.rand() - 20
    shear_x = 0.4 * np.random.rand() - 0.2
    shear_y = 0.4 * np.random.rand() - 0.2
    move_x = 2 * np.random.rand() - 1
    move_y = 2 * np.random.rand() - 1

    if len(img.shape) == 2:
        rows, cols = img.shape
    else:
        rows, cols, ch = img.shape

    rotate_M = cv2.getRotationMatrix2D((cols/2,rows/2),rotation_deg,1.2)
    shear_M = np.float32([
            [1, shear_x, 0],
            [shear_y, 1, 0]
        ])
    move_M = np.float32([
            [1, 0, move_x],
            [0, 1, move_y]
        ])
    img = cv2.warpAffine(img,rotate_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))
    res = cv2.warpAffine(img,move_M,(cols,rows))
    #res = cv2.resize(img,(rows,cols), fx=2.0,fy=1.5)
    if len(img.shape) == 2:
        res = np.reshape(res, (rows, cols, 1))
    return res

if __name__=="__main__":
    img = cv2.imread('gori.jpg', 0)
    print(img.shape)
    img = affine(img)
    cv2.imwrite('gori_aff.jpg',img)
    #plot_log('result/log.csv')



