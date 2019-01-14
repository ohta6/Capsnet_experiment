import numpy as np
import cv2
from matplotlib import pyplot as plt
import csv
import math
import pandas

def plot_log(filename, show=True):

    data = pandas.read_csv(filename)

    fig = plt.figure(figsize=(4,6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for key in data.keys():
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for key in data.keys():
        if key.find('acc') >= 0:  # acc
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()


def combine_images(generated_images, height=None, width=None):
    num = generated_images.shape[0]
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
    elif width is not None and height is None:  # height not given
        height = int(math.ceil(float(num)/width))
    elif height is not None and width is None:  # width not given
        width = int(math.ceil(float(num)/height))

    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

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

    rotate_M = cv2.getRotationMatrix2D((cols/2,rows/2),rotation_deg,0.75)
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



