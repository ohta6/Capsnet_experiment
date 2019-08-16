# encode:UTF-8
import keras
from keras.utils import to_categorical

def load_mnist():
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)

def load_fashion_mnist():
    # the data, shuffled and split between train and test sets
    fashion_mnist = keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)

def load_svhn():
    from scipy import io
    train_file = 'svhn/train_32x32.mat'
    test_file = 'svhn/test_32x32.mat'
    matdata_train = io.loadmat(train_file, squeeze_me=True)
    matdata_test = io.loadmat(test_file, squeeze_me=True)
    x_train = matdata_train['X']
    y_train = matdata_train['y']
    y_train = np.where(y_train == 10, 0, y_train)
    x_test = matdata_test['X']
    y_test = matdata_test['y']
    y_test = np.where(y_test == 10, 0, y_test)
    x_train = x_train.transpose(3, 0, 1, 2).astype('float32') / 255.
    x_test = x_test.transpose(3, 0, 1, 2).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)

def load_cifar10():
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255.
    x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)

def load_food101():
    from os.path import join
    from os import listdir
    import cv2
    import numpy as np
    size = 96
    train_dir = "food-101/train" + str(size)
    test_dir = "food-101/test" + str(size)
    foods = listdir(train_dir)
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i, food in enumerate(foods):
        food_dir = join(train_dir, food)
        for img_path in listdir(food_dir):
            x_train.append(cv2.imread(join(food_dir, img_path)))
            y_train.append(i)
        food_dir = join(test_dir, food)
        for img_path in listdir(food_dir):
            x_test.append(cv2.imread(join(food_dir, img_path)))
            y_test.append(i)
    x_train = np.array(x_train, "float32").reshape(-1, size, size, 3) / 255.
    x_test = np.array(x_test, "float32").reshape(-1, size, size, 3) / 255.
    y_train = to_categorical(np.array(y_train, "float32"))
    y_test = to_categorical(np.array(y_test, "float32"))
    return (x_train, y_train), (x_test, y_test)
        
        
