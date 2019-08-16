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

def stop_watch(func) :
    @wraps(func)
    def wrapper(*args, **kargs) :
        start = time.time()
        result = func(*args,**kargs)
        elapsed_time =  time.time() - start
        print("{}は{}秒かかりました".format(func.__name__, elapsed_time))
        return result
    return wrapper

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

def test_generator(x, batch_size):
    test_datagen = ImageDataGenerator()
    """
    test_datagen = ImageDataGenerator(samplewise_center=True,
                                      samplewise_std_normalization=True)
    """
    generator = test_datagen.flow(x, batch_size=batch_size, shuffle=False)
    return generator

@stop_watch
def pred(model, generator, batchsize=100):
    y_pred, x_recon = model.predict_generator(generator, steps=100)
    return y_pred, x_recon

def save_pred_and_recon(x_test, y_test, model, args, png_name='/real_and_recon.png'):
    #y_pred, x_recon = model.predict(x_test, batch_size=100)
    generator = test_generator(x_test, batch_size=100)
    y_pred, x_recon = pred(model, generator)
    acc = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0]
    print('Test acc:', acc)
    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    print('Reconstructed images are saved to %s%s' % (args.save_dir, png_name))
    print('-' * 30 + 'End: test' + '-' * 30)
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + png_name)
    #plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    sparcity = show_model_sparsity(model)
    with open(args.save_dir + "/test_acc.txt", "a") as f:
        f.write('Test acc:' + str(acc) + '\n')
        f.write('model sparcity:' + str(sparcity) + '\n')

def show_model_sparsity(model):
    layers = model.layers
    capsule_w = model.layers[5].get_weights()[0]
    capsule_w = capsule_w.reshape([-1, 16, 8])
    total_param = reduce(mul, capsule_w.shape)
    zero_param = np.sum(capsule_w==0.0)
    count = 0
    for w in capsule_w:
        if (np.sum(w==0.0)/(16*8)) == 1.0:
            count += 1
    print(count/capsule_w.shape[0])
    return zero_param/total_param

def manipulate_latent(model, data, args):
    print('-'*30 + 'Begin: manipulate' + '-'*30)
    x_test, y_test = data
    index = np.argmax(y_test, 1) == args.digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 10, 16])
    x_recons = []
    for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
        for dim in range(16):
            tmp = np.copy(noise)
            tmp[:,:,dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = combine_images(x_recons, height=11, width=16)
    image = img*255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.digit)
    print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.digit))
    print('-' * 30 + 'End: manipulate' + '-' * 30)

def save_for_gif(model, data, args):
    x_test, y_test = data
    index = np.argmax(y_test, 1) == args.digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 10, 16])
    for dim in range(16):
        os.mkdir(args.save_dir+'/digit{}_dim{}'.format(args.digit, dim))
        for i,r in enumerate(range(-25, 26)):
            tmp = np.copy(noise)
            tmp[:,:,dim] = r/100
            x_recon = model.predict([x, y, tmp])
            img = np.uint8(x_recon[0]*255)
            cv2.imwrite(args.save_dir+'/digit{}_dim{}/{:0=10}.png'.format(args.digit, dim, i), img)

if __name__=="__main__":
    img = cv2.imread('gori.jpg', 0)
    print(img.shape)
    img = affine(img)
    cv2.imwrite('gori_aff.jpg',img)
    #plot_log('result/log.csv')



