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
from utils import affine

def test(model, data, args):
    x_test, y_test = data
    #print(x_test.shape)
    x_deform_test = np.array([affine(x) for x in x_test])
    save_pred_and_recon(x_test, y_test, model, args)
    #save_pred_and_recon(x_deform_test, y_test, model, args, png_name='/deformed_and_recon.png')

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
def test_generator(x, batch_size):
    test_datagen = ImageDataGenerator(samplewise_center=True,
                                      samplewise_std_normalization=True)
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
    print(y_pred.shape)
    acc = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0]
    print('Test acc:', acc)
    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    print('Reconstructed images are saved to %s%s' % (args.save_dir, png_name))
    print('-' * 30 + 'End: test' + '-' * 30)
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + png_name)
    #plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    """
    sparcity = show_model_sparsity(model)
    with open(args.save_dir + "/test_acc.txt", "a") as f:
        f.write('Test acc:' + str(acc) + '\n')
        f.write('model sparcity:' + str(sparcity) + '\n')
    with open("test_acc_{}.txt".format(args.dataset), "a") as f:
        f.write(str(acc) + '\n')
    with open("model_sparcity_{}.txt".format(args.dataset), "a") as f:
        f.write(str(sparcity) + '\n')
    """

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
