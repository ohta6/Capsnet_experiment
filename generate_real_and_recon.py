import numpy as np
from keras import layers, models, optimizers, callbacks, regularizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from utils import combine_images, affine, manipulate_latent, save_pred_and_recon
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from sklearn.model_selection import train_test_split
from functools import reduce
from operator import mul

from data_loaders import load_mnist, load_fashion_mnist, load_svhn, load_cifar10
from capsulenet import CapsNet
K.set_image_data_format('channels_last')

if __name__ == "__main__":
    import os
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks
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
                        help='0=mnist, 1=fashion_mnist, 2=SVHN')
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--l1', default=0.0, type=float,
                        help="coeff l1 regularization")
    parser.add_argument('--retrain', action='store_true',
                        help="Retrain and make weights sparse")
    parser.add_argument('--retrain_coeff', default=0.000001, type=float,
                        help="閾値")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w1', '--weights1', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('-w2', '--weights2', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    if args.dataset == 0:
        (x_train, y_train), (x_test, y_test) = load_mnist()
    elif args.dataset == 1:
        (x_train, y_train), (x_test, y_test) = load_fashion_mnist()
    elif args.dataset == 2:
        (x_train, y_train), (x_test, y_test) = load_svhn()
    # define model
    model, eval_model, manipulate_model = CapsNet(input_shape=x_train.shape[1:],
                                                  n_class=len(np.unique(np.argmax(y_train, 1))),
                                                  routings=args.routings,
                                                  l1=args.l1)
    model.summary()

    flags = [0]*10
    index = [0]*10
    digits = np.where(y_test == 1)[1]
    for i, num in enumerate(digits):
        num = int(num)
        if flags[num]:
            continue
        else:
            flags[num] = 1
            index[num] = i
        if np.all(flags):
            break

    x_deform_test = np.array([affine(x) for x in x_test])
    print(index)
    print(x_test[index].shape)
    input_img = np.concatenate([x_test[index], x_deform_test[index]])
    input_img = combine_images(input_img, height=2, width=10)
    input_img = input_img * 255
    Image.fromarray(input_img.astype(np.uint8)).save(args.save_dir + '/input.png')

    model.load_weights(args.weights1)
    _, x_recon = eval_model.predict(x_test, batch_size=100)
    _, x_deform_recon = eval_model.predict(x_deform_test, batch_size=100)
    recon_img = np.concatenate([x_recon[index], x_deform_recon[index]])
    recon_img = combine_images(recon_img, height=2, width=10)
    recon_img = recon_img * 255
    Image.fromarray(recon_img.astype(np.uint8)).save(args.save_dir + '/recon.png')

    model.load_weights(args.weights2)
    _, x_l1_recon = eval_model.predict(x_test, batch_size=100)
    _, x_l1_deform_recon = eval_model.predict(x_deform_test, batch_size=100)
    l1_recon_img = np.concatenate([x_l1_recon[index], x_l1_deform_recon[index]])
    l1_recon_img = combine_images(l1_recon_img, height=2, width=10)
    l1_recon_img = l1_recon_img * 255
    Image.fromarray(l1_recon_img.astype(np.uint8)).save(args.save_dir + '/l1_recon.png')
