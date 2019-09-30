# encode:UTF-8
import numpy as np
from capsulenet import CapsNet, CapsNet_for_big
from train import train
from test import test, test_all
from utils import get_args
from data_loaders import load_mnist, load_fashion_mnist, load_svhn, load_cifar10, load_food101
from tensorflow.python import debug as tf_debug
from keras import backend as K
"""
sess = K.get_session()
sess = tf_debug.LocalCLIDebugWrapperSession(sess)
K.set_session(sess)
"""


if __name__ == "__main__":
    import os
    #import argparse

    args = get_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    if args.dataset == 0:
        (x_train, y_train), (x_test, y_test) = load_mnist()
    elif args.dataset == 1:
        (x_train, y_train), (x_test, y_test) = load_fashion_mnist()
    elif args.dataset == 2:
        (x_train, y_train), (x_test, y_test) = load_svhn()
    elif args.dataset == 3:
        (x_train, y_train), (x_test, y_test) = load_cifar10()
    elif args.dataset == 4:
        (x_train, y_train), (x_test, y_test) = load_food101()

    x_train = x_train[:args.train_num]
    y_train = y_train[:args.train_num]
    # define model
    if args.dataset != 4:
        model, eval_model, manipulate_model = CapsNet(input_shape=x_train.shape[1:],
                                                      n_class=len(np.unique(np.argmax(y_train, 1))),
                                                      routings=args.routings,
                                                      l1=args.l1)
    else:
        model, eval_model, manipulate_model = CapsNet_for_big(input_shape=x_train.shape[1:],
                                                      n_class=len(np.unique(np.argmax(y_train, 1))),
                                                      routings=args.routings,
                                                      l1=args.l1)
    model.summary()

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        train(model=model, data=(x_train, y_train), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            #print('No weights are provided. Will test using random initialized weights.')
            test_all(eval_model, (x_test, y_test), args.dataset)
        #manipulate_latent(manipulate_model, (x_test, y_test), args)
        #save_for_gif(manipulate_model, (x_test, y_test), args)
        #test(model=eval_model, data=(x_test, y_test), args=args)


