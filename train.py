import numpy as np
from keras import layers, models, optimizers, callbacks, regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import matplotlib.pyplot as plt
from utils import affine, get_args
from capsulenet import CapsNet, CapsNet_for_big, margin_loss
from PIL import Image
from sklearn.model_selection import train_test_split
from functools import reduce
from operator import mul

K.set_image_data_format('channels_last')

def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train) = data
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.15)

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=False, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    if args.retrain:
        capsule_weights = model.layers[5].get_weights()[0]
        capsule_mask = np.abs(capsule_weights) < args.retrain_coeff
        pruning = Pruning(capsule_mask)

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    """
    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
    """

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction,
                                           samplewise_center=True,
                                           samplewise_std_normalization=True)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    def valid_generator(x, y, batch_size):
        valid_datagen = ImageDataGenerator(samplewise_center=True,
                                           samplewise_std_normalization=True)  # shift up to 2 pixel for MNIST
        generator = valid_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])
    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    if args.retrain:
        model.fit_generator(generator=train_generator(x_train, y_train,
                                                      args.batch_size,
                                                      args.shift_fraction),
                            steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                            epochs=args.epochs,
                            validation_data=valid_generator(x_valid, y_valid, 100),
                            validation_steps=75,
                            callbacks=[log, tb, checkpoint, lr_decay, pruning])
    else:
        model.fit_generator(generator=train_generator(x_train, y_train,
                                                      args.batch_size,
                                                      args.shift_fraction),
                            steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                            epochs=args.epochs,
                            validation_data=valid_generator(x_valid, y_valid, 100),
                            validation_steps=75,
                            callbacks=[log, tb, checkpoint, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model
