from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.merge import _Merge
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K

def make_generator():
    """Creates a generator model that takes a 100-dimensional noise vector as a "seed",
    and outputs images of size 28x28x1."""
    model = Sequential()
    model.add(Dense(1024, input_dim=100))
    model.add(LeakyReLU())
    model.add(Dense(128 * 7 * 7))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    if K.image_data_format() == 'channels_first':
        model.add(Reshape((128, 7, 7), input_shape=(128 * 7 * 7,)))
        bn_axis = 1
    else:
        model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
        bn_axis = -1
    model.add(Conv2DTranspose(128, (5, 5), strides=2, padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())
    model.add(Convolution2D(64, (5, 5), padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(64, (5, 5), strides=2, padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())
    # Because we normalized training inputs to lie in the range [-1, 1],
    # the tanh function should be used for the output of the generator to ensure
    # its output also lies in this range.
    model.add(Convolution2D(1, (5, 5), padding='same', activation='tanh'))
    return model


def make_discriminator():
    """Creates a discriminator model that takes an image as input and outputs a single
    value, representing whether the input is real or generated. Unlike normal GANs, the
    output is not sigmoid and does not represent a probability! Instead, the output
    should be as large and negative as possible for generated inputs and as large and
    positive as possible for real inputs.
    Note that the improved WGAN paper suggests that BatchNormalization should not be
    used in the discriminator."""
    model = Sequential()
    if K.image_data_format() == 'channels_first':
        model.add(Convolution2D(64, (5, 5), padding='same', input_shape=(1, 28, 28)))
    else:
        model.add(Convolution2D(64, (5, 5), padding='same', input_shape=(28, 28, 1)))
    model.add(LeakyReLU())
    model.add(Convolution2D(128, (5, 5), kernel_initializer='he_normal',
                            strides=[2, 2]))
    model.add(LeakyReLU())
    model.add(Convolution2D(128, (5, 5), kernel_initializer='he_normal', padding='same',
                            strides=[2, 2]))
    model.add(LeakyReLU())
    model.add(Flatten())
    model.add(Dense(1024, kernel_initializer='he_normal'))
    model.add(LeakyReLU())
    model.add(Dense(1, kernel_initializer='he_normal'))
    return model
