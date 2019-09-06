import numpy as np
from keras.datasets import mnist
from keras import backend as K

def get_mnist():
    print('---loading MNIST---')
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = np.concatenate((X_train, X_test), axis=0)
    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]))
    else:
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    print('---loaded----')
    return X_train
