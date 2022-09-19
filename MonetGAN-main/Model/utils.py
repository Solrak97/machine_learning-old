import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_addons as tfa


def downsample(filters, size, apply_instancenorm=True):

    initializer = keras.initializers.LecunUniform()
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    base = keras.Sequential()
    base.add(layers.Conv2D(filters, size, strides=2, padding='same',
                           kernel_initializer=initializer, use_bias=False))

    if apply_instancenorm:
        base.add(tfa.layers.InstanceNormalization(
            gamma_initializer=gamma_init))

    base.add(layers.LeakyReLU())

    return base


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    result.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    if apply_dropout:
        result.add(layers.Dropout(0.5))

    result.add(layers.ReLU())

    return result
