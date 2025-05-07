import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import math


class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_prob=0.0, name='drop_path'):
        super().__init__(name=name)
        self.drop_prob = drop_prob
        self.name = name

    def get_config(self):
        config = super().get_config()
        config.update({
            'drop_prob': self.drop_prob,
            'name': self.name
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x, training=False):
        if self.drop_prob == 0.0 or not training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = [tf.shape(x)[0]] + [1] * (len(x.shape) - 1)
        random_tensor = keep_prob + tf.random.uniform(shape, dtype=x.dtype)
        random_tensor = tf.floor(random_tensor)
        if keep_prob > 0.0:
            random_tensor = tf.math.divide(random_tensor, keep_prob)
        return x * random_tensor

@tf.keras.utils.register_keras_serializable()
class ConvBNLayer(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size=3, stride=1, padding='valid', use_bias=False, name = 'conv_BN'):
        super().__init__(name=name)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.name = name

    def get_config(self):
        config = super().get_config()
        config.update({
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'use_bias': self.use_bias,
            'name': self.name
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        super().build(input_shape)
        self.conv = tf.keras.layers.Conv2D(
            filters=self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding=self.padding,
            use_bias=self.use_bias,
            name=f"{self.name}_conv"
        )
        self.bn = tf.keras.layers.BatchNormalization(name=f"{self.name}_bn" )
        self.act = tf.keras.layers.Activation('gelu',name=f"{self.name}_gelu" )
        self.built = True

    def call(self, inputs, training=False):
        out = self.conv(inputs)
        out = self.bn(out, training=training)
        out = self.act(out)
        return out

@tf.keras.utils.register_keras_serializable()
class Mlp(tf.keras.layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0, name='mlp'):
        super().__init__(name=name)
        self.in_features = in_features
        self.hidden_features = hidden_features or in_features
        self.out_features = out_features or in_features
        self.drop = drop
        self.name = name

    def get_config(self):
        config = super().get_config()
        config.update({
            'in_features': self.in_features,
            'hidden_features': self.hidden_features,
            'out_features': self.out_features,
            'drop': self.drop,
            'name': self.name
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build(self, input_shape):
        super().build(input_shape)
        self.fc1 = tf.keras.layers.Dense(self.hidden_features, name=self.name+'dense1')
        self.act = tf.keras.layers.Activation('gelu', name=self.name+'gelu')
        self.dropout1 = tf.keras.layers.Dropout(self.drop, name=self.name+'dropout1')
        self.fc2 = tf.keras.layers.Dense(self.out_features, name=self.name+'dense2')
        self.dropout2 = tf.keras.layers.Dropout(self.drop, name=self.name+'dropout2')
        self.built = True

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x, training=training)
        x = self.fc2(x)
        x = self.dropout2(x, training=training)
        return x