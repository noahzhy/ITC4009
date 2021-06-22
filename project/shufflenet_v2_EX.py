import keras
import tensorflow as tf

from keras import backend as K
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.activations import *
from tensorflow.keras.utils import plot_model


def channel_split(x, num_splits=2):
    if num_splits == 2:
        return tf.split(x, axis=-1, num_or_size_splits=num_splits)
    else:
        raise ValueError('Error! num_splits should be 2')


def channel_shuffle(x):
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2
    x = K.reshape(x, [-1, height, width, 2, channels_per_split])
    x = K.permute_dimensions(x, (0, 1, 2, 4, 3))
    x = K.reshape(x, [-1, height, width, channels])
    return x


def conv_bn_relu(inputs, out_channels, kernel_size=3, stride=1, relu="relu"):
    x = Conv2D(out_channels, kernel_size=kernel_size, strides=stride, padding='same', use_bias=False)(inputs)
    x = bn_relu(x, relu=relu)
    return x


def bn_relu(inputs, relu="relu"):
    x = BatchNormalization()(inputs)
    if relu == "relu":
        x = ReLU()(x)
    elif relu == "relu6":
        x = tf.nn.relu6(x)
    return x


def Conv_DWConv_Conv(inputs, out_channels, stride=1, dwconv_ks=3):
    x = conv_bn_relu(inputs, out_channels, kernel_size=1, relu="relu")
    x = DepthwiseConv2D(kernel_size=dwconv_ks, strides=stride, padding="same", use_bias=False)(x)
    x = bn_relu(x, relu=None)
    x = conv_bn_relu(x, out_channels, kernel_size=1, relu="relu")
    return x


def shufflenet_unit(inputs, out_channel, stride=1):
    out_channel //= 2
    top, bottom = channel_split(inputs)
    top = Conv_DWConv_Conv(top, out_channel, stride)

    if stride == 2:
        bottom = DepthwiseConv2D(kernel_size=3, strides=stride, padding="same", use_bias=False)(inputs)
        bottom = bn_relu(bottom, None)
        bottom = conv_bn_relu(bottom, out_channel, kernel_size=1, relu="relu")

    out = Concatenate()([top, bottom])
    out = channel_shuffle(out)
    return out


def stage(x, num_stages, out_channels):
    x = shufflenet_unit(x, out_channels, stride=2)
    for i in range(num_stages):
        x = shufflenet_unit(x, out_channels, stride=1)
    return x


def shufflenet_v2(inputs, out_channels: list, num_class=1000):
    x = Conv2D(24, (3, 3), strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    x = stage(x, 3, out_channels[0])
    x = stage(x, 7, out_channels[1])
    x = stage(x, 3, out_channels[2])

    x = conv_bn_relu(x, out_channels[3], kernel_size=1, relu="relu")

    x = GlobalAveragePooling2D()(x)
    x = Dense(num_class, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=x)
    return model


def shufflenetV2_x(inputs, scale=1):
    if scale == 0.5:
        out_channels = [48, 96, 192, 1024]
    elif scale == 1:
        out_channels = [116, 232, 464, 1024]
    elif scale == 1.5:
        out_channels = [176, 352, 704, 1024]
    elif scale == 2:
        out_channels = [244, 488, 976, 2048]

    return shufflenet_v2(inputs, out_channels=out_channels)


if __name__ == '__main__':
    inputs = Input(shape=(224, 224, 3))
    model = shufflenetV2_x(inputs, scale=1)
    model.summary()
    plot_model(model, to_file='shuffleNet_v2.png',
               show_layer_names=True, show_shapes=True)
