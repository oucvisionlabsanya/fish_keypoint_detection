import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Add,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    UpSampling2D,
    Concatenate,
    Softmax,
)
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import numpy as np

# 此版本代码为最早期版本，由于部分代码丢失，最新的代码将于后续更新
def build_stacked_hourglass(input_shape=(256, 256, 3),
                            dim_output=3,
                            n_hourglass=3, n_hourglass_layer=3,
                            dim_feature=128,
                            jumpwire_mode='add',
                            final_activation='sigmoid'):
    '''
    :param input_shape: -
    :param n_hourglass: Number of hourglass structures
    :param n_hourglass_layer: Order of a single hourglass
    :param dim_feature: Number of feature map's channels
    :param jumpwire_mode: Add: summing jumpwires together; concat: concatenat jumpwires together the 1x1 conv to half the channels
           Not including jumpwires in identity blocks
    :param final_activation: Sigmoid: multiple keypoints in one category. Or you just want to use sigmoid; softmax: only one keypoints for each category.
    :return: A Stacked Hourglass model
    '''

    _input = Input(shape=input_shape)
    x = _input

    x = Conv2D(filters=dim_feature, kernel_size=(3, 3), padding='same', activation='relu',
               kernel_initializer=glorot_uniform(0))(x)
    x = Conv2D(filters=dim_feature, kernel_size=(3, 3), padding='same', activation='relu',
               kernel_initializer=glorot_uniform(0))(x)
    x = Conv2D(filters=dim_feature, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu',
               kernel_initializer=glorot_uniform(0))(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x, _outputs = _stacked_hourglass(n_hourglass=n_hourglass,
                                     n_hourglass_layer=n_hourglass_layer,
                                     dim_output=dim_output,
                                     dim_feature=dim_feature,
                                     jumpwire_mode=jumpwire_mode,
                                     final_activation=final_activation)(x)

    model = Model(_input, _outputs)

    return model


def _stacked_hourglass(n_hourglass=3, n_hourglass_layer=3,
                       dim_output=3,
                       dim_feature=256,
                       jumpwire_mode='add',
                       final_activation='sigmoid'):
    def f(inputs):
        dim_features = []
        for i in range(n_hourglass_layer):
            dim_features.append(dim_feature)

        x = inputs

        side_outputs = []
        for i in range(n_hourglass):
            x, side_output = _hourglass(n_layers=n_hourglass_layer,
                                        dim_output=dim_output,
                                        dim_features=dim_features,
                                        final_dim_feature=dim_feature,
                                        jumpwire_mode=jumpwire_mode,
                                        final_activation=final_activation)(x)
            side_outputs.append(side_output)

        return x, side_outputs

    return f


# A single hourglass
def _hourglass(n_layers=3,
               dim_features=(256, 256, 256), final_dim_feature=256, dim_output=3,
               jumpwire_mode='add', final_activation='sigmoid'):
    def f(inputs):
        assert n_layers == len(dim_features)

        x = inputs

        func = _identity_block(dim_feature=dim_features[-1])
        for i in range(n_layers):
            func = _hourglass_layer(inner_func=func,
                                    output_dim=dim_features[i],
                                    jumpwire_mode=jumpwire_mode)
        x = func(x)
        x = Conv2D(filters=final_dim_feature, kernel_size=(1, 1), padding='same', activation='relu',
                   kernel_initializer=glorot_uniform(0))(x)

        # side output
        side_output = Conv2D(filters=dim_output, kernel_size=(1, 1), padding='same', activation=None,
                             kernel_initializer=glorot_uniform(0))(x)
        if final_activation == 'softmax':
            side_output = Softmax(axis=(-3, -2))(side_output)
        elif final_activation == 'sigmoid':
            side_output = Activation(activation='sigmoid')(side_output)
        else:
            raise Exception('Unexpected activation: {}'.format(final_activation))
        final_side_output = UpSampling2D(size=(4, 4), interpolation='bilinear')(side_output)

        x = Conv2D(filters=final_dim_feature, kernel_size=(1, 1), padding='same', activation='relu',
                   kernel_initializer=glorot_uniform(0))(x)
        side_x = Conv2D(filters=final_dim_feature, kernel_size=(1, 1), padding='same', activation='relu',
                        kernel_initializer=glorot_uniform(0))(side_output)
        if jumpwire_mode == 'add':
            output = Add()([x, side_x])
        elif jumpwire_mode == 'concat':
            output = Concatenate(axis=-1)([x, side_x])
            output = Conv2D(filters=final_dim_feature, kernel_size=(1, 1), padding='same', activation='relu',
                            kernel_initializer=glorot_uniform(0))(output)
        else:
            raise Exception('Undefined jumpwire_mode: ' + jumpwire_mode)

        return output, final_side_output

    return f


# One layer in a single hourglass
def _hourglass_layer(inner_func, output_dim,
                     upper_down_block_num=(3, 3),
                     jumpwire_mode='add'):
    def f(inputs):
        x_up = inputs
        x_down = inputs

        # upper rout
        for i in range(upper_down_block_num[0] - 1):
            x_up = _identity_block(dim_feature=output_dim)(x_up)
        x_up = _identity_block(dim_feature=output_dim)(x_up)

        # down rout
        x_down = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(x_down)
        for i in range(upper_down_block_num[1]):
            x_down = _identity_block(dim_feature=output_dim)(x_down)
        x_down = inner_func(x_down)
        x_down = _identity_block(dim_feature=output_dim)(x_down)
        x_down = UpSampling2D(size=(2, 2), interpolation='bilinear')(x_down)

        if jumpwire_mode == 'add':
            output = Add()([x_up, x_down])
        elif jumpwire_mode == 'concat':
            output = Concatenate(axis=-1)([x_up, x_down])
            output = Conv2D(filters=output_dim, kernel_size=(1, 1), padding='same', activation='relu',
                            kernel_initializer=glorot_uniform(0))(output)
        else:
            raise Exception('Undefined jumpwire_mode: ' + jumpwire_mode)

        return output

    return f


# A ResNet basic residual block
# filters: filters for each conv layer
# kernel_size: kernel_size for each conv layer
def _identity_block(dim_feature=128, filters=(64, 64), kernel_size=((1, 1), (3, 3), (1, 1))):
    def f(inputs):
        assert len(filters) + 1 == len(kernel_size)

        x = inputs
        x_shortcut = inputs

        # convs
        for i in range(len(filters)):
            x = Conv2D(filters=filters[i], kernel_size=kernel_size[i], padding='same',
                       kernel_initializer=glorot_uniform(0))(x)
            x = BatchNormalization(axis=3)(x)
            x = Activation('relu')(x)

        x = Conv2D(filters=dim_feature, kernel_size=kernel_size[-1], padding='same',
                   kernel_initializer=glorot_uniform(0))(x)
        x = BatchNormalization(axis=3)(x)

        # shortcut
        output = Add()([x, x_shortcut])
        output = Activation('relu')(output)

        return output

    return f


# for label in [0,1]
def keypoint_loss(gamma=1.0, alpha=0.9, smooth=1e-6, loss_shrink=0.001):
    # without smooth loss may be inf
    def loss_func_keypoint(y_true, y_pred):
        loss_pos = - K.sum(alpha * K.pow(y_true, gamma) * K.log(y_pred + smooth))
        loss_neg = - K.sum((1.0-alpha) * K.pow((1.0 - y_true), gamma) * K.log(1.0 - y_pred + smooth))
        loss = (loss_pos + loss_neg) * loss_shrink  # if don't shrink, the loss may be too big to show
        return loss
    return loss_func_keypoint


# for label in {0,1}
def focal_loss(gamma=1.0, alpha=0.9, smooth=1e-6):
    def loss_func_focal(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return - K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + smooth)) \
               - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + smooth))
    return loss_func_focal


if __name__ == '__main__':
    model = build_stacked_hourglass(dim_output=7, n_hourglass=3, n_hourglass_layer=3,
                                    jumpwire_mode='concat', final_activation='sigmoid')
    model.summary()
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file='model.png', show_shapes=False)
