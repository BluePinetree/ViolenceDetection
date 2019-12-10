from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Concatenate, BatchNormalization, AveragePooling3D, Dropout, ReLU, Softmax, Lambda

VALID_ENDPOINTS = (
      'Conv3d_1a_7x7',
      'MaxPool3d_2a_3x3',
      'Conv3d_2b_1x1',
      'Conv3d_2c_3x3',
      'MaxPool3d_3a_3x3',
      'Mixed_3b',
      'Mixed_3c',
      'MaxPool3d_4a_3x3',
      'Mixed_4b',
      'Mixed_4c',
      'Mixed_4d',
      'Mixed_4e',
      'Mixed_4f',
      'MaxPool3d_5a_2x2',
      'Mixed_5b',
      'Mixed_5c',
      'Logits',
      'Predictions',
  )


def Unit3D(inputs,
           output_channels,
           kernel_shape=(1,1,1),
           strides=(1,1,1),
           activation_fn=ReLU(),
           use_batch_norm=True,
           use_bias=False,
           is_training=True,
           name='unit_3d'):

    net = Conv3D(filters=output_channels, kernel_size=kernel_shape,
                 strides=strides, padding='same',
                 use_bias=use_bias, name=name)(inputs)

    if use_batch_norm:
        net = BatchNormalization()(net, training=is_training)

    if activation_fn is not None:
        net = activation_fn(net)

    return net

def inceptionI3D(inputs, num_classes=2, spatial_squeeze=True, is_training=True,
                 dropout_keep_prob=1.0, final_endpoint='Logits', name='inception_i3d'):

    if final_endpoint not in VALID_ENDPOINTS:
        raise ValueError(f'Unknown final endpoint {final_endpoint}')

    net = inputs
    end_points = {}
    end_point = 'Conv3d_1a_7x7'
    net = Unit3D(net, 64, kernel_shape=(7,7,7), strides=(2,2,2),
                 use_batch_norm=True, name=end_point, is_training=is_training)
    end_points[end_point] = net
    if final_endpoint == end_point : return net, end_points

    end_point = 'MaxPool3d_2a_3x3'
    net = MaxPooling3D(pool_size=(1,3,3), strides=(1,2,2), padding='same', name=end_point)(net)
    end_points[end_point] = net
    if final_endpoint == end_point : return net, end_points

    end_point = 'Conv3d_2b_1x1'
    net = Unit3D(net, output_channels=64, kernel_shape=(1,1,1), name=end_point, is_training=is_training)
    end_points[end_point] = net
    if final_endpoint == end_point : return net, end_points

    end_point = 'Conv3d_2c_3x3'
    net = Unit3D(net, output_channels=192, kernel_shape=(3,3,3), name=end_point, is_training=is_training)
    end_points[end_point] = net
    if final_endpoint == end_point : return net, end_points

    end_point = 'MaxPool3d_3a_3x3'
    net = MaxPooling3D(pool_size=(1,3,3), strides=(1,2,2), padding='same', name=end_point)(net)
    end_points[end_point] = net
    if final_endpoint == end_point : return net, end_points

    end_point = 'Mixed_3b'
    with tf.compat.v1.variable_scope(end_point):
        with tf.compat.v1.variable_scope('Branch_0'):
            branch_0 = Unit3D(net, output_channels=64, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0a_1x1-0', is_training=is_training)
        with tf.compat.v1.variable_scope('Branch_1'):
            branch_1 = Unit3D(net, output_channels=96, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0a_1x1-1', is_training=is_training)
            branch_1 = Unit3D(branch_1, output_channels=128, kernel_shape=(3,3,3), name=f'{end_point}_Conv3d_0b_3x3-1', is_training=is_training)
        with tf.compat.v1.variable_scope('Branch_2'):
            branch_2 = Unit3D(net, output_channels=16, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0a_1x1-2', is_training=is_training)
            branch_2 = Unit3D(branch_2, output_channels=32, kernel_shape=(3,3,3), name=f'{end_point}_Conv3d_0b_3x3-2', is_training=is_training)
        with tf.compat.v1.variable_scope('Branch_3'):
            branch_3 = MaxPooling3D(pool_size=(3,3,3), strides=(1,1,1), padding='same', name=f'{end_point}_MaxPool3d_0a_3x3-3')(net)
            branch_3 = Unit3D(branch_3, output_channels=32, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0b_1x1-3', is_training=is_training)
        net = Concatenate(axis=4)([branch_0, branch_1, branch_2, branch_3])
    end_points[end_point] = net
    if final_endpoint == end_point : return net, end_points

    end_point = 'Mixed_3c'
    with tf.compat.v1.variable_scope(end_point):
        with tf.compat.v1.variable_scope('Branch_0'):
            branch_0 = Unit3D(net, output_channels=128, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0a_1x1-0', is_training=is_training)
        with tf.compat.v1.variable_scope('Branch_1'):
            branch_1 = Unit3D(net, output_channels=128, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0a_1x1-1', is_training=is_training)
            branch_1 = Unit3D(branch_1, output_channels=192, kernel_shape=(3,3,3), name=f'{end_point}_Conv3d_0b_3x3-1', is_training=is_training)
        with tf.compat.v1.variable_scope('Branch_2'):
            branch_2 = Unit3D(net, output_channels=32, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0a_1x1-2', is_training=is_training)
            branch_2 = Unit3D(branch_2, output_channels=96, kernel_shape=(3,3,3), name=f'{end_point}_Conv3d_0b_3x3-2', is_training=is_training)
        with tf.compat.v1.variable_scope('Branch_3'):
            branch_3 = MaxPooling3D(pool_size=(3,3,3), strides=(1,1,1), padding='same', name=f'{end_point}_MaxPool3d_0a_3x3-3')(net)
            branch_3 = Unit3D(branch_3, output_channels=64, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0b_1x1-3', is_training=is_training)
        net = Concatenate(axis=4)([branch_0, branch_1, branch_2, branch_3])
    end_points[end_point] = net
    if final_endpoint == end_point : return net, end_points

    end_point = 'MaxPool3d_4a_3x3'
    net = MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2), padding='same', name=end_point)(net)
    end_points[end_point] = net
    if final_endpoint == end_point : return net, end_points

    end_point = 'Mixed_4b'
    with tf.compat.v1.variable_scope(end_point):
        with tf.compat.v1.variable_scope('Branch_0'):
            branch_0 = Unit3D(net, output_channels=192, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0a_1x1-0', is_training=is_training)
        with tf.compat.v1.variable_scope('Branch_1'):
            branch_1 = Unit3D(net, output_channels=96, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0a_1x1-1', is_training=is_training)
            branch_1 = Unit3D(branch_1, output_channels=208, kernel_shape=(3,3,3), name=f'{end_point}_Conv3d_0b_3x3-1', is_training=is_training)
        with tf.compat.v1.variable_scope('Branch_2'):
            branch_2 = Unit3D(net, output_channels=16, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0a_1x1-2', is_training=is_training)
            branch_2 = Unit3D(branch_2, output_channels=48, kernel_shape=(3,3,3), name=f'{end_point}_Conv3d_0b_3x3-2', is_training=is_training)
        with tf.compat.v1.variable_scope('Branch_3'):
            branch_3 = MaxPooling3D(pool_size=(3,3,3), strides=(1,1,1), padding='same', name=f'{end_point}_MaxPool3d_0a_3x3-3')(net)
            branch_3 = Unit3D(branch_3, output_channels=64, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0b_1x1-3', is_training=is_training)
        net = Concatenate(axis=4)([branch_0, branch_1, branch_2, branch_3])
    end_points[end_point] = net
    if final_endpoint == end_point : return net, end_points

    end_point = 'Mixed_4c'
    with tf.compat.v1.variable_scope(end_point):
        with tf.compat.v1.variable_scope('Branch_0'):
            branch_0 = Unit3D(net, output_channels=160, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0a_1x1-0', is_training=is_training)
        with tf.compat.v1.variable_scope('Branch_1'):
            branch_1 = Unit3D(net, output_channels=112, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0a_1x1-1', is_training=is_training)
            branch_1 = Unit3D(branch_1, output_channels=224, kernel_shape=(3,3,3), name=f'{end_point}_Conv3d_0b_3x3-1', is_training=is_training)
        with tf.compat.v1.variable_scope('Branch_2'):
            branch_2 = Unit3D(net, output_channels=24, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0a_1x1-2', is_training=is_training)
            branch_2 = Unit3D(branch_2, output_channels=64, kernel_shape=(3,3,3), name=f'{end_point}_Conv3d_0b_3x3-2', is_training=is_training)
        with tf.compat.v1.variable_scope('Branch_3'):
            branch_3 = MaxPooling3D(pool_size=(3,3,3), strides=(1,1,1), padding='same', name=f'{end_point}_MaxPool3d_0a_3x3-3')(net)
            branch_3 = Unit3D(branch_3, output_channels=64, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0b_1x1-3', is_training=is_training)
        net = Concatenate(axis=4)([branch_0, branch_1, branch_2, branch_3])
    end_points[end_point] = net
    if final_endpoint == end_point : return net, end_points

    end_point = 'Mixed_4d'
    with tf.compat.v1.variable_scope(end_point):
        with tf.compat.v1.variable_scope('Branch_0'):
            branch_0 = Unit3D(net, output_channels=128, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0a_1x1-0', is_training=is_training)
        with tf.compat.v1.variable_scope('Branch_1'):
            branch_1 = Unit3D(net, output_channels=128, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0a_1x1-1', is_training=is_training)
            branch_1 = Unit3D(branch_1, output_channels=256, kernel_shape=(3,3,3), name=f'{end_point}_Conv3d_0b_3x3-1', is_training=is_training)
        with tf.compat.v1.variable_scope('Branch_2'):
            branch_2 = Unit3D(net, output_channels=24, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0a_1x1-2', is_training=is_training)
            branch_2 = Unit3D(branch_2, output_channels=64, kernel_shape=(3,3,3), name=f'{end_point}_Conv3d_0b_3x3-2', is_training=is_training)
        with tf.compat.v1.variable_scope('Branch_3'):
            branch_3 = MaxPooling3D(pool_size=(3,3,3), strides=(1,1,1), padding='same', name=f'{end_point}_MaxPool3d_0a_3x3-3')(net)
            branch_3 = Unit3D(branch_3, output_channels=64, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0b_1x1-3', is_training=is_training)
        net = Concatenate(axis=4)([branch_0, branch_1, branch_2, branch_3])
    end_points[end_point] = net
    if final_endpoint == end_point : return net, end_points

    end_point = 'Mixed_4e'
    with tf.compat.v1.variable_scope(end_point):
        with tf.compat.v1.variable_scope('Branch_0'):
            branch_0 = Unit3D(net, output_channels=112, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0a_1x1-0', is_training=is_training)
        with tf.compat.v1.variable_scope('Branch_1'):
            branch_1 = Unit3D(net, output_channels=144, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0a_1x1-1', is_training=is_training)
            branch_1 = Unit3D(branch_1, output_channels=288, kernel_shape=(3,3,3), name=f'{end_point}_Conv3d_0b_3x3-1', is_training=is_training)
        with tf.compat.v1.variable_scope('Branch_2'):
            branch_2 = Unit3D(net, output_channels=32, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0a_1x1-2', is_training=is_training)
            branch_2 = Unit3D(branch_2, output_channels=64, kernel_shape=(3,3,3), name=f'{end_point}_Conv3d_0b_3x3-2', is_training=is_training)
        with tf.compat.v1.variable_scope('Branch_3'):
            branch_3 = MaxPooling3D(pool_size=(3,3,3), strides=(1,1,1), padding='same', name=f'{end_point}_MaxPool3d_0a_3x3-3')(net)
            branch_3 = Unit3D(branch_3, output_channels=64, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0b_1x1-3', is_training=is_training)
        net = Concatenate(axis=4)([branch_0, branch_1, branch_2, branch_3])
    end_points[end_point] = net
    if final_endpoint == end_point : return net, end_points

    end_point = 'Mixed_4f'
    with tf.compat.v1.variable_scope(end_point):
        with tf.compat.v1.variable_scope('Branch_0'):
            branch_0 = Unit3D(net, output_channels=256, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0a_1x1-0', is_training=is_training)
        with tf.compat.v1.variable_scope('Branch_1'):
            branch_1 = Unit3D(net, output_channels=160, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0a_1x1-1', is_training=is_training)
            branch_1 = Unit3D(branch_1, output_channels=320, kernel_shape=(3,3,3), name=f'{end_point}_Conv3d_0b_3x3-1', is_training=is_training)
        with tf.compat.v1.variable_scope('Branch_2'):
            branch_2 = Unit3D(net, output_channels=32, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0a_1x1-2', is_training=is_training)
            branch_2 = Unit3D(branch_2, output_channels=128, kernel_shape=(3,3,3), name=f'{end_point}_Conv3d_0b_3x3-2', is_training=is_training)
        with tf.compat.v1.variable_scope('Branch_3'):
            branch_3 = MaxPooling3D(pool_size=(3,3,3), strides=(1,1,1), padding='same', name=f'{end_point}_MaxPool3d_0a_3x3-3')(net)
            branch_3 = Unit3D(branch_3, output_channels=128, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0b_1x1-3', is_training=is_training)
        net = Concatenate(axis=4)([branch_0, branch_1, branch_2, branch_3])
    end_points[end_point] = net
    if final_endpoint == end_point : return net, end_points

    end_point = 'MaxPool3d_5a_2x2'
    net = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='same', name=end_point)(net)
    end_points[end_point] = net
    if final_endpoint == end_point : return net, end_points

    end_point = 'Mixed_5b'
    with tf.compat.v1.variable_scope(end_point):
        with tf.compat.v1.variable_scope('Branch_0'):
            branch_0 = Unit3D(net, output_channels=256, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0a_1x1-0', is_training=is_training)
        with tf.compat.v1.variable_scope('Branch_1'):
            branch_1 = Unit3D(net, output_channels=160, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0a_1x1-1', is_training=is_training)
            branch_1 = Unit3D(branch_1, output_channels=320, kernel_shape=(3,3,3), name=f'{end_point}_Conv3d_0b_3x3-1', is_training=is_training)
        with tf.compat.v1.variable_scope('Branch_2'):
            branch_2 = Unit3D(net, output_channels=32, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0a_1x1-2', is_training=is_training)
            branch_2 = Unit3D(branch_2, output_channels=128, kernel_shape=(3,3,3), name=f'{end_point}_Conv3d_0b_3x3-2', is_training=is_training)
        with tf.compat.v1.variable_scope('Branch_3'):
            branch_3 = MaxPooling3D(pool_size=(3,3,3), strides=(1,1,1), padding='same', name=f'{end_point}_MaxPool3d_0a_3x3-3')(net)
            branch_3 = Unit3D(branch_3, output_channels=128, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0b_1x1-3', is_training=is_training)
        net = Concatenate(axis=4)([branch_0, branch_1, branch_2, branch_3])
    end_points[end_point] = net
    if final_endpoint == end_point : return net, end_points

    end_point = 'Mixed_5c'
    with tf.compat.v1.variable_scope(end_point):
        with tf.compat.v1.variable_scope('Branch_0'):
            branch_0 = Unit3D(net, output_channels=384, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0a_1x1-0', is_training=is_training)
        with tf.compat.v1.variable_scope('Branch_1'):
            branch_1 = Unit3D(net, output_channels=192, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0a_1x1-1', is_training=is_training)
            branch_1 = Unit3D(branch_1, output_channels=384, kernel_shape=(3,3,3), name=f'{end_point}_Conv3d_0b_3x3-1', is_training=is_training)
        with tf.compat.v1.variable_scope('Branch_2'):
            branch_2 = Unit3D(net, output_channels=48, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0a_1x1-2', is_training=is_training)
            branch_2 = Unit3D(branch_2, output_channels=128, kernel_shape=(3,3,3), name=f'{end_point}_Conv3d_0b_3x3-2', is_training=is_training)
        with tf.compat.v1.variable_scope('Branch_3'):
            branch_3 = MaxPooling3D(pool_size=(3,3,3), strides=(1,1,1), padding='same', name=f'{end_point}_MaxPool3d_0a_3x3-3')(net)
            branch_3 = Unit3D(branch_3, output_channels=128, kernel_shape=(1,1,1), name=f'{end_point}_Conv3d_0b_1x1-3', is_training=is_training)
        net = Concatenate(axis=4)([branch_0, branch_1, branch_2, branch_3])
    end_points[end_point] = net
    if final_endpoint == end_point : return net, end_points

    end_point = 'Logits'
    with tf.compat.v1.variable_scope(end_point):
        net = AveragePooling3D(pool_size=(2,7,7), strides=(1,1,1), padding='valid')(net)
        net = Dropout(rate=dropout_keep_prob)(net)
        logits = Unit3D(net, output_channels=num_classes, kernel_shape=(1,1,1),
                     activation_fn=None, use_batch_norm=False, use_bias=True,
                     name=f'{end_point}_Conv3d_0c_1x1', is_training=is_training)
        if spatial_squeeze:
            logits = Lambda(lambda x : tf.squeeze(x, axis=[2,3]), name='SpatialSqueeze')(logits)
            # logits = tf.squeeze(logits, [2,3], name='SpatialSqueeze')
    averaged_logits = Lambda(lambda x : tf.reduce_mean(x, axis=1))(logits)
    end_points[end_point] = averaged_logits
    if final_endpoint == end_point : return averaged_logits, end_points

    end_point = 'Predictions'
    predictions = Softmax()(averaged_logits)
    end_points[end_point] = predictions
    if final_endpoint == end_point : return \
        predictions, end_points

if __name__ == '__main__':
    inputs = Input(shape=[None, 224, 224, 3])
    logits, end_points = inceptionI3D(inputs, dropout_keep_prob=0.5)
    model = Model(inputs, logits)
    model.summary()
