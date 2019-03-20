import tensorflow as tf
from tflearn.layers.conv import global_avg_pool

slim = tf.contrib.slim

reduction_ratio = 4


def basenet(inputs):
    """
    backbone net of vgg16
    """
    # End_points collect relevant activations for external use.
    end_points = {}
    # Original VGG-16 blocks.
    with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME'):
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        end_points['conv1_2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        # Block 2.
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        end_points['conv2_2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        # Block 3.
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        end_points['conv3_3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool3')

        # Block 4.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        # Block se
        net = squeeze_excitation_layer(net, 512, reduction_ratio, 'squeeze_layer')
        end_points['conv4_3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        # Block 5.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        end_points['conv5_3'] = net
        net = slim.max_pool2d(net, [3, 3], 1, scope='pool5')

        # fc6 as conv, dilation is added
        net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='fc6')
        end_points['fc6'] = net

        # fc7 as conv
        net = slim.conv2d(net, 1024, [1, 1], scope='fc7')
        end_points['fc7'] = net

    return net, end_points;

def Fully_connected(x,units, layer_name='fully_connected'):
    with tf.name_scope(layer_name):
        return tf.layers.dense(inputs=x, use_bias=False, units=units)

def Relu(x):
    return tf.nn.relu(x)

def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

def Sigmoid(x):
    return tf.nn.sigmoid(x)

#添加ＳeNet模块　
def squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name) :

        squeeze = Global_Average_Pooling(input_x)

        excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
        excitation = Relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
        excitation = Sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1,1,1,out_dim])
        scale = input_x * excitation

        return scale


