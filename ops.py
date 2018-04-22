# Author : hellcat
# Time   : 18-1-21
# Usage  : 网络层函数封装
"""
conv2d
deconv2d
lrelu
linear
"""

import tensorflow as tf


# def batch_normal(x, train=True, epsilon=1e-5, decay=0.9, scope="batch_norm"):
#     return tf.contrib.layers.batch_norm(x,
#                                         decay=decay,
#                                         updates_collections=None,
#                                         epsilon=epsilon,
#                                         scale=True,
#                                         is_training=train,
#                                         scope=scope)

def batch_normal(x, epsilon=1e-5, momentum=0.9, train=True, scope='batch_norm'):
    with tf.variable_scope(scope):
        return tf.contrib.layers.batch_norm(x,
                                            decay=momentum,
                                            updates_collections=None,
                                            epsilon=epsilon,
                                            scale=True,
                                            is_training=train)
'''
Note: when training, the moving_mean and moving_variance need to be updated.
By default the update ops are placed in `tf.GraphKeys.UPDATE_OPS`, so they
need to be added as a dependency to the `train_op`. For example:

```python
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)
```

One can set updates_collections=None to force the updates in place, but that
can have a speed penalty, especially in distributed settings.
'''


# class batch_norm(object):
#     def __init__(self, epsilon=1e-5, decay=0.9, scope="batch_norm"):
#         with tf.variable_scope(scope):
#             self.epsilon = epsilon
#             self.decay = decay
#             # self.scope = scope
#
#     def __call__(self, x, scope, train=True):
#         return tf.contrib.layers.batch_norm(x,
#                                             decay=self.decay,
#                                             updates_collections=None,
#                                             epsilon=self.epsilon,
#                                             scale=True,
#                                             is_training=train,
#                                             scope=scope)


def concat(tensor_a, tensor_b):
    """
    组合Tensor,注意的是这里tensor_a的宽高应该大于等于tensor_b
    :param tensor_a: 前面的tensor
    :param tensor_b: 后面的tensor
    :return:
    """
    if tensor_a.get_shape().as_list()[1] > tensor_b.get_shape().as_list()[1]:
        return tf.concat([tf.slice(tensor_a,
                                   begin=[0, (int(tensor_a.shape[1]) - int(tensor_b.shape[1])) // 2,
                                          (int(tensor_a.shape[1]) - int(tensor_b.shape[1])) // 2, 0],
                                   size=[int(tensor_b.shape[0]), int(tensor_b.shape[1]),
                                         int(tensor_b.shape[2]), int(tensor_a.shape[3])],
                                   name='slice'),
                          tensor_b],
                         axis=3, name='concat')

    elif tensor_a.get_shape().as_list()[1] < tensor_b.get_shape().as_list()[1]:
        return tf.concat([tensor_a,
                          tf.slice(tensor_b,
                                   begin=[0, (int(tensor_b.shape[1]) - int(tensor_a.shape[1])) // 2,
                                          (int(tensor_b.shape[1]) - int(tensor_a.shape[1])) // 2, 0],
                                   size=[int(tensor_a.shape[0]), int(tensor_a.shape[1]),
                                         int(tensor_a.shape[2]), int(tensor_b.shape[3])],
                                   name='slice')],
                         axis=3, name='concat')
    else:
        return tf.concat([tensor_a, tensor_b], axis=3)


def conv_cond_concat(x, y):
    """
    广播并连接向量,用于ac_gan的标签对矩阵拼接
    :param x: features，例如shape：[n,16,16,128]
    :param y: 扩暂维度后的标签，例如shape：[n,1,1,10]
    :return: 拼接后features，例如：[n,16,16,138]
    """
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], axis=3)


def conv2d(input_, output_dim,
           k_h=5, k_w=5, s_h=2, s_w=2, stddev=0.02,
           scope="conv2d", with_w=False, with_bias=True):
    """
    卷积网络封装
    :param input_: 
    :param output_dim: 输出的feature数目
    :param k_h: 
    :param k_w: 
    :param s_h: 
    :param s_w: 
    :param stddev: 
    :param scope: 
    :param with_w: 
    :param with_bias: 是否含有bias层
    :return: 
    """

    with tf.variable_scope(scope):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, s_h, s_w, 1], padding='SAME')
        if with_bias:
            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        else:
            biases = None

    if with_w:
        return conv, w, biases
    else:
        return conv


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, s_h=2, s_w=2, stddev=0.02,
             scope="deconv2d", with_w=False):
    """
    转置卷积网络封装
    :param input_: 
    :param output_shape: 输出的shape
    :param k_h: 
    :param k_w: 
    :param s_h: 
    :param s_w: 
    :param stddev: 
    :param scope: 
    :param with_w: 
    :return: 
    """
    with tf.variable_scope(scope):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, s_h, s_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2):
    """
    Leak_Relu层封装
    :param x: 
    :param leak: 
    :return: 
    """
    return tf.maximum(x, leak*x)


def linear(input_, output_size,
           stddev=0.02, bias_start=0.0,
           scope=None, with_w=False):
    """
    全连接层封装
    :param input_: 
    :param output_size: 输出节点数目
    :param scope: 
    :param stddev: 
    :param bias_start: 使用常数初始化偏执，常数值设定
    :param with_w: 返回是否返回参数Variable
    :return: 
    """
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))

        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

