
# Author : hellcat
# Time   : 18-4-23

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
 
import numpy as np
np.set_printoptions(threshold=np.inf)
 
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
"""

import os
import math
from ops import *
import numpy as np
import tensorflow as tf
import utils
import TFR_process

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

BATCH_SIZE = TFR_process.BATCH_SIZE
# IMAGE_SIZE = 64
# IMAGE_CHANNEL = 3
EPOCH = 60
IMAGE_PATH = "./Data_Set/cartoon_faces"
NUM_IMAGE = len(os.listdir(IMAGE_PATH))
STEPS = NUM_IMAGE // BATCH_SIZE


def conv_out_size_same(size, stride):
    """向上取整，用于确定上采样层输出的shape"""
    return int(math.ceil(float(size) / float(stride)))


def dcgan(learning_rate=0.0002,
          beat1=0.5, z_dim=100,
          c_dim=3, batch_size=BATCH_SIZE,
          gf_dim=64, df_dim=64,
          input_height=48, input_width=48):
    end_points = {}

    # batch_size, df_dim=64
    def discriminator(image, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse) as scope:
            h0 = lrelu(conv2d(image, df_dim, scope='d_h0_conv'))
            h1 = lrelu(batch_normal(conv2d(h0, df_dim * 2, scope='d_h1_conv'), scope='d_bn1'))
            h2 = lrelu(batch_normal(conv2d(h1, df_dim * 4, scope='d_h2_conv'), scope='d_bn2'))
            h3 = lrelu(batch_normal(conv2d(h2, df_dim * 8, scope='d_h3_conv'), scope='d_bn3'))
            h4 = linear(tf.reshape(h3, [batch_size, -1]), 1, scope='d_h4_lin')
            return tf.nn.sigmoid(h4), h4

    # input_height=48, input_width=48, c_dim=3, gf_dim=64
    def generator(z, train=True):
        """生成器"""
        with tf.variable_scope("generator") as scope:
            if not train:
                scope.reuse_variables()
            s_h, s_w = input_height, input_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            z_ = linear(
                z, gf_dim * 8 * s_h16 * s_w16, scope='g_h0_lin')
            h0 = tf.reshape(z_, [-1, s_h16, s_w16, gf_dim * 8])
            h0 = tf.nn.relu(batch_normal(h0, train=train, scope='g_bn0'))
            h1 = deconv2d(h0, [batch_size, s_h8, s_w8, gf_dim * 4], scope='g_h1')
            h1 = tf.nn.relu(batch_normal(h1, train=train, scope='g_bn1'))
            h2 = deconv2d(h1, [batch_size, s_h4, s_w4, gf_dim * 2], scope='g_h2')
            h2 = tf.nn.relu(batch_normal(h2, train=train, scope='g_bn2'))
            h3 = deconv2d(h2, [batch_size, s_h2, s_w2, gf_dim * 1], scope='g_h3')
            h3 = tf.nn.relu(batch_normal(h3, train=train, scope='g_bn3'))
            h4 = deconv2d(h3, [batch_size, s_h, s_w, c_dim], scope='g_h4')
            return tf.nn.tanh(h4)

    inputs, _ = TFR_process.batch_from_tfr()
    # z_dim
    z = tf.placeholder(tf.float32, [None, z_dim], name='z')

    d, d_logits = discriminator(inputs, reuse=False)
    g = generator(z)
    d_, d_logits_ = discriminator(g, reuse=True)
    s = generator(z, train=False)

    # 损失函数生成
    # D的real损失：使真实图片进入D后输出为1,只训练D的参数
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits, labels=tf.ones_like(d)))
    # D的fake损失：噪声经由G后进入D，使D的输出为0，只训练D的参数
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_, labels=tf.zeros_like(d_)))

    # G的损失：噪声经由G后进入D，使D的输出为1，只训练G的参数
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_, labels=tf.ones_like(d_)))
    # D的损失：D的real损失 + D的fake损失，只训练D的参数
    d_loss = tf.add(d_loss_real, d_loss_fake)

    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if var.name.startswith('generator')]
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

    d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beat1) \
        .minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beat1) \
        .minimize(g_loss, var_list=g_vars)

    z_sum = tf.summary.histogram("z", z)
    d_sum = tf.summary.histogram("d", d)
    d__sum = tf.summary.histogram("d_", d_)
    g_sum = tf.summary.image("G", g)
    d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
    d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
    g_loss_sum = tf.summary.scalar("g_loss", g_loss)
    d_loss_sum = tf.summary.scalar("d_loss", d_loss)
    g_total_sum = tf.summary.merge([z_sum, d__sum, g_sum, d_loss_fake_sum, g_loss_sum])
    d_total_sum = tf.summary.merge([z_sum, d_sum, d_loss_real_sum, d_loss_sum])

    end_points['initial_z'] = z
    end_points['sample_output'] = s
    end_points['d_loss_real'] = d_loss_real
    end_points['d_loss_fake'] = d_loss_fake
    end_points['g_loss'] = g_loss
    end_points['d_optim'] = d_optim
    end_points['g_optim'] = g_optim
    end_points['g_sum'] = g_total_sum
    end_points['d_sum'] = d_total_sum

    return end_points


def dcgan_train(z_dim=100, batch_size=BATCH_SIZE):

    end_points = dcgan(z_dim=z_dim, batch_size=batch_size)

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter("./logs", sess.graph)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())  # <-----线程相关不要忘了它

        sess.run(init_op)

        if not os.path.exists("./logs/model"):
            os.makedirs("./logs/model")
        ckpt = tf.train.get_checkpoint_state("./logs/model")
        if ckpt is not None:
            print("[*] Success to read {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("[*] Failed to find a checkpoint")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for epoch in range(EPOCH):
            for step in range(STEPS):
                counter = epoch * STEPS + step
                batch_z = np.random.uniform(-1, 1, [TFR_process.BATCH_SIZE, z_dim]).astype(np.float32)

                _, summary_str = sess.run([end_points['d_optim'], end_points['d_sum']],
                                          feed_dict={end_points['initial_z']: batch_z})
                writer.add_summary(summary_str, counter)

                _, summary_str = sess.run([end_points['g_optim'], end_points['g_sum']],
                                          feed_dict={end_points['initial_z']: batch_z})
                writer.add_summary(summary_str, counter)
                _, summary_str = sess.run([end_points['g_optim'], end_points['g_sum']],
                                          feed_dict={end_points['initial_z']: batch_z})
                writer.add_summary(summary_str, counter)

                error_d_fake = end_points['d_loss_fake'].eval({end_points['initial_z']: batch_z})
                error_d_real = end_points['d_loss_real'].eval()
                error_g = end_points['g_loss'].eval({end_points['initial_z']: batch_z})
                print("epoch: {0}, step: {1}".format(epoch, step))
                print("d_loss: {0:.8f}, g_loss: {1:.8f}".format(error_d_fake + error_d_real, error_g))

                if np.mod(step, 50) == 0:
                    # save(epoch, step, counter + counter_)
                    saver.save(sess, "./logs/model/DCGAN.model", global_step=epoch * STEPS + step)

                    sample_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
                    samples = sess.run(end_points['sample_output'], feed_dict={end_points['initial_z']: sample_z})
                    utils.save_images(samples, utils.image_manifold_size(samples.shape[0]),
                                      './train_{:02d}_{:04d}.png'.format(epoch, step))

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    dcgan_train()

