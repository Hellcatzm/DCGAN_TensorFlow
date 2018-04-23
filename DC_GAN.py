# Author : hellcat
# Time   : 18-1-21

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
 
import numpy as np
np.set_printoptions(threshold=np.inf)
"""

import os
import math
from ops import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import utils
import TFR_process

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

BATCH_SIZE = TFR_process.BATCH_SIZE
IMAGE_SIZE = 64
IMAGE_CHANNEL = 3
EPOCH = 60
IMAGE_PATH = "./Data_Set/cartoon_faces"
NUM_IMAGE = len(os.listdir(IMAGE_PATH))
STEPS = NUM_IMAGE // BATCH_SIZE


def conv_out_size_same(size, stride):
    """向上取整，用于确定上采样层输出的shape"""
    return int(math.ceil(float(size) / float(stride)))


class DCGAN(object):

    def __init__(self,
                 sess,
                 learning_rate=0.0002, beat1=0.5,
                 z_dim=100,
                 c_dim=3, batch_size=BATCH_SIZE,
                 gf_dim=64, gfc_dim=1024,
                 df_dim=64, dfc_dim=1024,
                 input_height=48, input_width=48):

        self.sess = sess
        self.z_dim = z_dim  # 噪声向量长度
        self.c_dim = c_dim  # 图片channel数目
        self.gf_dim = gf_dim  # G生成通道基准
        self.gfc_dim = gfc_dim  # ac_gan中最初还原的向量长度
        self.df_dim = df_dim  # D生成通道基准
        self.dfc_dim = dfc_dim  # ac_gan中最后一层全连接的输入维度的向量长度
        self.batch_size = batch_size  # 训练批次图数目
        self.input_height = input_height  # 图片高度
        self.input_width = input_width  # 图片宽度
        
        self.inputs, _ = TFR_process.batch_from_tfr()
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        
        # real数据通过判别器
        d, d_logits = self.discriminator(self.inputs, reuse=False)
        # fake数据生成
        g = self.generator(self.z)
        # fake数据通过判别器，注意来源不同的数据流流经同一结构，要reuse
        d_, d_logits_ = self.discriminator(g, reuse=True)
        # 用生成器生成示例的节点，其数据来源于上面的g相同，故图不需要reuse
        self.s = self.generator(self.z, train=False)

        # 损失函数生成
        # D的real损失：使真实图片进入D后输出为1,只训练D的参数
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits, labels=tf.ones_like(d)))
        # D的fake损失：噪声经由G后进入D，使D的输出为0，只训练D的参数
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_, labels=tf.zeros_like(d_)))

        # G的损失：噪声经由G后进入D，使D的输出为1，只训练G的参数
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_, labels=tf.ones_like(d_)))
        # D的损失：D的real损失 + D的fake损失，只训练D的参数
        self.d_loss = tf.add(self.d_loss_real, self.d_loss_fake)

        t_vars = tf.trainable_variables()
        g_vars = [var for var in t_vars if var.name.startswith('generator')]
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

        self.d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beat1) \
            .minimize(self.d_loss, var_list=d_vars)
        self.g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beat1) \
            .minimize(self.g_loss, var_list=g_vars)

        z_sum = tf.summary.histogram("z", self.z)
        d_sum = tf.summary.histogram("d", d)
        d__sum = tf.summary.histogram("d_", d_)
        g_sum = tf.summary.image("G", g)
        d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.g_sum = tf.summary.merge([z_sum, d__sum, g_sum, d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([z_sum, d_sum, d_loss_real_sum, d_loss_sum])

        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

    def train(self):
        self.sess.run([tf.global_variables_initializer(),
                       tf.local_variables_initializer()])  # <-----线程相关不要忘了它
        
       with self.sess.as_default():
        
            # 加载预训练模型
            if not os.path.exists("./logs/model"):
                os.makedirs("./logs/model")
            ckpt = tf.train.get_checkpoint_state("./logs/model")
            if ckpt is not None:
                print("[*] Success to read {}".format(ckpt.model_checkpoint_path))
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            else:
                print("[*] Failed to find a checkpoint")

            # 线程相关对象初始化
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

            # 进入循环
            for epoch in range(EPOCH):
                for step in range(STEPS):
                    counter = epoch*STEPS+step
                    # 准备数据
                    batch_z = np.random.uniform(-1, 1, [TFR_process.BATCH_SIZE, self.z_dim]).astype(np.float32)
                    
                    # 训练部分，每训练一次判别器需要训练两次生成器 
                    _, summary_str = self.sess.run([self.d_optim, self.d_sum], feed_dict={self.z: batch_z})
                    self.writer.add_summary(summary_str, counter) 
                    _, summary_str = self.sess.run([self.g_optim, self.g_sum], feed_dict={self.z: batch_z})
                    self.writer.add_summary(summary_str, counter)
                    _, summary_str = self.sess.run([self.g_optim, self.g_sum], feed_dict={self.z: batch_z})
                    self.writer.add_summary(summary_str, counter)
                    
                    # 获取loss值进行展示
                    error_d_fake = self.d_loss_fake.eval({self.z: batch_z})
                    error_d_real = self.d_loss_real.eval()
                    error_g = self.g_loss.eval({self.z: batch_z})
                    print("epoch: {0}, step: {1}".format(epoch, step))
                    print("d_loss: {0:.8f}, g_loss: {1:.8f}".format(error_d_fake + error_d_real, error_g))
                    
                    # 模型保存与中间结果展示
                    if np.mod(step, 50) == 0:
                        # self.save(epoch, step, counter + counter_)
                        self.saver.save(self.sess, "./logs/model/DCGAN.model", global_step=epoch*STEPS+step)

                        sample_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
                        samples = self.sess.run(self.s, feed_dict={self.z: sample_z})
                        utils.save_images(samples, utils.image_manifold_size(samples.shape[0]),
                                          './train_{:02d}_{:04d}.png'.format(epoch, step))
            # 线程控制对象关闭
            coord.request_stop()
            coord.join(threads)

    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse) as scope:

            h0 = lrelu(conv2d(image, self.df_dim, scope='d_h0_conv'))
            h1 = lrelu(batch_normal(conv2d(h0, self.df_dim * 2, scope='d_h1_conv'), scope='d_bn1'))
            h2 = lrelu(batch_normal(conv2d(h1, self.df_dim * 4, scope='d_h2_conv'), scope='d_bn2'))
            h3 = lrelu(batch_normal(conv2d(h2, self.df_dim * 8, scope='d_h3_conv'), scope='d_bn3'))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, scope='d_h4_lin')

            return tf.nn.sigmoid(h4), h4

    def generator(self, z, train=True):
        """生成器"""
        with tf.variable_scope("generator") as scope:
            if not train:
                scope.reuse_variables()

            s_h, s_w = self.input_height, self.input_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            z_ = linear(
                z, self.gf_dim * 8 * s_h16 * s_w16, scope='g_h0_lin')

            h0 = tf.reshape(z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(batch_normal(h0, train=train, scope='g_bn0'))

            h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], scope='g_h1')
            h1 = tf.nn.relu(batch_normal(h1, train=train, scope='g_bn1'))

            h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], scope='g_h2')
            h2 = tf.nn.relu(batch_normal(h2, train=train, scope='g_bn2'))

            h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], scope='g_h3')
            h3 = tf.nn.relu(batch_normal(h3, train=train, scope='g_bn3'))

            h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], scope='g_h4')

            return tf.nn.tanh(h4)


if __name__ == "__main__":
    # sess = tf.Session(config=config)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess_ = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    dc_gan = DCGAN(sess_)
    dc_gan.train()

