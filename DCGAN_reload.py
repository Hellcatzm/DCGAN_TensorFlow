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
import utils
import numpy as np
import tensorflow as tf
from DCGAN_function import dcgan

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def reload_dcgan():
    if not os.path.exists("./logs/model"):
        tf.logging.info("[*] Failed to find direct './logs/model'")
        return -1
    end_points = dcgan()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        ckpt = tf.train.get_checkpoint_state("./logs/model")
        if ckpt is not None:
            tf.logging.info("[*] Success to read {}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            tf.logging.info("[*] Failed to find a checkpoint")
        sample_z = np.random.uniform(-1, 1, size=(64, 100))
        samples = sess.run(end_points['sample_output'], feed_dict={end_points['initial_z']: sample_z})
        utils.save_images(samples, utils.image_manifold_size(samples.shape[0]),
                          './reload.png')

if __name__ == '__main__':
    reload_dcgan()
