# coding=utf-8
"""Evaluate the attack success rates under different DNNs models """

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import random
import numpy as np
# from scipy.misc import imread, imresize, imsave
from imageio import imread #, imresize, imsave
import pandas as pd

from DNNModels import InceptionV3Model, InceptionV4Model, InceptionResnetModel, ResNetModel
from func import *
import warnings
from tqdm import tqdm


warnings.filterwarnings('ignore')

RESULT_FILE = 'result.json'

tf.flags.DEFINE_integer('batch_size', 50, 'How many images process at one time.')

tf.flags.DEFINE_integer('size', 3, 'Number of randomly sampled images')

tf.flags.DEFINE_float('portion', 0.6, 'protion for the mixed image')

tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_string('output_dir', './outputs', 'Output directory with images.')

tf.flags.DEFINE_integer('percentile', 90, "Name of the percentile")

tf.flags.DEFINE_string('model_name', "InceptionV3Model", "Name of the model")

tf.flags.DEFINE_string('attack_method', "", "Name of the model")

tf.flags.DEFINE_integer('sigma', 10, "Name of the model")

tf.flags.DEFINE_string('mix_op', 'mixup', 'Output directory with images.')

FLAGS = tf.flags.FLAGS

slim = tf.contrib.slim
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

checkpoint_path = './models'
model_checkpoint_map = {
    'InceptionV3Model': os.path.join(checkpoint_path, 'InceptionV3Model.ckpt'),
    'ens3_adv_InceptionV3Model': os.path.join(checkpoint_path, 'ens3_adv_InceptionV3Model_rename.ckpt'),
    'ens4_adv_InceptionV3Model': os.path.join(checkpoint_path, 'ens4_adv_InceptionV3Model_rename.ckpt'),
    'InceptionV4Model': os.path.join(checkpoint_path, 'InceptionV4Model.ckpt'),
    'InceptionResnetModel': os.path.join(checkpoint_path, 'InceptionResnetModel_2016_08_30.ckpt'),
    'ens_adv_InceptionResnetModel': os.path.join(checkpoint_path, 'ens_adv_InceptionResnetModel_rename.ckpt'),
    'ResNetModel': os.path.join(checkpoint_path, 'ResNetModel_101.ckpt')}


def load_labels(file_name):
    dev = pd.read_csv(file_name)
    f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label'] for i in range(len(dev))}
    return f2l


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.
    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
    Yields:
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*')):
        with tf.gfile.Open(filepath, 'rb') as f:
            image = imread(f, pilmode='RGB').astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images


if __name__ == '__main__':
    f2l = load_labels('./dev_data/val_rs.csv')
    if 'ens' in FLAGS.attack_method:
        input_dir = os.path.join(FLAGS.output_dir,
                                 'ens_attacks',
                                 f'method_{FLAGS.attack_method}')
        result_file = os.path.join(FLAGS.output_dir, 'ens_attacks')
    else:
        input_dir = os.path.join(FLAGS.output_dir,
                                 FLAGS.mix_op,
                                 f'model_{FLAGS.model_name}_method_{FLAGS.attack_method}_{FLAGS.percentile}_{FLAGS.size}_sigma_{FLAGS.sigma}_portion_{FLAGS.portion}')
        result_file = os.path.join(FLAGS.output_dir, FLAGS.mix_op)

    batch_shape = [FLAGS.batch_size, FLAGS.image_width, FLAGS.image_height, 3]
    num_classes = 1001
    tf.logging.set_verbosity(tf.logging.ERROR)

    with tf.Graph().as_default():
        x_input = tf.placeholder(tf.float32, shape=batch_shape)

        with slim.arg_scope(InceptionV3Model.InceptionV3Model_arg_scope()):
            logits_v3, end_points_v3 = InceptionV3Model.InceptionV3Model(
                x_input, num_classes=num_classes, is_training=False)

        with slim.arg_scope(InceptionV3Model.InceptionV3Model_arg_scope()):
            logits_ens3_adv_v3, end_points_ens3_adv_v3 = InceptionV3Model.InceptionV3Model(
                x_input, num_classes=num_classes, is_training=False, scope='Ens3AdvInceptionV3')

        with slim.arg_scope(InceptionV3Model.InceptionV3Model_arg_scope()):
            logits_ens4_adv_v3, end_points_ens4_adv_v3 = InceptionV3Model.InceptionV3Model(
                x_input, num_classes=num_classes, is_training=False, scope='Ens4AdvInceptionV3')

        with slim.arg_scope(InceptionV4Model.InceptionV4Model_arg_scope()):
            logits_v4, end_points_v4 = InceptionV4Model.InceptionV4Model(
                x_input, num_classes=num_classes, is_training=False)

        with slim.arg_scope(InceptionResnetModel.InceptionResnetModel_arg_scope()):
            logits_res_v2, end_points_res_v2 = InceptionResnetModel.InceptionResnetModel(
                x_input, num_classes=num_classes, is_training=False)

        with slim.arg_scope(InceptionResnetModel.InceptionResnetModel_arg_scope()):
            logits_ens_adv_res_v2, end_points_ens_adv_res_v2 = InceptionResnetModel.InceptionResnetModel(
                x_input, num_classes=num_classes, is_training=False, scope='EnsAdvInceptionResnetV2')

        with slim.arg_scope(ResNetModel.resnet_arg_scope()):
            logits_resnet, end_points_resnet = ResNetModel.ResNetModel_101(
                x_input, num_classes=num_classes, is_training=False)

        pred_v3 = tf.argmax(end_points_v3['Predictions'], 1)
        pred_ens3_adv_v3 = tf.argmax(end_points_ens3_adv_v3['Predictions'], 1)
        pred_ens4_adv_v3 = tf.argmax(end_points_ens4_adv_v3['Predictions'], 1)
        pred_v4 = tf.argmax(end_points_v4['Predictions'], 1)
        pred_res_v2 = tf.argmax(end_points_res_v2['Predictions'], 1)
        pred_ens_adv_res_v2 = tf.argmax(end_points_ens_adv_res_v2['Predictions'], 1)
        pred_resnet = tf.argmax(end_points_resnet['Predictions'], 1)

        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
        s2 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
        s4 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        s5 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        s6 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))
        s7 = tf.train.Saver(slim.get_model_variables(scope='ResNetModel'))

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            s1.restore(sess, model_checkpoint_map['InceptionV3Model'])
            s2.restore(sess, model_checkpoint_map['ens3_adv_InceptionV3Model'])
            s3.restore(sess, model_checkpoint_map['ens4_adv_InceptionV3Model'])
            s4.restore(sess, model_checkpoint_map['InceptionV4Model'])
            s5.restore(sess, model_checkpoint_map['InceptionResnetModel'])
            s6.restore(sess, model_checkpoint_map['ens_adv_InceptionResnetModel'])
            s7.restore(sess, model_checkpoint_map['ResNetModel'])

            model_name = ['InceptionV3Model', 'InceptionV4Model', 'InceptionResnetModel',
                          'ResNetModel', 'ens3_adv_InceptionV3Model', 'ens4_adv_InceptionV3Model',
                          'ens_adv_InceptionResnetModel']
            success_count = np.zeros(len(model_name))

            idx = 0
            for filenames, images in tqdm(load_images(input_dir, batch_shape),
                                          desc=f"Evaluate {FLAGS.model_name}_{FLAGS.percentile}_{FLAGS.size}",
                                          ncols=100):
                idx += 1
                # print("start the i={} eval".format(idx))
                v3, ens3_adv_v3, ens4_adv_v3, v4, res_v2, ens_adv_res_v2, resnet = sess.run(
                    (pred_v3, pred_ens3_adv_v3, pred_ens4_adv_v3, pred_v4, pred_res_v2,
                     pred_ens_adv_res_v2, pred_resnet), feed_dict={x_input: images})

                for filename, l1, l2, l3, l4, l5, l6, l7 in zip(filenames, v3, ens3_adv_v3,
                                                                    ens4_adv_v3, v4, res_v2, ens_adv_res_v2,
                                                                    resnet):
                    label = f2l[filename]
                    l = [l1, l4, l5, l7, l2, l3, l6]
                    for i in range(len(model_name)):
                        if l[i] != label:
                            success_count[i] += 1
            result = dict()
            # result[f'percentile_{FLAGS.percentile}'] = dict()
            for i in range(len(model_name)):
                print("Attack Success Rate for {0} : {1:.1f}%".format(model_name[i], success_count[i] / 1000. * 100))
                result[f'{model_name[i]}'] = "{0:.1f}%".format(success_count[i] / 1000. * 100)
            save_json_file(result,
                           os.path.join(result_file,
                                        f'result_{FLAGS.model_name}_method_{FLAGS.attack_method}_{FLAGS.percentile}_{FLAGS.size}_sigma_{FLAGS.sigma}_portion_{FLAGS.portion}.json'))
