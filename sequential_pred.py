# -*- coding: utf-8 -*-

"""Make predictions sequentially"""

import argparse
import os
import sys
import math
import numpy as np
from tl_util.image_preprocess import parse_function, val_preprocess
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets


CURR_DIR = os.path.join(os.getcwd(), 'predictions')
VGG_MEAN = [123.68, 116.78, 103.94]
IMAGE_SIZE = 224

parser = argparse.ArgumentParser()
parser.add_argument('--model_save_path', default=os.path.join(CURR_DIR, 'checkpoints'), type=str)
parser.add_argument('--test_dir', default=os.path.join(CURR_DIR, 'test_images'), type=str)
parser.add_argument('--save_dir', default=CURR_DIR, type=str)
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)


def main(args):
    num_classes = 2  # depend on your own problem

    # --------------------------------------------------------------------------
    # In TensorFlow, you first want to define the computational graph with all the
    # necessary operations: loss, training op, accuracy...
    # Any tensor created in the `graph.as_default()` scope will be part of `graph`
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(2018)  # to keep consistent results

        # Indicates whether we are in training or in testing mode
        is_training = tf.placeholder(tf.bool)

        # Use feed_dict to pass in test images
        X = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
        vgg = tf.contrib.slim.nets.vgg
        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=args.weight_decay)):
            logits, end_points = vgg.vgg_16(X, num_classes=num_classes, is_training=is_training,
                                            dropout_keep_prob=args.dropout_keep_prob)

        if num_classes == 2:
            pred = tf.argmax(tf.nn.softmax(logits), axis=1)
        else:
            pred = logits

        # Create a model saver
        saver = tf.train.Saver()

    files = []
    total_file = 0
    # Count how many images in total
    for filename in os.listdir(args.test_dir):
        if filename.endswith('.png'):  # or filename.endswith('.jpg'):
            total_file += 1
            files.append(filename)
    print(files)
    with open(os.path.join(args.save_dir, 'image_filenames.txt'), 'w') as fout:
        for item in files:
            fout.write('%s\n' % item.split('.png')[0])

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())

        # Try to load model checkpoints if there are any
        try:
            saver.restore(sess, os.path.join(args.model_save_path, 'phase2'))
            print('Model restored.')
        except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError) as e:
            print(e)
            sys.exit(1)

        counter = 0
        res = None
        for i in range(math.ceil(total_file / 10)):
            filenames = []
            test_images = []
            for j in range(10):
                filename = files[i*10+j]
                filenames.append(filename.split('.png')[0])
                resized_image, _ = parse_function(os.path.join(args.test_dir, filename), '')
                centered_image, _ = val_preprocess(resized_image, '')
                test_images.append(centered_image.eval())
                counter += 1
                if counter == total_file:
                    break
            print(filenames)
            test_images = np.array(test_images)
            predictions = sess.run(pred, feed_dict={is_training: False, X: test_images})
            if i == 0:
                res = predictions
            else:
                res = np.concatenate((res, predictions), axis=0)

        np.save(os.path.join(args.save_dir, 'pred_array.npy'), res)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
