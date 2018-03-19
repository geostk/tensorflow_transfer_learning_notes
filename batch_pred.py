# -*- coding: utf-8 -*-

"""
See https://www.tensorflow.org/versions/master/performance/datasets_performance#summary_of_best_practices for reference
on how to increase performance.

Test results on the same dataset: on a 16 core machine, using Dataset API can shorten the running time
from 16h 47m 11s to 1m 29s.
"""

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets  # need to explicitly import nets, otherwise you will see an AttributeError
# The problem is caused by the fact that https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/
# __init__.py does not export nets.


CURR_DIR = os.path.join(os.getcwd(), 'predictions')
VGG_MEAN = [123.68, 116.78, 103.94]
 
parser = argparse.ArgumentParser()
parser.add_argument('--model_save_path', default=os.path.join(CURR_DIR, 'checkpoints'), type=str)
parser.add_argument('--test_dir', default=os.path.join(CURR_DIR, 'test_images'), type=str)
parser.add_argument('--save_dir', default=CURR_DIR, type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=16, type=int)
# Set num_workers to be the number of CPU cores you have should give you the most efficient results.
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)


def main(args):
    num_classes = 2  # depend on your own problem
    test_filenames = [x for x in os.listdir(args.test_dir) if x.endswith('.png')]
    test_filepaths = [os.path.join(args.test_dir, x) for x in test_filenames]
    
    # --------------------------------------------------------------------------
    # In TensorFlow, you first want to define the computational graph with all the
    # necessary operations: loss, training op, accuracy...
    # Any tensor created in the `graph.as_default()` scope will be part of `graph`
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(2018)  # to keep consistent results

        # Standard preprocessing for VGG on ImageNet taken from here:
        # https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/vgg_preprocessing.py
        # Also see the VGG paper for more details: https://arxiv.org/pdf/1409.1556.pdf

        # Preprocessing (for both training and validation):
        # (1) Decode the image from jpg format
        # (2) Resize the image so its smaller side is 256 pixels long
        def _parse_function(filepath, filename):
            image_string = tf.read_file(filepath)
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)  # (1)
            image = tf.cast(image_decoded, tf.float32)

            smallest_side = 256.0
            height, width = tf.shape(image)[0], tf.shape(image)[1]
            height = tf.to_float(height)
            width = tf.to_float(width)

            scale = tf.cond(tf.greater(height, width),
                            lambda: smallest_side / width,
                            lambda: smallest_side / height)
            new_height = tf.to_int32(height * scale)
            new_width = tf.to_int32(width * scale)

            resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
            return resized_image, filename

        # Preprocessing (for validation)
        # (3) Take a central 224x224 crop to the scaled image
        # (4) Substract the per color mean `VGG_MEAN`
        # Note: we don't normalize the data here, as VGG was trained without normalization
        def val_preprocess(image, filename):
            crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)  # (3)

            means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
            centered_image = crop_image - means  # (4)

            return centered_image, filename

        test_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(test_filepaths), 
                                                           tf.constant(test_filenames)))
        test_dataset = test_dataset.map(_parse_function, num_parallel_calls=args.num_workers)
        test_dataset = test_dataset.prefetch(args.batch_size)
        test_dataset = test_dataset.map(val_preprocess, num_parallel_calls=args.num_workers)
        test_dataset = test_dataset.prefetch(args.batch_size)
        test_dataset = test_dataset.batch(args.batch_size)
        test_dataset = test_dataset.prefetch(args.batch_size)
        """
        Note: Use the prefetch transformation to overlap the work of a producer and consumer. 
        In particular, we recommend adding prefetch(n) (where n is the number of elements / batches consumed by 
        a training step) to the end of your input pipeline to overlap the transformations performed on the CPU 
        with the training done on the accelerator.
        The `prefetch` step can be skipped after `batch` step since it won't bring too much acceleration. But adding
        `prefetch` steps after `map` functions can improve the performance.
        """
 
        iterator = tf.contrib.data.Iterator.from_structure(test_dataset.output_types,
                                                           test_dataset.output_shapes)

        # iterator = test_dataset.make_one_shot_iterator()
        test_images, test_image_names = iterator.get_next()
        test_init_op = iterator.make_initializer(test_dataset)

        # ----------------------------------------------------------------------
        # Indicates whether we are in training or in test mode
        is_training = tf.placeholder(tf.bool)

        # ---------------------------------------------------------------------
        # Now that we have set up the data, it's time to set up the model.
        # For this example, we'll use VGG-16 pretrained on ImageNet. We will remove the
        # last fully connected layer (fc8) and replace it with our own, with an
        # output size num_classes=8
        # We will first train the last layer for a few epochs.
        # Then we will train the entire model on our dataset for a few epochs.

        # Get the pretrained model, specifying the num_classes argument to create a new
        # fully connected replacing the last one, called "vgg_16/fc8"
        # Each model has a different architecture, so "vgg_16/fc8" will change in another model.
        # Here, logits gives us directly the predicted scores we wanted from the images.
        # We pass a scope to initialize "vgg_16/fc8" weights with he_initializer
        # X = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
        vgg = tf.contrib.slim.nets.vgg
        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=args.weight_decay)):
            logits, _ = vgg.vgg_16(test_images, num_classes=num_classes, is_training=is_training,
                                   dropout_keep_prob=args.dropout_keep_prob)
        if num_classes > 2:  # multi-label classification
            pred = logits
        else:
            pred = tf.argmax(tf.nn.softmax(logits), axis=1)

        # Create a model saver
        saver = tf.train.Saver()

    # --------------------------------------------------------------------------
    # Now that we have built the graph and finalized it, we define the session.
    # The session is the interface to *run* the computational graph.
    # We can call our training operations with `sess.run(train_op)` for instance
    with tf.Session(graph=graph) as sess:
        sess.run(test_init_op)

        # Try to load model checkpoints if there are any
        try:
            saver.restore(sess, os.path.join(args.model_save_path, 'phase2'))
            print('Model restored.')
        except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError) as e:
            print(e)
            sys.exit(1)
        
        res = None
        files = []
        while True:
            try:
                predictions, image_names = sess.run([pred, test_image_names], feed_dict={is_training: False})
                print(image_names)
                if res is None:
                    res = predictions
                else:
                    res = np.concatenate((res, predictions), axis=0)
                files.append(image_names)
            except tf.errors.OutOfRangeError:
                break
        
        np.save(os.path.join(args.save_dir, 'pred array batch.npy'), res)
        
        flat_files = [item for sublist in files for item in sublist]
        with open(os.path.join(args.save_dir, 'test_image_filenames.txt'), 'w') as fout:
            for item in flat_files:
                fout.write('%s\n' % item.decode('utf-8').split('.png')[0])
                # items are binary objects, not strings; need to decode them first


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
