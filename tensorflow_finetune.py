# -*- coding: utf-8 -*-

"""
Modified based on https://gist.github.com/omoindrot/dedc857cdc0e680dfb1be99762990c9c/.
Modifications:
1. Added model saver and summary writer
2. Made the code compatible to tf v1.4 - Use tf.data.Dataset instead of tf.contrib.data.Dataset since the fuctions are
going to be deprecated in future versions
3. Added learning rate decay
4. Added data augmentation (TODO: still need to be refined)
5. Provided options for other optimization algorithms

Reference for learning rate decay:
https://kwotsin.github.io/tech/2017/02/11/transfer-learning.html

==> Original Introduction <==
Example TensorFlow script for finetuning a VGG model on your own data.
Uses tf.contrib.data module which is in release v1.2
Based on PyTorch example from Justin Johnson:
https://gist.github.com/jcjohnson/6e41e8512c17eae5da50aebef3378a4c

Required packages: tensorflow (v1.2)
Download the weights trained on ImageNet for VGG:
```
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xvf vgg_16_2016_08_28.tar.gz
rm vgg_16_2016_08_28.tar.gz
```
For this example we will use a tiny dataset of images from the COCO dataset.
We have chosen eight types of animals (bear, bird, cat, dog, giraffe, horse,
sheep, and zebra); for each of these categories we have selected 100 training
images and 25 validation images from the COCO dataset. You can download and
unpack the data (176 MB) by running:
```
wget cs231n.stanford.edu/coco-animals.zip
unzip coco-animals.zip
rm coco-animals.zip
```
The training data is stored on disk; each category has its own folder on disk
and the images for that category are stored as .jpg files in the category folder.
In other words, the directory structure looks something like this:
coco-animals/
  train/
    bear/
      COCO_train2014_000000005785.jpg
      COCO_train2014_000000015870.jpg
      [...]
    bird/
    cat/
    dog/
    giraffe/
    horse/
    sheep/
    zebra/
  val/
    bear/
    bird/
    cat/
    dog/
    giraffe/
    horse/
    sheep/
    zebra/
"""

import argparse
import os
import sys
from tl_util.image_preprocess import *
from tl_util.tl_vgg_utils import *
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets  # need to explicitly import nets, otherwise you will see an AttributeError
# The problem is caused by the fact that https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/
# __init__.py does not export nets.


parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='/home/your/train/dir')
parser.add_argument('--val_dir', default='/home/your/validation/dir')
parser.add_argument('--train_from_vgg', default=False, type=bool)
parser.add_argument('--pretrained_model_path', default='vgg_16.ckpt', type=str)
parser.add_argument('--new_model_save_path', default='/home/your/new/model/save/dir/', type=str)
parser.add_argument('--optimizer', default='GradientDescent', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=16, type=int)
# Set num_workers to be the number of CPU cores you have should give you the most efficient results.
parser.add_argument('--data_augmentation', default=False, type=bool)
parser.add_argument('--num_epochs1', default=10, type=int)
parser.add_argument('--num_epochs2', default=10, type=int)
parser.add_argument('--initial_learning_rate1', default=1e-3, type=float)
parser.add_argument('--lr1_decay_factor', default=0.8, type=float)
parser.add_argument('--num_epochs1_before_decay', default=5, type=int)
parser.add_argument('--initial_learning_rate2', default=1e-5, type=float)
parser.add_argument('--lr2_decay_factor', default=0.8, type=float)
parser.add_argument('--num_epochs2_before_decay', default=10, type=int)
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)


def main(args):
    # Get the list of filenames and corresponding list of labels for training and validation
    train_filenames, train_labels = list_images(args.train_dir)
    val_filenames, val_labels = list_images(args.val_dir)
    assert set(train_labels) == set(val_labels),\
        "Train and validation labels don't match:\n{}\n{}".format(set(train_labels), set(val_labels))

    num_samples = len(train_filenames)
    num_classes = len(set(train_labels))

    # --------------------------------------------------------------------------
    # In TensorFlow, you first want to define the computational graph with all the
    # necessary operations: loss, training op, accuracy...
    # Any tensor created in the `graph.as_default()` scope will be part of `graph`
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(2018)  # to keep consistent results

        # ----------------------------------------------------------------------
        # DATASET CREATION using tf.data.Dataset
        # https://github.com/tensorflow/tensorflow/blob/r1.6/tensorflow/python/data/ops/dataset_ops.py

        # The tf.data.Dataset framework uses queues in the background to feed in data to the model.
        # We initialize the dataset with a list of filenames and labels, and then apply
        # the preprocessing functions defined in image_preprocessing.py.
        # Behind the scenes, queues will load the filenames, preprocess them with multiple
        # threads and apply the preprocessing in parallel, and then batch the data.

        # TODO: Verify if we need to prefetch the dataset every time after using map function.

        # Training dataset
        train_filenames = tf.constant(train_filenames)
        train_labels = tf.constant(train_labels)
        train_dataset0 = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
        train_dataset0 = train_dataset0.map(parse_function, num_parallel_calls=args.num_workers)
        train_dataset0 = train_dataset0.prefetch(args.batch_size)
        train_dataset0 = train_dataset0.map(training_preprocess, num_parallel_calls=args.num_workers)
        train_dataset0 = train_dataset0.prefetch(args.batch_size)

        if args.data_augmentation:
            train_dataset1 = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
            train_dataset1 = train_dataset1.map(parse_function, num_parallel_calls=args.num_workers)
            train_dataset1 = train_dataset1.prefetch(args.batch_size)
            train_dataset1 = train_dataset1.map(training_scaling, num_parallel_calls=args.num_workers)
            train_dataset1 = train_dataset1.prefetch(args.batch_size)
            train_dataset1 = train_dataset1.map(training_add_shades, num_parallel_calls=args.num_workers)
            train_dataset1 = train_dataset1.prefetch(args.batch_size)

            train_dataset2 = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
            train_dataset2 = train_dataset2.map(parse_function, num_parallel_calls=args.num_workers)
            train_dataset2 = train_dataset2.prefetch(args.batch_size)
            train_dataset2 = train_dataset2.map(training_flip, num_parallel_calls=args.num_workers)
            train_dataset2 = train_dataset2.prefetch(args.batch_size)
            train_dataset2 = train_dataset2.map(training_add_shades, num_parallel_calls=args.num_workers)
            train_dataset2 = train_dataset2.prefetch(args.batch_size)

            train_dataset = train_dataset0.concatenate(train_dataset1)
            train_dataset = train_dataset.concatenate(train_dataset2)
        else:
            train_dataset = train_dataset0

        train_dataset = train_dataset.shuffle(buffer_size=10000)  # don't forget to shuffle
        batched_train_dataset = train_dataset.batch(args.batch_size)

        # Validation dataset
        val_filenames = tf.constant(val_filenames)
        val_labels = tf.constant(val_labels)
        val_dataset = tf.data.Dataset.from_tensor_slices((val_filenames, val_labels))
        val_dataset = val_dataset.map(parse_function, num_parallel_calls=args.num_workers)
        val_dataset = val_dataset.prefetch(args.batch_size)
        val_dataset = val_dataset.map(val_preprocess, num_parallel_calls=args.num_workers)
        val_dataset = val_dataset.prefetch(args.batch_size)
        batched_val_dataset = val_dataset.batch(args.batch_size)

        # ---------------------------------------------------------------------
        # Now we define an iterator that can operator on either dataset.
        # The iterator can be reinitialized by calling:
        #     - sess.run(train_init_op) for 1 epoch on the training set
        #     - sess.run(val_init_op)   for 1 epoch on the validation set
        # Once this is done, we don't need to feed any value for images and labels
        # as they are automatically pulled out from the iterator queues.

        # A reinitializable iterator is defined by its structure. We could use the
        # `output_types` and `output_shapes` properties of either `train_dataset`
        # or `validation_dataset` here, because they are compatible.
        iterator = tf.contrib.data.Iterator.from_structure(batched_train_dataset.output_types,
                                                           batched_train_dataset.output_shapes)
        images, labels = iterator.get_next()

        train_init_op = iterator.make_initializer(batched_train_dataset)
        val_init_op = iterator.make_initializer(batched_val_dataset)

        # Indicates whether we are in training or in testing mode
        is_training = tf.placeholder(tf.bool)

        # ---------------------------------------------------------------------
        # Now that we have set up the data, it's time to set up the model.
        # For this example, we'll use VGG-16 pretrained on ImageNet. We will remove the
        # last fully connected layer (fc8) and replace it with our own, with an output size num_classes.
        # We will first train the last layer for a few epochs.
        # Then we will train the entire model on our dataset for a few epochs.

        # Get the pretrained model, specifying the num_classes argument to create a new
        # fully connected replacing the last one, called "vgg_16/fc8"
        # Each model has a different architecture, so "vgg_16/fc8" will change in another model.
        # Here, logits gives us directly the predicted scores we wanted from the images.
        # We pass a scope to initialize "vgg_16/fc8" weights with he_initializer
        vgg = tf.contrib.slim.nets.vgg
        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=args.weight_decay)):
            logits, _ = vgg.vgg_16(images, num_classes=num_classes, is_training=is_training,
                                   dropout_keep_prob=args.dropout_keep_prob)

        if args.train_from_vgg:
            # Specify where the model checkpoint is (pretrained weights)
            model_path = args.pretrained_model_path
            assert(os.path.isfile(model_path))

            # Restore only the layers up to fc7 (included)
            # Calling function `init_fn(sess)` will load all the pretrained weights.
            variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['vgg_16/fc8'])
            init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)

            # Initialization operation from scratch for the new "fc8" layers
            # `get_variables` will only return the variables whose name starts with the given pattern
            fc8_variables = tf.contrib.framework.get_variables('vgg_16/fc8')
            fc8_init = tf.variables_initializer(fc8_variables)

        # ---------------------------------------------------------------------
        # Using tf.losses, any loss is added to the tf.GraphKeys.LOSSES collection
        # We can then call the total loss easily
        tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        loss = tf.losses.get_total_loss()
        # # For multi-label problem:
        # tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)
        # loss = tf.losses.get_total_loss()
        # # For weighted loss:
        # tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits,
        #                                        weights=tf.where(tf.equal(labels, 0),
        #                                                         tf.fill(tf.shape(labels), 0.01),  # For illustration
        #                                                         tf.fill(tf.shape(labels), 1.0)))
        # loss = tf.losses.get_total_loss()

        global_step = tf.Variable(0, trainable=False, name='global_step')
        num_batches_per_epoch = num_samples / args.batch_size

        if args.train_from_vgg:
            gs_init = tf.variables_initializer([global_step])

            # First we want to train only the reinitialized last layer fc8 for a few epochs.
            # We minimize the loss only with respect to the fc8 variables (weight and bias).
            decay_steps1 = int(args.num_epochs1_before_decay * num_batches_per_epoch)
            lr1 = tf.train.exponential_decay(
                learning_rate=args.initial_learning_rate1,
                global_step=global_step,
                decay_steps=decay_steps1,
                decay_rate=args.lr1_decay_factor,
                staircase=True
            )
            if args.optimizer == 'GradientDescent':
                fc8_optimizer = tf.train.GradientDescentOptimizer(lr1)
            elif args.optimizer == 'Momentum':
                fc8_optimizer = tf.train.MomentumOptimizer(lr1, 0.9)  # TODO: Add tuning for beta
            else:
                print('Optimizer not found.')
                sys.exit(1)

            fc8_train_op = fc8_optimizer.minimize(loss, global_step=global_step, var_list=fc8_variables)

        # Then we want to finetune the entire model for a few epochs.
        # We minimize the loss with respect to all the variables.
        decay_steps2 = int(args.num_epochs2_before_decay * num_batches_per_epoch)
        lr2 = tf.train.exponential_decay(
            learning_rate=args.initial_learning_rate2,
            global_step=global_step,
            decay_steps=decay_steps2,
            decay_rate=args.lr2_decay_factor,
            staircase=True
        )
        if args.optimizer == 'GradientDescent':
            full_optimizer = tf.train.GradientDescentOptimizer(lr2)
        elif args.optimizer == 'Momentum':
            full_optimizer = tf.train.MomentumOptimizer(lr2, 0.9)  # TODO: Add tuning for beta
        else:
            print('Optimizer not found.')
            sys.exit(1)

        full_train_op = full_optimizer.minimize(loss, global_step=global_step)

        # Evaluation metrics
        prediction = tf.to_int32(tf.argmax(logits, 1))
        correct_prediction = tf.equal(prediction, labels)
        # # For multi-label problem: use Intersection/Union (IoU)
        # condition = tf.greater_equal(logits, 0.5)  # TODO: tune the threshold (currently pred = 1 if logits >= 0.5)
        # # Since logits has a dynamic size, we should use tf.fill here instead of tf.constant
        # prediction = tf.where(condition, tf.fill(tf.shape(logits), 1), tf.fill(tf.shape(logits), 0))
        # intersection = tf.logical_and(tf.cast(labels, tf.bool), tf.case(prediction, tf.bool))
        # union = tf.logical_or(tf.cast(labels, tf.bool), tf.cast(prediction, tf.bool))
        # intersection_sum = tf.reduce_sum(tf.cast(intersection, tf.int32), axis=1)
        # union_sum = tf.reduce_sum(tf.cast(union, tf.int32), axis=1)
        # correct_prediction = tf.divide(intersection_sum, union_sum)

        loss_eval = tf.placeholder(tf.float32, shape=(), name='loss_ph')
        tf.summary.scalar('loss', loss_eval)
        accuracy = tf.placeholder(tf.float32, shape=(), name='acc_ph')
        tf.summary.scalar('accuracy', accuracy)
        lr_eval = tf.placeholder(tf.float32, shape=(), name='lr_ph')
        tf.summary.scalar('learning_rate', lr_eval)
        # merged = tf.summary.merge_all()
        train_merged = tf.summary.merge([loss_eval, accuracy, lr_eval])
        val_merged = tf.summary.merge([loss_eval, accuracy])

        # Create a model saver
        saver = tf.train.Saver()

        if args.train_from_vgg:
            tf.get_default_graph().finalize()

    # --------------------------------------------------------------------------
    # Now that we have built the graph and finalized it (if training from pretrained VGG), we define the session.
    # The session is the interface to *run* the computational graph.
    # We can call our training operations with `sess.run(train_op)` for instance.
    with tf.Session(graph=graph) as sess:
        if args.train_from_vgg:
            sess.run(gs_init)  # initialize the global step
            init_fn(sess)  # load the pretrained weights
            sess.run(fc8_init)  # initialize the new fc8 layer
        else:
            sess.run(tf.global_variables_initializer())

        # Create a summary writer
        train_writer = tf.summary.FileWriter(args.new_model_save_path + "summaries/train/", sess.graph)
        val_writer = tf.summary.FileWriter(args.new_model_save_path + "summaries/val/", sess.graph)

        if args.train_from_vgg:
            # Update only the last layer for a few epochs.
            for epoch in range(args.num_epochs1):
                # Run an epoch over the training data.
                print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs1))
                # Here we initialize the iterator with the training dataset.
                # This means that we can go through an entire epoch until the iterator becomes empty.
                sess.run(train_init_op)
                while True:
                    try:
                        _ = sess.run(fc8_train_op, {is_training: True})
                    except tf.errors.OutOfRangeError:
                        break

                # Create summary for accuracy, loss and learning rate (lr only for training summary)
                loss_train, acc_train, lr_train = create_summary(sess, correct_prediction, loss, lr1,
                                                                 is_training, train_init_op)
                loss_val, acc_val = create_summary(sess, correct_prediction, loss, None, is_training, val_init_op)
                train_summ = sess.run(train_merged, feed_dict={loss_eval: loss_train, accuracy: acc_train,
                                                               lr_eval: lr_train})
                val_summ = sess.run(val_merged, feed_dict={loss_eval: loss_val, accuracy: acc_val})
                train_writer.add_summary(train_summ, epoch)
                val_writer.add_summary(val_summ, epoch)

                if (epoch+1) % 10 == 0:  # save model checkpoints every 10 epochs
                    saver.save(sess, args.new_model_save_path + 'phase1')

            # Train the entire model for a few more epochs, continuing with the *same* weights.
            for epoch in range(args.num_epochs2):
                print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs2))
                sess.run(train_init_op)
                while True:
                    try:
                        _ = sess.run(full_train_op, {is_training: True})
                    except tf.errors.OutOfRangeError:
                        break

                # Create summary for accuracy, loss and learning rate (lr only for training summary)
                loss_train, acc_train, lr_train = create_summary(sess, correct_prediction, loss, lr2,
                                                                 is_training, train_init_op)
                loss_val, acc_val = create_summary(sess, correct_prediction, loss, None, is_training, val_init_op)
                train_summ = sess.run(train_merged, feed_dict={loss_eval: loss_train, accuracy: acc_train,
                                                               lr_eval: lr_train})
                val_summ = sess.run(val_merged, feed_dict={loss_eval: loss_val, accuracy: acc_val})
                train_writer.add_summary(train_summ, epoch+args.num_epochs1)
                val_writer.add_summary(val_summ, epoch+args.num_epochs1)

                if (epoch + 1) % 10 == 0:  # save model checkpoints every 10 epochs
                    saver.save(sess, args.new_model_save_path + 'phase2')

        else:
            # Try to load model checkpoints if there are any
            try:
                saver.restore(sess, os.path.join(args.model_save_path, 'phase1'))  # change to `phase2` if needed
                print('Model restored.')
            except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError) as e:
                print(e)
                sys.exit(1)

            # Train the entire model for a few more epochs, continuing with the *same* weights.
            for epoch in range(args.num_epochs2):
                print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs2))
                sess.run(train_init_op)
                while True:
                    try:
                        _ = sess.run(full_train_op, {is_training: True})
                    except tf.errors.OutOfRangeError:
                        break

                # Create summary for accuracy, loss and learning rate (lr only for training summary)
                loss_train, acc_train, lr_train = create_summary(sess, correct_prediction, loss, lr2,
                                                                 is_training, train_init_op)
                loss_val, acc_val = create_summary(sess, correct_prediction, loss, None, is_training, val_init_op)
                train_summ = sess.run(train_merged, feed_dict={loss_eval: loss_train, accuracy: acc_train,
                                                               lr_eval: lr_train})
                val_summ = sess.run(val_merged, feed_dict={loss_eval: loss_val, accuracy: acc_val})
                train_writer.add_summary(train_summ, epoch+100)  # depend on how many epochs you have trained before
                val_writer.add_summary(val_summ, epoch+100)  # depend on how many epochs you have trained before
                # TODO: See if there is a way to automatically find out how many epochs we have trained before

                if (epoch + 1) % 10 == 0:  # save model checkpoints every 10 epochs
                    saver.save(sess, args.new_model_save_path + 'phase2')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
