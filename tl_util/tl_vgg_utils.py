# -*- coding: utf-8 -*-

"""Utility functions for Transfer Learning with VGG"""

import os
import pandas as pd
import tensorflow as tf


def list_images(directory):
    """Get all the images and labels in directory/label/*.png

    Args:
        directory(str): Path to directory for all the training and validation images
    Returns:
        filenames(list): a list of filenames
        labels(list): a list of labels that are paired with filenames
    """
    labels = os.listdir(directory)
    # Sort the labels so that training and validation get them in the same order
    labels.sort()

    # Pair files and labels
    files_and_labels = []
    for label in labels:
        for f in os.listdir(os.path.join(directory, label)):
            files_and_labels.append((os.path.join(directory, label, f), label))

    filenames, labels = zip(*files_and_labels)  # Create 2 tuples in the same order
    filenames = list(filenames)
    labels = list(labels)
    unique_labels = list(set(labels))

    # Convert labels from string to integer
    label_to_int = {}
    for i, label in enumerate(unique_labels):
        label_to_int[label] = i

    labels = [label_to_int[l] for l in labels]

    return filenames, labels


def list_images_multilabel(directory, label_file):
    """Get all the images and labels in directory/label/*.png

    Args:
        directory(str): Path to directory for all the training and validation images
        label_file(str): Path to label.csv
    Returns:
        filenames(list): a list of filenames
        labels(list): a list of labels that are paired with filenames
    """
    labels = pd.read_csv(label_file)

    # Pair files and labels
    files_and_labels = []
    for f in os.listdir(directory):
        label = labels.loc[labels['image_id'] == f.split('.')[0]][['A', 'B', 'C', 'D']].values.tolist()[0]
        files_and_labels.append((os.path.join(directory, f), label))

    filenames, labels = zip(*files_and_labels)  # Create 2 tuples in the same order
    filenames = list(filenames)
    labels = list(labels)

    return filenames, labels


def create_summary(sess, correct_prediction, loss, lr, is_training, dataset_init_op):
    """Create model summaries of accuracy and loss

    Args:
        sess(object): a tensorflow session
        correct_prediction(tensor): a tensor that indicates whether predictions match with true labels in a batch
        loss(tensor): a tensor of loss for each sample in a batch
        lr(tensor): current learning rate; if None, indicates it's validation and we don't need to print lr
        is_training(tensor): a placeholder that indicates whether we are in training or testing mode
        dataset_init_op(object): an initializer for training or validation dataset

    Returns:
        loss_eval(float): loss value for current epoch
        acc(float): accuracy value for current epoch
    """
    # Initialize the correct dataset
    sess.run(dataset_init_op)
    num_correct, num_samples, total_cost = 0, 0, 0
    while True:
        try:
            correct_pred = sess.run(correct_prediction, {is_training:True})
            current_cost = sess.run(loss, {is_training:True})
            num_correct += correct_pred.sum()
            num_samples += correct_pred.shape[0]
            total_cost += current_cost * correct_pred.shape[0]
        except tf.errors.OutOfRangeError:  # Iterate through all samples in an epoch
            break

    loss_eval = float(total_cost) / num_samples  # average loss per sample
    acc = float(num_correct) / num_samples
    # Note: The graph has been finalized after we define it, so we cannot modify it anywhere else, i.e. we cannot use
    # tf.summary.scalar here

    if lr:
        lr_eval = sess.run(lr, {is_training:True})
        return loss_eval, acc, lr_eval
    else:
        return loss_eval, acc
