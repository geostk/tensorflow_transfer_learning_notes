# -*- coding: utf-8 -*-

"""Utility functions for image preprocessing and data augmentation operations
Standard preprocessing for VGG on ImageNet taken from here:
https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/vgg_preprocessing.py
Also see the VGG paper for more details: https://arxiv.org/pdf/1409.1556.pdf
Data augmentation TensorFlow implementation reference:
https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9

Preprocessing (for both training and validation):
(1) Decode the image from jpg/png format
(2) Resize the image so its smaller side is 256 pixels long

Preprocessing (for training)
(3) Take a random 224x224 crop to the scaled image
(4) Horizontally flip the image with probability 1/2 -- may apply other data augmentation techniques instead
(5) Substract the per color mean `VGG_MEAN`
Note: we don't normalize the data here, as VGG was trained without normalization

Preprocessing (for validation)
(3) Take a central 224x224 crop to the scaled image
(4) Substract the per color mean `VGG_MEAN`
Note: we don't normalize the data here, as VGG was trained without normalization

Note: we input label to every preprocessing function and output it same as input solely for the purpose of using
tf.data.Dataset API
"""

import tensorflow as tf

VGG_MEAN = [123.68, 116.78, 103.94]  # RGB Version
IMAGE_SIZE = 224  # VGG was trained on 224x224 images
SMALLEST_SIDE = 256.0
SCALE = 0.9


def parse_function(filename, label):
    """Read an image and resize it to have the smallest side as certain pixels specified by SMALLEST_SIDE

    Args:
        filename(str): filename for an image
        label(str): true label for that image

    Returns:
        resized_image(tensor): resized image stored as a tensor
        label(str): same as input
    """
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)                      # (1)
    image = tf.cast(image_decoded, tf.float32)

    smallest_side = SMALLEST_SIDE
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    height = tf.to_float(height)
    width = tf.to_float(width)

    # Downsize the image and retain the aspect ratio
    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,  # if height >= width, scale width to SMALLEST_SIDE px
                    lambda: smallest_side / height)  # if width > height, scale height to SMALLEST_SIDE px
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)

    resized_image = tf.image.resize_images(image, [new_height, new_width])              # (2)
    return resized_image, label


def _center_image(image):
    """Substract the per color channel mean `VGG_MEAN` from the input image

    Args:
        image(tensor): an image

    Returns:
        centered_image(tensor): image after centering
    """
    means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])  # create the per-channel-mean tensor
    centered_image = image - means                                                      # (5)
    return centered_image


def training_preprocess(image, label):
    """Perform training preprocessing

    Args:
        image(tensor): an image
        label(str): true label for that image

    Returns:
        centered_image(tensor): image after preprocessing
        label(str): same as input
    """
    crop_image = tf.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 3])                     # (3)

    centered_image = _center_image(crop_image)

    return centered_image, label


def training_scaling(image, label):
    """Zoom in on the image; zoom level specified by SCALE

    Args:
        image(tensor): an image
        label(str): true label for that image

    Returns:
        centered_image(tensor): image after preprocessing
        label(str): same as input

    TODO:
        Apply different scaling to a batch of training images
    """
    # To scale centrally
    x1 = y1 = 0.5 - 0.5 * SCALE
    x2 = y2 = 0.5 + 0.5 * SCALE

    # Since we will be using tf.data.Dataset API(to be specific, the map function), we will perform preprocessing step
    # on one image at a time. So currently we will apply the same SCALE factor to all the training images(which is not
    # a good idea). See the above note.
    boxes = tf.constant([[y1, x1, y2, x2]], dtype=tf.float32)
    box_ind = tf.zeros([1], dtype=tf.int32)
    crop_size = [IMAGE_SIZE, IMAGE_SIZE]

    # Convert the image tensor from 3d to 4d first so that we can use the crop_and_resize function, then convert it
    # back to 3d for output
    image_4d = tf.expand_dims(image, 0)  # insert at the first dimension (placeholder for number of samples in a batch)
    scaled_image = tf.image.crop_and_resize(image_4d, boxes, box_ind, crop_size)
    scaled_image = tf.squeeze(scaled_image, axis=0)  # remove the first dimension to bring the image back to 3d

    centered_image = _center_image(scaled_image)

    return centered_image, label


def training_flip(image, label):
    """Flip the image horizontally

    Args:
        image(tensor): an image
        label(str): true label for that image

    Returns:
        centered_image(tensor): image after preprocessing
        label(str): same as input
    """
    crop_image = tf.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 3])                     # (3)
    fliped_image = tf.image.flip_left_right(crop_image)                                 # (4)

    centered_image = _center_image(fliped_image)                                        # (5)

    return centered_image, label


def training_add_shades(image, label):
    """Add lightening conditions

    Args:
        image(tensor): an image
        label(str): true label for that image

    Returns:
        centered_image(tensor): image after preprocessing
        label(str): same as input
    """
    crop_image = tf.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 3])                     # (3)
    shaded_image = tf.image.random_brightness(crop_image, max_delta=32./255.)
    shaded_image = tf.image.random_saturation(shaded_image, lower=0.5, upper=1.5)
    shaded_image = tf.image.random_hue(shaded_image, max_delta=0.2)
    shaded_image = tf.image.random_contrast(shaded_image, lower=0.5, upper=1.5)

    centered_image = _center_image(shaded_image)                                        # (5)

    return centered_image, label


def val_preprocess(image, label):
    """Perform validation preprocessing

    Args:
        image(tensor): an image
        label(str): true label for that image

    Returns:
        centered_image(tensor): image after preprocessing
        label(str): same as input
    """
    crop_image = tf.image.resize_image_with_crop_or_pad(image, IMAGE_SIZE, IMAGE_SIZE)  # (3)

    centered_image = _center_image(crop_image)                                          # (5)

    return centered_image, label
