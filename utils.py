#By Mehjabin, originally from preprocess_v_1 colab

import os
#import cv2
from PIL import Image,ImageFilter
import numpy as np
import random
import imgaug.augmenters as iaa
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy

#ORIG - takes path to image/annotation folders, saves preprocessed images to output folder
"""
def preprocess_images(image_folder, annotation_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = os.listdir(image_folder)

    for image_file in image_files:
        annotation_file = image_file.replace('.png', '_Annotation.png')

        image_path = os.path.join(image_folder, image_file)
        annotation_path = os.path.join(annotation_folder, annotation_file)

        if not os.path.isfile(annotation_path):
            continue

        # Load the image and annotation
        image = Image.open(image_path)
        annotation = Image.open(annotation_path)

        # Crop the image according to the annotation
        bbox = annotation.getbbox()
        if bbox:
            image_cropped = image.crop(bbox)
        else:
            # Handle the case where no bounding box is found
            image_cropped = image

        # Convert to grayscale
        image_gray = image_cropped.convert('L')

        # Apply blur
        image_blurred = image_gray.filter(ImageFilter.BLUR)

        # Augmentation
        aug = iaa.Sequential([
            iaa.GaussianBlur(sigma=(0, 3.0)),
            iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
            iaa.Affine(rotate=(-45, 45)),
            iaa.Multiply((0.8, 1.2))
        ])

        image_augmented = aug.augment_image(np.array(image_blurred))

        # Save the preprocessed image
        output_path = os.path.join(output_folder, image_file)
        image_augmented = Image.fromarray(image_augmented)
        image_augmented.save(output_path)
"""


#MODIFIED - Takes Image (normal and annotation) and returns them preprocessed
#[took out getting paths/loading images at start, and saving files at end]
#annotation is not used? is intentional? i performed same steps to annotations
def preprocess_images(image, annotation,dim):

        # Crop the image according to the annotation
        """
        bbox = annotation.getbbox()
        if bbox:
            image_cropped = image.crop(bbox)
            ann_cropped = annotation.crop(bbox)
        else:
            # Handle the case where no bounding box is found
            image_cropped = image
            ann_cropped = annotation
        """
        
        # Convert to grayscale
        image_gray = image.convert('L')
        ann_gray = annotation.convert('L')

        # Apply blur
        image_blurred = image_gray.filter(ImageFilter.BLUR)
        ann_blurred = ann_gray.filter(ImageFilter.BLUR)

        # Augmentation
        aug = iaa.Sequential([
            iaa.GaussianBlur(sigma=(0, 3.0)),
            iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
            iaa.Affine(rotate=(-45, 45)),
            iaa.Multiply((0.8, 1.2))
        ])
        
        #Crop/resize
        image_augmented = aug.augment_image(np.array(image_blurred))
        ann_augmented = aug.augment_image(np.array(ann_blurred))

        image_augmented = tf.convert_to_tensor(image_augmented)
        image_augmented = tf.expand_dims(image_augmented,-1)
        image_augmented = tf.image.resize_with_crop_or_pad(image_augmented,*dim)
        
        ann_augmented = tf.convert_to_tensor(ann_augmented)
        ann_augmented = tf.expand_dims(ann_augmented,-1)
        ann_augmented = tf.image.resize_with_crop_or_pad(ann_augmented,*dim)
        
        ann_augmented = tf.cast(ann_augmented,tf.float32)/255.0
        ann_augmented=tf.cast(ann_augmented,tf.int32)

        #print(tf.shape(image_augmented), tf.shape(ann_augmented))
        
        return image_augmented,ann_augmented
    
    
def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true = tf.cast(y_true, dtype=tf.float32)  # Convert labels to float32
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss