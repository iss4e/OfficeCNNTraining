import os
import glob
import random
import numpy as np

import cv2

import caffe
from caffe.proto import caffe_pb2
import lmdb
import sys
import pudb

"""
This script divides the training images into 2 sets and stores them in lmdb databases for training and validation. The path 'train_data' should be set to the user's folder of images
"""

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    old_size = img.shape[:2]
    desired_size = img_width
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    img = cv2.resize(img, (new_size[1], new_size[0]), interpolation = cv2.INTER_CUBIC)

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h - (delta_h//2)
    left, right = delta_w//2, delta_w - (delta_w//2)
    color = [0,0,0]

    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return new_img


def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())

def read_label(img_path):
    txt_file = img_path.split(".")[0] + ".txt"

    with open(txt_file, 'r') as f:
        label = f.read()
        label = label.strip()
        return int(label)

train_lmdb = '../input/train_lmdb'
validation_lmdb = '../input/validation_lmdb'

os.system('rm -rf  ' + train_lmdb)
os.system('rm -rf  ' + validation_lmdb)

# USER-SET PARAMETER
# path should be set by user to the folder containing images + labels
# Note that the path should end with /*jpg to glob for images ONLY
train_data = [img for img in glob.glob("/home/sasha-d/research_2018/model2/large_dataset/*jpg")]

#Shuffle train_data
random.shuffle(train_data)

print('Creating train_lmdb')

in_db = lmdb.open(train_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(train_data):
        if in_idx %  6 == 0:
            continue
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        label = read_label(img_path)
        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(in_idx).encode(), datum.SerializeToString())
        print('{:0>5d}'.format(in_idx) + ':' + img_path)
in_db.close()


print('\nCreating validation_lmdb')

in_db = lmdb.open(validation_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(train_data):
        if in_idx % 6 != 0:
            continue
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        label = read_label(img_path)

        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(in_idx).encode(), datum.SerializeToString())
        print('{:0>5d}'.format(in_idx) + ':' + img_path)
in_db.close()

print('\nFinished processing all images')
