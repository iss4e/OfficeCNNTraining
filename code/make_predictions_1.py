'''
Title           :make_predictions_1.py
Description     :This script makes predictions using the 1st trained model and generates a submission file.
Author          :Adil Moujahid
Date Created    :20160623
Date Modified   :20160625
version         :0.2
usage           :python make_predictions_1.py
python_version  :2.7.11
'''

import os
import glob
import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
import pudb

caffe.set_mode_cpu() 
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

def read_label(img_path):
    txt_file = img_path.split(".")[0] + ".txt"

    with open(txt_file, 'r') as f:
        label = f.read()
        label = label.strip()
        return int(label)


'''
Reading mean image, caffe model and its weights 
'''
#Read mean image
mean_blob = caffe_pb2.BlobProto()
with open('/home/sasha-d/research_2018/model2/input/mean.binaryproto','rb') as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))


#Read model architecture and trained model's weights
net = caffe.Net('/home/sasha-d/research_2018/model2/my_code/caffenet_deploy_2.prototxt',
                '/home/sasha-d/research_2018/model2/output/caffe_model2_iter_10000.caffemodel',
                caffe.TEST)

#Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))

'''
Making predicitions
'''
#Reading image paths
test_img_paths = [img_path for img_path in glob.glob("/home/sasha-d/research_2018/model2/small_dataset_2/*.jpg")]

#Making predictions
test_names = []
preds = []
labels = []

for img_path in test_img_paths:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    pred_probas = out['prob']

    test_names +=  [img_path]

    preds += [pred_probas.argmax()]
    
    label = read_label(img_path)

    labels += [label]
    # pu.db

    print(img_path)
    print(pred_probas.argmax())
    print(label)
    print('-------')

'''
Making submission file
'''
with open("/home/sasha-d/research_2018/model2/model2_small_dataset_2.csv","w+") as f:
    f.write("id,label\n")
    for i in range(len(test_names)):
        f.write(str(test_names[i])+","+str(preds[i])+","+str(labels[i]) + "\n")
f.close()
