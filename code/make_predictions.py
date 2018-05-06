import os
import glob
import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
import pudb

"""
This script makes predictions for an image being occupied or unoccupied, given an input dataset of images.
These images should also be labelled, as this allows validation of the model's accuracy.
The output is a csv file with the image path, predicted classes, and actual labelled values.
Note: 1 = OCCUPIED
      0 = UNOCCUPIED
The main path that must be set by the user is the 'test_img_paths', pointing to the folder w/ labelled images
"""
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
with open('../input/mean.binaryproto','rb') as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))


#Read model architecture and trained model's weights
net = caffe.Net('../modelfiles/caffenet_deploy.prototxt',
                '../output/caffe_model2_iter_10000.caffemodel',
                caffe.TEST)

#Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))

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
with open("../validation/small_dataset.csv","w+") as f:
    f.write("image path,predicted label, actual label\n")
    for i in range(len(test_names)):
        f.write(str(test_names[i])+","+str(preds[i])+","+str(labels[i]) + "\n")
f.close()
