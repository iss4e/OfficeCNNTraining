#! /usr/bin/env python3

import caffe
import numpy as np
import sys
import os

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( '/home/sasha-d/research_2018/model2/input/mean.binaryproto', 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
out = arr[0]
np.save( 'image_mean.npy', out )

