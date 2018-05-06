## Training a CNN for overhead images using transfer learning
*Note*: The source http://adilmoujahid.com/posts/2016/06/introduction-deep-learning-python-caffe/ 
was used heavily for the creation of this repo, and should be consulted for further details.
___
The following steps should be performed on a computer with cv2 installed (locally/VM):
1. Download the 'bvlc_reference_caffenet model' and place it in the `models/` folder. 
This is the trained model whose weights that will be adjusted with transfer learning. It can be found here http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel.
2. Load a large dataset of labelled images onto your computer.
3. Change the `train_data` path in `code/create_lmdb.py` to point to this uploaded dataset of images. 
Run the command `python create_lmdb.py` 
The following paths should now exist: `input/train_lmdb` and `input/validation_lmdb`.
4. To generate the mean file, run  
`opt/movidius/caffe/build/tools/compute_image_mean -backend=lmdb input/train_lmdb input/mean.binaryproto`  
within your current directory

The following should be performed on a dedicated GPU cluster/AWS setup:  
1. Copy over the repository to your setup, EXCLUDING the image dataset (only the database files are needed)
2. Replace the relative paths ('PATH_TO_REPO') in `code/caffenet_train_val.prototxt` and `code/solver.prototxt` 
with the absolute path to your cloned repository.
3. Run the command  
`PATH_TO_CAFFE/caffe/build/tools/caffe train --solver=modelfiles/solver.prototxt
--weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel 2>&1 |
tee logs/model_train.log`  
replacing the PATH_TO_CAFFE with the path to caffe on your cluster/AWS setup

Once trained, copy the newly created trained files + logs onto your local computer, 
and perform the following steps on your local computer:
1. Plot the learning curve by running: 
`python code/plot_learning_curve.py logs/model_train.log validation/caffe_model_learning_curve.png`
2. Copy over the final weights from `snapshots/` (should be a .caffemodel file) of the model into `output/`
3. Using a new **UNSEEN** labelled dataset, one can check the numerical accuracy of the model. In `code/make_predictions.py`, 
modify the `test_img_paths` to point to your dataset. Run `python make_predictions.py`. 
You should get a csv file in the `validation` folder with the true labels and predicted labels for each image.

___
#### Movidius-Specific Instructions
*Note*: For more details, see https://movidius.github.io/blog/deploying-custom-caffe-models/.

Once trained, we can profile the model by running:  
`mvNCProfile -s 12 modelfiles/caffenet_deploy.prototxt -w output/caffe_model_iter_10000.caffemodel`.  
This will provide useful information about the performance of our trained network.

To deploy our trained model, run:  
`mvNCCompile -s 12 caffenet_deploy.prototxt -w caffe_model_iter_10000.caffemodel`  
This will generate a 'graph' file that can be used by the Movidius stick.
