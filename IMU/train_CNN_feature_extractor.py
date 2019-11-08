'''
File: train_CNN_feature_extractor.py
Author: Federico Cruciani
Date: 03/10/2019
Version: 1.0
Description: 
	Example Script training CNN feature extractor on UCI HAPT dataset. 	 
'''

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
#local imports
import utils
import Classifiers
import HAPT_Dataset as UCI_HAPT
from HAPT_Dataset import ucihapt_datapath
import sys




if len( sys.argv ) == 4:
	#load parameters from command line arguments
	n_layers = int(sys.argv[1])
	k = int(sys.argv[2])
	nfilters = int(sys.argv[3])
else:
	#Use default parameters
	n_layers = 3
	k = 24
	nfilters = 24

if (n_layers in [1,2,3,4]) and (k in [2,8,16,24,32,64]) and (nfilters in [12,24,48,96,128]):
	train_uuids = UCI_HAPT.get_train_uuids()
	test_uuids = UCI_HAPT.get_test_uuids()

	tr_uuids = train_uuids[0:18]
	#Keep 3 users from training set as validation
	vl_uuids = train_uuids[18:21]

	#Number of cores used for parallel loading
	num_threads = 20

	print("Loading Training set")
	(X_train,y_train) = UCI_HAPT.get_all_data_multi_thread_resampling_3D(tr_uuids,num_threads)
	print("Loading Validation set")
	(X_vld,y_vld) = UCI_HAPT.get_all_data_multi_thread_resampling_3D(vl_uuids,num_threads)
	print("Loading test set")
	(X_test,y_test) = UCI_HAPT.get_all_data_multi_thread_resampling_3D(test_uuids,num_threads)

	classes = ["WALKING", "W. UPSTAIRS", "W. DOWNSTAIRS", "SITTING", "STANDING", "LAYING","TRANSITION"]

	clf = Classifiers.IMU_CNN_3D_FEATURE_EXTRACTOR(patience=1000,num_filters=nfilters,layers=n_layers,kern_size=k,divide_kernel_size=True,suffix="50Hz")
	y_train_inv = [ np.argmax(y) for y in y_train]
	class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train_inv),y_train_inv)
	d_class_weights = dict(enumerate(class_weights))
	clf.fit_class_weights(X_train,y_train,X_vld,y_vld,batch_size=32,class_weight=d_class_weights,epochs=6000)

	#Loading saved model
	clf.loadBestWeights()

	y_true = [ [np.argmax(y)] for y in y_test]#one_hot(labels_test)
	#Evalyate CNN on test set
	y_predictions = clf.predict(X_test,batch_size=1)
	y_predictions_inv = [ [np.argmax(y)] for y in y_predictions]

	cr = classification_report(np.array(true), np.array(predictions),target_names=classes,digits=4)
	print(cr)

else:
	print("Not valid argument: n-layers kernel_size num_filters")