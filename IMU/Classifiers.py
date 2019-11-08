'''
File: Classifiers.py
Author: Federico Cruciani
Date: 05/03/2019
Version: 1.1
Description: 
	Class implementing Keras model classifiers.	 
'''

import numpy as np
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras import backend as keras_backend
from keras.utils import plot_model
from keras.layers import *
from datetime import datetime
import keras
#For multi GPU training
#from keras.utils import multi_gpu_model


basepath = './keras_logs/'


##Base class
class BaseClassifier:
	def __init__(self,name,patience=25,fontSize=16):
		self.name = name
		#stop criterion
		self.early_stopping = EarlyStopping(monitor='val_loss',patience=patience)
		self.csv_log_file = basepath+self.name+"_training_log.csv"
		self.csv_logger = CSVLogger(self.csv_log_file)
		#Checkpoint where weights are saved
		self.bestmodelweights = basepath+ name+"weights_best.hdf5"
		#CheckPoint: during trainng saves model minimizing loss on validation set
		self.checkpoint = ModelCheckpoint(self.bestmodelweights, monitor='val_loss',verbose=1,save_best_only=True, mode='min')
		####Model init is done in specialized class

	#Class weights are used to deal with the dataset that is slightly imbalanced.
	def fit_class_weights(self,X_tr,y_tr,X_vld,y_vld,class_weight,batch_size=256,epochs=500,verbose=0):
		self.history = self.model.fit( X_tr, y_tr, validation_data=(X_vld,y_vld), class_weight=class_weight, batch_size=batch_size, epochs=epochs, shuffle=False, verbose=2, callbacks = [self.logger, self.early_stopping, self.checkpoint, self.csv_logger] )

	
	def loadBestWeights(self):
		print(self.checkpoint)
		self.model.load_weights(self.bestmodelweights)
		#self.model.compile( loss='mse', optimizer='adam' )

	def predict(self,X_test,batch_size=32):
		predictions = self.model.predict(X_test,batch_size)
		return predictions

	def save(self):
		self.model.save(basepath+self.name+".h5")

	def load(self):
		self.model = load_model(basepath+self.name+".h5")

	#This function is used to retrieve output of hidden layer instead of output
	# when using the model as feature extractor.
	def get_layer_output(self, x, layer_name):
		get_output = keras_backend.function([self.model.layers[0].input], [self.name2layer[layer_name].output])
		output = get_output([x])[0]
		return output


	
#Classifier taking IMU data ACC+GYRO input_shape: 128x6
class IMU_CNN_FEATURE_EXTRACTOR(BaseClassifier):
	def __init__(self,patience,num_filters,layers=3,num_classes=7,kern_size=2,divide_kernel_size=True,fontSize=16,suffix=""):
		self.name = "IMU_"+str(layers)+"-CNN_NOBN_k"+str(kern_size)+"_"+str(num_filters)+"_filters_SGD_dropout_"+suffix
		super().__init__(self.name,patience,fontSize)
		self.model = Sequential()
		filters = num_filters
		self.model.add( Conv1D(filters,input_shape=(128,6),kernel_size=kern_size,padding='same',activation='relu', name="layer_1") )
		self.model.add( Dropout(0.5) )
		self.model.add(MaxPooling1D())
		for i in range(2,layers+1):
			filters = filters*2
			if divide_kernel_size:
				kern_size = int(kern_size / 2)
			layer_name = "layer_"+str(i)
			self.model.add( Conv1D(filters,kernel_size=kern_size,padding='same',activation='relu', name=layer_name) )
			self.model.add( Dropout(0.5) )
			self.model.add(MaxPooling1D())
		#Automatic features
		self.model.add(Flatten(name="automatic_features"))
		self.model.add( Dense(64,activation='relu', name="layer_dense") )
		self.model.add( Dense(num_classes,activation='softmax',  name="output_layer"))
		self.model.compile( loss='mse',metrics=['mse','acc'], optimizer='sgd' )
		self.model.summary()
		self.name2layer = {}
		for layer in self.model.layers:
			self.name2layer[layer.name] = layer


#Classifier taking IMU data ACC+GYRO + 3D magnitude of ACC and GYRO input_shape: 128x8
class IMU_CNN_3D_FEATURE_EXTRACTOR(BaseClassifier):
	def __init__(self,patience,num_filters,layers=3,num_classes=7,kern_size=2,divide_kernel_size=True,fontSize=16,suffix=""):
		self.name = "IMU_3D_"+str(layers)+"-CNN_NOBN_k"+str(kern_size)+"_"+str(num_filters)+"_filters_SGD_dropout_"+suffix
		super().__init__(self.name,patience,fontSize)
		self.model = Sequential()
		filters = num_filters
		self.model.add( Conv1D(filters,input_shape=(128,8),kernel_size=kern_size,padding='same',activation='relu', name="layer_1") )
		self.model.add( Dropout(0.5) )
		self.model.add(MaxPooling1D())
		for i in range(2,layers+1):
			filters = filters*2
			if divide_kernel_size and kern_size >= 4:
				kern_size = int(kern_size / 2)
			layer_name = "layer_"+str(i)
			self.model.add( Conv1D(filters,kernel_size=kern_size,padding='same',activation='relu', name=layer_name) )
			self.model.add( Dropout(0.5) )
			self.model.add(MaxPooling1D())
		#Automatic features
		self.model.add(Flatten(name="automatic_features"))
		self.model.add( Dense(64,activation='relu', name="layer_dense") )
		self.model.add( Dense(num_classes,activation='softmax',  name="output_layer"))
		self.model.compile( loss='mse',metrics=['mse','acc'], optimizer='sgd' )
		self.model.summary()
		self.name2layer = {}
		for layer in self.model.layers:
			self.name2layer[layer.name] = layer


