'''
File: CNN_RF.py
Author: Federico Cruciani
Date: 05/10/2019
Version: 1.0
Description: 
	Class implementing a Random Forest using CNN pre-trained feature extractor.	 
'''

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import Classifiers
import pickle


class CNN_RF_Classifier:
	def __init__(self,num_filters):
		self.featExtractor = Classifiers.IMU_CNN_3D_FEATURE_EXTRACTOR(suffix="40Hz",num_filters=num_filters,patience=250,layers=3,kern_size=32,divide_kernel_size=True)
		self.featExtractor.loadBestWeights()
		#Output of grid search:
		#{'bootstrap': False, 'max_depth': 142, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 60}
		#Initializing RF
		self.clf_RF = RandomForestClassifier(n_estimators=60,min_samples_split=2,min_samples_leaf=2,max_features='auto',max_depth=142,bootstrap=False)

	def save_classifier(self,name="RF_model"):
		f = open(name+'_rnd_opt.pickle', 'wb')
		pickle.dump(self.clf_RF, f, -1)
		f.close()

	def load_classifier(self,name="RF_model"):
		f = open(name+'_rnd_opt.pickle', 'rb')
		self.clf_RF = pickle.load(f)
		f.close()

	def fit(self, X_train, y_train ):
		##taking input 128x8
		X_auto_features = self.featExtractor.get_layer_output(X_train,"automatic_features")
		self.clf_RF.fit(X_auto_features,y_train)

	def score(self,X_test,y_test):
		##taking input 128x8
		X_auto_features = self.featExtractor.get_layer_output(X_test,"automatic_features")
		self.clf_RF.score(X_auto_features,y_test)

	def predict(self,X_test):
		##taking input 128x8
		X_auto_features = self.featExtractor.get_layer_output(X_test,"automatic_features")
		return self.clf_RF.predict(X_auto_features)