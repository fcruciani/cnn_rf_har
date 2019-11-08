'''
File: HAPT_Dataset.py
Author: Federico Cruciani
Date: 03/10/2019
Version: 1.0
Description: 
	utility functions to load the 
	Human Activities and Postural Transitions (HAPT) dataset	 
'''
import numpy as np
import pandas as pd 
from os.path import expanduser
from keras.utils import to_categorical
from multiprocessing import Pool as ThreadPool
import math
import scipy.signal

home = expanduser("~")


'''
Dataset Info - Labels:
1 WALKING
2 W_UPSTAIRS
3 W_DOWNSTAIRS
4 SITTING
5 STANDING
6 LAYING
7 STAND_TO_SIT
8 SIT_TO_STAND
9 SIT_TO_LIE
10 LIE_TO_SIT
11 STAND_TO_LIE
12 LIE_TO_STAND
'''

#Modify this line with the right path.
#Dataset available at: http://archive.ics.uci.edu/ml/machine-learning-databases/00341/
ucihapt_datapath = home+"/HAPT_Dataset/"


def get_test_uuids():
	test_uuids = pd.read_csv(ucihapt_datapath+"Test/subject_id_test.txt",names=["UUID"])
	all_test_uuids = np.unique(test_uuids.values)
	return all_test_uuids

def get_train_uuids():
	train_uuids = pd.read_csv(ucihapt_datapath+"Train/subject_id_train.txt",names=["UUID"])
	all_train_uuids = np.unique(train_uuids.values)
	return all_train_uuids


#Get Data no resampling
def get_all_data_multi_thread_noresampling_3D(uuids, n_threads):
	print("Loading data")
	print("Initiating pool")
	print("resampling 50 -> 40 Hz disabled. Doing 3D ")
	uuids_list = [ [x] for x in uuids ]
	pool = ThreadPool(n_threads)
	print("Pool map")
	test_points = pool.map( get_all_data_noresampling_3D,uuids_list)
	print("Pool map")
	pool.close()
	print("Pool join")
	pool.join()

	#Merging data from treads
	print("Merging threads' data")
	X_list = []
	y_list = []
	for res in test_points:
		#dataset_size += len(res[1])
		X_list.extend(res[0])
		y_list.extend(res[1])

	X_es = np.zeros((len(y_list),128,8))
	X_es[:,:] = [x for x in X_list  ]
	y_es = np.zeros(len(y_list))
	y_es[:] = [y for y in y_list]
	y_scaled = to_categorical(y_es, num_classes=7)
	return (X_es, y_scaled)

def get_all_data_noresampling_3D(uuids):
	gt_df = pd.read_csv(ucihapt_datapath+"RawData/labels.txt",sep="\s",names=['EXP_ID','USER_ID','LABEL','START','END'],engine='python')
	#exclude other uuids
	#print( gt_df.head() )
	filtered_df = pd.DataFrame(columns=['EXP_ID','USER_ID','LABEL','START','END'])
	for uuid in uuids:
		data_uuid = gt_df[ gt_df['USER_ID'] == uuid ]
		filtered_df = pd.concat([filtered_df,data_uuid], ignore_index=True)

	X_list = []
	y_list = []
	for index, row in filtered_df.iterrows():
		exp_id = row['EXP_ID']
		user_id = row['USER_ID']
		start = row['START']
		end = row['END']
		label = row['LABEL']
		str_user_id = str(user_id)
		if user_id < 10:
			str_user_id = "0"+str(user_id)
		str_exp_id = str(exp_id)
		if exp_id < 10:
			str_exp_id = "0"+str(exp_id)
		accfile = ucihapt_datapath+"RawData/acc_exp"+str_exp_id+"_user"+str_user_id+".txt"
		gyrfile = ucihapt_datapath+"RawData/gyro_exp"+str_exp_id+"_user"+str_user_id+".txt"
		#print(accfile)
		acc_data_df = pd.read_csv(accfile,names=['x','y','z'],sep='\s|,', engine='python')
		gyr_data_df = pd.read_csv(gyrfile,names=['x','y','z'],sep='\s|,', engine='python')
		acc_x = acc_data_df['x'].values
		acc_y = acc_data_df['y'].values
		acc_z = acc_data_df['z'].values
		gyr_x = gyr_data_df['x'].values
		gyr_y = gyr_data_df['y'].values
		gyr_z = gyr_data_df['z'].values

		acc_mag = []
		gyr_mag = []
		for i in range(len(acc_x)):
			acc_mag.append( math.sqrt( (acc_x[i]*acc_x[i]) + (acc_y[i]*acc_y[i]) + (acc_z[i]*acc_z[i]) ) )
			gyr_mag.append( math.sqrt( (gyr_x[i]*gyr_x[i]) + (gyr_y[i]*gyr_y[i]) + (gyr_z[i]*gyr_z[i]) ) )

		until = start + 128
		while until < end:
			X_point = np.zeros((128,8))
			X_point[:,0] = acc_x[start:until]
			X_point[:,1] = acc_y[start:until]			
			X_point[:,2] = acc_z[start:until]
			X_point[:,3] = gyr_x[start:until]
			X_point[:,4] = gyr_y[start:until]
			X_point[:,5] = gyr_z[start:until]
			X_point[:,6] = acc_mag[start:until]
			X_point[:,7] = gyr_mag[start:until]
			X_list.append(X_point)
			#Remapping id from 1-12 to 0-6
			if label < 7:
				y_list.append(label-1)
			else:
				y_list.append(6) #considering all trainsitions as NULL class 6
			start += 64
			until += 64
	X_es = np.zeros((len(y_list),128,8))
	X_es[:,:] = [x for x in X_list  ]
	y_es = np.zeros(len(y_list))
	y_es[:] = [y for y in y_list]
	print("Finished loading: ",uuids)
	return (X_es, y_es)

#Loads data resampling from 50 to 40 Hz
def get_all_data_multi_thread_resampling_3D(uuids, n_threads):
	print("Loading Test set")
	print("Initiating pool")
	print("resampling 50 -> 40 Hz Enabled. Doing 3D ")
	uuids_list = [ [x] for x in uuids ]
	pool = ThreadPool(n_threads)
	print("Pool map")
	test_points = pool.map( get_all_data_noresampling_3D,uuids_list)
	print("Pool map")
	pool.close()
	print("Pool join")
	pool.join()

	#Merging data from treads
	print("Merging threads' data")
	X_list = []
	y_list = []
	for res in test_points:
		#dataset_size += len(res[1])
		X_list.extend(res[0])
		y_list.extend(res[1])

	X_es = np.zeros((len(y_list),128,8))
	X_es[:,:] = [x for x in X_list  ]
	y_es = np.zeros(len(y_list))
	y_es[:] = [y for y in y_list]
	y_scaled = to_categorical(y_es, num_classes=7)
	return (X_es, y_scaled)

def get_all_data_resampling_3D(uuids,resampling=True):
	#Load groundtruth
	gt_df = pd.read_csv(ucihapt_datapath+"RawData/labels.txt",sep="\s",names=['EXP_ID','USER_ID','LABEL','START','END'],engine='python')

	#Filter data: only uuids
	#Empty data frame 
	filtered_df = pd.DataFrame(columns=['EXP_ID','USER_ID','LABEL','START','END'])
	for uuid in uuids:
		#add data for user ID is in list
		data_uuid = gt_df[ gt_df['USER_ID'] == uuid ]
		filtered_df = pd.concat([filtered_df,data_uuid], ignore_index=True)

	X_list = []
	y_list = []
	#Iterating filtered groundtruth
	for index, row in filtered_df.iterrows():
		exp_id = row['EXP_ID'] #Used to retrive raw data file
		user_id = row['USER_ID'] #Used to retrieve raw data file
		start = row['START'] #Start of data segment with this label
		end = row['END'] #End of segment
		label = row['LABEL'] #Label of this segment
		str_user_id = str(user_id)
		if user_id < 10:
			str_user_id = "0"+str(user_id)
		str_exp_id = str(exp_id)
		if exp_id < 10:
			str_exp_id = "0"+str(exp_id)
		#Load raw data file
		accfile = ucihapt_datapath+"RawData/acc_exp"+str_exp_id+"_user"+str_user_id+".txt"
		gyrfile = ucihapt_datapath+"RawData/gyro_exp"+str_exp_id+"_user"+str_user_id+".txt"
		acc_data_df = pd.read_csv(accfile,names=['x','y','z'],sep='\s|,', engine='python')
		gyr_data_df = pd.read_csv(gyrfile,names=['x','y','z'],sep='\s|,', engine='python')
		acc_x = acc_data_df['x'].values
		acc_y = acc_data_df['y'].values
		acc_z = acc_data_df['z'].values
		gyr_x = gyr_data_df['x'].values
		gyr_y = gyr_data_df['y'].values
		gyr_z = gyr_data_df['z'].values

		#Isolate relevant data
		acc_x = acc_x[ start:end ]
		acc_y = acc_z[ start:end ]
		acc_z = acc_y[ start:end ]
		gyr_x = gyr_x[ start:end ]
		gyr_y = gyr_y[ start:end ]
		gyr_z = gyr_z[ start:end ] 

		#Calculate 3D magnitude of the signals
		acc_mag = []
		gyr_mag = []
		for i in range(len(acc_x)):
			acc_mag.append( math.sqrt( (acc_x[i]*acc_x[i]) + (acc_y[i]*acc_y[i]) + (acc_z[i]*acc_z[i]) ) )
			gyr_mag.append( math.sqrt( (gyr_x[i]*gyr_x[i]) + (gyr_y[i]*gyr_y[i]) + (gyr_z[i]*gyr_z[i]) ) )

		#Resampling factor: 50 / 40 = 1.25
		#downsampling from 50 to 40 Hz for Extrasensory compatibility
		num_samples_50Hz = end - start
		num_samples_40Hz = num_samples_50Hz / 1.25

		##DOWNSAMPLING from 50 to 40 Hz
		acc_x = scipy.signal.resample( acc_x, int(num_samples_40Hz) )
		acc_x = scipy.signal.resample( acc_y, int(num_samples_40Hz) )
		acc_x = scipy.signal.resample( acc_z, int(num_samples_40Hz) )
		gyr_x = scipy.signal.resample( gyr_x, int(num_samples_40Hz) )
		gyr_x = scipy.signal.resample( gyr_y, int(num_samples_40Hz) )
		gyr_x = scipy.signal.resample( gyr_z, int(num_samples_40Hz) )
		acc_mag = scipy.signal.resample( acc_mag, int(num_samples_40Hz) )
		gyr_mag = scipy.signal.resample( gyr_mag, int(num_samples_40Hz) )
		
		segment_start = 0
		segment_end = num_samples_40Hz
		#Performing segmentation: sliding window 50% overlap		
		until = segment_start + 128
		while until < segment_end:
			X_point = np.zeros((128,8))
			X_point[:,0] = acc_x[segment_start:until]
			X_point[:,1] = acc_y[segment_start:until]			
			X_point[:,2] = acc_z[segment_start:until]
			X_point[:,3] = gyr_x[segment_start:until]
			X_point[:,4] = gyr_y[segment_start:until]
			X_point[:,5] = gyr_z[segment_start:until]
			X_point[:,6] = acc_mag[segment_start:until]
			X_point[:,7] = gyr_mag[segment_start:until]
			X_list.append(X_point)
			#All activities + transitions
			if label < 7:
				#all activities except transitions
				y_list.append(label-1)
			else:
				#putting all transitions in same class
				y_list.append(6)
			segment_start += 64
			until += 64
	X_es = np.zeros((len(y_list),128,8))
	X_es[:,:] = [x for x in X_list  ]
	y_es = np.zeros(len(y_list))
	y_es[:] = [y for y in y_list]
	#y_scaled = to_categorical(y_es, num_classes=7)
	print("Finished loading: ",uuids)
	return (X_es, y_es)



