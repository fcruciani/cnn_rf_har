'''
File: ES_Dataset.py
Author: Federico Cruciani
Date: 05/03/2019
Version: 1.1
Description: 
	Utility Class ito load Extrasensory dataset: raw data from accelerometer and gyroscope.	 
'''

import numpy as np
import pandas as pd
from os.path import expanduser
from sklearn.utils import shuffle
from keras.utils import to_categorical
import scipy.signal
import random
import os
import math

#get actual home path for current user
home = expanduser("~")


# Change this to the right path.
#Data available at: http://extrasensory.ucsd.edu/
rawdatafolder = home+'/extrasensory_data/raw_acc/raw_data/'
gyrodatafolder = home+'/extrasensory_data/raw_gyro/proc_gyro/'

#In ExtraSensory Android user acceleration is in [m/s^2], for iOS is in [g]
#This dictionary indicates for which users data are normalized to [g].
normalized = {'E65577C1-8D5D-4F70-AF23-B3ADB9D3DBA3':True,'BE3CA5A6-A561-4BBD-B7C9-5DF6805400FC':True,'C48CE857-A0DD-4DDB-BEA5-3A25449B2153':True,'61359772-D8D8-480D-B623-7C636EAD0C81':False,'9759096F-1119-4E19-A0AD-6F16989C7E1C':True,'665514DE-49DC-421F-8DCB-145D0B2609AD':False,'81536B0A-8DBF-4D8A-AC24-9543E2E4C8E0':False,'8023FE1A-D3B0-4E2C-A57A-9321B7FC755F':True,'5152A2DF-FAF3-4BA8-9CA9-E66B32671A53':True,'5119D0F8-FCA8-4184-A4EB-19421A40DE0D':True,'78A91A4E-4A51-4065-BDA7-94755F0BB3BB':True,'33A85C34-CFE4-4732-9E73-0A7AC861B27A':False,'24E40C4C-A349-4F9F-93AB-01D00FB994AF':True,'1DBB0F6F-1F81-4A50-9DF4-CD62ACFA4842':False,'0A986513-7828-4D53-AA1F-E02D6DF9561B':True,'797D145F-3858-4A7F-A7C2-A4EB721E133C':False,'99B204C0-DD5C-4BB7-83E8-A37281B8D769':False,'098A72A5-E3E5-4F54-A152-BBDA0DF7B694':True,'74B86067-5D4B-43CF-82CF-341B76BEA0F4':False,'4FC32141-E888-4BFF-8804-12559A491D8C':True,'59818CD2-24D7-4D32-B133-24C2FE3801E5':False, 'D7D20E2E-FC78-405D-B346-DBD3FD8FC92B':True,'F50235E0-DD67-4F2A-B00B-1F31ADA998B9':False,'9DC38D04-E82E-4F29-AB52-B476535226F2':True,'99B204C0-DD5C-4BB7-83E8-A37281B8D769':False, '4FC32141-E888-4BFF-8804-12559A491D8C':True,'098A72A5-E3E5-4F54-A152-BBDA0DF7B694':True,'74B86067-5D4B-43CF-82CF-341B76BEA0F4':False,'0BFC35E2-4817-4865-BFA7-764742302A2D':False,'A76A5AF5-5A93-4CF2-A16E-62353BB70E8A':True, '4FC32141-E888-4BFF-8804-12559A491D8C':True, 'B09E373F-8A54-44C8-895B-0039390B859F':True, '59EEFAE0-DEB0-4FFF-9250-54D2A03D0CF2': False, '7CE37510-56D0-4120-A1CF-0E23351428D2': True, 'ECECC2AB-D32F-4F90-B74C-E12A1C69BBE2': True, '0A986513-7828-4D53-AA1F-E02D6DF9561B': True, '481F4DD2-7689-43B9-A2AA-C8772227162B': False, '806289BC-AD52-4CC1-806C-0CDB14D65EB6': False, '136562B6-95B2-483D-88DC-065F28409FD2': True, '86A4F379-B305-473D-9D83-FC7D800180EF': False, '665514DE-49DC-421F-8DCB-145D0B2609AD': False, '3600D531-0C55-44A7-AE95-A7A38519464E': True, 'BEF6C611-50DA-4971-A040-87FB979F3FC1': False, 'A7599A50-24AE-46A6-8EA6-2576F1011D81': False, 'CF722AA9-2533-4E51-9FEB-9EAC84EE9AAC': True, '83CF687B-7CEC-434B-9FE8-00C3D5799BE6': True, '00EABED2-271D-49D8-B599-1D4A09240601': True, 'CCAF77F0-FABB-4F2F-9E24-D56AD0C5A82F': False, '7D9BB102-A612-4E2A-8E22-3159752F55D8': True, 'C48CE857-A0DD-4DDB-BEA5-3A25449B2153': True, '4E98F91F-4654-42EF-B908-A3389443F2E7': False, '96A358A0-FFF2-4239-B93E-C7425B901B47': False, 'FDAA70A1-42A3-4E3F-9AE3-3FDA412E03BF': True, '40E170A7-607B-4578-AF04-F021C3B0384A': False, 'CA820D43-E5E2-42EF-9798-BE56F776370B': True, 'B7F9D634-263E-4A97-87F9-6FFB4DDCB36C': False, '27E04243-B138-4F40-A164-F40B60165CF3': False, '2C32C23E-E30C-498A-8DD2-0EFB9150A02E': True, 'A5CDF89D-02A2-4EC1-89F8-F534FDABDD96': True, 'B9724848-C7E2-45F4-9B3F-A1F38D864495': True, '11B5EC4D-4133-4289-B475-4E737182A406': True, '1155FF54-63D3-4AB2-9863-8385D0BD0A13': False, 'CDA3BBF7-6631-45E8-85BA-EEB416B32A3C': True, '61976C24-1C50-4355-9C49-AAE44A7D09F6': True, '0E6184E1-90C0-48EE-B25A-F1ECB7B9714E': False, 'A5A30F76-581E-4757-97A2-957553A2C6AA': False, '1538C99F-BA1E-4EFB-A949-6C7C47701B20': True, '5EF64122-B513-46AE-BCF1-E62AAC285D2C': True }

def get_train_test_uuids(fold,include_gyro=True):
	gyro_ids_df = pd.read_csv("gyro_ids.txt",names=['UUID'])
	gyro_ids = gyro_ids_df['UUID'].values.tolist()
	if fold in ["0","1","2","3","4"]:
		print("Starting fold: ",fold)
		train_df_fold_iphone = pd.read_csv("./es_cv_5_folds/fold_"+fold+"_train_iphone_uuids.txt",names=['UUID'])
		train_df_fold_android = pd.read_csv("./es_cv_5_folds/fold_"+fold+"_train_android_uuids.txt",names=['UUID'])
		for uuid in train_df_fold_android['UUID'].values.tolist():
			if normalized[uuid] == True:
				print(uuid," : error")
			
		for uuid in train_df_fold_iphone['UUID'].values.tolist():
			if normalized[uuid] == False:
				print(uuid," : error")
			
		train_df_fold = pd.concat([train_df_fold_iphone,train_df_fold_android])
		train_uuids = train_df_fold['UUID'].values.tolist()

		test_df_fold_iphone = pd.read_csv("./es_cv_5_folds/fold_"+fold+"_test_iphone_uuids.txt",names=['UUID'])
		test_df_fold_android = pd.read_csv("./es_cv_5_folds/fold_"+fold+"_test_android_uuids.txt",names=['UUID'])
		for uuid in test_df_fold_android['UUID'].values.tolist():
			if normalized[uuid] == True:
				print(uuid," : error")
			
		for uuid in test_df_fold_iphone['UUID'].values.tolist():
			if normalized[uuid] == False:
				print(uuid," : error")
			
		test_df_fold = pd.concat([test_df_fold_iphone,test_df_fold_android])
		test_uuids = test_df_fold['UUID'].values.tolist()

		tr_uuids = []
		ts_uuids = []
		for uuid in train_uuids:
			if include_gyro == True:
				if uuid in gyro_ids:
					tr_uuids.append(uuid)
			else:
				tr_uuids.append(uuid)

		for uuid in test_uuids:
			if include_gyro == True:
				if uuid in gyro_ids:
					ts_uuids.append(uuid)
			else:
				ts_uuids.append(uuid)

		return tr_uuids, ts_uuids


def labelToIndex(label):
	if label == 'LYING_DOWN':
		return 0
	elif label == 'SITTING':
		return 1
	elif label == "WALKING":
		return 2
	elif label == "RUNNING":
		return 3
	elif label == "CYCLING":
		return 4


def get_all_dataset_IMU(uuids,show_progress=False,balanced=False,smote=False):
	gt_df = pd.read_csv("GROUNDTRUTH_EXTRASENSORY.csv",names=['UUID','label','steps','ts','class'])
	#exclude other uuids
	filtered_df = pd.DataFrame(columns=['UUID','label','steps','ts','class'])
	for cuuid in uuids:
		data_uuid = gt_df[ gt_df['UUID'] == cuuid ]
		filtered_df = pd.concat([filtered_df,data_uuid], ignore_index=True)

	#print("all data: ",len(gt_df))
	#print("only uuids: ", len(filtered_df))
	df_lyi_all = filtered_df[filtered_df['label'] == "LYING_DOWN" ] #0
	df_sit_all = filtered_df[filtered_df['label'] == "SITTING" ] #1
	df_wal_all = filtered_df[filtered_df['label'] == "WALKING" ] #2
	df_run_all = filtered_df[filtered_df['label'] == "RUNNING" ] #3
	df_cyc_all = filtered_df[filtered_df['label'] == "CYCLING" ] #4
	#df_sup_all = filtered_df[filtered_df['label'] == "STAIRS_UP" ] #4
	#df_sdo_all = filtered_df[filtered_df['label'] == "STAIRS_DOWN" ] #5

	if balanced == False:
			dataset_df = pd.concat([df_wal_all,df_sit_all,df_run_all,df_cyc_all,df_lyi_all], ignore_index=True)
	else:
		min_size = min( len(df_lyi_all),len(df_sit_all),len(df_wal_all),len(df_run_all),len(df_cyc_all) )
		if smote == False:
			dataset_df = pd.concat([df_wal_all.sample(n=min_size),df_lyi_all.sample(n=min_size),df_sit_all.sample(n=min_size),df_run_all.sample(n=min_size),df_cyc_all.sample(n=min_size)], ignore_index=True)
		else:
			dataset_df = pd.concat([df_wal_all.sample(n=min_size),df_lyi_all.sample(n=min_size),df_sit_all.sample(n=min_size),df_run_all.sample(n=min_size),df_cyc_all.sample(n=min_size)], ignore_index=True)
			dataset_df = pd.concat([dataset_df,df_wal_all.sample(n=min_size),df_lyi_all.sample(n=min_size),df_sit_all.sample(n=min_size),df_run_all.sample(n=min_size),df_cyc_all.sample(n=min_size)], ignore_index=True)
	
	X_list = []
	y_list =[]

	for index, fragment in dataset_df.iterrows():
		if show_progress:
			utils.printProgressBar(index, len(dataset_df), prefix = 'Progress:', suffix = 'Complete',length = 30)
		##check if data exists
		rawfile = rawdatafolder+fragment[0]+"/"+ str(fragment[3]) +".m_raw_acc.dat.gz"
		gyrofile = gyrodatafolder+fragment[0]+"/"+ str(fragment[3]) +".m_proc_gyro.dat.gz"
		if os.path.exists(gyrofile) and os.path.exists(rawfile):
			#feats = extrautils.extractAccGyroFeatures(fragment[0], fragment[3], normalized[fragment[0]],step=60 )
			raw_data = pd.read_csv(rawfile,names=['ts','x','y','z'],sep=' ', compression='gzip')
			raw_gyro = pd.read_csv(gyrofile,names=['ts','x','y','z'],sep=' ', compression='gzip')
			num_samples = min(len(raw_data),len(raw_gyro))
			label = fragment[1]
			gravity = 9.80665
			if normalized[fragment[0]] == True:
				df_x = raw_data[['x']]
				df_y = raw_data[['y']]
				df_z = raw_data[['z']]
			else:
				df_x = raw_data[['x']] / gravity
				df_y = raw_data[['y']] / gravity
				df_z = raw_data[['z']] / gravity

			acc_x = df_x['x'].values
			acc_y = df_y['y'].values
			acc_z = df_z['z'].values
			gyr_x = raw_gyro['x'].values
			gyr_y = raw_gyro['y'].values
			gyr_z = raw_gyro['z'].values

			#acceleration 3D magnitude
			acc_m = []
			#gyroscope 3D magnitude
			gyr_m = []
			for i in range(len(acc_x)):
				acc_m.append( math.sqrt( (acc_x[i]*acc_x[i])+(acc_y[i]*acc_y[i])+(acc_z[i]*acc_z[i]) ) )

			
			for i in range(len(gyr_x)):
				gyr_m.append( math.sqrt( (gyr_x[i]*gyr_x[i])+(gyr_y[i]*gyr_y[i])+(gyr_z[i]*gyr_z[i]) ) )
			

			#Note in some gyroscope raw data files are truncated
			end = min(len(acc_x),len(acc_y),len(acc_z),len(gyr_x),len(gyr_y),len(gyr_z))
			
			if balanced == True:
				start = random.randint(0,32)
			else:
				start = 0
			
			until = start + 128
			while until < end:
				X_point = np.zeros((128,8))
				X_point[:,0] = acc_x[start:until]
				X_point[:,1] = acc_y[start:until]			
				X_point[:,2] = acc_z[start:until]
				X_point[:,3] = gyr_x[start:until]
				X_point[:,4] = gyr_y[start:until]
				X_point[:,5] = gyr_z[start:until]
				X_point[:,6] = acc_m[start:until]
				X_point[:,7] = gyr_m[start:until]
				X_list.append(X_point)
				y_list.append(labelToIndex(label))
				start += 64
				until += 64
		else:
			print("Missing data: ",fragment[0]+" ts: "+ str(fragment[3]))
				
	#print("Done Loading data: ",uuid)
	X_es = np.zeros((len(X_list),128,8))
	X_es[:] = [x for x in X_list]
	y_es = np.zeros(len(y_list))
	y_es[:] = [y for y in y_list]
	X_shuffled, y_shuffled = shuffle(X_es, y_es)
	return (X_shuffled, y_shuffled)