import pycuda.autoinit
import pycuda.driver as cuda
import sys
from math import sqrt
from math import ceil
from pycuda.compiler import SourceModule
import csv
import random
import math
import operator
import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as sm
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix

mod = SourceModule("""

__global__ void EucledianDistance(float *test,float *train,float *result,int n_rows,int n_cols,int n_features)

{
      int bx = blockIdx.x, by = blockIdx.y;
      int k;
      int i = by * n_rows + bx;
      float tmp,s=0;

      for(k=0;k<n_features;k++)
      {
         tmp=test[bx * n_features + k]-train[by * n_features + k];
         s+=tmp*tmp;
      }
      result[i] = sqrt(s);
  }
 """)

eucDist = mod.get_function("EucledianDistance")

trainingSet_opt = []
testSet_opt = []
dataset_length = 0
 
def loadDataset(filename,start,last,trainingSet=[],testSet=[]):

	print("start: "+str(start)+" last: "+str(last)+"\n")

	with open(filename, 'rt') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		dataset = dataset[1:len(dataset)]
		global dataset_length 
		dataset_length = len(dataset)
		print("No of instances in the dataset:"+str(len(dataset)))

		total_features = len(dataset[0]) - 1;
		print("Total no of features per instance:"+str(total_features))

		for x in range(len(dataset)-1):
			if dataset[x][-1] == '0':
				dataset[x][-1] = '0'
			else:
				dataset[x][-1] = '1'

		for x in range(len(dataset)-1):
			for y in range(total_features):
	            		dataset[x][y] = float(dataset[x][y])

			if x>=start and x<=last:
				testSet.append(dataset[x])
			else:
				trainingSet.append(dataset[x])
			

	#dataset = np.asarray(dataset)
	#dataset = dataset.astype(np.float32)	
	#dataset = np.append(arr = np.ones((len(dataset),1)).astype(int), values = dataset, axis=1)
	#y=[]
 
	#y = dataset[:, -1:]	

	#y = np.asarray(y)
	#y = y.astype(np.float32)
	#dataset_opt = dataset[: , 0:51]
	#dataset_opt = np.delete(arr = dataset_opt , obj = 3, axis = 1)
	#dataset_opt=  np.delete(arr=dataset_opt,obj=42, axis=1)
	#dataset_opt=np.delete(arr=dataset_opt,obj=44,axis=1)
	#dataset_opt=np.delete(arr=dataset_opt,obj=24,axis=1)
	#dataset_opt=np.delete(arr=dataset_opt,obj=18,axis=1)
	#dataset_opt=np.delete(arr=dataset_opt,obj=33,axis=1)
	#dataset_opt=np.delete(arr=dataset_opt,obj=44,axis=1)
	#dataset_opt=np.delete(arr=dataset_opt,obj=30,axis=1)
	#dataset_opt=np.delete(arr=dataset_opt,obj=30,axis=1)
	#dataset_opt=np.delete(arr=dataset_opt,obj=9,axis=1)
	#dataset_opt=np.delete(arr=dataset_opt,obj=14,axis=1)
	#dataset_opt=np.delete(arr=dataset_opt,obj=16,axis=1)
	#dataset_opt=np.delete(arr=dataset_opt,obj=32,axis=1)
	#dataset_opt=np.delete(arr=dataset_opt,obj=16,axis=1)
	#dataset_opt=np.delete(arr=dataset_opt,obj=23,axis=1)
	#dataset_opt=np.delete(arr=dataset_opt,obj=30,axis=1)
	#dataset_opt=np.delete(arr=dataset_opt,obj=27,axis=1)
	#dataset_opt=np.delete(arr=dataset_opt,obj=13,axis=1)
	#dataset_opt=np.delete(arr=dataset_opt,obj=10,axis=1)
	#dataset_opt=np.delete(arr=dataset_opt,obj=3,axis=1)
	#dataset_opt=np.delete(arr=dataset_opt,obj=21,axis=1)
	#dataset_opt=np.delete(arr=dataset_opt,obj=26,axis=1)
	#dataset_opt=np.delete(arr=dataset_opt,obj=7,axis=1)
	#dataset_opt=np.delete(arr=dataset_opt,obj=3,axis=1)
	#dataset_opt=np.delete(arr=dataset_opt,obj=14,axis=1)
	#dataset_opt=np.delete(arr=dataset_opt,obj=9,axis=1)
	#dataset_opt=np.delete(arr=dataset_opt,obj=19,axis=1)

	#regressor_OLS = sm.OLS(endog = y , exog = dataset_opt).fit()
	#print(regressor_OLS.summary())

	#print("dataset_opt_shape :"+str(dataset_opt.shape))
	#print("y_shape :"+str(y.shape))
	#dataset_opt=np.delete(arr=dataset_opt,obj=0,axis=1)
	#dataset_opt = np.append(arr = dataset_opt, values = y, axis=1)
	
	#dataset_opt = list(dataset_opt)	

	#total_features = len(dataset_opt[0]) - 1

	#for x in range(len(dataset_opt)-1):
        #                for y in range(total_features):
        #                        dataset_opt[x][y] = float(dataset_opt[x][y])
                        #if x < 1001:
                        #        trainingSet_opt.append(dataset_opt[x])
                        #else:
                        #        testSet_opt.append(dataset_opt[x])


	return total_features
 
def getNeighbors(trainingSet, dist, k):
	distances = []
	for x in range(len(trainingSet)):
		distances.append((trainingSet[x], dist[x]))
	
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors
 
def FindtheClass(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		neighbor_class = neighbors[x][-1]
		if neighbor_class in classVotes:
			classVotes[neighbor_class] += 1
		else:
			classVotes[neighbor_class] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]
 
def FindtheAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet)))
	
def main():
	# prepare data
	trainingSet=[]
	testSet=[]
	testSet_trunc=[]
	trainingSet_trunc=[]
	filename = 'ransomeware.data'

	start = 0
	last = 1000
	
	total_features = loadDataset(filename,start,last, trainingSet, testSet)
	#print("Train set: " + repr(len(trainingSet)))
	#print("Test set: " + repr(len(testSet)))

	no_of_folds = 10
	subset_size = ceil(dataset_length/no_of_folds)

	start = 0
	last = subset_size - 1

	acc_values = defaultdict(list)
	#acc_values_xgboost = []
	final_cm = np.zeros((2,2))
	final_cm = final_cm.astype(float)

	start_time = time.time()	

	for i in range(no_of_folds):

		trainingSet = []
		testSet = []
		#print("start: "+str(start)+" last: "+str(last)+"\n")
		total_features = loadDataset(filename,start,last,trainingSet,testSet)
		
		print("Train set: " + repr(len(trainingSet)))
		print("Test set: " + repr(len(testSet)))
        	
		#create arrays of test and train data to sent to gpu
		for x in range(len(testSet)):
                	testSet_trunc.append(testSet[x][0:total_features])
		a = np.asarray(testSet_trunc)
		a=a.astype(np.float32)
		for x in range(len(trainingSet)):
                	trainingSet_trunc.append(trainingSet[x][0:total_features])
		b = np.asarray(trainingSet_trunc)
		b=b.astype(np.float32)
		a_gpu=cuda.mem_alloc(a.nbytes)
		b_gpu=cuda.mem_alloc(b.nbytes)
		n_rows = len(testSet)
		n_rows = np.int32(n_rows)
		n_cols = len(trainingSet)
		n_cols = np.int32(n_cols)
		dist_mat = [[0 for x in range(n_rows)] for y in range(n_cols)]
		dist_mat = np.asarray(dist_mat)
		dist_mat = dist_mat.astype(np.float32)
		dist_gpu=cuda.mem_alloc(dist_mat.nbytes)
		cuda.memcpy_htod(a_gpu,a)
		cuda.memcpy_htod(b_gpu,b)
		#kernel call to calculate the eucledian distance matrix
		#eucDist = mod.get_function("EucledianDistance")
		grid = (int(n_rows),int(n_cols),1)
		n_features = total_features;
		n_features = np.int32(n_features)

		eucDist(a_gpu,b_gpu,dist_gpu,n_rows,n_cols,n_features,block=(1,1,1),grid=grid)
	        #Get the result from gpu into an array
		cuda.memcpy_dtoh(dist_mat,dist_gpu)
		dist_mat = np.transpose(dist_mat)
		euc_distance = list(dist_mat)

		# generate predictions
		#acc_values = []
		k_values = [2,3,5,7,9,11,13,17,19,23,25,39,60,80,100,120,140,160,180,200,300,400,500,800,1000,1300]

		for k in k_values:
			predictions=[]
			for x in range(len(testSet)):
				neighbors = getNeighbors(trainingSet, euc_distance[x], k)
				result = FindtheClass(neighbors)
				predictions.append(result)
		
			accuracy = FindtheAccuracy(testSet, predictions)
			
			
			#cfm_result = confusion_matrix(testSet,predictions)
			#print(cfm_result)
			#print("\n")
			
			#if k not in accuracy:
			#	accuracy[k] = float(accuracy)
			#else:
			acc_values[k].append(float(accuracy))
			
			test_labels = []
			for x in range(len(testSet)):
				test_labels.append(testSet[x][-1])
			
			test_labels = np.asarray(test_labels)
			label_pred = np.asarray(predictions)
			cm = confusion_matrix(test_labels, label_pred)
			
			with open('k_cfm.data','a+') as f:
				f.write(str(k)+","+str(i)+"\n"+str(cm)+"\n")		
	
		#with open('k_vs_acc.data','w') as f:
		#	for x in range(len(k_values)):
		#		f.write(str(k_values[x])+","+str(acc_values[x])+"\n")
	

		#create arrays of testSet_opt and trainSet_opt data to sent to gpu
		#a = np.asarray(testSet_opt)
		#a=a.astype(np.float32)
		#b = np.asarray(trainingSet_opt)
		#b=b.astype(np.float32)
		#a_gpu=cuda.mem_alloc(a.nbytes)
		#b_gpu=cuda.mem_alloc(b.nbytes)
		#n_rows = len(testSet_opt)
		#n_rows = np.int32(n_rows)
		#n_cols = len(trainingSet_opt)
		#n_cols = np.int32(n_cols)
		#dist_mat = [[0 for x in range(n_rows)] for y in range(n_cols)]
		#dist_mat = np.asarray(dist_mat)
		#dist_mat = dist_mat.astype(np.float32)
		#dist_gpu=cuda.mem_alloc(dist_mat.nbytes)
		#cuda.memcpy_htod(a_gpu,a)
		#cuda.memcpy_htod(b_gpu,b)
		#kernel call to calculate the eucledian distance matrix
		#eucDist = mod.get_function("EucledianDistance")
		#grid = (int(n_rows),int(n_cols),1)
		#total_features = len(trainingSet_opt[0]) - 1
		#n_features = total_features;
		#n_features = np.int32(n_features)
		#eucDist(a_gpu,b_gpu,dist_gpu,n_rows,n_cols,n_features,block=(1,1,1),grid=grid)
	        #Get the result from gpu into an array
		#cuda.memcpy_dtoh(dist_mat,dist_gpu)
		#dist_mat = np.transpose(dist_mat)
		#euc_distance = list(dist_mat)
	
		# generate predictions
		#acc_values = []
		#k_values = [2,3,5,7,9,11,13,17,19,23,25]

		#for k in k_values:
		#	predictions=[]
		#	for x in range(len(testSet_opt)):
		#		neighbors = getNeighbors(trainingSet_opt, euc_distance[x], k)
		#		result = FindtheClass(neighbors)
		#		predictions.append(result)
		
		#	accuracy = FindtheAccuracy(testSet_opt, predictions)
		#	acc_values.append(float(accuracy))

		#with open('k_vs_acc_with_backward_elimination.data','w') as f:
		#	for x in range(len(k_values)):
		#		f.write(str(k_values[x])+","+str(acc_values[x])+"\n")
		

		# Testing for XGBooster
		
		#train_labels = []
		#for x in range(len(trainingSet)):
		#	train_labels.append(trainingSet[x][-1])
		
		#train_labels = np.asarray(train_labels)
		#trainingSet = np.asarray(trainingSet)
		#testSet = np.asarray(testSet)
		#model = XGBClassifier()
		#model.fit(trainingSet,train_labels)
		
		#label_pred = model.predict(testSet)
		#label_pred = label_pred.astype(float)
		#print(label_pred.type())
		#print(label_pred)
		#predictions = [round(value) for value in label_pred]
		
		#testSet = list(testSet)
		#accuracy = FindtheAccuracy(testSet, predictions)

		#acc_values_xgboost.append(accuracy)
	
		# End of XGBooster

		start = start + subset_size
		last = last + subset_size

	end_time = time.time()

	print("GPU Time : "+str(end_time-start_time)+"seconds")
	
	with open('k_vs_acc.data','w') as f:
		for x in range(len(k_values)):
			f.write(str(k_values[x])+","+str(acc_values[k_values[x]])+"\n")	

#	with open('xgbooster_acc.data','w') as f:
		#for x in range(len(acc_values_xgboost)):
		#	f.write(str(acc_values_xgboost[x])+"\n")
main()
