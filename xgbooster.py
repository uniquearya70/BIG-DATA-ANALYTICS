import csv
import random
import math
import operator
import numpy as np
import time
#from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler
#import statsmodels.formula.api as sm
#from collections import defaultdict
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from math import ceil
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

trainingSet_opt = []
testSet_opt = []
dataset_length = 0
dataset_main = []

def loadDataset(filename,start,last,trainingSet=[],testSet=[]):

	print("start: "+str(start)+" last: "+str(last)+"\n")

	with open(filename, 'rt') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		dataset = dataset[1:len(dataset)]
		global dataset_length
		dataset_length = len(dataset)
		
		global dataset_main
		for x in range(len(dataset)):
			dataset_main.append(dataset[x])

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
	return total_features

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
	acc_values_xgboost = []
	final_cm = np.zeros((2,2))
	final_cm = final_cm.astype(float)

	for i in range(no_of_folds):

		trainingSet = []
		testSet = []
		trainingSet_trunc = []
		testSet_trunc = []
                #print("start: "+str(start)+" last: "+str(last)+"\n")
		total_features = loadDataset(filename,start,last,trainingSet,testSet)

		print("Train set: " + repr(len(trainingSet)))
		print("Test set: " + repr(len(testSet)))
	        
		for x in range(len(trainingSet)):
			trainingSet_trunc.append(trainingSet[x][0:len(trainingSet[x])-1])
		
		for x in range(len(testSet)):
			testSet_trunc.append(testSet[x][0:len(testSet[x])-1])	
                # Testing for XGBooster

		train_labels = []
		for x in range(len(trainingSet)):
			train_labels.append(trainingSet[x][-1])

		train_labels = np.asarray(train_labels)
		trainingSet_trunc = np.asarray(trainingSet_trunc)
		testSet_trunc = np.asarray(testSet_trunc)
		model = XGBClassifier()
		model.fit(trainingSet_trunc,train_labels)

		label_pred = model.predict(testSet_trunc)
		#label_pred = label_pred.astype(float)
                #print(label_pred.type())
                #print(label_pred)
		#predictions = [round(value) for value in label_pred]

		testSet = list(testSet)
		accuracy = FindtheAccuracy(testSet, label_pred)

		acc_values_xgboost.append(accuracy)
		
		test_labels = []
		for x in range(len(testSet)):
			test_labels.append(testSet[x][-1])

		#print("test_labels= "+str(test_labels))
		#print("\npredictions= "+str(label_pred))
		test_labels = np.asarray(test_labels)
		label_pred = np.asarray(label_pred)
		cm = confusion_matrix(test_labels, label_pred)
		
		final_cm[0][0] = final_cm[0][0] + cm[0][0]
		final_cm[0][1] = final_cm[0][1] + cm[0][1]
		final_cm[1][0] = final_cm[1][0] + cm[1][0]
		final_cm[1][1] = final_cm[1][1] + cm[1][1]

		print("\nconfusion matrix = "+str(cm))
		print("\naccuracy= "+str(accuracy))

                # End of XGBooster  
		
		start = start + subset_size
		last = last + subset_size
	 
	with open('xgbooster_acc.data','w') as f:
		for x in range(len(acc_values_xgboost)):
			f.write(str(acc_values_xgboost[x])+"\n")

	total = 0
	for x in range(len(acc_values_xgboost)):
		total = total + acc_values_xgboost[x]

	avg_acc = float(total/no_of_folds)
	print("avg accuracy is = "+str(avg_acc))

	print("final_cm before = "+str(final_cm))

	final_cm[0][0] = ceil(final_cm[0][0]/no_of_folds)
	final_cm[0][1] = ceil(final_cm[0][1]/no_of_folds)
	final_cm[1][0] = ceil(final_cm[1][0]/no_of_folds)
	final_cm[1][1] = ceil(final_cm[1][1]/no_of_folds)

	print("final_cm after = "+str(final_cm))
main()
