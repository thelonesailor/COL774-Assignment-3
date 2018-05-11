import csv
import sys
import numpy as np
from random import shuffle
from svmutil import svm_train, svm_predict, svm_save_model


inputfile = "MNIST_train.csv"
csvfile = open(inputfile, 'r')
data = list(csv.reader(csvfile))

m = len(data)
n = len(data[0])-1
xtrain = []
ytrain = []
for i in range(m):
	for j in range(n+1):
		data[i][j] = int(data[i][j])

for i in range(m):
	image = data[i]
	ytrain.append(image[n])
	xtrain.append(image[:n])


inputfile = "MNIST_test.csv"
csvfile = open(inputfile, 'r')
data = list(csv.reader(csvfile))

mtest = len(data)
n = len(data[0])-1
xtest = []
ytest = []
for i in range(mtest):
	for j in range(n+1):
		data[i][j] = int(data[i][j])

for i in range(mtest):
	image = data[i]
	ytest.append(image[n])
	xtest.append(image[:n])

print("Linear")
mlinear = svm_train(ytrain, xtrain, '-s 0 -t 0 -c 1 -q')
print("trained")

p_label, p_acc, p_val = svm_predict(ytrain, xtrain, mlinear)
print("Train data")
print(p_acc)

p_label, p_acc, p_val = svm_predict(ytest, xtest, mlinear)
print("Test data")
print(p_acc)
