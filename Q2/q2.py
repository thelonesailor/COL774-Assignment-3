import sys
import numpy as np
import random
import math
import csv
from visualization import plot_decision_boundary 


csvfile = open("toy_trainX.csv", 'r')
trainx = list(csv.reader(csvfile))
csvfile = open("toy_trainY.csv", 'r')
trainy = list(csv.reader(csvfile))


assert len(trainx) == len(trainy)
m=len(trainx)

X=[]
Y=[]
for i in range(m):
	trainx[i] = np.array([float(trainx[i][0]), float(trainx[i][1])])
	X.append(trainx[i])

for i in range(m):
	trainy[i] = int(trainy[i][0])
	Y.append(trainy[i])

	if(trainy[i]==0):
		trainy[i]=[1,0]
	else:
		trainy[i] = [0, 1]

X=np.array(X)
Y=np.array(Y)


# number of inputs 
n=2
batchsize=m
layers=[5]

layers.append(2)
numlayers=len(layers)

th=[]
thd = []
O=[]

for i in range(numlayers):
	l=-1
	if(i==0):
		l=n
	else:
		l=layers[i-1]

	th.append([[random.uniform(-0.2, 0.2) for k in range(l)] for j in range(layers[i])])
	thd.append([[0 for k in range(l)] for j in range(layers[i])])
	O.append([[0 for u in range(m)] for j in range(layers[i])])
	# grad.append([np.array([0 for u in range(m)]) for j in range(layers[i])])

def sigmoid(x):
	ret= 1 / (1 + math.exp(-x))
	# print(ret)
	return ret

alpha=0
eta=0.07
t=0
Jold=0
epsilon=0.000001
while True:
	t+=1
	for a in range(0,m,batchsize):
		b=min(m,a+batchsize)
		for k in range(a,b):
			for i in range(numlayers):
				inpt = []
				if(i == 0):
					inpt = trainx[k]
					for j in range(layers[i]):
						temp = 0
						for u in range(n):
							temp += (inpt[u]* th[i][j][u])
						O[i][j][k] = sigmoid(temp)
				else:
					inpt = O[i-1]
					for j in range(layers[i]):
						temp=0
						for u in range(layers[i-1]):
							temp+=(inpt[u][k]*th[i][j][u])
						O[i][j][k]=sigmoid(temp)	

		
		# print(O)

		delta = [[[0 for k in range(m)] for j in range(layers[i])] for i in range(numlayers)]
		grad = [[] for i in range(numlayers)]

		for i in range(numlayers-1,-1,-1):
			for j in range(layers[i]):

				if(i==0):
					for u1 in range(n):
						gr=0
						for k in range(a, b):
							delt=0
							for u in range(layers[i+1]):
								delt += delta[i+1][u][k]*th[i+1][u][j]*O[i][j][k]*(1-O[i][j][k])
								gr -= delta[i+1][u][k]*th[i+1][u][j]*O[i][j][k]*(1-O[i][j][k])*trainx[k][u1]							

							delta[i][j][k]=delt			
		
						thd[i][j][u1]=eta*gr+alpha*thd[i][j][u1]
						th[i][j][u1] -= thd[i][j][u1]

				else:	
					for u1 in range(layers[i-1]):
						delt=0
						gr=0
						if(i==numlayers-1):
							for k in range(a, b):
								delt=0
								delt += (trainy[k][j]-O[i][j][k])*O[i][j][k]*(1-O[i][j][k])
								gr -= (trainy[k][j]-O[i][j][k])*O[i][j][k]*(1-O[i][j][k])*O[i-1][u1][k]
								delta[i][j][k]=delt
						else:
							for k in range(a, b):
								delt=0
								for u in range(layers[i+1]):
									delt += delta[i+1][u][k]*th[i+1][u][j]*O[i][j][k]*(1-O[i][j][k])
									gr -= delta[i+1][u][k]*th[i+1][u][j]*O[i][j][k]*(1-O[i][j][k])*O[i-1][u1][k]
								delta[i][j][k] = delt

						thd[i][j][u1] = eta*gr+alpha*thd[i][j][u1]
						th[i][j][u1] -= thd[i][j][u1]
						# print(gr)


	# batch only till here

	for k in range(m):
		for i in range(numlayers):
			inpt = []
			if(i == 0):
				inpt = trainx[k]
				for j in range(layers[i]):
					temp = 0
					for u in range(n):
						temp += (inpt[u] * th[i][j][u])
					O[i][j][k] = sigmoid(temp)
			else:
				inpt = O[i-1]
				for j in range(layers[i]):
					temp = 0
					for u in range(layers[i-1]):
						temp += (inpt[u][k]*th[i][j][u])
					O[i][j][k] = sigmoid(temp)


	J=0
	for k in range(m):
		for i in range(layers[numlayers-1]):
			temp = (O[numlayers-1][i][k]-trainy[k][i])
			J+=temp*temp
	J/=(2*m)

	print(t)
	print(J)
	if(t>1000 and abs(Jold-J)<epsilon):
		break		

	Jold=J


for k in range(m):
	for i in range(numlayers):
		inpt = []
		if(i == 0):
			inpt = trainx[k]
			for j in range(layers[i]):
				temp = 0
				for u in range(n):
					temp += (inpt[u] * th[i][j][u])
				O[i][j][k] = sigmoid(temp)
		else:
			inpt = O[i-1]
			for j in range(layers[i]):
				temp = 0
				for u in range(layers[i-1]):
					temp += (inpt[u][k]*th[i][j][u])
				O[i][j][k] = sigmoid(temp)

correct = 0
for k in range(m):
	ans=-1
	ma=-1
	for i in range(layers[numlayers-1]):
		temp = O[numlayers-1][i][k]
		if(temp>ma):
			ma=temp
			ans=i

	if(trainy[k][ans]==1):
		correct+=1

print("Accuracy on train data={}%".format(correct/m*100))
print((correct,m))				

csvfile = open("toy_testX.csv", 'r')
testx = list(csv.reader(csvfile))
csvfile = open("toy_testY.csv", 'r')
testy = list(csv.reader(csvfile))

assert len(testx) == len(testy)
mtest = len(testx)
X2=[]
Y2=[]
for i in range(mtest):
	testx[i] = np.array([float(testx[i][0]), float(testx[i][1])])
	X2.append(testx[i])

for i in range(mtest):
	testy[i] = int(testy[i][0])
	Y2.append(testy[i])

	if(testy[i] == 0):
		testy[i] = [1, 0]
	else:
		testy[i] = [0, 1]

X2 = np.array(X2)
Y2 = np.array(Y2)
O=[[[0 for u in range(mtest)] for j in range(layers[i])] for i in range(numlayers)]

for k in range(mtest):
	for i in range(numlayers):
		inpt = []
		if(i == 0):
			inpt = testx[k]
			for j in range(layers[i]):
				temp = 0
				for u in range(n):
					temp += (inpt[u] * th[i][j][u])
				O[i][j][k] = sigmoid(temp)
		else:
			inpt = O[i-1]
			for j in range(layers[i]):
				temp = 0
				for u in range(layers[i-1]):
					temp += (inpt[u][k]*th[i][j][u])
				O[i][j][k] = sigmoid(temp)

correct = 0
for k in range(mtest):
	ans = -1
	ma = -1
	for i in range(layers[numlayers-1]):
		temp = O[numlayers-1][i][k]
		if(temp > ma):
			ma = temp
			ans = i

	if(testy[k][ans] == 1):
		correct += 1

print("Accuracy on test data={}%".format(correct/mtest*100))
print((correct, mtest))

def f(xx):

	m2=len(xx)
	O=[[[0 for u in range(m2)] for j in range(layers[i])] for i in range(numlayers)]


	for k in range(m2):
		for i in range(numlayers):
			inpt = []
			if(i == 0):
				inpt = xx[k]
				for j in range(layers[i]):
					temp = 0
					for u in range(n):
						temp += (inpt[u] * th[i][j][u])
					O[i][j][k] = sigmoid(temp)
			else:
				inpt = O[i-1]
				for j in range(layers[i]):
					temp = 0
					for u in range(layers[i-1]):
						temp += (inpt[u][k]*th[i][j][u])
					O[i][j][k] = sigmoid(temp)


	res = []
	for k in range(m2):
		ans = -1
		ma = -1
		for i in range(layers[numlayers-1]):
			temp = O[numlayers-1][i][k]
			if(temp > ma):
				ma = temp
				ans = i

		res.append(ans)

	res=np.array(res)

	return res		


plot_decision_boundary(f,X,Y,'Train data '+str(layers[:numlayers-1]))
plot_decision_boundary(f,X2,Y2,'Test data '+str(layers[:numlayers-1]))


print("Logistic regression..")
from sklearn import linear_model
logistic=linear_model.LogisticRegression()
logistic.fit(X,Y)

Yp=logistic.predict(X)
ctrain=0
for i in range(m):
	if(Yp[i]==Y[i]):
		ctrain+=1

print("Train data accuracy={}".format(ctrain/m*100))

Yp2 = logistic.predict(X2)
ctest=0
for i in range(mtest):
	if(Yp2[i]==Y2[i]):
		ctest+=1
print("Test data accuracy={}".format(ctest/mtest*100))

plot_decision_boundary(logistic.predict,X,Y,'Train data '+str(layers[:numlayers-1]))
plot_decision_boundary(logistic.predict, X2, Y2, 'Test data '+str(layers[:numlayers-1]))
