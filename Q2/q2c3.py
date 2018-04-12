import sys
import numpy as np
import random
import math
import csv
from visualization import plot_decision_boundary
from random import shuffle

inputfile = "MNIST_train.csv"
csvfile = open(inputfile, 'r')
data = list(csv.reader(csvfile))

shuffle(data)
shuffle(data)

m = len(data)
n = len(data[0])-1
print("mtrain={}".format(m))

trainx = []
trainy = []
for i in range(m):
	for j in range(n+1):
		data[i][j] = float(data[i][j])

# X = []
# Y = []
for i in range(m):
	image = data[i]
	trainy.append(image[n])
	trainx.append(image[:n])
	# X.append(trainx[i])

	if(trainy[i] == 6):
		trainy[i] = [1, 0]
		# Y.append(0)
	else:
		trainy[i] = [0, 1]
		# Y.append(1)

# X = np.array(X)
# Y = np.array(Y)
# print(trainy)

# number of inputs
# n = 2
batchsize = 10
layers = [100]

layers.append(2)
numlayers = len(layers)

th = []
thd = []
O = []

m = 100

for i in range(numlayers):
	l = -1
	if(i == 0):
		l = n
	else:
		l = layers[i-1]

	th.append([[random.uniform(-0.1, 0.1) for k in range(l)]
            for j in range(layers[i])])
	thd.append([[0 for k in range(l)] for j in range(layers[i])])
	O.append([[0 for u in range(m)] for j in range(layers[i])])


def sigmoid(x,t):
	# print(x)
	ret=0
	if(t==0):
		ret = 1 / (1 + np.exp(-x))
	else:
		ret = max(0,x)

	# print(ret)
	return ret


alpha = 0
eta = 0.0001
print(eta)
t = 0
Jold = 0
epsilon = 0.001
b=0
while True:

	delta = [[[0 for k in range(m)] for j in range(layers[i])]
          for i in range(numlayers)]

	for a in range(0, m, batchsize):
		t += 1
		# eta = 0.01/math.sqrt(t)

		b = min(m, a+batchsize)
		print((a, b))
		for k in range(a, b):
			for i in range(numlayers):
				inpt = []
				if(i == 0):
					inpt = trainx[k]
					for j in range(layers[i]):
						# temp = 0
						# for u in range(n):
						# 	temp += (inpt[u] * th[i][j][u])
						temp = np.dot(inpt, th[i][j])
						O[i][j][k] = sigmoid(temp,1)
				else:
					inpt = O[i-1]
					for j in range(layers[i]):
						temp = 0
						thij=th[i][j]
						for u in range(layers[i-1]):
							temp += (inpt[u][k]*thij[u])
						if(i==numlayers-1):
							O[i][j][k] = sigmoid(temp,0)
						else:
							O[i][j][k] = sigmoid(temp, 1)


		# print(O)

		for i in range(numlayers-1, -1, -1):
			for j in range(layers[i]):

				if(i == 0):
					net=[0 for z in range(m)]
					for k in range(a, b):
						for u1 in range(n):
							net[k]+=trainx[k][u1]*th[i][j][u1]
						# net[k]=np.dot(trainx[k],th[i][j])	

					for u1 in range(n):
						gr = 0
						for k in range(a, b):
							delt = 0
							for u in range(layers[i+1]):
								subg=0
								if(net[k]>0):
									subg = 1
								elif(net[k]<0):
									subg=0	
								else:	
									subg = 0.5
								delt += delta[i+1][u][k]*th[i+1][u][j]*subg
								gr -= delta[i+1][u][k]*th[i+1][u][j] * subg*trainx[k][u1]

							delta[i][j][k] = delt

						# print(gr)

						th[i][j][u1] -= eta*gr

				else:

					net = [0 for z in range(m)]
					for k in range(a, b):
						for u1 in range(layers[i-1]):
							net[k] += O[i-1][u1][k]*th[i][j][u1]

					for u1 in range(layers[i-1]):
						delt = 0
						gr = 0
						if(i == numlayers-1):
							for k in range(a, b):
								delt = 0

								subg = O[i][j][k]*(1-O[i][j][k])
								delt += (trainy[k][j]-O[i][j][k])*subg
								gr -= (trainy[k][j]-O[i][j][k])*subg*O[i-1][u1][k]
								delta[i][j][k] = delt
							# print(gr)
						else:
							for k in range(a, b):
								delt = 0
								for u in range(layers[i+1]):
									subg = 0
									if(net[k] > 0):
										subg = 1
									elif(net[k] < 0):
										subg = 0
									else:
										subg=0.5	
									delt += delta[i+1][u][k]*th[i+1][u][j]*subg
									gr -= delta[i+1][u][k]*th[i+1][u][j] * 	subg*O[i-1][u1][k]
								delta[i][j][k] = delt


						th[i][j][u1] -= eta*gr
						# print(gr)

	# batch only till here
	# print(th)

	for k in range(m):
		for i in range(numlayers):
			inpt = []
			if(i == 0):
				inpt = trainx[k]
				for j in range(layers[i]):
					temp = 0
					for u in range(n):
						temp += (inpt[u] * th[i][j][u])
					O[i][j][k] = sigmoid(temp,1)
			else:
				inpt = O[i-1]
				for j in range(layers[i]):
					temp = 0
					for u in range(layers[i-1]):
						temp += (inpt[u][k]*th[i][j][u])
						if(i == numlayers-1):
							O[i][j][k] = sigmoid(temp, 0)
						else:
							O[i][j][k] = sigmoid(temp, 1)

	J = 0
	for k in range(m):
		for i in range(layers[numlayers-1]):
			temp = (O[numlayers-1][i][k]-trainy[k][i])
			J += temp*temp
	J /= (2*m)

	print(t)
	print(J)
	# print(th)

	# if(t > 30 or abs(Jold-J) < epsilon):
	if(abs(Jold-J) < epsilon):
		b+=1
	else:
		b=0	
	if(b==2):
		break

	Jold = J


for k in range(m):
	for i in range(numlayers):
		inpt = []
		if(i == 0):
			inpt = trainx[k]
			for j in range(layers[i]):
				temp = 0
				for u in range(n):
					temp += (inpt[u] * th[i][j][u])
				O[i][j][k] = sigmoid(temp,1)
		else:
			inpt = O[i-1]
			for j in range(layers[i]):
				temp = 0
				for u in range(layers[i-1]):
					temp += (inpt[u][k]*th[i][j][u])
				# O[i][j][k] = sigmoid(temp)
				if(i == numlayers-1):
					O[i][j][k] = sigmoid(temp, 0)
				else:
					O[i][j][k] = sigmoid(temp, 1)

correct = 0
for k in range(m):
	ans = -1
	ma = -1
	for i in range(layers[numlayers-1]):
		temp = O[numlayers-1][i][k]
		if(temp > ma):
			ma = temp
			ans = i
	# print(ans)
	if(trainy[k][ans] == 1):
		correct += 1

print("Accuracy on train data={}%".format(correct/m*100))
print((correct, m))


inputfile = "MNIST_test.csv"
csvfile = open(inputfile, 'r')
data = list(csv.reader(csvfile))
shuffle(data)
shuffle(data)

mtest = len(data)
mtest = 100

n = len(data[0])-1
testx = []
testy = []
for i in range(mtest):
	for j in range(n+1):
		data[i][j] = int(data[i][j])

# X2 = []
# Y2 = []
for i in range(mtest):
	image = data[i]
	testy.append(image[n])
	testx.append(image[:n])
	# X2.append(testx[i])
	# Y2.append(testy[i])

	if(testy[i] == 6):
		testy[i] = [1, 0]
	else:
		testy[i] = [0, 1]

# X2 = np.array(X2)
# Y2 = np.array(Y2)


O = [[[0 for u in range(mtest)] for j in range(layers[i])]
     for i in range(numlayers)]

for k in range(mtest):
	for i in range(numlayers):
		inpt = []
		if(i == 0):
			inpt = testx[k]
			for j in range(layers[i]):
				temp = 0
				for u in range(n):
					temp += (inpt[u] * th[i][j][u])
				O[i][j][k] = sigmoid(temp,1)
		else:
			inpt = O[i-1]
			for j in range(layers[i]):
				temp = 0
				for u in range(layers[i-1]):
					temp += (inpt[u][k]*th[i][j][u])
				# O[i][j][k] = sigmoid(temp)
				if(i == numlayers-1):
					O[i][j][k] = sigmoid(temp, 0)
				else:
					O[i][j][k] = sigmoid(temp, 1)


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


