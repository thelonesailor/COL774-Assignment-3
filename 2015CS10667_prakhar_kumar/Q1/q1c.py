import matplotlib.pyplot as plt
import numpy as np
from read_data import preprocess2
import math
import sys
import resource

numerical = [1, 3, 5, 11, 12, 13]
train_data = preprocess2("./dtree_data/train.csv")
valid_data = preprocess2("./dtree_data/valid.csv")
test_data = preprocess2("./dtree_data/test.csv")

print(train_data)

n = len(train_data[0])
m = len(train_data)
print("{} in train data".format(m))
error = 0

num_attr = n-1
values = [[] for i in range(num_attr+1)]
for d in train_data:
	if(d[0] == 0):
		error += 1
	for i in range(1, num_attr+1):
		v = d[i]
		if(i in numerical):
			continue
		if(v not in values[i]):
			values[i].append(v)

print("error ={}".format(error))

for i in range(1, num_attr+1):
	values[i] = sorted(values[i])
print(values)


def entropy(data):
	c = {}
	for d in data:
		y = d[0]
		if(y not in c):
			c[y] = 0
		c[y] += 1

	entropy = 0
	for y in c.keys():
		temp = c[y]/len(data)
		temp *= math.log2(temp)

		entropy -= temp
	return entropy


def findmajorityclass(data):
	c = {}
	for d in data:
		y = d[0]
		if(y not in c):
			c[y] = 0
		c[y] += 1

	my = 0
	mc = -1
	for y in c.keys():
		if(c[y] > mc):
			mc = c[y]
			my = y

	return my


def split(D, a):

	if(a not in numerical):
		sD = [[] for i in range(len(values[a]))]
		for d in D:
			d[a]
			for j in range(len(values[a])):
				if(d[a] == values[a][j]):
					sD[j].append(d)
					break
		return sD, -1
	else:
		sD = [[] for i in range(2)]
		
		v=[]
		for d in D:
			v.append(d[a])

		v=sorted(v)

		median=v[len(v)//2]	

		for d in D:	
			if(d[a] < median):
				sD[0].append(d)
			else:
				sD[1].append(d)

		return sD,median


heightnum = {}
# train_data
last = n-1


def grow_tree(D, lvl, done):
	print("{} {}".format(lvl, len(D)))

	if(len(D) == 0):
		print("returning")
		return node(isleaf=True, ans=-1, height=0)

	count = [0 for i in range(2)]

	for d in D:
		count[d[0]] += 1

	if(count[0] == len(D)):
		return node(ans=0, height=0, isleaf=True)
	elif(count[1] == len(D)):
		return node(ans=1, height=0, isleaf=True)

	e = entropy(D)
	mclass = findmajorityclass(D)

	maxgain = -100000000000
	childdata = []
	battr = -1
	med=-1
	for attr in range(1, num_attr+1):
		if(attr in done and attr not in numerical):
			continue

		sD,median = split(D, attr)

		gain = 0
		for a in sD:
			gain += len(a)*entropy(a)
		gain /= len(D)

		gain = e-gain
		if(gain > maxgain):
			maxgain = gain
			childdata = sD
			battr = attr
			med=median

	if(maxgain > 0):
		childs = []
		done = (list(done))
		done.append(battr)
		for i in range(len(childdata)):
			# print(len())
			childs.append(grow_tree(childdata[i], lvl+1, done))

		ht = 1
		ht = max(c.height for c in childs)+1

		if ht not in heightnum:
			heightnum[ht] = 1
		else:
			heightnum[ht] += 1

		print("returning")
		return node(col=battr, children=childs, height=ht, ans=mclass, isleaf=False,value=med)
	else:
		ht = 0
		if ht not in heightnum:
			heightnum[ht] = 1
		else:
			heightnum[ht] += 1

		print("returning")
		return node(isleaf=True, height=ht, ans=mclass)


class node:

	def __init__(self, children=None, col=-1, value=None, classes=None, ans=-1, isleaf=False, height=0,  numchild=0):
		self.children = children  
		self.col = col 		
		self.value = value  
		self.ans = ans 			
		self.isleaf = isleaf
		self.height = height

		self.numchild = numchild

		self.passed = [0, 0, 0, 0]
		self.correct = [0, 0, 0, 0]
		self.tmajor = [0, 0, 0, 0]


def accuracy(nodex, level, height, entry):
	if nodex.isleaf or level == height:
		return entry[0] == nodex.ans
	else:
		child = None
		if(nodex.col not in numerical):
			for i in range(len(values[nodex.col])):
				if(entry[nodex.col] == values[nodex.col][i]):
					child = nodex.children[i]
					break
		else:
			if(entry[nodex.col] < nodex.value):
				child = nodex.children[0]
			else:
				child = nodex.children[1]
		
		return accuracy(child, level+1, height, entry)


def accux(nodex, data, height):
	correct = 0
	for entry in data:
		if accuracy(nodex, 0, height, entry):
			correct += 1
	return float(correct)/len(data), correct


def levelnums(nodex, level):
	if level not in a:
		a[level] = 1
	else:
		a[level] += 1
	if nodex.isleaf:
		return
	else:
		for b in nodex.children:
			levelnums(b, level+1)
		return


# sys.setrecursionlimit=100000
resource.setrlimit(resource.RLIMIT_STACK, (2**29, -1))
sys.setrecursionlimit(10**6)

decision_tree = grow_tree(train_data, 0, [])
print("decision tree made")

a = {}
levelnums(decision_tree, 0)
maxh = max(i for i in a)
print("max height={}".format(maxh))
for i in range(1, len(a)):
	a[i] += a[i-1]

correct = 0
b = {}
for i in range(0, maxh+1):
	b[i], correct = accux(decision_tree, train_data, i)
trcorrect = correct

c = {}
for i in range(0, maxh+1):
	c[i], correct = accux(decision_tree, test_data, i)
tecorrect=correct

d = {}
for i in range(0, maxh+1):
	d[i], correct = accux(decision_tree, valid_data, i)
vcorrect = correct

lvd = len(valid_data)
print(len(valid_data))
lte = len(test_data)
print(len(test_data))
ltr = len(train_data)
print(len(train_data))

print("Without preprocessing:")
print("validation accuracy={}%".format(vcorrect/lvd*100))
print("test accuracy={}%".format(tecorrect/lte*100))
print("train accuracy={}%".format(trcorrect/ltr*100))


plt.ioff()
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)


ax1.clear()
ax1.plot(np.array([v for v in a.values()]), np.array(
	[v for v in d.values()]), label="Valid Data")
ax1.plot(np.array([v for v in a.values()]), np.array(
	[v for v in c.values()]), label="Test Data")
ax1.plot(np.array([v for v in a.values()]), np.array(
	[v for v in b.values()]), label="Train Data")

plt.xlabel('Number of Nodes')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Q1 (c)')
plt.show()


