import matplotlib.pyplot as plt
import numpy as np
from read_data import preprocess
import math,sys
import resource

train_data = preprocess("./dtree_data/train.csv")
valid_data = preprocess("./dtree_data/valid.csv")
test_data = preprocess("./dtree_data/test.csv")

print(train_data)

n=len(train_data[0])
m=len(train_data)
print("{} in train data".format(m))
error=0

num_attr=n-1
values=[[] for i in range(num_attr+1)]
for d in train_data:
	if(d[0]==0):
		error+=1
	for i in range(1, num_attr+1):
		v=d[i]
		if(v not in values[i]):
			values[i].append(v)

print("error ={}".format(error))

for i in range(1, num_attr+1):
	values[i]=sorted(values[i])
print(values)

def entropy(data):
	c={}
	for d in data:
		y=d[0]
		if(y not in c):
			c[y]=0
		c[y]+=1

	entropy=0
	for y in c.keys():
		temp=c[y]/len(data)
		temp*= math.log2(temp)
		
		entropy-=temp
	return entropy 

def findmajorityclass(data):
	c = {}
	for d in data:
		y = d[0]
		if(y not in c):
			c[y] = 0
		c[y] += 1

	my=0
	mc=-1
	for y in c.keys():
		if(c[y]>mc):
			mc=c[y]
			my=y

	return my

def split(D,a,t):

	if(t==1):	
		sD=[[] for i in range(len(values[a]))]
		for d in D:
			d[a]
			for j in range(len(values[a])):
				if(d[a]==values[a][j]):
					sD[j].append(d)
					break

		return sD

heightnum={}
# train_data
last=n-1
def grow_tree(D,lvl,done):
	print("{} {}".format(lvl,len(D)))

	if(len(D)==0):
		print("returning")
		return node(isleaf=True,ans=-1,height=0)

	count=[0 for i in range(2)]

	for d in D:
		count[d[0]]+=1


	if(count[0]==len(D)):
		return node(ans=0,height=0,isleaf=True)
	elif(count[1] == len(D)):
		return node(ans=1,height=0,isleaf=True)


	e=entropy(D)
	mclass=findmajorityclass(D)

	maxgain=-100000000000
	childdata=[]
	battr=-1
	for attr in range(1,num_attr+1):
		if(attr in done):
			continue

		sD=split(D,attr,1)

		gain=0
		for a in sD:
			gain+=len(a)*entropy(a)
		gain/=len(D)

		gain=e-gain
		if(gain>maxgain):
			maxgain=gain
			childdata=sD
			battr=attr

	if(maxgain>0):
		childs=[]
		done=(list(done))
		done.append(battr)
		for i in range(len(childdata)):
			# print(len())
			childs.append(grow_tree(childdata[i],lvl+1,done))

		ht=1
		ht=max(c.height for c in childs)+1

		if ht not in heightnum:
			heightnum[ht]=1
		else:
			heightnum[ht] += 1

		print("returning")
		return node(col=battr,children=childs,height=ht,ans=mclass,isleaf=False)	
	else:
		ht=0
		if ht not in heightnum:
			heightnum[ht] = 1
		else:
			heightnum[ht] += 1
		
		print("returning")
		return node(isleaf=True, height=ht, ans=mclass)

		



class node:

	
	def __init__(self, children=None, col=-1, value=None, classes=None, ans=-1, isleaf=False, height=0,  numchild=0):
		#  passed1=0, correct1=0, tmajor1=0, passed2=0, correct2=0, tmajor2=0, passed3=0, correct3=0, tmajor3=0):
		self.children = children # Child
		self.col = col 		# The attribute about which I will split
		# self.value = value  # The value about which data will be splitted
		# self.classes = classes  # None for every column except the last one <- Changed
		self.ans = ans 			# The class i m going to assign an observations if I stop at this node
		self.isleaf = isleaf
		self.height = height

		self.numchild = numchild  		

		self.passed=[0,0,0,0]
		self.correct = [0, 0, 0, 0]
		self.tmajor = [0, 0, 0, 0]


def accuracy(nodex, level, height, entry):
	if nodex.isleaf or level == height:
		return entry[0] == nodex.ans
	else:
		child = None
		for i in range(len(values[nodex.col])):
			if(entry[nodex.col] == values[nodex.col][i]):
				child=nodex.children[i]
				break


		return accuracy(child, level+1, height, entry)


def accux(nodex, data, height):
	correct = 0
	for entry in data:
		if accuracy(nodex, 0, height, entry):
			correct += 1
	return float(correct)/len(data),correct


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

decision_tree=grow_tree(train_data,0,[])
print("decision tree made")

a={}
levelnums(decision_tree, 0)
maxh=max(i for i in a)
print("max height={}".format(maxh))
for i in range(1,len(a)):
	a[i]+=a[i-1]

correct=0
b = {}
for i in range(0, maxh+1):
	b[i],correct = accux(decision_tree, train_data, i)

c = {}
for i in range(0, maxh+1):
	c[i], correct = accux(decision_tree, test_data, i)

d = {}
for i in range(0, maxh+1):
	d[i], correct = accux(decision_tree, valid_data, i)
vcorrect=correct

# print(a)
# print(b)


plt.ioff()
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)


ax1.clear()
ax1.plot(np.array([v for v in a.values()]), np.array([v for v in d.values()]), label="Valid Data")
ax1.plot(np.array([v for v in a.values()]), np.array([v for v in c.values()]), label="Test Data")
ax1.plot(np.array([v for v in a.values()]), np.array([v for v in b.values()]), label="Train Data")

plt.xlabel('Number of Nodes')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Q1 (a)')
plt.show()




# Pruning

def subtree(nodex):
	if(nodex.isleaf):
		nodex.numchild=0
	else:
		temp=0
		for a in nodex.children:
			subtree(a)
			temp+=a.numchild+1
		nodex.numchild=temp
	return


def classify(entry, nodex, level, p):
	if(nodex==None):return False

	if nodex.isleaf:
		nodex.passed[p] += 1

		if entry[0] == nodex.ans:
			nodex.correct[p] += 1
			# nodex.tmajor[p] += 1
			return True
		return False
	else:
		if entry[0] == nodex.ans:
			nodex.tmajor[p] += 1
		child = None

		attr=nodex.col
		for i in range(len(values[attr])):
			if(entry[attr] == values[attr][i]):
				child=nodex.children[i]
				break

		if(child==nodex):return False

		temp = classify(entry, child, level+1, p)
		nodex.passed[p] += 1
		if temp==True:
			nodex.correct[p] += 1
			return True

		return False


def init(nodex, p):
	nodex.correct[p]=0
	nodex.tmajor[p] = 0
	nodex.passed[p] = 0
	if nodex.isleaf:
		return
	for a in nodex.children:
		init(a,p)

init(decision_tree,1)
init(decision_tree, 2)
init(decision_tree, 3)

for d in valid_data:
	classify(d,decision_tree,0,1)
for d in test_data:
	classify(d, decision_tree, 0, 2)
for d in train_data:
	classify(d, decision_tree, 0, 3)


def update_correct(nodex,p):
	if nodex.isleaf:
		return nodex.correct[p]
	else:
		temp=0
		for a in nodex.children:
			temp += update_correct(a,p)
		nodex.correct[p] = temp
		return nodex.correct[p]


def update_numberofchildren(nodex):
	if nodex.isleaf:
		nodex.numchild = 0
	else:
		temp=0
		for a in nodex.children:
			update_numberofchildren(a)
			temp += a.numchild+1
		nodex.numchild = temp
	return


def numberofnodes(nodex):
	if nodex.isleaf:
		return 1
	else:
		temp=1
		for a in nodex.children:
			temp += numberofnodes(a)
		return temp

bestnode=None
cc=vcorrect
print(vcorrect)
better=False
def prune(nodex):
	global bestnode,cc,better
	
	if nodex.isleaf:
		return
	else:
		tempcorrect = cc - nodex.correct[1] + nodex.tmajor[1]
		if tempcorrect > cc:
			bestnode = nodex
			cc = tempcorrect
			better = True
		for a in nodex.children: 
			prune(a)
	
		return


def accuracy2(nodex, entry):
	if nodex.isleaf:
		return entry[0] == nodex.ans
	else:
		child = None
		attr = nodex.col
		for i in range(len(values[attr])):
			if(entry[attr] == values[attr]):
				child = nodex.children[i]
				break


		return accuracy2(child, entry)


def accux2(nodex, data):
	correct = 0
	for entry in data:
		if accuracy2(nodex, entry):
			correct += 1

	return float(correct)/len(data)


lvd = len(valid_data)
print(len(valid_data))
lte = len(test_data)
print(len(test_data))
ltr = len(train_data)
print(len(train_data))

nodes=[]
valid=[]
test=[]
train=[]

print("Before pruning:")
print("validation accuracy={}%".format(decision_tree.correct[1]/lvd*100))
print("test accuracy={}%".format(decision_tree.correct[2]/lte*100))
print("train accuracy={}%".format(decision_tree.correct[3]/ltr*100))

initial = numberofnodes(decision_tree)
pruned=0
while True:
	better = False
	prune(decision_tree)
	if not better:
		break
	else:
		bestnode.isleaf = True
		bestnode.children = []
		bestnode.correct[1] = bestnode.tmajor[1]
		bestnode.correct[2] = bestnode.tmajor[2]
		bestnode.correct[3] = bestnode.tmajor[3]
		y1 = update_correct(decision_tree,1)
		y2 = update_correct(decision_tree,2)
		y3 = update_correct(decision_tree,3)

		n = numberofnodes(decision_tree)
		
		nodes.append(n)
		valid.append(y1/lvd)
		test.append(y2/lte)
		train.append(y3/ltr)


		pruned += bestnode.numchild
		# print("No. of nodes pruned ", pruned)
		# print("No. of nodes in tree ", n)
		update_numberofchildren(decision_tree)


print("After pruning:")
print("validation accuracy={}%".format(decision_tree.correct[1]/lvd*100))
print("test accuracy={}%".format(decision_tree.correct[2]/lte*100))
print("train accuracy={}%".format(decision_tree.correct[3]/ltr*100))

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

ax1.clear()
ax1.plot(nodes, valid, label="Valid Data")
ax1.plot(nodes, test, label="Test Data")
ax1.plot(nodes, train, label="Train Data")

plt.xlabel('Number of Nodes')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Q1 (b)')
plt.show()
