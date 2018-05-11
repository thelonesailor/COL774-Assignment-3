from sklearn.tree import DecisionTreeClassifier
from read_data import preprocess
import math
import sys

train_data = preprocess("./dtree_data/train.csv")
valid_data = preprocess("./dtree_data/valid.csv")
test_data = preprocess("./dtree_data/test.csv")


vals = [1e-6,1e-5,2e-5, 5e-5, 1e-4, 3e-4, 8e-4, 1e-3, 5e-3, 9e-3, 2e-2, 6e-2, 1e-1, 7e-1]
for frac in vals:
	dtree = DecisionTreeClassifier(criterion='entropy', min_samples_split=frac)
	dtree.fit(train_data[:, 1:], train_data[:, 0])
	train_acc = dtree.score(train_data[:, 1:], train_data[:, 0])
	valid_acc = dtree.score(valid_data[:, 1:], valid_data[:, 0])
	test_acc = dtree.score(test_data[:, 1:], test_data[:, 0])

	print(frac, train_acc, valid_acc, test_acc)
# valid_acc increases	then decreases

print("---------------------------")
vals = [1e-6, 1e-5, 2e-5, 5e-5, 1e-4, 3e-4, 8e-4, 1e-3, 5e-3, 9e-3, 2e-2, 6e-2, 1e-1]
for frac in vals:
	dtree = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=frac)
	dtree.fit(train_data[:, 1:], train_data[:, 0])
	train_acc = dtree.score(train_data[:, 1:], train_data[:, 0])
	valid_acc = dtree.score(valid_data[:, 1:], valid_data[:, 0])
	test_acc = dtree.score(test_data[:, 1:], test_data[:, 0])

	print(frac, train_acc, valid_acc, test_acc)
# valid_acc increases	then decreases

print("---------------------------")

vals=[1,3,5,8,10,13,15,20,25,30,35,40,45,50,70]
for frac in vals:
	dtree = DecisionTreeClassifier(criterion='entropy', max_depth=frac)
	dtree.fit(train_data[:, 1:], train_data[:, 0])
	train_acc = dtree.score(train_data[:, 1:], train_data[:, 0])
	valid_acc = dtree.score(valid_data[:, 1:], valid_data[:, 0])
	test_acc = dtree.score(test_data[:, 1:], test_data[:, 0])

	print(frac, train_acc, valid_acc, test_acc)


print("---------------------------")
v1 = [1e-10,1e-9,1e-8,2e-8,1e-7,1e-6,5e-6,1e-5]
v2 = [1e-4, 3e-4, 8e-4,9e-4, 1e-3,1.1e-3,1.2e-3,2e-3, 5e-3]
v3 = [7,8,9,10,11,12,13,14,15,17,19,20,21]

best=(0,0,0)
A=(0,0,0)
mv=-1
for f1 in v1:
	for f2 in v2:
		for f3 in v3:
			dtree = DecisionTreeClassifier(criterion='entropy', max_depth=f3, min_samples_leaf=f2, min_samples_split=f1)
			dtree.fit(train_data[:, 1:], train_data[:, 0])
			train_acc = dtree.score(train_data[:, 1:], train_data[:, 0])
			valid_acc = dtree.score(valid_data[:, 1:], valid_data[:, 0])
			test_acc = dtree.score(test_data[:, 1:], test_data[:, 0])

			if(valid_acc>mv):
				mv=valid_acc
				A = (train_acc, valid_acc, test_acc)
				best = (f1,f2,f3)
			print((mv,f1,f2,f3))
			print(best)

print(A)
print(mv)
print(best)
