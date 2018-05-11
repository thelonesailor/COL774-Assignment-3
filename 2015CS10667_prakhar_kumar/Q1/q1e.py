from sklearn.ensemble import RandomForestClassifier
from read_data import preprocess


train_data = preprocess("./dtree_data/train.csv")
valid_data = preprocess("./dtree_data/valid.csv")
test_data = preprocess("./dtree_data/test.csv")


# for f in range(2, 30, 1):
# 	forest = RandomForestClassifier(
# 		criterion='entropy',  n_estimators=f, bootstrap=False)
# 	forest.fit(train_data[:, 1:], train_data[:, 0])
# 	train_acc = forest.score(train_data[:, 1:], train_data[:, 0])
# 	valid_acc = forest.score(valid_data[:, 1:], valid_data[:, 0])
# 	test_acc = forest.score(test_data[:, 1:], test_data[:, 0])

# 	print(f,train_acc, valid_acc, test_acc)

# print ("----------------------------------------")
# for f in range(1, 14):
# 	forest = RandomForestClassifier(
# 		criterion='entropy', max_features=f, bootstrap=False)
# 	forest.fit(train_data[:, 1:], train_data[:, 0])
# 	train_acc = forest.score(train_data[:, 1:], train_data[:, 0])
# 	valid_acc = forest.score(valid_data[:, 1:], valid_data[:, 0])
# 	test_acc = forest.score(test_data[:, 1:], test_data[:, 0])

# 	print(f, train_acc, valid_acc, test_acc)

# print ("----------------------------------------")
# for f in range(2, 30, 1):
# 	forest = RandomForestClassifier(
# 		criterion='entropy',  n_estimators=f, bootstrap=True)
# 	forest.fit(train_data[:, 1:], train_data[:, 0])
# 	train_acc = forest.score(train_data[:, 1:], train_data[:, 0])
# 	valid_acc = forest.score(valid_data[:, 1:], valid_data[:, 0])
# 	test_acc = forest.score(test_data[:, 1:], test_data[:, 0])

# 	print(f,train_acc, valid_acc, test_acc)

# print ("----------------------------------------")
# for f in range(1, 15):
# 	forest = RandomForestClassifier(
# 		criterion='entropy', n_estimators=6, max_features=1, bootstrap=True)
# 	forest.fit(train_data[:, 1:], train_data[:, 0])
# 	train_acc = forest.score(train_data[:, 1:], train_data[:, 0])
# 	valid_acc = forest.score(valid_data[:, 1:], valid_data[:, 0])
# 	test_acc = forest.score(test_data[:, 1:], test_data[:, 0])

# 	print(f,train_acc, valid_acc, test_acc)

max_acc = (0,0,0)
max_1, max_2, max_3 = -1, -1, False
for f1 in range(1, 15):
	for f2 in range(4, 35, 1):
		for f3 in [True, False]:
			forest = RandomForestClassifier(criterion='entropy', max_features=f1, n_estimators=f2, bootstrap=f3)
			forest.fit(train_data[:, 1:], train_data[:, 0])
			train_acc = forest.score(train_data[:, 1:], train_data[:, 0])
			valid_acc = forest.score(valid_data[:, 1:], valid_data[:, 0])
			test_acc = forest.score(test_data[:, 1:], test_data[:, 0])

			a,b,c=max_acc
			if valid_acc > b:
				max_acc = (train_acc, valid_acc, test_acc)
				max_1 = f1
				max_2 = f2
				max_3 = f3

		print (max_acc, max_1, max_2, max_3)


print (max_acc, max_1, max_2, max_3)
