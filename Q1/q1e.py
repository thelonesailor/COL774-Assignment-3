from sklearn.ensemble import RandomForestClassifier
from read_data import preprocess


train_data = preprocess("./dtree_data/train.csv")
valid_data = preprocess("./dtree_data/valid.csv")
test_data = preprocess("./dtree_data/test.csv")


for frac in range(2, 30, 1):
	forest = RandomForestClassifier(
		criterion='entropy',  n_estimators=frac, bootstrap=False)
	forest.fit(train_data[:, 1:], train_data[:, 0])
	train_acc = forest.score(train_data[:, 1:], train_data[:, 0])
	valid_acc = forest.score(valid_data[:, 1:], valid_data[:, 0])
	test_acc = forest.score(test_data[:, 1:], test_data[:, 0])

	print(frac,train_acc, valid_acc, test_acc)

print ("----------------------------------------")
for frac in range(1, 14):
	forest = RandomForestClassifier(
		criterion='entropy', max_features=frac, bootstrap=False)
	forest.fit(train_data[:, 1:], train_data[:, 0])
	train_acc = forest.score(train_data[:, 1:], train_data[:, 0])
	valid_acc = forest.score(valid_data[:, 1:], valid_data[:, 0])
	test_acc = forest.score(test_data[:, 1:], test_data[:, 0])

	print(frac, train_acc, valid_acc, test_acc)

print ("----------------------------------------")
for frac in range(2, 30, 1):
	forest = RandomForestClassifier(
		criterion='entropy',  n_estimators=frac, bootstrap=True)
	forest.fit(train_data[:, 1:], train_data[:, 0])
	train_acc = forest.score(train_data[:, 1:], train_data[:, 0])
	valid_acc = forest.score(valid_data[:, 1:], valid_data[:, 0])
	test_acc = forest.score(test_data[:, 1:], test_data[:, 0])

	print(frac,train_acc, valid_acc, test_acc)

print ("----------------------------------------")
for frac in range(1, 15):
	forest = RandomForestClassifier(
		criterion='entropy', n_estimators=6, max_features=1, bootstrap=True)
	forest.fit(train_data[:, 1:], train_data[:, 0])
	train_acc = forest.score(train_data[:, 1:], train_data[:, 0])
	valid_acc = forest.score(valid_data[:, 1:], valid_data[:, 0])
	test_acc = forest.score(test_data[:, 1:], test_data[:, 0])

	print(frac,train_acc, valid_acc, test_acc)

# max_acc = 0
# max_i, max_j, max_k = -1, -1, False
# for i in range(1, 14):
# 	for j in range(2, 30, 2):
# 		for k in [True, False]:
# 			forest = RandomForestClassifier(criterion='entropy', max_features=i, n_estimators=j, bootstrap=k)
# 			forest.fit(train_data[:, 1:], train_data[:, 0])
# 			valid_acc = forest.score(valid_data[:, 1:], valid_data[:, 0])
# 			if valid_acc > max_acc:
# 				max_acc = valid_acc
# 				max_i = i
# 				max_j = j
# 				max_k = k

# 		print (max_acc, max_i, max_j, max_k)
