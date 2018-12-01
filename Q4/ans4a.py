import numpy as np
# import matplotlib
import matplotlib.pyplot as plt

def load_data():
	train_X = []
	train_Y = []
	test_X = []
	test_Y = []
	with open("./../Q1/data/SpambaseFull/train.txt","r") as fp:
		for line in fp:
			line = line.split(",")
			train_X.append(line[:-1])
			train_Y.append(int(line[-1]))

	with open("./../Q1/data/SpambaseFull/test.txt","r") as fp:
		for line in fp:
			line = line.split(",")
			test_X.append(line[:-1])
			test_Y.append(int(line[-1]))

	train_X = np.asarray(train_X, dtype=np.float32)
	train_Y = np.asarray(train_Y)
	train_Y = np.expand_dims(train_Y,axis=1)
	test_X = np.asarray(test_X, dtype=np.float32)
	test_Y = np.asarray(test_Y)
	test_Y = np.expand_dims(test_Y,axis=1)
	# train_X = np.astype(float)
	# test_X = np.astype(float)

	return train_X, train_Y, test_X, test_Y

def sign(pred):
	if( pred > 0 ):
		return 1
	else:
		return -1

def main():
	# N is batch size; D_in is input dimension;
	# H is hidden dimension; D_out is output dimension.
	train_X, train_Y, test_X, test_Y = load_data()
	shuffle_indices = np.random.permutation(np.arange(len(train_Y)))
	train_X = train_X[shuffle_indices]
	train_Y = train_Y[shuffle_indices]

	W = np.zeros(train_X[0].shape)
	b = np.zeros((1))

	MisClassifications = []
	for t in range(100):
		count = 0
		for i in range(len(train_X)):
			pred = np.dot(W.T,train_X[i])+b
			pred = sign(pred)
			if(pred == train_Y[i]):
				continue
			else:
				count += 1
				W += train_Y[i]*train_X[i]
				b += train_Y[i]
		MisClassifications.append(count)

	print(MisClassifications)
	plt.plot(MisClassifications)
	plt.show()

	count = 0
	for i in range(len(train_X)):
		pred = np.dot(W.T,train_X[i])+b
		pred = sign(pred)
		if(pred == train_Y[i]):
			continue
		else:
			count += 1
	print(count)
	print("Training Accuracy: ",float(len(train_Y)-count)/len(train_Y))

	count = 0
	for i in range(len(test_X)):
		pred = np.dot(W.T,test_X[i])+b
		pred = sign(pred)
		if(pred == test_Y[i]):
			continue
		else:
			count += 1
	print(count)
	print("Testing Accuracy: ",float(len(test_Y)-count)/len(test_Y))

	print(W)
	arg_W = np.argsort(W)

	top5 = []
	bottom5 = []
	features = []
	with open("features","r") as fp:
		for line in fp:
			features.append(line.split()[0])

	for i in range(len(arg_W)-1,len(arg_W)-6,-1):
		top5.append(features[arg_W[i]])
	for i in range(5,0,-1):
		bottom5.append(features[arg_W[i]])

	print("Top 5: ",top5)
	print("Bottom 5: ",bottom5)


if __name__ == '__main__':
	main()