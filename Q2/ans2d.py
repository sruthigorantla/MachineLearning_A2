import numpy as np
import ans2c as a2c
import matplotlib.pyplot as plt
import sys

def load_data():
	train_X = []
	train_Y = []
	test_X = []
	test_Y = []
	with open("./../Q1/data/SpambaseFull/train.txt","r") as fp:
		for line in fp:
			line = line.split(",")
			train_X.append(line[:-1])
			if(int(line[-1]) == 1):
				train_Y.append(int(line[-1]))
			else:
				train_Y.append(0)

	with open("./../Q1/data/SpambaseFull/test.txt","r") as fp:
		count = 0
		for line in fp:
			line = line.split(",")
			test_X.append(line[:-1])
			if(int(line[-1]) == 1):
				test_Y.append(int(line[-1]))
			else:
				count += 1
				test_Y.append(0)
	train_X = np.asarray(train_X, dtype=np.float32)
	train_Y = np.asarray(train_Y)
	train_Y = np.expand_dims(train_Y,axis=1)
	test_X = np.asarray(test_X, dtype=np.float32)
	test_Y = np.asarray(test_Y)
	test_Y = np.expand_dims(test_Y,axis=1)
	# train_X = np.astype(float)
	# test_X = np.astype(float)

	return train_X, train_Y, test_X, test_Y

def main():
	# Load SpambaseFull data
	train_X, train_Y, test_X, test_Y = load_data()
	
	shuffle_indices = np.random.permutation(np.arange(len(train_Y)))
	train_X = train_X[shuffle_indices]
	train_Y = train_Y[shuffle_indices]
	# Instantiate a network
	net = a2c.Net2c()

	# Iterate
	Cost = []
	Test_cost = []
	i = 0
	while(i<10000):		
		i += 1
		# Compute gradients
		batch_size = 10
		cost_avg = 0
		for j in range(0,len(train_X),batch_size):
			preds = net.forward(train_X[j:j+batch_size])
			cost = net.backward(preds, train_Y[j:j+batch_size])
			net.step(1e-2)
			cost_avg += cost
		cost_avg /= (float(len(train_X))/batch_size)
		
		if(i%1000 == 0):
			print("Iteration: ",i,end="\t")
			print("Cost: ",cost_avg)
			preds = net.forward(test_X)
			cost = net.backward(preds, test_Y)
			print("Test loss: ",cost)
			Cost.append(cost_avg)
			Test_cost.append(cost)

	plt.plot(Cost,Test_cost)
	plt.show()

	preds = net.forward(train_X)
	# print(preds)
	outputs = []
	for i in range(len(preds)):
		if(preds[i] < 0.5):
			outputs.append([0])
		else:
			outputs.append([1])
	outputs = np.asarray(outputs)
	# print(outputs.shape)
	accuracy = (outputs == train_Y).all(axis=1).mean()
	print(accuracy)

	preds = net.forward(test_X)
	# print(preds)
	outputs = []
	for i in range(len(preds)):
		if(preds[i] < 0.5):
			outputs.append([0])
		else:
			outputs.append([1])
	outputs = np.asarray(outputs)
	# print(outputs.shape)
	accuracy = (outputs == test_Y).all(axis=1).mean()
	print(accuracy)

if __name__ == '__main__':
	main()