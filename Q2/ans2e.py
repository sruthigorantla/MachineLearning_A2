import torch
from torch.autograd import Variable
import numpy as np

dtype = torch.FloatTensor

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
	print(count, len(test_Y)-count)

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
	# N is batch size; D_in is input dimension;
	# H is hidden dimension; D_out is output dimension.
	train_X, train_Y, test_X, test_Y = load_data()
	shuffle_indices = np.random.permutation(np.arange(len(train_Y)))
	train_X = train_X[shuffle_indices]
	train_Y = train_Y[shuffle_indices]
	N = 64
	D_in = len(train_X[0])
	H = 20
	D_out = 1

	x = Variable(torch.from_numpy(train_X).type(dtype), requires_grad=False)
	y = Variable(torch.from_numpy(train_Y).type(dtype), requires_grad=False)

	test_x = Variable(torch.from_numpy(test_X).type(dtype), requires_grad=False)
	w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
	w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)


	model = torch.nn.Sequential(
          torch.nn.Linear(D_in, H),
          # torch.nn.ReLU(),
          torch.nn.Linear(H, D_out),
        )
	loss_fn = torch.nn.MSELoss(size_average=False)

	for l2 in [0.0001, 0.001, 0.01, 0.1, 1]:
		print("L2 regularization: ",l2)
		optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=l2)
		for t in range(1000):
			# Forward pass: compute predicted y by passing x to the model.
			y_pred = model(x)

			# Compute and print loss.
			loss = loss_fn(y_pred, y)
			# print(t, loss.data[0])

			# Before the backward pass, use the optimizer object to zero all of the
			# gradients for the variables it will update (which are the learnable weights
			# of the model)
			optimizer.zero_grad()

			# Backward pass: compute gradient of the loss with respect to model parameters
			loss.backward()

			# Calling the step function on an Optimizer makes an update to its parameters
			optimizer.step()

		# learning_rate = 1e-6
		# for t in range(1000):	
		# 	y_pred = x.mm(w1).clamp(min=0).mm(w2)
		# 	loss = (y_pred - y).pow(2).sum()
		# 	if(t%100 == 0):
		# 		print(t, loss.data[0])
		# 	loss.backward()

		# 	w1.data -= learning_rate * w1.grad.data
		# 	w2.data -= learning_rate * w2.grad.data

		# 	w1.grad.data.zero_()
		# 	w2.grad.data.zero_()

		# preds = x.mm(w1).clamp(min=0).mm(w2)

		preds = model(x)
		count = 0
		for i in range(len(y)):
			if(preds.data[i][0]>0.5):
				if(train_Y[i] == 1):
					count += 1
			else:
				if(train_Y[i] == 0):
					count += 1
		accuracy = float(count)/len(train_Y)
		print("Train accuracy: ",accuracy)
		
		preds = model(test_x)
		count = 0
		for i in range(len(test_Y)):
			if(preds.data[i][0]>0.5):
				if(test_Y[i] == 1):
					count += 1
			else:
				if(test_Y[i] == 0):
					count += 1
		accuracy = float(count)/len(test_Y)
		print("Test accuracy: ",accuracy)

if __name__ == '__main__':
	main()
