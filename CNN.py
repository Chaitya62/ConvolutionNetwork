import numpy as np 
import scipy
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits


"""
CNN with 2 CNN layer 
and one Fully connected layer
"""



df = load_digits()

filter1 = np.random.random(9)
filter1 = filter1.reshape(9,1)
bias1 = np.random.random(1)[0]

filter2 = np.random.random(9)
filter2 = filter2.reshape(9, 1)
bias2 = np.random.random(1)[0]

W1  = np.random.random(16*10)
W1 = W1.reshape(16,10)



dat = df.data[0].reshape(8, 8)

def cnn_forwardprop(data, W, bias):
	data1 =[]
	pre = int((W.shape[0])**0.5)
	d_shape = data.shape;
	for i in range(d_shape[0]-pre+1):
		d =  []
		for j in range(d_shape[1]-pre+1):
			temp = data[i:i+pre,j:j+pre]
			temp = temp.reshape(1, pre*pre)
			dotp = (temp.dot(W))[0][0]
			dotp += bias
			d.append(dotp)
		data1.append(d)
	data1 =  np.array(data1)
	return data1

def relu_activation(data):
	z = np.zeros_like(data)
	return np.where(data > z, data, z)


def flatten_layer(data):
	return data.reshape(data.size,)
	

def fc_layer(data, W):
	return data.dot(W)

def sigmoid(X):
	return (1/(1 + (np.exp(-1*X))))



def forward_prop():
	for digit in df.data:
		digit = digit.reshape(8, 8)
		h = cnn_forwardprop(digit, filter1, bias1)
		h = relu_activation(h)
		h = cnn_forwardprop(h, filter2, bias2)
		h = flatten_layer(h)
		h = sigmoid(h)
		h = fc_layer(h, W1)
		h = sigmoid(h)
		h = h.reshape(10,)
		print(h)
		
	


forward_prop()
