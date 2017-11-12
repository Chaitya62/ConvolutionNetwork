import numpy as np 
from pprint import pprint


np.random.seed(10)
class NeuralNetwork:

	def __init__(self, learning_rate, epoches):
		self.layers = []
		self.weights = []
		self.learning_rate = learning_rate
		self.data = []
		self.target = []	
		self.epoches= epoches	
		self.layer_val = []
		self.deltas = []

	def layer(self, nodes, output_layer=False, input_layer=False):
		
		
		if not input_layer:
		#	print("layers : {}, {}".format(self.layers[-1], nodes))
			self.weights.append(np.random.rand(self.layers[-1], nodes))
		self.layers.append(nodes)

	def feedforward(self, data):
		self.layer_val = []
		self.layer_val.append(self.sigmoid(data))

		for i, weight in enumerate(self.weights):
			#print(self.bias[i])
			self.layer_val.append(self.sigmoid((self.layer_val[i]).dot(weight)))



		return self.layer_val[-1]

		#for i in self.layer_val:
		#	print(i.shape)

	def relu(self, X):
		z = np.zeros_like(X)
		return np.where(X > z, X, 0.01*X)

	def sigmoid(self, X):

		#return (1/(1 + np.exp(-1*X)))
		return self.relu(X)
		#return X

	def relu_prime(self, X):
		z = np.zeros_like(X)
		return np.where(X>z, 1, 0.01)

	def sigmoid_prime(self, X):
		#return (np.exp(-1*X) / np.power((1 + np.exp((-1*X))),2))
		#return X
		return self.relu_prime(X)

	def backprop(self, target):

		
		self.layer_val.reverse()
		self.weights.reverse()

	#	print("weight : ",self.weights)


		for i, a in enumerate(self.layer_val):
			if(i >= len(self.layer_val)-1): break
			if i is 0: 
				self.deltas.append((a-target)*self.sigmoid_prime(a))		
			else:

				self.deltas.append(((self.deltas[i-1]).dot((self.weights[i-1].T)))
				*self.sigmoid_prime(self.layer_val[i]))

#		print("weights : ", self.weights)
		for i,a in enumerate(self.layer_val):
			if (i < len(self.layer_val)-1):
				#self.bias[i] = self.deltas[i]
				self.weights[i] -= (self.learning_rate * ((self.layer_val[i].T).dot(self.deltas[i])))

		self.weights.reverse()
		self.layer_val.reverse()
		self.deltas = []

	def costFunction(self, data, target):
		return sum(0.5 * (target-data)**2)


	def training(self, dataset):

		for i,data in enumerate(dataset):
			self.layer_val = []
			self.feedforward(data)
			self.backprop(target[i])



	def train(self, dataset, target):

	
		
		for i in range(self.epoches):


			

			#print("epoch: ", i)
			self.target = target
			self.training(dataset)

			ans = []
			for data in dataset:
				ans.append(self.feedforward(data))

			ans = np.array(ans)
			print("COST : ", self.costFunction(ans, target)) 

		#	print(self.layer_val[-1])



# data = [[0, 1], [1, 0], [1, 1],[0, 1], [1, 0], [1, 1], [0, 0], [0, 0], [1, 1], [0, 1], [1 ,0], [1, 1], [0, 1], [1 ,0]]
# target = [1, 1, 0,1, 1, 0, 0, 0,0, 1, 1 , 0, 1, 1]
# dx = []
# tar = []
# for i,d in enumerate(data):
# 	for j in range(np.random.randint(10)):
# 		dx.append(d)
# 		tar.append(target[i])
# data = dx
# target = tar
# t = []

# for i in target:
# 	d = [0 , 0]
# 	d[i] = 1
# 	d = np.array(d)
# 	d = d.reshape(2, 1)
# 	t.append(d)

# target = t

data = [1, 2, 3, 4, 5, 6, 7, 30]
target = [2, 4, 6, 8, 10, 12, 14, 60]

data = []
target = []
for i in range(100):
	data.append(i)
	target.append(11*i+i*2)

data = np.array(data)
data = data.reshape(len(data), 1)

target = np.array(target)
target = target.reshape(len(target),1)




nn = NeuralNetwork(0.0000001, 10)
nn.layer(1, input_layer=True)
nn.layer(2)
nn.layer(2)
nn.layer(1, output_layer=True)

nn.train(data, target)


nn.feedforward(np.array([2]))
print("answer : ", 11*2+4)
print(nn.layer_val[-1])

# (nn.feedforward(np.array([0, 1])))
# print(nn.layer_val[-1])


# (nn.feedforward(np.array([1, 0])))
# print(nn.layer_val[-1])


# (nn.feedforward(np.array([0, 0])))
# print(nn.layer_val[-1])


# (nn.feedforward(np.array([1, 1])))
# print(nn.layer_val[-1])