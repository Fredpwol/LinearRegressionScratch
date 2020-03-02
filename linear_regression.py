import numpy as np 



class Linear_Regressor(object):

	def __init__(self):
		self.weights = None
		self.biases = None


	def train(self, X, y, epochs=50,alpha=0.005):
		self.weights = 0
		self.biases = 0

		len_nums = len(X)
		loss = (1 / len_nums) * np.sum((y - outputs) ** 2)
		outputs = self.weights * X + self.biases

		for i in range(epochs):
			print(f"Epoch: {i} val loss ======================== {loss}")

			delta_weights = alpha * (np.sum((outputs - y) * X))
			delta_bias = alpha * (np.sum(outputs - y))

			self.weights -= delta_weights / len_nums
			self.biases -= delta_bias / len_nums

			outputs = self.weights * X + self.biases
			loss = (1 / len_nums) * np.sum((y - outputs) ** 2)


	def predict(self,X):
		assert self.weights != None and self.biases != None
		outputs = self.weights * X + self.biases
		return outputs


