import numpy as np 



class Linear_Regressor(object):

	def __init__(self):
		self.weights = None
		self.biases = None


	def train(self, X, y, epochs=50,alpha=0.005,verbose=1):
		self.weights = 0
		self.biases = 0

		len_nums = len(X)
		outputs = self.weights * X + self.biases
		loss = self.cost_function(len_nums,y,outputs)

		for i in range(epochs):
			if verbose:
				print(f"Epoch: {i} val loss ======================== {loss}")

			delta_weights = alpha * (np.sum((outputs - y) * X))
			delta_bias = alpha * (np.sum(outputs - y))

			self.weights -= delta_weights / len_nums
			self.biases -= delta_bias / len_nums

			outputs = self.weights * X + self.biases
			loss = self.cost_function(len_nums,y,outputs)


	def predict(self,X):
		assert self.weights != None and self.biases != None
		outputs = self.weights * X + self.biases
		return outputs

	@staticmethod
	def cost_function(n,y,outputs):
		avg_loss = 0.0
		for i in range(n):
			error = (y[i] - outputs[i])**2
			avg_loss += error

		avg_loss = avg_loss / (2*n)
		return avg_loss 

