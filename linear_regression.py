import numpy as np 



class Linear_Regressor(object):

	def __init__(self):
		self.weight = None
		self.bias = None


	def train(self, X, y, epochs=50,alpha=0.005,verbose=1):
		self.weight = 0
		self.bias = 0

		len_nums = len(X)
		outputs = self.weight * X + self.bias
		loss = self.cost_function(len_nums,y,outputs)

		for i in range(epochs):
			if verbose:
				print(f"Epoch: {i} val loss ========================> {loss}")


			delta_weight = alpha * (np.sum((outputs - y) * 2*X))
			delta_bias = alpha * (np.sum(outputs - y)*2)

			self.weight -= delta_weight / len_nums
			self.bias -= delta_bias / len_nums

			outputs = self.weight * X + self.bias
			loss = self.cost_function(len_nums,y,outputs)


	def predict(self,X):
		assert self.weight != None and self.bias != None
		outputs = self.weight * X + self.bias
		return outputs

	@staticmethod
	def cost_function(n,y,outputs):
		avg_loss = 0.0
		for i in range(n):
			error = (outputs[i] - y[i])**2
			avg_loss += error

		avg_loss = avg_loss / (2*n)
		return avg_loss 

