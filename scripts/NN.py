import numpy as np
import pandas as pd
import math
import statistics
import random
import sys
from .io import return_training_data, unpack, return_seqs
"""

Adapted code from:

1. https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
&
2. https://hackernoon.com/building-a-feedforward-neural-network-from-scratch-in-python-d3526457156b.

Defining the neuralnetwork class and defining its methods.

"""
#Making a NN class with methods for fitting
class NeuralNetwork:
	#3 initial attributes: an object with lists
	def __init__(self, network = list(), output = list(), seed=1):
		self.network = network
		self.output = output
		self.seed = seed

	#Make weights for two more layers: hidden and output. Defaults = autoencoder settings
	def make_weights(self, network, n_inputs = 8, n_hidden = 3, n_outputs = 8):
		#Set random seed
		random.seed(self.seed)
		#Make weights for range in each inputs (ie 9 = 8 + 1 bias), for each hidden neuron (ie 3)
		hidden_layer = [{'weights':[random.uniform(0, 1) for i in range(n_inputs + 1)]} for i in range(n_hidden)]
		network.append(hidden_layer)
		#Make weights for range in each hidden (ie 4 = 3 + 1 bias), for each output neuron (ie 8)
		output_layer = [{'weights':[random.uniform(0, 1) for i in range(n_hidden + 1)]} for i in range(n_outputs)]
		network.append(output_layer)
		#Should be length 2 network
		self.network = network

	#Pass in inputs, and then get the dot product between weights of the respective layer and your inputs = 'z'
	def activate(self, weights, inputs):
		activation = weights[-1]
		#assume that last term is the bias, which doesn't have a weight
		for i in range(len(weights)-1):
			   #dot product
			   activation += weights[i] * inputs[i]
		#Should be a scalar
		return activation

	#Simple functions for sigmoid function and its derivative
	def sigmoid_transfer (self, activation):
		#activation is 'z' from lecture notation
		return 1.0 / (1.0 + math.exp(-activation))

	#derivative
	def sigmoid_derivative (self, input):
		return input * (1.0 - input)

	#first step in fitting: feedforward. Affine.
	def feedforward(self, network, data_row):
		inputs = data_row
		for layer in network:
		#this is 2 in the simple autoencoder case I have
			new_inputs = []
			#Iterate through each neuron
			for neuron in layer:
				#Get activation = 'z'
				activation = self.activate(neuron['weights'], inputs)
				#sigmoid transfer to get activation 'a'
				neuron['output'] = self.sigmoid_transfer(activation)
				#append to new inputs (which at beginning is empty list)
				new_inputs.append(neuron['output'])
			#Feeding forward means that the outputs turn into inputs for next layer
			inputs = new_inputs
		#After the last layer = output layer by definition, set self output to the inputs
		self.output = inputs

	#Second step: backpropagation algorithm
	#Expected is the label here (supervised learning)
	def backprop(self, network, expected, autoencode = True):
		#Start from the end of your network
		for i in reversed(range(len(network))):
			layer = network[i]
			#Want to get MCEE at the end
			errors = list()
			#You are at a hidden layer
			if i != len(network)-1:
				for j in range(len(layer)):
					error = 0.0
					#For each neuron in the layer AFTER the current one (e.g. in case of multiple hidden layers):
					for neuron in network[i + 1]:
						#Append error as the weighted sum (weight coming from neuron j in current layer we are looking at),
						#times the later-layer neuron 'delta' (defined below)
						error += (neuron['weights'][j] * neuron['delta'])
					#Append to list
					errors.append(error)
			else:
			#The final output layer
				for j in range(len(layer)):
					neuron = layer[j]
					#Logic to handle different parts of the assignment
					if autoencode:
						#Autoencoder output = 8 output neurons
						errors.append(expected[j] - neuron['output'])
					else:
						#Only one neuron output for DNA sequences
						errors.append(expected - neuron['output'])
			for j in range(len(layer)):
			#This would be dependent on different loss function/ definitions of 'error'.
				neuron = layer[j]
				#I don't multiply by the sigmoid derivative if we're not at an output layer.
				#Intuitively, this makes sense.
				if i != len(network)-1:
					neuron['delta'] = errors[j] * self.sigmoid_derivative(neuron['output'])
				else:
					#At output, just set the delta to be the direct difference between expected-neuron output
					neuron['delta'] = errors[j]

	# Update network weights with error
	def update_weights(self, network, row, l_rate):
		#Iterate thru network
		for i in range(len(network)):
			#Only consider n-1 inputs
			inputs = row[:-1]
			if i != 0:
			#At hidden + output layers, inputs are the neuronal outputs from previous layers
				inputs = [neuron['output'] for neuron in network[i - 1]]
			for neuron in network[i]:
				#Across each neuron in a layer, do this
				for j in range(len(inputs)):
					#Update the weights for each of the inputs
					neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
				#Separately update the bias term
				neuron['weights'][-1] += l_rate * neuron['delta']

	#For fitting a neural network. conv = threshold for convergence based on validation error.
	#Need to pass in training and validation datasets.
	def fit(self, network, l_rate, n_epoch, conv = 1e-5, autoencode = True, nucleotide_dict = [], train = [], val = []):
		#Ultimate output is errors (by each epoch )
		errors = []
		for epoch in range(n_epoch):
			#Calculate MCEE
			cross_error = 0
			#Logic for autoencoding
			if autoencode:
				#row is a row in the ID matrix
				for row in train:
					self.feedforward(network, row)
					#Autoencoder logic
					expected = row
					#Handling cross-entropy error
					for i in range(len(expected)):
						if expected[i] == 0:
							cross_error += -math.log(1 - self.output[i])
						else:
							cross_error += -math.log(self.output[i])
					#Backprop and update the weights
					self.backprop (network, expected)
					self.update_weights(network, row, l_rate)
				#Calculate mean cross error and print
				MCEE = cross_error/len(train)
				print('>epoch=%d, lrate=%.3f, MCEE=%.3f' % (epoch, l_rate, MCEE))
				errors.append(MCEE)
				#Convergence condition
				if MCEE < conv:
					print ('Convergence for autoencoder potentially reached. Exiting now..')
					return errors
			#NN against DNA logic
			else:
				#Assume training data comes as dataframe
				for index, row in train.iterrows():
					#One-hot encoding
					input = unpack(row['seq'], nucleotide_dict)
					#Expected is in the second column (labeled 'class')
					expected = float(row['class'])
					#Feedforward, backprop, update
					self.feedforward(network, input)
					#autoencode = False => only have one output
					self.backprop (network, expected, autoencode = False)
					self.update_weights(network, input, l_rate)
				#Validation error: call to evaluate method defined below
				val_error = self.evaluate(val, network, l_rate, nucleotide_dict)
				print('>epoch=%d, lrate=%.3f, validation_error=%.3f' % (epoch, l_rate, val_error))
				#Keep track of previous errors and print them out; no functional
				#significance but just for terminal tracking
				if len(errors) > 0:
					previous_error = errors[-1]
					print('old error was', previous_error)
					#Convergence condition: if validation error spikes up, or falls below conv threshold
					if val_error > min(errors) * 1e3 or val_error < conv:
						print ('Convergence for NN potentially reached by validation error. Exiting now..')
						return errors
				errors.append(val_error)
		#This statement is reached if number of epochs is reached
		return errors

	#Predict class for a row of data
	def single_predict(self, network, input, autoencode = True, raw = False):
		#Autoencoder case
		if autoencode:
			self.feedforward(network, input)
			return self.output #8-long
		#DNA NN
		else:
			self.feedforward(network, input)
			#Want the raw prediction for test seqs
			if raw:
				return self.output[0]
			#Return the rounded output for purposes of calculating 'accuracy' for k-folds CV
			return round(self.output[0])

	#Handling cross-validation
	def cross_validate(self, train, val, n_folds, n_inputs, n_hidden, n_outputs, l_rate, n_epoch, nucleotide_dict = []):
		#Partition a shuffled dataset into equal partitions of n-folds
		shuffled = train.sample(frac = 1)
		steps = list(range(0, shuffled.shape[0], shuffled.shape[0]//n_folds))
		#Print out the partitions of dataset
		print('steps are', steps)
		k_fold_acc = []
		#Do this for number of folds - 1
		for i in range(n_folds - 1):
			train = shuffled.copy(deep = True)
			#Initialize a network
			self.network = list()
			self.make_weights(self.network, n_inputs, n_hidden, n_outputs)
			#Begin and end specify how to subset the data
			begin, end = steps[i], steps[i + 1]
			#Look visually at prediction set to make sure it's approximately balanced
			predict = train[begin:end]
			print('prediction set is', predict)
			#Negative intersection of prediction dataset gives the 'true' training dataset
			train = train[~train['seq'].isin(predict['seq'])]
			#Where the fitting occurs
			self.fit(self.network, l_rate, n_epoch, autoencode = False, train = train, val = val, nucleotide_dict = nucleotide_dict)
			#Call evaluate to get prediction accuracy
			pred_acc = self.evaluate(predict, self.network, l_rate, nucleotide_dict, acc = True)
			#print('The accuracy for this', n_folds, '-fold cross validation was', pred_acc)
			k_fold_acc.append(pred_acc)
		#Calculate the mean to get mean k-folds CV accuracy
		total_acc = statistics.mean(k_fold_acc)
		#Print out the results for this particular round of cross-validation and return it
		print('final', n_folds, '-fold cross validation accuracy was', total_acc)
		return total_acc

	#Evaluating error for DNA NN scenarios
	def evaluate(self, dataset, network, l_rate, nucleotide_dict, acc = False):
		cross_error = 0
		#Not calculating accuracy for cross-validation; single fitting
		if not acc:
		#Do manual labor of backprop and weight updating in this eval function
			for index, row in dataset.iterrows():
				#Call unpack for one-hot-encoding handling
				input = unpack(row['seq'], nucleotide_dict)
				expected = float(row['class'])
				#Feedforward input and calculate cross entropy error
				self.feedforward(network, input)
				if expected == 0:
					#Adjust output if output is completely wrong, add to cross-entropy error
					try:
						cross_error += -math.log(1 - self.output[0])
					except ValueError:
						print('Output was 1, but the actual class label was 0. Adjusting output by subtracting small number..')
						#Arbitrary small number
						self.output[0] -= 1e-15
						cross_error += -math.log(1 - self.output[0])
				else:
					try:
						cross_error += -math.log(self.output[0])
					except ValueError:
						print('Output was 0, but the actual class label was 1. Adjusting output by adding small number..')
						self.output[0] += 1e-15
						cross_error += -math.log(self.output[0])
				#Backprop and updating weights
				self.backprop (network, expected, autoencode = False)
				self.update_weights(network, input, l_rate)
			#Returning validation error as being equal to MCEE for validation set
			length_of_data = dataset.shape[0]
			val_error = cross_error/length_of_data
			return val_error
		#Calculating accuracy for k-folds CV
		else:
			acc = 0
			length_rows = dataset.shape[0]
			#Same logic as above
			for index, row in dataset.iterrows():
				input = unpack(row['seq'], nucleotide_dict)
				expected = float(row['class'])
				#Calculate prediction for a particular sequence
				prediction = self.single_predict(network, input, autoencode = False)
				if prediction == expected:
					#Add up accuracy.
					acc += 1
			#Return normalized accuracy across length of the dataset (ie prediction dataset)
			return acc/length_rows

	#Wrapper around k-folds CV (model selection), across a range of learning rates.
	def model_selection(self, train, val, n_folds, n_inputs, n_hidden, n_outputs,
						n_epoch, nucleotide_dict = [], mode = 'lr', lr_range = [0.5, 1, 1.5, 2]):
		#The only 'mode' I currently implemented
		if mode == 'lr':
			model_k_score = {}
			#Make and append to a dictionary of learning rates
			for rate in lr_range:
				#Return k-folds CV score for a particular learning rate
				k_score = self.cross_validate(train, val, n_folds, n_inputs, n_hidden, n_outputs, rate, n_epoch, nucleotide_dict)
				model_k_score[rate] = k_score
			#Print out the results for each learning rate and return the dictionary
			print('The dictionary of scores for', n_folds, '-fold cross validation looks like', model_k_score)
			return model_k_score

	#Test out a FULLY-FITTED model on the test sequences, in the path specified
	def test_predictions(self, path = './data/rap1-lieb-test.txt', nucleotide_dict = []):
		test_seqs = return_seqs(path)
		predictions = []
		#Return prediction for each sequence as a scalar
		for seq in test_seqs:
			seq = unpack(seq, nucleotide_dict)
			score = self.single_predict(self.network, seq, autoencode = False, raw = True)
			#Not a rounded 0/1
			predictions.append(score)
		#Print out what final scores are and write out to tsv file
		print('Final list of scores looks like', predictions)
		pred_df = pd.DataFrame({'seqs': test_seqs, 'predicted_scores': predictions})
		#writing to file code here, no index or column names
		pred_df.to_csv('BW_predictions_bmi_203_w2020.tsv', sep='\t', index = False,
						header = False)
		return pred_df
