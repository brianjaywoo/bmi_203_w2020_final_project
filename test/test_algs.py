"""
Code testing for two main code documents: my 'io.py' & 'NN.py' document from
main scripts module.
"""

#Imports
from scripts import NN
from scripts import io
import numpy as np

#Test the neural network class
def test_NN():
	"""
	Test various methods in my NN class to make sure that network architecture is
	behaving as expected.
	"""
	#Test overall structure of new network
	nn = NN.NeuralNetwork()
	nn.make_weights(nn.network)
	assert len(nn.network) == 2, 'This should be 2.'
	#Get the hidden layer.
	hidden = nn.network[0]
	assert len(hidden) == 3, 'Length of hidden layer'
	#Test activation
	test_vec = [1, 0, 0, 0, 0, 0, 0, 0]
	#Get the first neuron in the hidden layer and input its weights with activation by test vector above
	test_act = nn.activate(hidden[0]['weights'], test_vec)
	assert isinstance(test_act, float), 'Activation should be a real scalar.'

	#Testing feedforward
	nn.feedforward(nn.network, test_vec)
	hidden = nn.network[0]
	assert isinstance (hidden, list), 'Should be a list'
	assert len(hidden) == 3

	#Test backprop output
	nn.backprop(nn.network, test_vec)
	output = nn.network[1]
	assert len(output) == 8

	#Testing fit: using a large learning rate, and train/val data from fastas, examine error structures
	train, val = io.return_training_data(), io.return_training_data()
	nucleotide_dict = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
	lr = 3
	epochs = 30
	inputs = 68
	hidden = 10
	output = 1
	nn.network = list()
	nn.make_weights(nn.network, inputs, hidden, output)
	errors = nn.fit(nn.network, lr, epochs, autoencode = False, nucleotide_dict = nucleotide_dict, train = train, val= val)
	assert len(errors) <= epochs, 'Single error for each epoch.'
	assert min(errors) >= max(errors)/1e3, 'The maximum and minimum errors should differ at most by 1,000.'

	#Testing predictions using trained neural net
	training_accuracy = nn.evaluate(train, nn.network, lr, nucleotide_dict, acc = True)
	assert training_accuracy <= 1, 'Accuracy should not go above 1.'

	#Testing model selection method (which in turn indirectly tests cross validation)
	k_dict = nn.model_selection(train, val, 4, inputs, hidden, output, epochs, nucleotide_dict, lr_range = [0, 1, 100])
	k_max = [key for (key, value) in k_dict.items() if value == max(k_dict.values())]
	assert k_max == 1, 'This is the only learning rate in set [0, 1, 100] that should come close to convergence.'

#Test sequence reading in and one-hot-encoding
def test_io_and_encoding():
	"""
	Test various methods in my io.py file to make sure that sequence reading in is
	behaving as expected.
	"""
	#Test reading in negative sequence files, the fasta for which looks a little
	#different than expected.
	negs = io.return_negs()
	pos = io.return_seqs()

	assert len(negs) == 137, 'This length should match the length of the positive seqs.'
	assert len(pos) == 137, 'This length should match the length of the negative seqs.'
	for i in negs:
		assert len(i) == 17, 'Each sequence should be 17bp long.'
	for i in pos:
		assert len(i) == 17, 'Each sequence should be 17bp long.'

	#Test a test encoding dictionary.
	test_string = 'ABC'
	test_dict = {'A': '1', 'B': '2', 'C': '3'}
	test_array = io.one_hot_encode (test_string, test_dict)
	test_input = io.unpack(test_string, test_dict)

	assert test_array == np.array([['1'], ['2'], ['3']]), 'This is what the test one hot encoding should look like.'
	assert test_input == ['123'], 'This is what the output of the unpacked input should look like.'
