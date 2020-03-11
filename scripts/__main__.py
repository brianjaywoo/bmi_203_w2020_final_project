import numpy as np
import pandas as pd
import random
import sys
import matplotlib.pyplot as plt
from .NN import NeuralNetwork
from .io import return_negs, return_seqs, one_hot_encode, return_training_data, unpack, plot_utility

'''

Code adapted from

1. https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
&
2. https://hackernoon.com/building-a-feedforward-neural-network-from-scratch-in-python-d3526457156b.

With guidance about loss functions and derivative references coming from:
3. https://peterroelants.github.io/posts/cross-entropy-logistic/.

This main module has three different functions depending on the logical indicator
passed in.

1. -I: autoencode identity matrix 8x8 using 8x3x8 affine NN structure
2. -N: Singly test and fit neural network architecture with hyperparameters specified below
3. -K: Perform k-folds cross validation, select best hyperparameter, and then predict on test dataset of sequences.
'''

#Argument check
if len(sys.argv) < 4:
    print("Usage: python -m scripts [-I|-N|-K] <lr> <epochs>")
    sys.exit('Incorrect usage')

#Read in system arguments
flow = sys.argv[1]
lr = float(sys.argv[2])
epochs = int(sys.argv[3])

#Hyperparameter settings: importantly, use 20 neurons in hidden layer for decent testing time.
#And I don't directly penalize weights here so convergence can be difficult with larger hidden input

#One-hot-encoding
nucleotide_dict = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
n_folds = 4

#Setting input length (should be 68)
data_for_example = list(return_training_data()['seq'])
example_data = random.sample(data_for_example, 1)[0]
input_length = len(unpack(example_data, nucleotide_dict)) #68 = 17 * 4

#NN architecture: single hidden layer. Affine.
n_hidden = 20 #Tangible testing number
n_outputs = 1
train, val = return_training_data(), return_training_data()

#Identity matrix 8x8
if flow == '-I':
    ident = np.identity(8)

    #Initialize the identity matrix
    test = NeuralNetwork()
    #Make random weights in range of (0, 1)
    test.make_weights(network = test.network)
    print('Network post initialization looks like', test.network, 'and the number of layers is', len(test.network))

    #Get errors and their lengths (= the number of epochs for which fitting ran )
    errors = test.fit(test.network, lr, epochs, train = ident)
    error_length = len(errors)

    #Reconstruct my autoencoded identity matrix
    autoencoded_ident = list()
    for row in ident:
        prediction = test.single_predict(test.network, row)
        autoencoded_ident.append(prediction)

    #Make a raw and rounded dataframe from the autoencoded output
    auto_df = pd.DataFrame(autoencoded_ident)
    rounded_df = auto_df.apply(round)

    #Report 3 things: raw, rounded, autoencoded cross entropy mean error vs epochs
    print('The autoencoded identity matrix looks like', auto_df)
    print('After rounding, it looks like', rounded_df)

    #MCEE is mean cross-entropy error
    plot_utility(list(range(error_length)), errors, name = 'autoencode_cross_vs_epochs.png',
                title = 'Mean identity matrix autoencoder cross entropy error vs epochs', ylab = 'MCEE',
                xticks = list(range(0, epochs, epochs//10)), ylim = (0, max(errors)))

#Demonstrating NN fitting and MCEE for fed-in learning rate (default = 1)
elif flow == '-N':
    #Making the neural network and running it as a first pass on training and validation data
    nn = NeuralNetwork()
    nn.make_weights(network = nn.network, n_inputs = input_length, n_hidden = n_hidden, n_outputs = n_outputs)
    #Also has 2 layers
    print('Network has number of layers', len(nn.network))
    #Like above: get errors and plot out the validation dataset errors against epochs.
    #Ideally, this should go down with epochs and not spike up (that means the NN is
    #learning something specific to the training dataset that isn't shared with the validation.)
    errors = nn.fit(nn.network, lr, epochs, train = train, val = val, autoencode = False, nucleotide_dict = nucleotide_dict)
    error_length = len(errors)

    #Plot with 10 ticks on x-axis
    plot_utility(list(range(error_length)), errors, name = 'validation_cross_vs_epochs.png',
                title = 'Mean cross entropy error for validation dataset vs epochs', ylab = 'MCEE',
                xticks = list(range(0, epochs, epochs//10)), ylim = (0, max(errors)))

#K-fold cross validation followed by predictions on test dataset.
elif flow == '-K':
    nn = NeuralNetwork()
    #Go through the process of model selection across a different range of learning rates.
    #Default: 0.5-2, in steps of 0.5 = 4 total. I expect 2 to be too high a learning
    #rate (I am scaling up each step by 2 times), and 0.5 to be too low (I am scaling
    #each step by 1/2.)

    #Train for less epochs here because I want to select for hyperparameters that don't
    #take forever to train.
    k_fold_dict = nn.model_selection(train, val, n_folds, n_inputs = input_length, n_hidden = n_hidden, n_outputs = n_outputs,
                        n_epoch = epochs//4, nucleotide_dict = nucleotide_dict)

    #Plot out the mean k-folds cross validation accuracy, by learning rate
    k_keys, k_vals = list(k_fold_dict.keys()), list(k_fold_dict.values())
    plot_utility(k_keys, k_vals, name = 'k_vs_lr.png',
                title = 'k-fold-val-mean-accuracy by learning rate', xticks = k_keys,
                ylab = 'mean' + str(n_folds) + '-folds accuracy')

    """

    Let's now use this best learning rate and use it to test predictions. We will fit our model using the entire
    dataset and use the fitted model to output predictions on the test file. We will then return the test
    prediction score dataframe and export as tsv.

    """

    #Choose the best hyperparameter and train the model
    best_lr = [key for (key,value) in k_fold_dict.items() if value == max(k_fold_dict.values())][0]

    #Re-initialize a network and make weights
    nn = NeuralNetwork()
    nn.make_weights(network = nn.network, n_inputs = input_length, n_hidden = n_hidden, n_outputs = n_outputs)

    #Fit using the best learning rate determined from cross-validation experiments
    nn.fit(nn.network, best_lr, epochs * 2, train = train, val = val, autoencode = False, nucleotide_dict = nucleotide_dict)

    #For each sequence in test document, output a scalar in the range of (0, 1), where 0 = not bound
    test_predictions = nn.test_predictions(nucleotide_dict = nucleotide_dict)

    #print('The test prediction score df looks like', test_predictions)
