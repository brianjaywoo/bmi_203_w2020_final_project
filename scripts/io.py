import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import sys
from Bio import SeqIO

"""

Code here adapted from:

https://2-bitbio.com/2018/06/one-hot-encode-dna-sequence-using.html

Code to help read in and parse sequence formats specific to this assignment.

"""

#Fully display all rows in dataframe: optional
#pd.set_option('display.max_rows', None)

#Parse and return a specified sampling of sequences of length 17 from the negative
#fasta file.

def return_negs (path = './data/yeast-upstream-1k-negative.fa', number_to_return = 137, length = 17):
	#set a seed
	random.seed(1)
	#parse fasta file
	seqs = SeqIO.parse(path, "fasta")
	full_negs = []
	#For each sequence, append the raw DNA sequence to a list
	for seq in seqs:
		full_negs.append(seq.seq)
	sampled_negs = []
	#Sample one of these DNA sequences from the full list
	for i in range(number_to_return):
		sequence = str(random.sample(full_negs, 1)[0])
		end = len(sequence)
		#In the range of the sequence length, choose to start somewhere anywhere
		#in the sequence (assumption: it's ALL negative )
		begin = random.randrange(1, end - length)
		to = begin + length
		#Subsequence is begin, to begin + length
		subseq = sequence[begin:to]
		#append sampled subsequence and then return the full list of sampled negs
		sampled_negs.append(subseq)
	return sampled_negs

#return seqs from a text file of DNA sequences
def return_seqs (path = './data/rap1-lieb-positives.txt'):
	with open(path) as f:
		#Split and strip off new lines
		split_file = f.readlines()
		seqs = [i.strip() for i in split_file]
	return seqs

#Return a balanced sample of data. Always will have 137 positive sequences, and
#however many specified negative sequences.
def return_training_data (length = 137):
	test_negs = return_negs()
	test_pos = return_seqs()
	#Label vectors
	neg_df = pd.DataFrame({'seq': test_negs, 'class': ['0'] * length})
	pos_df = pd.DataFrame({'seq': test_pos, 'class': ['1'] * length})
	#concat data and return as dataframe
	full_data = pd.concat ([neg_df, pos_df])
	return full_data

#one-hot encoding logic: given a dictionary, return array of values (vectors of 0/1)
def one_hot_encode (string, dictionary):
	new_list = []
	#Each nucleotide gets a different encoding
	for nucl in string:
		value = dictionary[nucl]
		new_list.append(value)
	#Return an array with shape 17 * 4
	return np.array(new_list)

#unpack an input string by calling above 'one_hot_encode', and flattens list
def unpack (input, dict):
	input = one_hot_encode(input, dict)
	#List flattening
	input = [item for sublist in input for item in sublist]
	return input

#Helper function to plot mostly error vs epochs plots. Weird '1.02' limit is
#so to help see CV accuracies of 1
def plot_utility (keys, values, name, title, ylab, xticks, ylim = (0, 1.02)):
	plt.plot(keys, values, linewidth = 3)
	plt.xticks(rotation=90)
	plt.xticks(xticks)
	plt.title(title)
	plt.ylim(ylim)
	plt.ylabel(ylab)
	plt.savefig(name)
	plt.clf()
