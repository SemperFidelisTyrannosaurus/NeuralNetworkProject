"""
Methods to generate and load the dataset for the convolutional network.
"""
import numpy as np
import random

def dna_string_to_array(str):
	"""Convert a string containing only A, C, T, or G characters the encoding 
	used by the neural network. In this encoding, there is a channel for each 
	possible character. At a position where the character is present a 1.0 will
	set, otherwise a -1.0 will be set."""
	lookup = { 
	'A' : np.array([[1.0], [-1.0], [-1.0], [-1.0]]), 
	'C' : np.array([[-1.0], [1.0], [-1.0], [-1.0]]),
	'T' : np.array([[-1.0], [-1.0], [1.0], [-1.0]]),
	'G' : np.array([[-1.0], [-1.0], [-1.0], [1.0]]) }
	return np.concatenate([lookup[ch] for ch in str.upper()])

def random_read(infile, nbytes, start_offset, num_samples):
	"""Try to read random num_samples valid characters from file infile until a
	valid sequence is read. This function is trying to avoid reading the whole 
	3GB sequence into memory.""" 
	seq = ""
	while len(seq) < num_samples:
		# Assuming 1 byte chars.
		startidx = random.randint(start_offset, nbytes - num_samples)
		infile.seek(startidx)

		seq = ""
		for line in infile:
			for ch in line:
				if ch == 'A' or ch == 'C' or ch == 'T' or ch == 'G' or ch == 'a' or ch == 'c' or ch == 't' or ch == 'g':
					seq = seq + ch
			if len(seq) >= num_samples:
				break

	return seq[:num_samples].upper()+'\n'

def gen_sequences(infilename, num_reads, num_samples, label, outfilename):
	"""Generate a data set containing num_reads random reads of length 
	num_samples as a text file containing one line per read. Read from sequence 
	in file named infilename. Save the array to file with name outfilename."""
	with open(infilename) as infile:
		nbytes = infile.seek(0, os.SEEK_END)
		infile.seek(0)
		line = infile.readline()
		with open(outfilename, 'w') as outfile:
			outfile.write(str(label) + '\n')
			for i in range(num_reads):
				outfile.write(random_read(infile, nbytes, len(line), num_samples))

def gen_datafiles():
	"""Generate a data set to be used for the project. For each class of data,
	a training and a test data file will be generated."""
	num_reads = 1000
	num_samples = 100
	gen_sequences('hg38.fa', num_reads, num_samples, 1, 'hg38_train.txt')
	gen_sequences('HIV-1.fasta', num_reads, num_samples, 0, 'HIV-1_train.txt')
	gen_sequences('hg38.fa', num_reads, num_samples, 1, 'hg38_test.txt')
	gen_sequences('HIV-1.fasta', num_reads, num_samples, 0, 'HIV-1_test.txt')

def load_labeled_data(files):
	"""Load data from files, matched with labels."""
	x = []
	y = []
	for filename in files:
		data = []
		with open(filename) as infile:
			label = int(infile.readline())
			for line in infile:	
				data.append(dna_string_to_array(line.strip()))
		y += [label]*len(data)
		x += data

	return (np.array(x), np.array(y))
