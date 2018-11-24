
import numpy as np
import tensorflow as tf

def np_example():
	"""Demonstrate possible mehtod for encoding input for project.
	The four letter alphabet is encoded along the 2nd dimension (rows).
	Columns are concatenated together to form strings to be used as a signal or
	filter.
	When convolved, if the filter and signal match at a position, then the 
	resulting bin will be equal to the length of the filter.
	"""

	# Create the alphabet
	A = np.array([[1], [-1], [-1], [-1]])
	C = np.array([[-1], [1], [-1], [-1]])
	T = np.array([[-1], [-1], [1], [-1]])
	G = np.array([[-1], [-1], [-1], [1]])

	x = np.concatenate((G, A, C, T, C, T, A, C, G, A, C, C, T),axis=1)
	filt = np.fliplr(np.concatenate((G, A, C), axis=1))

	np.set_printoptions(precision=2)
	print("Input signals x:\n{}\n".format(x))
	print("Example filter filt:\n{}\n".format(filt))

	filtered = np.zeros(x.shape)
	for row in range(4):
		filtered[row,:] = np.convolve(x[row,:], filt[row,:], mode='same') 

	print("Filtered signals:\n{}\n".format(filtered))
	print("Normalized vertical sum:\n{}\n".format(np.sum(filtered,axis=0)/4/3))

	"""
	Example output:

	Input signals x:
	[[-1  1 -1 -1 -1 -1  1 -1 -1  1 -1 -1 -1]
	 [-1 -1  1 -1  1 -1 -1  1 -1 -1  1  1 -1]
	 [-1 -1 -1  1 -1  1 -1 -1 -1 -1 -1 -1  1]
	 [ 1 -1 -1 -1 -1 -1 -1 -1  1 -1 -1 -1 -1]]

	Example filter filt:
	[[-1  1 -1]
	 [ 1 -1 -1]
	 [-1 -1 -1]
	 [-1 -1  1]]

	Filtered signals:
	[[-2.  3. -1.  1.  1. -1.  3. -1. -1.  3. -1.  1.  0.]
	 [ 0.  3. -1.  1. -1. -1.  3. -1. -1.  3.  1. -3.  0.]
	 [ 2.  3.  1.  1. -1.  1.  1.  3.  3.  3.  3.  1.  0.]
	 [ 0.  3.  1.  1.  1.  1.  1. -1. -1.  3.  1.  1.  0.]]

	Normalized vertical sum:
	[0.   1.   0.   0.33 0.   0.   0.67 0.   0.   1.   0.33 0.   0.  ]
	"""

np_example()
