

 # Creating a python file that creates data files discretised for Bayesian Network Analysis, as described by Milns et al, 2010.

import numpy as np
import os as os
from os.path import isfile, join

def remove_txt(path):
    file = os.path.basename(path)
    out = os.path.splitext(file)
    final = out[0].replace(' ', '_')
    return final

# Now define a function that will discretise the data as desired

def milns_discrete(matrix):
    dims = matrix.shape
    num = np.count_nonzero(matrix)
    size = float(matrix.size)
    percent = float(num)/size
    m, n = dims[0], dims[1]
    output = np.zeros((m,n), dtype=int)
    if percent < 0.58:
        for i in range(0, m):
            for j in range(0, n):
                if matrix[i, j] != 0:
                    output[i, j] = 1
    else:
        mean = matrix[np.nonzero(matrix)].mean()
        for i in range(0, m):
            for j in range(0, n):
                if matrix[i,j] != 0:
                    if matrix[i,j] <= mean:
                        output[i,j] = 1
                    else:
                        output[i,j] = 2
    return output

# Works as intended. Now implement for files in a directory:

current_directory = os.path.dirname(os.path.realpath(__file__))

files = [f for f in os.listdir(current_directory) if isfile(join(current_directory, f))]
files.remove("Milns_discretiser.py")
for file in files:
    output = milns_discrete(np.loadtxt("{0}/{1}".format(current_directory, file)))
    np.savetxt("{0}/discretised/discrete_{1}.txt".format(current_directory, remove_txt(file)), output, fmt='%.1f', delimiter='\t')




