
# Want a python function that loads in a bipartite graph and its data set, then sifts out a random 'half dataset'
# including equal numbers of members from both layers

import numpy as np
import random as rd
import os as os
from os.path import isfile, join
import shutil

rd.seed(0)

# As the first 6 nodes are in one layer and the remaining 6 in the other, can get away with taking 3 of the first 6 then
# 3 of the next 6
# However no guarantee they are linked???? - Will need to re-run
# Re-running on vas1-2
# define a function that selects which nodes to select
# as network bipartite get away with selecting 3 nodes in one layer, then 3 of those these are connected to

def get_nodes(adj_matrix, nodes):
    # so, first half in one layer, second half in the other
    node = set(np.arange(0, nodes))
    middle = int(float(nodes)/2.0)
    variables = rd.sample(node, middle)
    # Now read the adjacency matrix
    choice = []
    # adj_matrix must be square
    n = np.shape(adj_matrix)[0]
    for j in range(0, n):
        if j in variables:
            for k in range(0, n):
                if adj_matrix[j][k] != 0:
                    choice.append(k)
    # Now choose from choice
    second_nodes = rd.sample(choice, middle)
    variables = variables + second_nodes
    variables = set(variables)
    return variables

def get_columns(array):
    ni = np.shape(array)[1]
    nj = np.shape(array)[0]
    columns = []
    for i in range(0, ni):
        columni = []
        for j in range(0, nj):
            columni.append(array[j][i])
        columns.append(columni)
    return columns
# Above function works

def get_alternating(length):
    # returns an nxn array
    output = np.zeros((length, length), dtype=int)
    for i in range(0, length):
        for j in range(0, length):
            if j % 2 == 0:
                output[i][j] = 1
    np.savetxt(r"C:\Users\User\Documents\GitHub\BL4201-SH-Project\TestDirectory\trial.txt", output, fmt='%.0f')
    return output

# get_alternating(12)
#
# # Now try and remove the columns for this
# columns = get_columns(get_alternating(12))
#
# network_structure = np.zeros((12,12))
# for i in range(0, 12):
#     for j in range(0, 12):
#         network_structure[i][j] = rd.choice((0, 1))
#
# nodes = get_nodes(network_structure, 6)
# print(f"Nodes are {nodes}")
#
# output = []
# for node in nodes:
#     output.append(columns[node])
# print(output)



def remove_txt(path):
    file = os.path.basename(path)
    out = os.path.splitext(file)
    final = out[0].replace(' ', '_')
    return final

# Now loop over files in the directory

network_structure = np.loadtxt("C:/Users/User/Documents/GitHub/BL4201-SH-Project/gene network structure static network with n6 L15 N10 I1000 in0.txt")

current_directory = os.path.dirname(os.path.realpath(__file__))
print(f"Current directory is {current_directory}")

# get list of files in local directory
files = [f for f in os.listdir(current_directory) if isfile(join(current_directory, f))]
# Now remove all files not with .txt extension
for item in files:
    if item.endswith(".txt"):
        pass
    else:
        files.remove(item)
# Now get the nodes
nodes = get_nodes(network_structure, 6)
print(f"The nodes selected here are {nodes}")
# now loop over the files

for file in files:
    input_array = np.loadtxt(f"{current_directory}/{file}")
    columns = get_columns(input_array)
    output = []
    for node in nodes:
        output.append(columns[node])
    output = np.array(output)
    output = np.transpose(output)
    np.savetxt(f"{current_directory}/truncated_data/truncated_{file}", output, delimiter='\t', fmt='%.6f', newline='\n')
    dimensions = np.shape(output)
    print(f"The dimensions are {dimensions}")

# Works!!!!