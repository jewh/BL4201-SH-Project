
# Python file for automatically evaluating the network recovery.

# Compares with existing true network and evaluates the structural recovery

import numpy as np
import os as os
from os.path import isfile, join
import networkx as nx
import matplotlib.pyplot as plt
import shutil

# define a function that converts output network structures (.txt) into matrices
# For now this is probably going to have to be two different structures

def output_to_array(path, nvariables): # returns a list
    outputs = []
    with open(path, 'r') as file:
        file_lines = file.readlines()
        for num, line in enumerate(file_lines):
            if 'Influence scores' in line:  # As best network overall not always associated with one?
                # Can correctly index the lines using this method.
                output = np.zeros((nvariables, nvariables), dtype=float)
                # count the number of spaces after the lines
                for number, jine in enumerate(file_lines[num+3:len(file_lines)-1]):
                    #print(jine)
                    if jine == '\n':
                        break
                    else:
                        start = jine.find('(') + len('(') #returns position of the desired tag
                        # print(jine[0:start])
                        middlestart = jine.find(',') # find first comma after the parent number
                        # print(jine[start:middlestart])
                        middleend = jine.find('(', middlestart) + len('(') # find the next open bracket
                        # print(jine[middlestart:middleend])
                        end = jine.find(',0)', middleend) #+ len(',0)')
                        # print(jine[middleend:end])
                        influence = float(jine.replace(jine[0:end+len(',0)')],'').replace(' ', ''))
                        for i in range(0, nvariables):
                            for j in range(0, nvariables):
                                if int(jine[start:middlestart]) == i:
                                    if int(jine[middleend:end]) == j:
                                        output[i, j] = 1 # influence
                outputs.append(output) # if multiple output arrays, return as list
    file.close()
    return outputs

# define a function that calculates the structural recovery score, from two matrices of edge weights and influence scores

def structural_recovery(actual, solved, kind):
    score = 0
    false_positives = 0
    false_negatives = 0
    max = actual.shape[0]
    for i in range(0, max):
        for j in range(0, max):
            score = score + abs(actual[i][j])  # This adds up actual matrix weights
            if actual[i][j] != 0 and solved[i][j] == 0:
                score += -(abs(actual[i][j]))
                false_negatives += 1
            elif actual[i][j] == 0 and solved[i][j] != 0:
                score += -(abs(solved[i][j]))
                false_positives += 1
                # adding influence scores, assuming influence scores represent a confidence in the link being there
    out = np.array(score, false_positives, false_negatives)
    if kind == 'metric':
        return out
    elif kind == 'hamming_distance': # add a way of calculating hamming distance for reference
        hamming = false_negatives + false_positives
        return np.array(hamming, false_positives, false_negatives) 

# The functions appear to work! Now how to save them
# Now want to run the function over a directory of data

# Also define a function that returns the output matrix as a graph
def draw_network(matrix, directory, file, instance):
    g = nx.from_numpy_matrix(matrix, create_using=nx.DiGraph())
    plt.figure()
    nx.draw(g, with_labels=True, font_weight='bold', pos=nx.circular_layout(g))
    plt.savefig("{0}/{1} structure in{2}.png".format(directory, file, instance))
    plt.show()

current_directory = os.path.dirname(os.path.realpath(__file__))

files = [f for f in os.listdir(current_directory) if isfile(join(current_directory, f))]
files.remove("Network_evaluator.py")

# structure = open("C:\Users\User\Documents\GitHub\BL4201-SH-Project\extinction_networks\instance_0/network structure static network with n6 L15 N10 I1000 in0.txt", 'r')
# output_file = open("{0}/summaries/summary.txt".format(current_directory), "w+")
for file in files:
    out = output_to_array("{0}/{1}".format(current_directory, file), 38) # used six variables for all the networks involved
    draw_network(out[0], current_directory, file, 0)
    for i in range(0, len(out)-1):
        print("memes")
        draw_network(out[i], current_directory, file, i)
#         for j in range(0, 3):
#             output_file.write("{0}\t".format(structural_recovery(out[i], structure)[j]))
#             print("memes")
#         output_file.write("{0}\n".format(file))
# output_file.close()
# structure.close()
