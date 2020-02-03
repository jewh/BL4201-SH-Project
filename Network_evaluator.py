
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
                        # print("breaking")
                        break
                    else:
                        start = jine.find('Influence score for   (') + len('Influence score for   (') #returns position of the desired tag
                        middlestart = jine.find(',0) ->   (') # this returns -1
                        middleend = middlestart + len(',0) ->   (')
                        end = jine.find(',0)', middleend) + len(',0)')
                        influence = float(jine.replace(jine[0:end],'').replace(' ', ''))
                        for i in range(0, nvariables):
                            for j in range(0, nvariables):
                                if int(jine[start:middlestart]) == i:
                                    if int(jine[middleend:end-len(',0)')]) == j:
                                        output[i, j] = influence
                outputs.append(output) # if multiple output arrays, return as list
    file.close()
    return outputs

# define a function that calculates the structural recovery score, from two matrices of edge weights and influence scores

def structural_recovery(actual, solved):
    score = 0
    max = actual.shape[0]
    for i in range(0, max):
        for j in range(0, max):
            score = score + abs(actual[i][j])  # This adds up actual matrix weights
            if actual[i][j] != 0 and solved[i][j] == 0:
                score += -(abs(actual[i][j]))
            elif actual[i][j] == 0 and solved[i][j] != 0:
                score += -(abs(solved[i][j]))
                # adding influence scores, assuming influence scores represent a confidence in the link being there
    return score


# The functions appear to work! Now how to save them
# Now want to run the function over a directory of data

# Also define a function that returns the output matrix as a graph

def remove_txt(path):
    file = os.path.basename(path)
    out = os.path.splitext(file)
    final = out[0].replace(' ', '_')
    return final

def draw_network(matrix, directory, file, instance):
    g = nx.from_numpy_matrix(matrix, create_using=nx.DiGraph())
    plt.figure()
    nx.draw(g, with_labels=True, font_weight='bold', pos=nx.circular_layout(g))
    plt.savefig("{0}/SolvedStructures/{1} structure in{2}.png".format(directory, remove_txt(file), instance))


current_directory = os.path.dirname(os.path.realpath(__file__))

files = [f for f in os.listdir(current_directory) if isfile(join(current_directory, f))]
files.remove("Network_evaluator.py")

for file in files:
    out = output_to_array("{0}/{1}".format(current_directory, file), 6) # used six variables for all the networks involved
    for i in range(0, len(out)-1):
        draw_network(out[i], current_directory, file, i)



