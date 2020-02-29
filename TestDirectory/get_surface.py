# Python script to read the score from banjo networks, tracking the score over iterations

import numpy as np
import matplotlib.pyplot as plt
import os as os
from os.path import isfile, join


#Define a function that returns the string between two characters and returns a number
def between(string, string1, string2):
    start = string.index(string1)
    end = string.index(string2)
    if start <= end:
        out = float(string[start+len(string1):end])
    else:
        out = float(string[end+len(string2):start])
    return out

# Now find the network scores in banjo output file and plot them against the number of iterations
# define a function that returns the number of top scoring networks and how many individual runs
def get_iterations(path):
    filein = open(path, 'r')
    iteration_values = []
    # Find values by:
    count = 0
    for line in filein:
        if '#' in line:
            if '<' in line:
                pass
            else:
                iteration_values.append(between(line, '#', ','))
            if '#1,' in line:
                count += 1 # just checks how many repeats there are on the file
    filein.close()
    length = int(max(iteration_values))
    return count, length

# Define a function to pull out scores and then run over the list of files
def get_scores(path):
    count = get_iterations(path)[0]
    length = get_iterations(path)[1]
    # Now create a 2d array where we can store network score for each best scoring network
    data = np.zeros(length, dtype=float) # might need to transpose this, will come back later
    # Now fill this again
    filein = open(path, 'r')
    for line in filein:
        ticker = -1
        if '1#,' in line:
            ticker += 1
        if '#' in line:
            if '<' in line:
                pass
            else:
                data[int(between(line, '#', ',')-1)] = between(line, 'score: ', ', f')
    # Now plot the data on one figure
    filein.close()
    return data


# get the current directory for file names
current_directory = os.path.dirname(os.path.realpath(__file__))

# get list of files in local directory
files = [f for f in os.listdir(current_directory) if isfile(join(current_directory, f))]
# Now remove all files not with .txt extension
for item in files:
    if item.endswith(".txt"):
        pass
    else:
        files.remove(item)

# Now run across the list of files
# Create data object for the scores in this directory
scores = np.zeros(len(files), dtype=float)

for file in files:
    # Now just sift out what sample the file belongs to
    if str(file).find('Positive') >= 0:
        colour = 'g'
        #label = 'Positive Control'
    elif str(file).find('Neg') >= 0:
        colour = 'r'
        #label = 'Negative Control'
    else:
        colour = 'b'
        #label = 'Sample'
    # So red = neg control, green = pos control, blue = test
    y = get_scores("{0}/{1}".format(current_directory, file))
    x = np.arange(0, get_iterations("{0}/{1}".format(current_directory, file))[1], step=1)
    plt.plot(x, y, colour)
    plt.xlabel("Best network number")
    plt.ylabel("BDe")
    plt.title("BDe Scores for Network {0}".format(int(between(file, '_in', 'Report'))))
plt.show()