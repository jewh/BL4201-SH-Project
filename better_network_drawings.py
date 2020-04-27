
# python file that simply draws clearer network structures and saves them as png
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os as os
from os.path import isfile, join
import networkx as nx

#Define a function that returns the string between two characters and returns a number

def between(string, string1, string2, integer=True):
    start = string.index(string1)
    end = string.index(string2)
    if start <= end:
        out = string[start+len(string1):end]
    else:
        out = string[end+len(string2):start]
    if integer:
        out = int(out)
        return out
    else:
        return out

# define a function that plots graph diagrams
def draw_graph(path):
    current_directory = os.path.dirname(os.path.realpath(__file__))
    array = np.loadtxt(path)
    graph = nx.DiGraph()
    # Get the minimum value in the array
    non_zero = array[np.nonzero(array)]
    minimum = np.amin(non_zero)
    # get the dimensions of the square array
    n = np.shape(array)[0]
    for i in range(0, n):
        for j in range(0, n):
            if array[i][j] != 0:
                weight = (abs(array[i][j]) / minimum) ** 2
                # exponent just controls the difference in edge width between lowest and highest
                if array[i][j] < 0:
                    colour = 'm'
                elif array[i][j] > 0:
                    colour = 'g'
                graph.add_edge(i, j, color=colour, weight=weight)
    edges = graph.edges()
    colours = [graph[u][v]['color'] for u, v in edges]
    weights = [graph[u][v]['weight'] for u, v in edges]
    pos = nx.circular_layout(graph)
    nx.draw_networkx(graph, with_labels=True, pos=pos, edge_color=colours, width=weights)
    # Now get legend
    # custom_lines = [Line2D([0], [0], color='g', lw=2),
    #                 Line2D([0], [0], color='m', lw=2)]
    # plt.legend(custom_lines, ['Positive Effects', 'Negative Effects'])
    plt.title(f"{between(path, '/', ' structure', integer=False)} Network Number {between(path, ' in', '.txt')}")
    plt.savefig(f"{current_directory}/figures/ColouredEdges{between(path, '/', ' structure', integer=False)}Network{between(path, ' in', '.txt')}.png")
    plt.close()



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


# Now load in each file as numpy array
for file in files:
    draw_graph("{0}/{1}".format(current_directory, file))
    
