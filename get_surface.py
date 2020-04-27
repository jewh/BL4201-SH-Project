# Python script to read the score from banjo networks, tracking the score over iterations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
import os as os
from os.path import isfile, join
import seaborn as sbn
import pandas as pd
import networkx as nx



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
                # print(between(line, 'score: ', ', f'))
    # Now plot the data on one figure
    filein.close()
    data = data[::-1]
    return data


# Define a function to get the array for each best network tracked during a banjo search
# Return a list of best networks for the file

def get_bestnetworks(path, type):
    # Load the file
    filein = open(path, 'r')
    filein = filein.readlines()
    # Now find the number of variables, and the number of best networks
    # specify which rows to find them in using the below id tags
    id = '- Number of best networks tracked:'
    pid = '- Number of variables:'
    for line in filein:
        if id in line:
            num_networks = int(line.replace(id, ''))
        elif pid in line:
            num_variables = int(line.replace(pid, ''))
    # Now find all adjacency matrices for the n best networks
    # Create list to store the arrays
    nbestnetworks = []
    # Loop over the number of tracked best networks and the lines in the output file
    # This is probably an awful method of removing duplicates, however define a list to store all the network scores,
    # then remove duplicates and get the size of the set
    number_tracked = []
    for num, line in enumerate(filein):
        for i in range(1, num_networks+1):
            if '#{0},'.format(i) in line:
                number_tracked.append(i)
                # Create array to store the adjacency matrix in
                adj_matrix = np.zeros((num_variables, num_variables), dtype=int)
                for number, jine in enumerate(filein[num+2:len(filein)-1]):
                    # stop the loop if empty line, and go to next i
                    if jine == '\n':
                        break
                    elif type == 'static':
                        # Now we wish to identify the parent node and its children
                        # Get the parent identity:
                        # print("Jine = {0}".format(jine))
                        # assume that jine[0] is either an integer or a space
                        # Therefore find the next space along:
                        upper = jine[1:].index(' ')
                        child = int(jine[0:upper+1])
                        # print("child = {0}".format(child))
                        # Now remove the child and the two spaces either side, what is left are the parents
                        # split() converts to a list
                        parents = jine.split()
                        parents.remove(str(child))
                        # Now remove the first entry of the list, which is the number of parents per node
                        del parents[0]
                        # print("parents are = {0}".format(parents))
                        # Now fill up the adjacency matrix for all children nodes
                        for j in range(0, len(parents)):
                            k = int(parents[j])
                            # add a 1 for each link
                            # Note a_ij = 1 iff i -> j in the network
                            adj_matrix[k][child] = 1
                        # print(adj_matrix)
                    elif type == 'dynamic':
                        # Find the child node (upper allows for multiple digits)
                        upper = jine[1:].index(' ')
                        child = int(jine[0:upper + 1])
                        # As parents specified after the :, find :
                        base = jine.index(':')
                        parents = jine[base+1:].split()
                        del parents[0]
                        # Now fill up the adjacency matrix for all children nodes
                        for j in range(0, len(parents)):
                            k = int(parents[j])
                            # add a 1 for each link
                            # Note a_ij = 1 iff i -> j in the network
                            adj_matrix[k][child] = 1
                nbestnetworks.append(adj_matrix)
                continue
    # Now remove the duplicates
    # Find the size of the 'basic set' i.e. set of elements which are multiplied in nbestnetworks
    ind = len(set(number_tracked))
    # Now remove them
    nbestnetworks = nbestnetworks[0:ind]
    # Now reverse the list
    nbestnetworks = nbestnetworks[::-1]
    # Return a list of adjacency matrices
    return nbestnetworks

# Above appears to recover the adjacency matrices


# Now define a small function that calculates hamming distance between two 2d arrays

def hamming_distance(array1, array2):
    distance = 0
    # Assume both matrices of same dimension and square in R^2
    n = np.shape(array1)[0]
    for i in range(0, n):
        for j in range(0, n):
            if array1[i][j] != array2[i][j]:
                distance += 1
    return distance

# Now a function that uses this in a loop to get consecutive hamming distances

def consecutive_distances(list):
    # list must be a list of arrays of equal dimension
    out = np.zeros(len(list), dtype=float)
    for i in range(0, len(list)-2):
        out[i+1] = hamming_distance(list[i], list[i+1])
    out[0] = 1
    out[len(list)-1] = 1
    return out

# Additionally define a function that plots the number of links in the array
# This is because hamming distance does not distinguish between [0 , 1] -> [0] and [0] -> [0 , 1], which we wish to detect

def number_links(list):
    out = np.zeros(len(list), dtype=float)
    for i in range(0, len(list)):
        out[i] = np.count_nonzero(list[i])
    return out

# Define a function that normalises the BDe scores for a banjo output file

def normalise_score(path):
    scores = get_scores(path)
    # as the scores list is reversed, desire the intial value
    initial_score = scores[0]
    # Now define output array
    out = np.zeros(len(scores), dtype=float)
    # Now fill up with normalised scores
    for i in range(0, len(scores)):
        out[i] = scores[i]/initial_score
    return out

# Define a function that computes the length of curves composed of linear sections

def get_lengths(list_of_lists):
    # assume that the list of lists is input
    # define an output list
    output = []
    for li in list_of_lists:
        y = li
        # set up the x array
        x = np.arange(0, len(y))
        # initialise the length
        length = 0.0
        # assume the length of y = len x
        for i in range(0, len(y)-1):
            dy = y[i+1] - y[i]
            dx = x[i+1] - x[i]
            # get the euclidean distance
            length += np.sqrt(dy**2 + dx**2)
        length = length/float(len(y))
        output.append(length)
    return output

# Define a smal function to map each dataset type to an index

def get_index(label):
    if label == 'Positive Control':
        return 1
    elif label == 'Negative Control':
        return 2
    elif label == 'Sample':
        return 0

# similarly, a function that gets the repeat number

def get_repeat(file, label, upper):
    if label == 'Positive Control':
        id = int(between(file, '_Control_', upper))
    elif label == 'Negative Control':
        id = int(between(file, '_Control_', upper))
    elif label == 'Sample':
        id = int(between(file, '_Output_', upper))
    return id

# Define a function that calculates the roughness for each curve in a dataset, then exports to file format for R

def get_roughs(list_of_lists):
    # assumes input is a list of lists
    # calculate the roughness for each curve
    roughness = []
    for list in list_of_lists:
        total = 0
        for j in range(1, len(list)-1):
            total += abs(list[j]-1)
        roughness.append(float(total)/float(len(list)))
    return roughness

# define a function to make a list of a certain string

def list_of_string(string, n):
    output = []
    for i in range(0, n):
        output.append(string)
    return output

# Define a function that will generate a jagged curve

def get_jagged(n):
    if n % 2 == 0:
        return 0
    else:
        return 2

plt.style.use('ggplot')
matplotlib.rcParams['font.sans-serif'] = "Arial"

# for plotting individual files specify where the data is
data_directory = r"C:\Users\User\Documents\GitHub\BL4201-SH-Project\Dump\banjooutputs"
items = [f"{data_directory}/dynamic.combinedsongbirdReport.txt", f"{data_directory}/static.A3report.txt"]
length = len(items)
labels = ["Moorland", "Farmland"]
# specify the kind of inference (dynamic/static)
kind = ['dynamic', 'static']
# Now plot for each file
figure = plt.figure()

gs = figure.add_gridspec(2,2)
ax1 = figure.add_subplot(gs[0,0])
ax2 = figure.add_subplot(gs[1,0])
ax3 = figure.add_subplot(gs[0,1])
ax4 = figure.add_subplot(gs[1,1])


hammings = []
nums = []
scores = []

for k in range(0, length):
    label = labels[k]
    result = normalise_score(items[k])
    scores.append(result)
    # xscore = np.arange(0, len(result))
    # plt.plot(xscore, result)
    # plt.xlabel("Best network number")
    # plt.ylabel("Normalised BDe Score")
    # plt.title("BDe Score during Static BNI Search on Rat Gene Expression Data")
    # plt.show()

    # Try for a specific file:
    type = kind[k]
    networks = get_bestnetworks(items[k], type)
    # Above returns 52 matrices, i.e. 2x26 i.e. returns duplicates of the run
    # Now get the hamming distance for each sequential pairs of networks in the list, and plot
    hamming_distances = np.zeros(len(networks), dtype=float)
    for i in range(0, len(networks)-2):
        hamming_distances[i+1] = hamming_distance(networks[i], networks[i+1])
    hamming_distances[0] = 1
    hamming_distances[len(networks)-1] = 1
    hammings.append(hamming_distances)
    p = get_roughs([hamming_distances])
    print(f"P = {p}")

    links = number_links(networks)
    nums.append(links)
    # x = np.arange(0, len(links))
    # Now create a subplot environment to plot both of them
    # ax1.plot(xscore, result)
    # ax1.set_ylabel("Normalised BDe Score", fontsize=10)
    # ax1.title.set_text("All Local Moves Proposer + SimAnneal Searcher")
    # ax2.plot(x, links, color="#00aa00")
    # ax2.plot(x, hamming_distances, color="#348abdff")
    # ax2.set_ylabel("Number of Network Edges/\nConsecutive Hamming Distance", fontsize=10)
    # custom_lines = [Line2D([0], [0], color='#00aa00', lw=2),
    #                 Line2D([0], [0], color='#348abdff', lw=2)]
    # ax2.legend(custom_lines, ['Number of Edges', 'Consecutive Hamming Distance'], fontsize=8)
    # plt.xlabel("Best Network Number", fontsize=10)
    # plt.suptitle(f"Topology of BNI Search Space on Milns et al 2010 Data", fontsize=14, fontweight='bold')
    # plt.show()
    continue

x = np.arange(0, len(scores[0]))
ax1.plot(x, scores[0])
ax1.set_ylabel("Normalised BDe Score", fontsize=10)
ax1.set_title("Zebrafinch Auditory System", fontsize=11)

ax2.plot(x, hammings[0], color="#348abdff")
ax2.plot(x, nums[0], color="#00aa00")
ax2.set_ylabel("Number of Network Edges/\nConsecutive Hamming Distance", fontsize=10)

x = np.arange(0, len(scores[1]))
ax3.plot(x, scores[1])
ax3.set_title("Rat Brain Microarray", fontsize=11)

ax4.plot(x, hammings[1], color="#348abdff")
ax4.plot(x, nums[1], color="#00aa00")

custom_lines = [Line2D([0], [0], color='#00aa00', lw=2), Line2D([0], [0], color='#348abdff', lw=2)]
# ax2.legend(custom_lines, ['Number of Edges', 'Consecutive Hamming Distance'], fontsize=8)


# plt.legend(custom_lines, ['Number\nof Edges', 'Consecutive\nHamming\nDistance'], fontsize=8, loc='bottom')

ax2.set_xlabel("Best Network Number", fontsize=10)
ax4.set_xlabel("Best Network Number", fontsize=10)
plt.suptitle(f"Topology of BNI Search Spaces for Canonical Examples\n ", fontsize=14, fontweight='bold')
# plt.tight_layout()
plt.savefig("C:/Users/User/Documents/GitHub/BL4201-SH-Project/TestDirectory/figures/canonicalexamples.png", bbox_inches='tight')
plt.show()

    # ax2.xlabel("Best network number")
    # ax2.ylabel("Number of Links/Hamming Distance")
    # plt.show()
    # plt.title("Search Space Topology of Static BNI Search on Rat Gene Expression Data")
    # custom_lines = [Line2D([0], [0], color='#e24a33ff', lw=2),
    #                Line2D([0], [0], color='#348abdff', lw=2)]
    # plt.legend(custom_lines, ['Number of Links', 'Consecutive Hamming Distance'])
    # plt.show()
    # # Get the average clustering coefficient of the top-scoring network
    # top = len(result)
    # # top scoring network is [top-1]
    #
    # top_network = result[top-1]
    # top_graph = nx.from_numpy_matrix(top_network, create_using=nx.DiGraph())
    # clustering = nx.average_clustering(top_graph)
    # print(f"Global Clustering Coefficient = {clustering}")
    # # compare with erdo-renyi graph on same number of nodes
    # reference = nx.gnm_random_graph(np.shape(top_network)[0], 25, directed=True)
    # erdos_clustering = nx.average_clustering(reference)
    # print(f"Erdos-Renyi has clustering = {erdos_clustering}")

    # get roughness for the file
    # hamming = consecutive_distances(result)
    # length = len(hamming)
    # roughness = 0.0
    # for i in range(1, length-1):
    #     roughness += float(abs(hamming[i]-1))
    # roughness = roughness/float(length)
    # print(f"The roughness for the data set is {roughness}")


# # get the current directory for file names
# current_directory = os.path.dirname(os.path.realpath(__file__))
#
# # get list of files in local directory
# files = [f for f in os.listdir(current_directory) if isfile(join(current_directory, f))]
# # Now remove all files not with .txt extension
# for item in files:
#     if item.endswith(".txt"):
#         pass
#     else:
#         files.remove(item)
# # Now remove the python file
# files.remove("get_surface.py")
# # remove the roughness file
# # files.remove("roughness.txt")
# # Now run across the list of files
# # Specify the plot style
# plt.style.use('ggplot')
# # Specify the kind of plot you'd like to run - options are 'normalised_scores', 'consecutive_distances', or 'numlinks'
# plot_type = 'normalised_scores'
# # specify the file type to analyse too, options are 'static' or 'dynamic'
# type = 'dynamic'
#specify upper bound in string to get the output number
# options are '_gene_reg_' or '_static_', or '_dynamic_'
# upper = '_static_'
# #
# # #
# # # # Create a list to store the number of peaks each search found, for each dataset
# # # sample_peaks = []   # for the sample
# # # pos_peaks = []      # for positive control
# # # neg_peaks = []   # for negative control
# # #
# # # # Create lists that store the consecutive hamming distances for each dataset type
# sample_distances = []
# pos_distances = []
# neg_distances = []
# dag_distances = []
# cyclic_distances = []
# truncated_distances = []
# other_distances = []
# twodev = []
#
# # #
# # # # Create an array to store lengths
# # lengths = np.zeros((3,10), dtype=float)
# #
# for file in files:
#     print(file)
#     # Now just sift out what sample the file belongs to
#     if str(file).find('truncated') >= 0:
#         identifier = '_Control_'
#         output = float(between(file, identifier, upper))
#         colour = (0.0, output / 10.0, 0.0)
#         label = 'Truncated Data'
#         # peaks = pos_peaks
#         distances = truncated_distances
#     elif str(file).find('Positive') >= 0:
#         identifier = '_Control_'
#         # output = float(between(file, identifier, upper))
#         # colour = (0.0, output/10.0, 0.0)
#         label = 'Positive Control'
#         # peaks = pos_peaks
#         distances = pos_distances
#     elif str(file).find('Neg') >= 0:
#         identifier = '_Control_'
#         # output = float(between(file, identifier, upper))
#         # colour = (output/10.0, 0.0, 0.0)
#         label = 'Negative Control'
#         # peaks = neg_peaks
#         distances = neg_distances
#     elif str(file).find('Output') >= 0:
#         identifier = '_Output_'
#         # output = float(between(file, identifier, upper))
#         # colour = (0.0, 0.0, output/10.0)
#         label = 'Sample'
#         # peaks = sample_peaks
#         distances = sample_distances
#     elif str(file).find("Cyclic") >= 0:
#         identifier = '_Output_'
#         output = float(between(file, identifier, upper))
#         colour = (output / 10.0, 0.0, 0.0)
#         distances = cyclic_distances
#     elif str(file).find("DAG") >= 0:
#         identifier = '_Output_'
#         output = float(between(file, identifier, upper))
#         colour = (0.0, output / 10.0, 0.0)
#         distances = dag_distances
#     elif str(file).find("LtGr") >= 0:
#         distances = other_distances
#         output = float(between(file, "whitenoise_", "Report"))
#         colour = (output/20.0, 0.0, 0.0)
#         label = "neural"
#     else:
#         print("Did not find positive, neg, or output")
#     # So red = neg control, green = pos control, blue = test
#     # Now specify y axis variable depending on the plot type
#     if plot_type == 'normalised_scores':
#         y = normalise_score("{0}/{1}".format(current_directory, file))
#         print("got y")
#         # specify the axis label
#         y_label = 'Normalised BDe Score'
#     elif plot_type == 'consecutive_distances':
#         networks = get_bestnetworks("{0}/{1}".format(current_directory, file), type)
#         # length = float(len(networks))
#         # twodev.append(2.0/length)
#         y = consecutive_distances(networks)
#         distances.append(y)
#         y_label = 'Consecutive Hamming Distance'
#     elif plot_type == 'numlinks':
#         networks = get_bestnetworks("{0}/{1}".format(current_directory, file), type)
#         y = number_links(networks)
#         y_label = 'Number of Network Links'
#         # peaks.append(y[len(y)-1])
#         x = np.arange(0, len(y))
#         # id = get_repeat(file, label, upper)
#         # lengths[get_index(label)][id] = get_lengths(y)
#     # now plot
#     x = np.arange(0, len(y))
#     plt.plot(x, y, label='{0} {1}'.format(label, int(output)))
# plt.xlabel("Best network number")
# plt.ylabel(y_label)
# # Now create a legend with line colours not explicilty tied to the data
# # custom_lines = [Line2D([0], [0], color='b', lw=3),
# #                 Line2D([0], [0], color='g', lw=3),
# #                 Line2D([0], [0], color='r', lw=3)]
# # plt.legend(custom_lines, ['Sample', 'Positive Control', 'Negative Control'])
# plt.title("{0} during Dynamic Banjo Search \n on Zebrafinch Auditory Data".format(y_label))
# # plt.savefig("{0}/figures/{1} during Banjo Search on dynamic neural data".format(current_directory, y_label))
# plt.show()

# # Below a way to generate subplots for an entire directory:
# # Create a figure
# fig = plt.figure()
# gs = fig.add_gridspec(2,1)
# ax1 = fig.add_subplot(gs[0, 0])
# ax2 = fig.add_subplot(gs[1, 0])
# # ax3 = fig.add_subplot(gs[0:,2])
# # bit of styling:
# medianprops = dict(linestyle='-', linewidth=1.5, color='black')
# boxprops = dict(facecolor="#00BFC4")
# # and a list to store the hamming distance plots in
# distances = []
#
# for file in files:
#     # get the path to file and get the scores and best networks
#     path = f"{current_directory}/{file}"
#     scores = normalise_score(path)
#     nets = get_bestnetworks(path, type)
#     # Now get x axis
#     x = np.arange(0, len(scores))
#     # Now get the number of links and hamming distances
#     numlinks = number_links(nets)
#     hamming = consecutive_distances(nets)
#     distances.append(hamming)
#     # Now plot on each axis
#     ax1.plot(x, scores, color="#F8766D", lw=1)
#     ax2.plot(x, numlinks, color="#00aa00", lw=1)
#     ax2.plot(x, hamming, color="#348abdff", lw=1)
# # Now plot the P values
# P = get_roughs(distances)
# # ax3.boxplot(P, patch_artist=True, medianprops=medianprops, boxprops=boxprops)
# # Now set the labels
# ax1.set_ylabel("Normalised BDe Score", fontsize=10)
# ax2.set_ylabel("Number of Network Edges/\nConsecutive Hamming Distance", fontsize=10)
# # ax3.set_ylabel("P")
# # And create a legend
# custom_lines = [Line2D([0], [0], color='#00aa00', lw=2),
#                 Line2D([0], [0], color='#348abdff', lw=2)]
# ax2.legend(custom_lines, ['Number of Edges', 'Consecutive Hamming Distance'], fontsize=8)
# # Now set global text
# plt.xlabel("Best Network Number", fontsize=10)
# plt.suptitle(f"Topology of DBNI Search Space on Zebrafinch Auditory Data", fontsize=14, fontweight='bold')
# plt.show()



# # #
# # # # # now save the peaks as a numpy array
# # # # if plot_type == 'numlinks':
# # # #
# # # #     peak_totals = np.zeros(3, dtype=int)
# # # #     peak_totals[0] = len(set(sample_peaks))
# # # #     peak_totals[1] = len(set(pos_peaks))
# # # #     peak_totals[2] = len(set(neg_peaks))
# # # #
# # # #     # x = np.array(['Sample', 'Positive \n Control', 'Negative \n Control'])
# # # #     #     # for i in range(0, 3):
# # # #     #     #     sbn.catplot(x=x, y=lengths[i])
# # # #     plt.boxplot(lengths.transpose())
# # # #     plt.title("Curve Lengths {0}".format(output))
# # # #     plt.savefig("{0}/figures/Curve Lengths.png".format(current_directory, label))
# # # #     plt.show()
# # #
# # #
# # # Write code to get an idea of how this roughness metric behaves
# # # Generate constant lists of 0 and a rough curve
# #
# # # constant_length = []
# # # jagged_length = []
# # # for i in range(0, 10):
# # #     constant_length.append(get_length(list_of_string(0, 50)))
# # #     appendix = []
# # #     for j in range(0, 50):
# # #         appendix.append(get_jagged(j))
# # #     jagged_length.append(get_length(appendix))
# # #
# # # # Write code to output this to a csv file for R analysis
# # # out = pd.DataFrame()
# # # data = constant_length + jagged_length
# # # id = list_of_string("Constant", len(constant_length)) + list_of_string("Jagged", len(jagged_length))
# # # out["Metric"] = data
# # # out["id"] = id
# # # out.to_csv("C:/Users/User/Documents/GitHub/BL4201-SH-Project/TestDirectory/R/jaggedlength.csv")
# #
# Write code to output the roughness metric for a search instance
# sample_rough = get_roughs(sample_distances)
# pos_rough = get_roughs(pos_distances)
# # trun_rough = get_roughs(neg_distances)
# neg_rough = get_roughs(neg_distances)
#
# # cyclic_rough = get_roughs(cyclic_distances)
# # dag_rough = get_roughs(dag_distances)
#
# other_rough = get_roughs(other_distances)
#
# print("The output roughness is:")
# print(f"Sample Roughness = {sample_rough}")
# print(f"Pos Control Roughness = {pos_rough}")
# # # print(f"Truncated Sample Roughness = {trun_rough}")
# # print(f"Neg Control Roughness = {neg_rough}")
# # print(f"The neural roughness is {other_rough}")
# # print(f"The two dev is {twodev}")
# #
# # print(f"Cyclic Roughness = {cyclic_rough}")
# # print(f"DAG Roughness = {dag_rough}")
#
#
# # out_file = open(f"{current_directory}/roughness.txt", 'a')
# # # Now write to this file
# # out_file.write(f'{sample_roughness}\t{pos_roughness}\t{neg_roughness}\n')
# # out_file.close()
#
# # Assuming everything in the directory is from the same instance, can get the instance via
# file = files[0]
# #instance = int(between(file, '_in', 'Report'))
#
# # Write code to ouput the roughness of each run into a csv file for R analysis
# data = pd.DataFrame()
# # Now add two lists
# # One is a sample identifiers variable, and another is the value of curve roughness
# roughness = sample_rough + pos_rough + neg_rough
# identifiers = list_of_string("One Layer", len(sample_rough)) + list_of_string("Two Layers", len(pos_rough)) #+ list_of_string("Random Noise", len(neg_rough))
# data["Roughness"] = roughness
# data["id"] = identifiers
# # Check it looks correct
# print(data.head())
# # Now save to a csv file
# data.to_csv(f"{current_directory}/R/generegdynamicroughness_total.csv")

#
# Now input to it the