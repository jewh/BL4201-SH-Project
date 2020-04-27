
# python file to get q3 discretisation for each file

import numpy as np
import os as os
from os.path import isfile, join
import pandas as pd

# define a function that discretises


def discretise(number, q2, q3):
    if number <= q2:
        return 0
    elif number <= q3:
        return 1
    else:
        return 2

# now that'll discretise a file


def q3_discretise(input):
    arr = np.loadtxt(input)
    # get the quantiles
    q2 = np.quantile(arr, float(1/3))
    q3 = np.quantile(arr, float(2/3))
    # get the dimensions of the input
    dims = np.shape(arr)
    output = np.zeros(dims, dtype=int)
    for i in range(0, dims[0]):
        for j in range(0, dims[1]):
            output[i][j] = discretise(arr[i][j], q2, q3)
    return output

# create a function that loops over an array and counts the number of each instance

def count_instance(arr, inst):
    dims = np.shape(arr)
    counter = 0
    for i in range(0, dims[0]):
        if arr[i] == inst:
            counter += 1
    return counter


# define a function to make a list of a certain string

def list_of_string(string, n):
    output = []
    for i in range(0, n):
        output.append(string)
    return output


# And a function that gets strings between a string
def between(string, string1, string2):
    start = string.index(string1)
    end = string.index(string2)
    if start <= end:
        out = float(string[start+len(string1):end])
    else:
        out = float(string[end+len(string2):start])
    return out

# define a function to get our summary metric for a file
def summary(list_of_metrics):
    # calculate the mean to get the deviation
    mean = np.mean(list_of_metrics)
    # initialise the output
    out = 0.0
    for metric in list_of_metrics:
        out += abs(metric - mean)
    return out




current_directory = os.path.dirname(os.path.realpath(__file__))

# # get list of files in local directory
# files = [f for f in os.listdir(current_directory) if isfile(join(current_directory, f))]
# # Now remove all files not with .txt extension
# for item in files:
#     if item.endswith(".txt"):
#         pass
#     else:
#         files.remove(item)
#
# # Now loop over the files, discretise each, and count the number of discretised states in each column for each
# # make a list to store the data frames
#
# data = []
# metrics = []
#
# for file in files:
#
#     q3_arr = q3_discretise(f"{current_directory}/{file}")
#
#     # transpose so we can get the columns easily
#     q3_arr = np.transpose(q3_arr)
#     # Now count the instances of each in each column
#     nodes = np.shape(q3_arr)[0]
#
#     count = np.zeros((nodes, 3), dtype=int)
#     # initialise a variable so we can get the total number of counts for the file
#     total = 0
#     for i in range(0, nodes):
#         for j in range(0, 3):
#             count[i][j] = count_instance(q3_arr[i], j)
#             total += count_instance(q3_arr[i], j)
#     # reset to float for later calculation
#     total = float(total)
#     # get so it's the number of observations per node
#     total = total/float(nodes)
#     # This should return the number of each state per node
#     # Now add to a dataframe for csv export of node instances
#     df = pd.DataFrame()
#     # And another for storing metric scores
#     score = pd.DataFrame()
#     # Add a list to store the metric score per node in
#     metric_list = []
#     # Get a list for the number of each state for each node, and for each node
#     node = []
#     node_counts = []
#     state = []
#     # Now fill them up
#     for i in range(0, nodes):
#         # get the number of each state for this node
#         num_0 = count[i][0]
#         num_1 = count[i][1]
#         num_2 = count[i][2]
#         state += [0, 1, 2]
#         node_counts += [num_0, num_1, num_2]
#         node += list_of_string(i, 3)
#         # calculate a score for our metric
#         # first calculate the fraction of the total dataset that each num_i represents
#         num_0 = float(num_0)
#         num_1 = float(num_1)
#         num_2 = float(num_2)
#         fraction_0 = num_0/total
#         fraction_1 = num_1/total
#         fraction_2 = num_2/total
#         # Now update the metric
#         metric = abs(fraction_0 - 1.0/3.0) + abs(fraction_1 - 1.0/3.0) + abs(fraction_2 - 1.0/3.0)
#         metric = metric/3.0
#         metric_list.append(metric)
#     # Calculate the file's summary metric score
#     summary_score = np.mean(metric_list)
#     # Now put these numbers into df
#     df["node"] = node
#     df["state"] = state
#     df["count"] = node_counts
#     # Now so we can identify the file, get the identifier and the instance
#
#     # Now associate each file with an identifier
#     if str(file).find("Output") >= 0:
#         label = "Extinction"
#     elif str(file).find("Positive") >= 0:
#         label = "No Extinction"
#     elif str(file).find("Neg") >= 0:
#         label = "Random Noise"
#     else:
#         print("You won't find a label here")
#     # Get the length of the data frame
#     length_of_df = len(node)
#     df["id"] = list_of_string(label, length_of_df)
#     # Now get the instance, which is the network number
#     instance = int(between(file, ' in', '.txt'))
#     df["Network"] = list_of_string(instance, length_of_df)
#     # Now append to the overall list
#     data.append(df)
#
#     # Additionally, we need to add the file's metric score to the overall scores list
#     initial_state = "constant"
#     # so similarly add:
#     score["id"] = [label]
#     score["Network"] = [instance]
#     score["score"] = [summary_score]
#     score["initial_state"] = [initial_state]
#     metrics.append(score)
#
# # Now once out of the for loop, we combine all the dataframes into one dataframe
#
# counts = pd.concat(data)
# scores = pd.concat(metrics)
#
# # check it looks correct by heading it
#
# print(counts.head())
#
# # Now save to a .csv file for R analysis
#
# # counts.to_csv(f"{current_directory}/R/q3_distribution_non_perturbed_extinction_networks.csv")
# print("State distribution saved")
# print("_________________________________________________________________________________________________________________")
#
# print(scores.head())
# scores.to_csv(f"{current_directory}/R/q3_score_mean_non_perturbed_extinction_networks.csv")
# print("Scores saved")

# comment out either above or below, below allows analysis of specific file(s)

# data_directory = r"C:\Users\User\Documents\GitHub\BL4201-SH-Project\RISData"
# items = [f"{data_directory}/RISdata.txt", f"{data_directory}/RISgrassdata.txt"]
# length = len(items)
# labels = ["heather", "grass"]
# node_states = []
# scores = []
#
# for i in range(0, length):
#     q3_arr = q3_discretise(items[i])
#     label = labels[i]
#     # transpose so we can get the columns easily
#     q3_arr = np.transpose(q3_arr)
#     # Now count the instances of each in each column
#     nodes = np.shape(q3_arr)[0]
#
#     count = np.zeros((nodes, 3), dtype=int)
#     # initialise a variable so we can get the total number of counts for the file
#     total = 0
#     for i in range(0, nodes):
#         for j in range(0, 3):
#             count[i][j] = count_instance(q3_arr[i], j)
#             total += count_instance(q3_arr[i], j)
#     # reset to float for later calculation
#     total = float(total)
#     # get so it's the number of observations per node
#     total = total/float(nodes)
#     # This should return the number of each state per node
#     # Now add to a dataframe for csv export of node instances
#     df = pd.DataFrame()
#     # And another for storing metric scores
#     score = pd.DataFrame()
#     # Add a list to store the metric score per node in
#     metric_list = []
#     # Get a list for the number of each state for each node, and for each node
#     node = []
#     node_counts = []
#     state = []
#     id = []
#     # Now fill them up
#     for i in range(0, nodes):
#         # get the number of each state for this node
#         num_0 = count[i][0]
#         num_1 = count[i][1]
#         num_2 = count[i][2]
#         state += [0, 1, 2]
#         node_counts += [num_0, num_1, num_2]
#         node += list_of_string(i, 3)
#         # calculate a score for our metric
#         # first calculate the fraction of the total dataset that each num_i represents
#         num_0 = float(num_0)
#         num_1 = float(num_1)
#         num_2 = float(num_2)
#         fraction_0 = num_0/total
#         fraction_1 = num_1/total
#         fraction_2 = num_2/total
#         # Now update the metric
#         metric = abs(fraction_0 - 1.0/3.0) + abs(fraction_1 - 1.0/3.0) + abs(fraction_2 - 1.0/3.0)
#         metric = metric/3.0
#         metric_list.append(metric)
#
#         id += list_of_string(label, 3)
#     # Calculate the file's summary metric score
#     summary_score = np.mean(metric_list)
#     # put the nodes into a
#     df["node"] = node
#     df["state"] = state
#     df["count"] = node_counts
#     # associate the file with a label
#     df["id"] = id
#     node_states.append(df)
#
#     score["id"] = [label]
#     score["score"] = [summary_score]
#     scores.append(score)
#
# # now assemble and save
# counts = pd.concat(node_states)
# scores = pd.concat(scores)
#
# print(counts.head())
# print(scores.head())
#
# counts.to_csv(f"{current_directory}/R/q3_states_RISdata.csv")
# scores.to_csv(f"{current_directory}/R/q3_score_mean_RISdata.csv")