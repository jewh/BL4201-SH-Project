
import random as rd
import numpy as np
import scipy as sc # Need scipy for the random graph generation
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
from timeit import default_timer
from tqdm import tqdm

# Time to execute given by:

# start = default_timer()

rd.seed(0)

nodes = 6
links = 15
alpha = 5.0
beta = 5.0


def create_network(nodes, links, alpha, beta):
    # Generate a random graph with n nodes and m directed edges:
    network = nx.gnm_random_graph(nodes, links, directed=True)
    for (i, j) in network.edges():
        network.edges[i, j]['weight'] = rd.betavariate(alpha, beta) * (-1) ** (
            rd.choice((1, 2)))  # Does this actually specify only values on the zeros?
    jacobian = nx.to_numpy_matrix(network, dtype=float)
    return jacobian

def get_dag(nodes, links, alpha, beta):
    network = create_network(nodes, links, alpha, beta)
    g = nx.from_numpy_matrix(network, create_using=nx.DiGraph())
    total = 1
    while not nx.is_directed_acyclic_graph(g):
        total += 1
        network = create_network(nodes, links, alpha, beta)
        g = nx.from_numpy_matrix(network, create_using=nx.DiGraph())
    output = nx.to_numpy_matrix(g)
    print(f"{total} networks checked")
    unchecked = 3.0**(float(nodes)*float((nodes-1))/2.0) - total
    print(f"{unchecked} networks unchecked")
    return output

def get_cyclic(nodes, links, alpha, beta):
    network = create_network(nodes, links, alpha, beta)
    g = nx.from_numpy_matrix(network, create_using=nx.DiGraph())
    total = 1
    while nx.is_directed_acyclic_graph(g):
        total += 1
        network = create_network(nodes, links, alpha, beta)
        g = nx.from_numpy_matrix(network, create_using=nx.DiGraph())
    output = nx.to_numpy_matrix(g)
    print(f"{total} networks checked")
    unchecked = 3.0 ** (float(nodes) * float((nodes - 1)) / 2.0) - total
    print(f"{unchecked} networks unchecked")
    return output

def draw_network(jacobian, nodes, links, alpha, beta, i):
    # print("drawing")
    g = nx.from_numpy_matrix(jacobian, create_using=nx.DiGraph())
    plt.figure()
    pos = nx.circular_layout(g)
    nx.draw(g, with_labels=True, font_weight='bold', pos=pos)  # is nx.circular_layout for alternative
    plt.suptitle("Network on {0} nodes, with {1} links".format(nodes, links))
    plt.title("Interactions a beta distribution where alpha = {0}, beta = {1}".format(alpha, beta))
    plt.show()

for i in range(0, 3):
    draw_network(get_cyclic(nodes, links, alpha, beta), nodes, links, alpha, beta, i)

