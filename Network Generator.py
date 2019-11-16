
#
import random as rd
import numpy as np
import scipy as sc # Need scipy for the random graph generation
import matplotlib.pyplot as plt
import networkx as nx

rd.seed(0)

# Now create random networks with varying topologies:

# Create a class of networks where interactions can cause extinction between populations

# Define a function which generates guassian noise for species-species interactions, with noise scaling inversely proportional to the number of species i


def noisy_interaction(interaction, abundance, noise_factor):
    abundance = float(abundance)
    interaction = float(interaction)
    sd = rd.random()*float(noise_factor)/(abundance + 1)
    out = rd.normalvariate(interaction, sd)
    return out


def non_unitary_heaviside(x1, x2):
    if x1 == 0 or x1 < 0:
        out = x2
    else:
        out = x1
    return out


class InteractionNetwork:

    def __init__(self, nodes, links, alpha, beta):
        self.nodes = nodes
        self.links = links
        self.alpha = alpha
        self.beta = beta

    def create_network(self):
        # Generate a random graph with n nodes and m directed edges:
        network = nx.gnm_random_graph(self.nodes, self.links, directed=True)
        for (i, j) in network.edges():
            network.edges[i, j]['weight'] = rd.betavariate(self.alpha, self.beta)*(-1)**(rd.choice((1, 2, 4, 6)))
        jacobian = nx.to_numpy_matrix(network, dtype=float)
        return jacobian

    # ***This function appears to work as intended, maybe need to alter the distribution of interaction strengths though***
    # Now create function for visualising the network structure
    def true_network(self):
        jacobian = self.create_network()
        g = nx.from_numpy_matrix(jacobian, create_using=nx.DiGraph())
        plt.subplot(111)
        nx.draw(g, with_labels=True, font_weight='bold', pos=nx.circular_layout(g))
        plt.suptitle("Network on {0} nodes, with {1} links".format(self.nodes, self.links))
        plt.title("With interactions given by a beta distribution where alpha = {0}, beta = {1}".format(self.alpha, self.beta))
        plt.show()


    # Now create class of network equillibrium states, for given network types.


class EvolvedNetwork:

    # set network parameters as input parameters

    def __init__(self, kind, nodes, links, noise, iterations, time=1000, alpha=5.0, beta=5.0):
        self.kind = kind   # kind of network - e.g. one with layers, one with extinction, etc.
        self.nodes = nodes  # number of network nodes
        self.links = links  # number of network edges
        self.noise = noise  # noise of interactions
        self.iterations = iterations    # number of replicate samples
        self.alpha = alpha    # describes network weight distribution
        self.beta = beta      # describes network weight distribution
        self.time = time      # length of system evolution

    def evolve_system(self):
        network = InteractionNetwork(self.nodes, self.links, self.alpha, self.beta)
        jacobian = network.create_network()
        # Create a 'population' vector describing the population at time t, which will be our output
        out = np.zeros((self.nodes, self.iterations), dtype=float)
        # Create an initial state for this population
        for i in range(0, self.nodes):
            for j in range(0, self.iterations):
                out[i, j] = 10.0  # Set to an arbitrary constant value for now, can change later, just want reproducibility
        t = 0  # create time counter
        # Now evolve the system, for different kinds of network:
        if self.kind == 'extinction':
            while t < self.time:
                t = t + 1
                for i in range(0, self.nodes):
                    for j in range(0, self.iterations):
                        if out[i, j] == 0:  # This keeps nodes extinct
                            break
                        else:
                            for k in range(0, self.nodes):
                                out[i, j] = non_unitary_heaviside(out[i, j] + noisy_interaction(jacobian[k, i], out[k, j], self.noise)*out[k, j], 0.0) # heaviside function creates extinction
        else:
            print('ERROR: {} is not a valid kind of network!'.format(self.kind))
        return out

    def output_file(self):
        file = open("{0} network n{1} L{2} N{3} I{4}.txt".format(self.kind, self.nodes, self.links, self.noise, self.iterations), 'w+')
        file.write(self.evolve_system())
        return file

out = EvolvedNetwork('extinction', 5, 10, 4.0, 10)
print(out.evolve_system())
print(out.output_file())

