
#
import random as rd
import numpy as np
import scipy as sc # Need scipy for the random graph generation
import matplotlib.pyplot as plt
import networkx as nx
from timeit import default_timer

#Time to execute given by:

start = default_timer()

rd.seed(0)

# Now create random networks with varying topologies:

# Create a class of networks where interactions can cause extinction between populations

# Define a function which generates guassian noise for species-species interactions, with noise scaling inversely proportional to the number of species i


def noisy_interaction(interaction, abundance, noise_factor):
    abundance = float(abundance)
    interaction = float(interaction)
    sd = rd.random()*float(noise_factor)/(abundance + 1)
    out = int(rd.normalvariate(interaction, sd))
    return out


def non_unitary_heaviside(x1, x2):
    if x1 == 0 or x1 < 0:
        out = x2
    else:
        out = x1
    return out

#Create a function that bounds interactions by a number, for some global bound (positive int)

pos_real_bound = 1000.0

def bound(x):
    if abs(x) < pos_real_bound:
        return x
    elif x < pos_real_bound:
        return -1*pos_real_bound
    elif x > pos_real_bound:
        return pos_real_bound



# Now create class of network equillibrium states, for given network types.


class EvolvedNetwork:

    # set network parameters as input parameters

    def __init__(self, kind, nodes, links, noise, iterations, instance, time=1000, alpha=5.0, beta=5.0):
        self.kind = kind   # kind of network - e.g. one with layers, one with extinction, etc.
        self.nodes = nodes  # number of network nodes
        self.links = links  # number of network edges
        self.noise = noise  # noise of interactions
        self.iterations = iterations    # number of replicate samples
        self.alpha = alpha    # describes network weight distribution
        self.beta = beta      # describes network weight distribution
        self.time = time      # length of system evolution
        self.instance = instance # creates different instances, hopefully to create different networks for same parameters

    def create_network(self):
        # Generate a random graph with n nodes and m directed edges:
        network = nx.gnm_random_graph(self.nodes, self.links, directed=True)
        for (i, j) in network.edges():
            network.edges[i, j]['weight'] = rd.betavariate(self.alpha, self.beta)*(-1)**(rd.choice((1, 2)))
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

    def evolve_system(self):
        jacobian = self.create_network()
        # Create a 'population' vector describing the population at time t, which will be our output
        out = np.zeros((self.nodes, self.iterations), dtype=int)
        # And a positive control network too.
        control = np.zeros((self.nodes, self.iterations), dtype=int)
        # And a totally random, negative control
        neg_control = np.zeros((self.nodes, self.iterations), dtype=int)
        # Create an initial state for this population
        for i in range(0, self.nodes):
            for j in range(0, self.iterations):
                out[i, j] = 10.0  # Set to an arbitrary constant value for now, can change later, just want reproducibility
                control[i, j] = 10.0
        t = 0  # create time counter
        # Now evolve the system, for different kinds of network:
        if self.kind == 'extinction':
            while t < self.time:
                t = t + 1
                for i in range(0, self.nodes):
                    for j in range(0, self.iterations):
                        neg_control[i, j] = rd.uniform(-1*pos_real_bound, pos_real_bound+1.0) # creates a random negative control network for testing
                        if out[i, j] == 0:  # This keeps nodes extinct
                            break
                        else:
                            for k in range(0, self.nodes):
                                out[i, j] = bound(non_unitary_heaviside(out[i, j] + noisy_interaction(jacobian[k, i], out[k, j], self.noise)*out[k, j], 0.0)) # heaviside function creates extinction
                                control[i, j] = bound(control[i, j] + noisy_interaction(jacobian[k, i], out[k, j], self.noise) * out[k, j])
                                print("------------it works fam-------------")
            # Now create .txt outputs for these networks.
            file_network = np.savetxt(
                f"Outputs - Extinction Networks\{self.kind} network structure with n{self.nodes} L{self.links} N{self.noise} I{self.iterations} in{self.instance}.txt",
                jacobian)
            file_out = np.savetxt(
                f"Outputs - Extinction Networks\{self.kind} network n{self.nodes} L{self.links} N{self.noise} I{self.iterations} in{self.instance}.txt", out)
            file_control = np.savetxt(
                f"Outputs - Extinction Networks\control {self.kind} network n{self.nodes} L{self.links} N{self.noise} I{self.iterations} in{self.instance}.txt",
                control)
            file_neg_control = np.savetxt(
                f"Outputs - Extinction Networks\ negative control {self.kind} network n{self.nodes} L{self.links} N{self.noise} I{self.iterations} in{self.instance}.txt",
                neg_control)
        else:
            print('ERROR: {} is not a valid kind of network!'.format(self.kind))


#Now generate data for analysis
number_networks = 10
for i in range(0, number_networks):
    out = EvolvedNetwork('extinction', 6, 15, 4.0, 1000, i)
    out.evolve_system()
    print("#")
end = default_timer()
print("----%s----"%(end-start))







