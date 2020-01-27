
import random as rd
import numpy as np
import scipy as sc # Need scipy for the random graph generation
import matplotlib.pyplot as plt
import networkx as nx
from timeit import default_timer

# Time to execute given by:

start = default_timer()

rd.seed(0)

# Now create random networks with varying topologies:

# Create a class of networks where interactions can cause extinction between populations

# Define a function which generates guassian noise for species-species interactions, with noise scaling inversely proportional to the number of species i


def noisy_interaction(interaction, abundance, noise_factor):
    abundance = float(abundance)
    interaction = float(interaction)
    sd = rd.random( ) *float(noise_factor ) /(abundance + 1)
    out = rd.normalvariate(interaction, sd)
    return out

def non_unitary_heaviside(x1, x2):
    if x1 == 0 or x1 < 0:
        out = x2
    else:
        out = x1
    return out

# Create a function that bounds interactions by a number, for some global bound (positive int)
# Note this is necessary for us to represent the outputs in a decimal format readable by Banjo!


pos_real_bound = 1000.0


def bound(x):
    if abs(x) < pos_real_bound:
        return x
    elif x < pos_real_bound:
        return - 1 *pos_real_bound
    elif x > pos_real_bound:
        return pos_real_bound

# Create a function that returns a network structure for a given Jacobian matrix and outputs the result as an image file


def draw_network(jacobian, nodes, links, noise, alpha, beta, instance):
        g = nx.from_numpy_matrix(jacobian, create_using=nx.DiGraph())
        plt.figure()
        nx.draw(g, with_labels=True, font_weight='bold', pos=nx.circular_layout(g))
        plt.suptitle("Network on {0} nodes, with {1} links".format(nodes, links))
        plt.title("Interactions a beta distribution where alpha = {0}, beta = {1}".format(alpha, beta))
        plt.savefig("Network structure with n{0} L{1} N{2} in{3}.png".format(nodes, links, noise, instance))

# Create a function for saving .txt files of arrays, to trim down the code
# Specify the decimal place precision to 6 using fmt = '%.6f'

def save_txt(array, kind, nodes, links, noise, iterations, instance, marker):
    file_network = np.savetxt(
        f"{marker} {kind} network with n{nodes} L{links} N{int(noise)} I{iterations} in{instance}.txt",
        array, fmt='%.6f', delimiter='\t')
    return file_network


# Now create class of network equillibrium states, for given network types.


class ExtinctionNetwork:

    # set network parameters as input parameters

    def __init__(self, kind, nodes, links, noise, iterations, instance, time=1000, alpha=5.0, beta=5.0, number_replicates=10):
        self.kind = kind   # kind of data - time-series or static.
        self.nodes = nodes  # number of network nodes
        self.links = links  # number of network edges
        self.noise = noise  # noise of interactions
        self.iterations = iterations    # number of replicate samples
        self.alpha = alpha    # describes network weight distribution
        self.beta = beta      # describes network weight distribution
        self.time = time      # length of system evolution
        self.instance = instance # creates different instances, hopefully to create different networks for same parameters
        self.number_replicates = number_replicates # Number of replicate datasets generated for a given network structure

    def create_network(self):
        # Generate a random graph with n nodes and m directed edges:
        network = nx.gnm_random_graph(self.nodes, self.links, directed=True)
        for (i, j) in network.edges():
            network.edges[i, j]['weight'] = rd.betavariate(self.alpha, self.beta ) *(-1 )**(rd.choice((1, 2)))
        jacobian = nx.to_numpy_matrix(network, dtype=float)
        return jacobian

    # ***This function appears to work as intended, maybe need to alter the distribution of interaction strengths though***
    # Now create function for visualising the network structure


    def evolve_system(self):
        # Establish a network structure, and save this to an image file (.png)
        jacobian = self.create_network()
        draw_network(jacobian, self.nodes, self.links, self.noise, self.alpha, self.beta, self.instance)
        save_txt(jacobian, self.kind, self.nodes, self.links, self.noise, self.iterations, self.instance, 'network structure')
        # Create a 'population' vector describing the population at time t, which will be our output
        out = np.zeros((self.iterations, self.nodes))
        # And a positive control network too.
        control = np.zeros((self.iterations, self.nodes))
        # And a totally random, negative control
        neg_control = np.zeros((self.iterations, self.nodes))
        # Now as we desire a number of replicate datasets for each network, place all of below beneath a ticker:
        for iterate in range(0, self.number_replicates):
            # Create an initial state for this population
            for i in range(0, self.iterations):
                for j in range(0, self.nodes):
                    out[i, j] = 10.0  # Set to an arbitrary constant value for now, can change later, just want reproducibility
                    control[i, j] = 10.0
            t = 0  # create time counter
            # Now evolve the system, for different kinds of network:
            if self.kind == 'static':
                while t < self.time:
                    t = t + 1
                    for i in range(0, self.iterations):
                        for j in range(0, self.nodes):
                            neg_control[i, j] = rd.uniform(-1 * pos_real_bound, pos_real_bound + 1.0) # creates a random negative control network for testing
                            if out[i, j] == 0:
                                break # This keeps nodes extinct
                            else:
                                for k in range(0, self.nodes):
                                    out[i, j] = bound(non_unitary_heaviside
                                        (out[i, j] + noisy_interaction(jacobian[k, j], out[i, k], self.noise ) * out[i, k], 0.0)) # heaviside function creates extinction
                                    out[i, j] = out[i, j]
                                    control[i, j] = bound \
                                        (control[i, j] + noisy_interaction(jacobian[k, j], out[i, k], self.noise) * out
                                            [i, k])
                                    control[i, j] = control[i, j]
                # Now create .txt outputs for these networks.
                save_txt(out, self.kind, self.nodes, self.links, self.noise, self.iterations, self.instance, 'Extinction Network Output {0}'.format(iterate))
                save_txt(control, self.kind, self.nodes, self.links, self.noise, self.iterations, self.instance, 'Extinction Network Positive Control {0}'.format(iterate))
                save_txt(neg_control, self.kind, self.nodes, self.links, self.noise, self.iterations, self.instance, 'Extinction Network Neg Control {0}'.format(iterate))

        else:
            print('ERROR: {} is not a valid kind of network!'.format(self.kind))






# Now generate data for analysis

number_networks = 10
for i in range(0, number_networks):
    print("{0}%".format(100.0*i/float(number_networks)))
    out = ExtinctionNetwork('static', 6, 15, 4.0, 1000, i)
    out.evolve_system()
end = default_timer()
print("----%s---- " %(end - start))
