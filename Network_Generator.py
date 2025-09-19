
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

# Now create random networks with varying topologies:

# Create a class of networks where interactions can cause extinction between populations

# Define a function which generates guassian noise for species-species interactions, with noise scaling inversely proportional to the number of species i


def noisy_interaction(interaction, abundance, noise_factor):
    abundance = float(abundance)
    interaction = float(interaction)
    sd = float(noise_factor)/(abundance**2 + 1)
    out = rd.gauss(interaction, sd)
    return out

def non_unitary_heaviside(x1, x2=0.0):
    if x1 == 0 or x1 < 0:
        out = x2
    else:
        out = x1
    return out

# Create a function that bounds interactions by a number, for some global bound (positive int)
# Note this is necessary for us to represent the outputs in a decimal format readable by Banjo!


pos_real_bound = 100.0


def bound(x):
    if abs(x) <= pos_real_bound:
        return x
    elif x < pos_real_bound:
        return -1*pos_real_bound
    elif x > pos_real_bound:
        return pos_real_bound


# Create a function that returns a network structure for a given Jacobian matrix and outputs the result as an image file


def draw_network(jacobian, kind, nodes, links, noise, alpha, beta, instance):
       # print("drawing")
        g = nx.from_numpy_array(jacobian, create_using=nx.DiGraph())
        plt.figure()
        if kind == 'gene_reg':
            genes = np.arange(0, nodes)
            pos = nx.bipartite_layout(g, genes)
        else:
            pos = nx.circular_layout(g)
        nx.draw(g, with_labels=True, font_weight='bold', pos=pos) # is nx.circular_layout for alternative
        plt.suptitle("Network on {0} nodes, with {1} links".format(nodes, links))
        plt.title("Interactions a beta distribution where alpha = {0}, beta = {1}".format(alpha, beta))
        plt.savefig("Network structure with n{0} L{1} N{2} in{3}.png".format(nodes, links, noise, instance))
       # plt.show()

# Create a function for saving .txt files of arrays, to trim down the code
# Specify the decimal place precision to 6 using fmt = '%.6f'

def save_txt(array, kind, nodes, links, noise, iterations, instance, marker):
    file_network = np.savetxt(
        f"{marker} {kind} network with n{nodes} L{links} N{int(noise)} I{iterations} in{instance}.txt",
        array, fmt='%.6f', delimiter='\t')
    return file_network

# define a function that gets a 1xn array of random integers

def rand_int_array(arr, n):
    length = len(arr)
    for i in range(0, length):
        arr[i] = rd.randint(0, n+1)
    arr = np.array(arr, dtype=float)
    return arr
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
            network.edges[i, j]['weight'] = rd.betavariate(self.alpha, self.beta )*(-1)**(rd.choice((1, 2))) # Does this actually specify only values on the zeros?
        jacobian = nx.to_numpy_array(network, dtype=float)
        return jacobian

    # ***This function appears to work as intended, maybe need to alter the distribution of interaction strengths though***
    # Create a function that generates random bipartite graphs for given numbers of nodes and probability of forming a link,
    # which will be given by links/(nodes in one part * nodes in second part)

    def create_bipartite(self):
        # keep the number of 'proteins' higher than the number of 'genes' - will probably need to come back and change this to test later
        # should also test if one gene to disjoint groups of proteins affects inference
        # update 12/03/20 decided for computational ease to keep proteins = nodes, as not really interested in difference of size between two
        proteins = self.nodes
        # Cannot assign the exact number of links, rather give a probability of a link being formed
        p = self.links/(self.nodes*proteins)
        network = nx.bipartite.random_graph(self.nodes, int(proteins), p, directed=True)
        for (i, j) in network.edges():
            network.edges[i, j]['weight'] = rd.betavariate(self.alpha, self.beta )*(-1)**(rd.choice((1, 2)))
        # returns a square matrix
        jacobian = nx.to_numpy_array(network, dtype=float)
        return jacobian

    # Now create a function that creates random acylic directed graphs, and for comparison, one that generates cyclic graphs

    def get_dag(self):

        network = self.create_network()
        g = nx.from_numpy_array(network, create_using=nx.DiGraph())

        while not nx.is_directed_acyclic_graph(g): # so just checks all the generated graphs for if they're cyclic
            network = self.create_network()
            g = nx.from_numpy_array(network, create_using=nx.DiGraph())
        output = nx.to_numpy_array(g)

        return output
    
    # Below is an equivalent function that just finds the first cyclic graph on its run
    def get_cyclic(self):
        network = self.create_network()
        g = nx.from_numpy_array(network, create_using=nx.DiGraph())
        while nx.is_directed_acyclic_graph(g):
            network = self.create_network()
            g = nx.from_numpy_array(network, create_using=nx.DiGraph())
        output = nx.to_numpy_array(g)
        return output
    # Note that testing both functions above on 15 links, 6 nodes, they appear to generate non-equal networks
    def evolve_ecosystem(self):
        # Establish a network structure, and save this to an image file (.png)
        jacobian = self.create_network()
        kind = 'ecology'
        draw_network(jacobian, kind, self.nodes, self.links, self.noise, self.alpha, self.beta, self.instance)
        save_txt(jacobian, self.kind, self.nodes, self.links, self.noise, self.iterations, self.instance, 'network structure')
        # Now as we desire a number of replicate datasets for each network, place all of below beneath a ticker:
        if self.kind == 'static':
            for iterate in tqdm(range(0, self.number_replicates)):
                t = 0  # create time counter
                out = np.zeros((self.iterations, self.nodes))  # 10.0 being default initial state
                # And a positive control network too.
                control = np.zeros((self.iterations, self.nodes))  # 10.0 being default intial state
                # And a totally random, negative control
                neg_control = np.zeros((self.iterations, self.nodes))
                # Now as we desire a number of replicate datasets for each network, place all of below beneath a ticker:
                # Now evolve the system, for different kinds of network:
                while t < self.time:
                    t += 1
                    for i in range(0, self.iterations):
                        # generate random initial state, to test perturbations
                        out[i] = rand_int_array(out[i], 10)
                        control[i] = rand_int_array(control[i], 10)
                        for j in range(0, self.nodes):
                            neg_control[i, j] = rd.uniform(-1 * pos_real_bound, pos_real_bound + 1.0) # creates a random negative control network for testing
                            if out[i, j] == 0: # This keeps nodes extinct
                                break
                            else:
                                for k in range(0, self.nodes):
                                    out[i, j] = bound(non_unitary_heaviside(out[i, j] + noisy_interaction(jacobian[k, j], out[i, k], self.noise) * out[i, k]))  # heaviside function creates extinction
                                    control[i, j] = bound(control[i, j] + noisy_interaction(jacobian[k, j], out[i, k], self.noise)*out[i, k])
                # Now create .txt outputs for these networks.
                save_txt(out, self.kind, self.nodes, self.links, self.noise, self.iterations, self.instance, 'Extinction Network Output {0}'.format(iterate))
                save_txt(control, self.kind, self.nodes, self.links, self.noise, self.iterations, self.instance, 'Extinction Network Positive Control {0}'.format(iterate))
                save_txt(neg_control, self.kind, self.nodes, self.links, self.noise, self.iterations, self.instance, 'Extinction Network Neg Control {0}'.format(iterate))

        elif self.kind == 'dynamic':
            # Still generate a nodes*iterates sized array, but with time increasing as go down the file
            for iterate in tqdm(range(0, self.number_replicates)):
                out = np.full((self.iterations, self.nodes), 10.0)  # 10.0 being default initial state
                # And a positive control network too.
                control = np.full((self.iterations, self.nodes), 10.0)  # 10.0 being default intial state
                # And a totally random, negative control
                neg_control = np.zeros((self.iterations, self.nodes))
                # Now as we desire a number of replicate datasets for each network, place all of below beneath a ticker:
                # Now evolve the system, for different kinds of network:
                for i in range(1, self.iterations):
                    for j in range(0, self.nodes):
                        neg_control[i-1, j] = rd.uniform(-1 * pos_real_bound, pos_real_bound + 1.0) # creates a random negative control network for testing
                        if out[i-1, j] == 0: # This keeps nodes extinct
                            out[i, j] = 0
                        else:
                            sum_out = 0.0
                            for k in range(0, self.nodes): # cycle over all other species and add up interactions
                                sum_out += noisy_interaction(jacobian[k, j], out[i-1, k], self.noise) * out[i-1, k]
                                out[i, j] = bound(non_unitary_heaviside(out[i-1, j] + sum_out))  # heaviside function creates extinction
                                control[i, j] = bound(control[i-1, j] + sum_out)
                # Now create .txt outputs for these networks.
                save_txt(out, self.kind, self.nodes, self.links, self.noise, self.iterations, self.instance, 'Extinction Network Output {0}'.format(iterate))
                save_txt(control, self.kind, self.nodes, self.links, self.noise, self.iterations, self.instance, 'Extinction Network Positive Control {0}'.format(iterate))
                save_txt(neg_control, self.kind, self.nodes, self.links, self.noise, self.iterations, self.instance, 'Extinction Network Neg Control {0}'.format(iterate))
        else:
            print('ERROR: {} is not a valid kind of network!'.format(self.kind))

    def evolve_genesystem(self):
        jacobian = self.create_bipartite()
        draw_network(jacobian, self.kind, self.nodes, self.links, self.noise, self.alpha, self.beta, self.instance)
        save_txt(jacobian, self.kind, self.nodes, self.links, self.noise, self.iterations, self.instance,
                 'gene network structure')
        # need to get number of proteins for data saving
        # Lazy method for now, as the number of proteins >= number of genes
        # update 12/03/20 setting proteins = nodes
        proteins = self.nodes
        if self.kind == "static":
            for iterate in tqdm(range(0, self.number_replicates)):
                # Generate a positive control network where the nodes are observed and a test sample where they are not
                out_pos_control = np.full((self.iterations, self.nodes+proteins), 10.0, dtype=float)
                # In this array, the first n entries along the horizontal represent gene levels, and the next m represent protein levels
                out = np.full((self.iterations, self.nodes), 10.0, dtype=float)
                # Similarly, create a data set for only the n genes
                # Now fill these in with a simple dynamical model
                t = 0       # create time counter
                while t < self.time:
                    t += 1
                    for i in tqdm(range(0, self.iterations)):
                        for j in range(0, self.nodes):  # fill up the output array
                            for k in range(0, proteins):
                                out[i, j] += noisy_interaction(jacobian[k, j], out_pos_control[i, self.nodes+k], self.noise)*out_pos_control[i, self.nodes+k]
                                out[i ,j] = bound(out[i ,j])
                                # (assume for convention that the kjth entry represents k -> j)
                                out_pos_control[i, j] += noisy_interaction(jacobian[k, j], out_pos_control[i, self.nodes+k], self.noise)*out_pos_control[i, self.nodes+k]
                                out_pos_control[i, self.nodes+k] += noisy_interaction(jacobian[j, k], out_pos_control[i, j], self.noise)*out_pos_control[i, j]
                                out_pos_control[i ,j] = bound(out_pos_control[i ,j])
                                out_pos_control[i, self.nodes+k] = bound(out_pos_control[i, self.nodes+k])
                                # so interactions are going gene -> protein and protein -> gene but never gene -> gene or protein -> protein
                # Now save
                save_txt(out, self.kind, self.nodes, self.links, self.noise, self.iterations, self.instance,
                         'Gene Network Output {0}'.format(iterate))
                save_txt(out_pos_control, self.kind, self.nodes, self.links, self.noise, self.iterations, self.instance,
                         'Gene Network Positive Control {0}'.format(iterate))
        # Now define the dynamic equivalent
        elif self.kind == "dynamic":
            for iterate in tqdm(range(0, self.number_replicates)):
                # Generate a positive control network where the nodes are observed and a test sample where they are not
                out_pos_control = np.full((self.iterations, self.nodes+proteins), 10.0, dtype=float)
                # In this array, the first n entries along the horizontal represent gene levels, and the next m represent protein levels
                out = np.full((self.iterations, self.nodes), 10.0, dtype=float)
                # Similarly, create a data set for only the n genes
                # Now fill these in with a simple dynamical model
                for i in (range(0, self.iterations)):
                    for j in range(0, self.nodes):  # fill up the output array
                        for k in range(0, proteins):
                            out[i, j] += noisy_interaction(jacobian[k, j], out_pos_control[i, self.nodes+k], self.noise)*out_pos_control[i, self.nodes+k]
                            out[i ,j] = bound(out[i ,j])
                            # (assume for convention that the kjth entry represents k -> j)
                            out_pos_control[i, j] += noisy_interaction(jacobian[k, j], out_pos_control[i, self.nodes+k], self.noise)*out_pos_control[i, self.nodes+k]
                            out_pos_control[i, self.nodes+k] += noisy_interaction(jacobian[j, k], out_pos_control[i, j], self.noise)*out_pos_control[i, j]
                            out_pos_control[i ,j] = bound(out_pos_control[i ,j])
                            out_pos_control[i, self.nodes+k] = bound(out_pos_control[i, self.nodes+k])
                            # so interactions are going gene -> protein and protein -> gene but never gene -> gene or protein -> protein
                # Now save
                save_txt(out, self.kind, self.nodes, self.links, self.noise, self.iterations, self.instance,
                         'Gene Network Output {0}'.format(iterate))
                save_txt(out_pos_control, self.kind, self.nodes, self.links, self.noise, self.iterations, self.instance,
                         'Gene Network Positive Control {0}'.format(iterate))

    def evolve_DAG(self):
        dag_jacobian = self.get_dag()
        cyclic_jacobian = self.get_cyclic()
        # Seeing as I have a better drawing function locally than here, will just save as txt for now
        save_txt(dag_jacobian, self.kind, self.nodes, self.links, self.noise, self.iterations, self.instance,
                 'Acyclic structure')
        save_txt(cyclic_jacobian, self.kind, self.nodes, self.links, self.noise, self.iterations, self.instance,
                 'Cyclic structure')
        # Now evolve, for static case
        if self.kind == 'static':
            for iterate in tqdm(range(0, self.number_replicates)):
                t = 0  # create time counter
                dag = np.full((self.iterations, self.nodes), 10.0)  # 10.0 being default initial state
                # And a positive control network too.
                cyclic = np.full((self.iterations, self.nodes), 10.0)  # 10.0 being default intial state
                # And a totally random, negative control
                # Now as we desire a number of replicate datasets for each network, place all of below beneath a ticker:
                # Now evolve the system, for different kinds of network:
                while t < self.time:
                    t += 1
                    for i in range(0, self.iterations):
                        for j in range(0, self.nodes):
                            for k in range(0, self.nodes):
                                dag[i, j] = bound(dag[i, j] + noisy_interaction(dag_jacobian[k, j], dag[i, k], self.noise) * dag[i, k]) # heaviside function creates extinction
                                cyclic[i, j] = bound(cyclic[i, j] + noisy_interaction(cyclic_jacobian[k, j], cyclic[i, k], self.noise)*cyclic[i, k])
                # Now create .txt outputs for these networks.
                save_txt(dag, self.kind, self.nodes, self.links, self.noise, self.iterations, self.instance, 'DAG Network Output {0}'.format(iterate))
                save_txt(cyclic, self.kind, self.nodes, self.links, self.noise, self.iterations, self.instance, 'Cyclic Network Output {0}'.format(iterate))
        elif self.kind == 'dynamic':
            # exactly the same as above, but save as you go
            for iterate in tqdm(range(0, self.number_replicates)):
                dag = np.full((self.iterations, self.nodes), 10.0)  # 10.0 being default initial state
                # And a positive control network too.
                cyclic = np.full((self.iterations, self.nodes), 10.0)  # 10.0 being default intial state
                # Now as we desire a number of replicate datasets for each network, place all of below beneath a ticker:
                # Now evolve the system, for different kinds of network:
                for i in range(0, self.iterations):
                    # create a variable 'checker' which will record which nodes are equal
                    checker = 0
                    for j in range(0, self.nodes):
                        for k in range(0, self.nodes):
                            dag[i, j] = bound(dag[i, j] + noisy_interaction(dag_jacobian[k, j], dag[i, k], self.noise) * dag[i, k])
                            cyclic[i, j] = bound(cyclic[i, j] + noisy_interaction(cyclic_jacobian[k, j], cyclic[i, k], self.noise)*cyclic[i, k])
                # Now save
                # Note we don't check for successive equality here, as noise makes this impossible
                save_txt(dag, self.kind, self.nodes, self.links, self.noise, self.iterations, self.instance,
                         'DAG Network Output {0}'.format(iterate))
                save_txt(cyclic, self.kind, self.nodes, self.links, self.noise, self.iterations, self.instance,
                         'Cyclic Network Output {0}'.format(iterate))


# Now generate data for analysis


number_networks = 10
for i in range(0, number_networks):
    #print("{0}%".format(100.0*i/float(number_networks)))
    out = ExtinctionNetwork('dynamic', 6, 15, 10.0, 1000, i)
    out.evolve_genesystem()
# end = default_timer()
# print("----%s---- " %(end - start))
