
#Intended as a cleaner copy of Network Generator.py

import random as rd
import numpy as np
from timeit import default_timer

# Initialise the function's time

start = default_timer()

# Intialise the random module

rd.seed(0)

# Now define a function that we will use to dictate the interactions between species
# Essentially, this function returns an interaction strength, drawn from a normal distribution where the mean is the
# interaction strength as declared by the Jacobian, and the standard deviation is some noise factor.

def noisy_interaction(interaction, abundance, noise_factor):
    abundance = float(abundance)
    interaction = float(interaction)
    sd = rd.random()*float(noise_factor)/(abundance + 1) # So noise inversely scales with system size, and system size > 0
    out = rd.normalvariate(interaction, sd)
    return out

# Define a function that we will use for node extinction
# i.e., returns the population if pop > 0, returns 0 else
def non_unitary_heaviside(x1, x2):
    if x1 == 0 or x1 < 0:
        out = x2
    else:
        out = x1
    return out

# Now define a function that bounds interactions by some global variable bound

pos_real_bound = 1000.0

def bound(x):
    if abs(x) <= pos_real_bound:
        return x
    elif x < pos_real_bound:
        return -1*pos_real_bound
    elif x > pos_real_bound:
        return pos_real_bound
# Above function returns x if abs(x) =< bound, and returns the bound else

def gene_time_series(genes, proteins):
    gene_to_protein = np.zeros((genes, proteins), dtype=float)
    for i in range(0, genes):
        for j in range(0, proteins):



