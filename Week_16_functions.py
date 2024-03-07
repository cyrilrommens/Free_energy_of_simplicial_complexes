# Define functions to be used in week_16

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import genpareto
from collections import defaultdict
from itertools import combinations


# Define internal energy function
def energy_function(p, A):
    return np.sum(A * np.outer(p, p))

# Define shannon entropy function
def shannon_entropy(probabilities):
    # Remove any zero probabilities to avoid log(0) issues
    probabilities = probabilities[probabilities != 0]
    return -np.sum(probabilities * np.log2(probabilities))

# Define free energy function
def free_energy_function(p, inverse_matrix, temperature):
    # Remove any zero probabilities to avoid log(0) issues
    p = p[p != 0]
    return np.sum(inverse_matrix * np.outer(p, p)) - temperature * (-np.sum(p * np.log2(p)))


# Generate the clique complex
def build_clique_complex_new(correlation_matrix, threshold, max_clique_size):
    n = correlation_matrix.shape[0]

    # Use NumPy to create a boolean adjacency matrix based on the threshold
    adjacency_matrix = np.abs(correlation_matrix) > threshold

    # Create the graph directly from the adjacency matrix
    G = nx.from_numpy_matrix(adjacency_matrix)

    # Enumerate all cliques directly
    all_cliques = list(nx.enumerate_all_cliques(G))

    # Building the clique complex
    seen_cliques = {tuple(sorted(clique)) for clique in all_cliques if len(clique) <= max_clique_size}

    # Sort the list of sets based on the length of cliques and sorted vertices within each clique
    clique_complex = sorted(map(frozenset, seen_cliques), key=lambda x: (len(x), sorted(x)))

    return clique_complex

def generate_inverse_connectivity_matrix(clique_complex):
    # Convert the list of lists to a list of sets
    clique_complex = [set(inner_list) for inner_list in clique_complex]

    # Initialize the matrix
    size = len(clique_complex)
    matrix = np.zeros((size, size))

    # Fill the matrix
    for i in range(0, len(clique_complex)):
        for j in range(0, len(clique_complex)):
            if clique_complex[i].intersection(clique_complex[j]):
                matrix[i, j] = 1
                #matrix[j, i] = 1  # Ensure the matrix is symmetric

    # Compute the inverse connectivity matrix
    inverse_connectivity_matrix = np.linalg.inv(matrix)

    return matrix, inverse_connectivity_matrix

# Define probability generator
def generate_probability_list(clique_complex, size, pareto_constant, distribution_type='uniform'):
    if distribution_type == 'uniform':
        # Generate a list of random numbers
        probabilities = np.random.rand(size)

    elif distribution_type == 'custom':
        # Sample from a clique based distribution
        probabilities = nodes_probabilities(clique_complex)[1]

    else:
        raise ValueError("Invalid distribution_type. Supported types: 'uniform', 'custom'")

    # Ensure all values are positive (abs) and normalize to sum to 1
    probabilities = np.abs(probabilities).astype(float)  # Convert to float
    probabilities /= probabilities.sum()

    return probabilities

# Define simulated annealing for energy
def simulated_annealing_free_energy(clique_complex, distribution_type, pareto_constant, matrix, num_iterations, initial_temperature=1.0, cooling_rate=0.95):
    current_probabilities = generate_probability_list(clique_complex, len(clique_complex), 'custom')
    current_value = energy_function(current_probabilities, matrix)
    history = []

    for _ in range(num_iterations):
        temperature = initial_temperature * (cooling_rate ** _)

        # Generate a new set of probabilities
        new_probabilities = generate_probability_list(clique_complex, len(current_probabilities), pareto_constant, distribution_type)

        # Evaluate the entropy of the new set of probabilities
        new_value = free_energy_function(new_probabilities, matrix, 1)

        # Accept the new set of probabilities if its entropy is greater
        if new_value < current_value:
            current_probabilities = new_probabilities
            current_value = new_value

        history.append(current_value)

    return history, current_probabilities

# Generate a probability list according to the clique_complex
def nodes_probabilities(clique_complex, distribution_type='uniform', pareto_constant=-0.1):

    clique_dict = {}
    
    # Create a dictionary to group sets by their length
    sets_by_length = defaultdict(list)

    # Group sets by length
    for s in clique_complex:
        sets_by_length[len(s)].append(s)

    # Convert the dictionary values to lists
    result = list(sets_by_length.values())

    # Create empty list
    probabilities_clique_complex = []

    # Generate probability list per clique dimension and add
    probabilities_nodes = generate_probability_list(clique_complex, len(result[0]), pareto_constant, distribution_type)

    # Normalise the probability list for all clique dimensions together
    probabilities_nodes = np.abs(probabilities_nodes).astype(float)  # Convert to float
    probabilities_nodes /= probabilities_nodes.sum()

    for i in range(0, len(result[0])):
        clique_dict[clique_complex[i]]=probabilities_nodes[i]

    for clique in clique_complex:
        
        # Convert set to tuple and obtain all possible combinations with length smaller than the original set
        combinations_in_clique = [set(combination) for r in range(1, len(clique)) for combination in combinations(tuple(clique), r)]

        prior_element_prob = 1

        for element in combinations_in_clique:
            element_prob = clique_dict[frozenset(element)]  # to convert the set element into a string, so it is interpretable for dictionary
            prior_element_prob *= element_prob

        posterior_element_prob = prior_element_prob * np.random.rand() # Use a uniform distribution to sample the probability of each simplex

        clique_dict[frozenset(clique)]=posterior_element_prob

    probabilities_clique_complex = list(clique_dict.values())

    # Normalise so the sum equals 1, since probability distribution
    probabilities_clique_complex = np.abs(probabilities_clique_complex).astype(float)  # Convert to float
    probabilities_clique_complex /= probabilities_clique_complex.sum()

    return result, probabilities_clique_complex

# Compute analytical max entropy and min energy
def analytical_functionals(matrix, cutoff, max_dim):

    # Generate connection matrix and inverse
    clique_complex =  build_clique_complex_new(matrix, cutoff, max_dim)
    matrix, inverse_connectivity_matrix = generate_inverse_connectivity_matrix(clique_complex)

    # Maximum shannon entropy from uniform distribution
    n = len(inverse_connectivity_matrix)
    p_Smax = np.ones(n) / n
    max_entropy_value = shannon_entropy(p_Smax)

    # Minimum internal energy from analytical solution
    min_energy_probabilities = (np.inner(matrix,[1]*len(matrix)))/np.sum(matrix)
    min_energy_value = energy_function(min_energy_probabilities, inverse_connectivity_matrix)

    return max_entropy_value, min_energy_value

# Compute the free energy directly by approximating min_free_energy
def computing_functionals_direct_custom(matrix, cutoff, max_dim):
    clique_complex = build_clique_complex_new(matrix, cutoff, max_dim)
    inverse_connectivity_matrix = generate_inverse_connectivity_matrix(clique_complex)[1]
    free_energy_history, f_probabilities = simulated_annealing_free_energy(clique_complex, 'custom', -0.1, inverse_connectivity_matrix, 10, initial_temperature=1.0, cooling_rate=0.95)
    return clique_complex, free_energy_history[-1], f_probabilities