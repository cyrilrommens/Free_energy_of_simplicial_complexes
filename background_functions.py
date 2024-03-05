# Define functions to be used in entropy/energy optimisation and free energy calculation

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import genpareto

# Define shannon entropy function
def shannon_entropy(probabilities):
    # Remove any zero probabilities to avoid log(0) issues
    probabilities = probabilities[probabilities != 0]
    return -np.sum(probabilities * np.log2(probabilities))

# Define internal energy function
def energy_function(p, A):
    return np.sum(A * np.outer(p, p))

# Define free energy function
def free_energy_function(p, inverse_matrix, temperature):
    # Remove any zero probabilities to avoid log(0) issues
    p = p[p != 0]
    return np.sum(inverse_matrix * np.outer(p, p)) - temperature * (-np.sum(p * np.log2(p)))

# Define simulated annealing for energy
def simulated_annealing_energy(initial_probabilities, distribution_type, pareto_constant, matrix, num_iterations, initial_temperature=1.0, cooling_rate=0.95):
    current_probabilities = initial_probabilities
    current_value = energy_function(current_probabilities, matrix)
    history = [current_value]

    for _ in range(num_iterations):
        temperature = initial_temperature * (cooling_rate ** _)

        # Generate a new set of probabilities
        new_probabilities = generate_probability_list(len(current_probabilities), pareto_constant, distribution_type)

        # Evaluate the entropy of the new set of probabilities
        new_value = energy_function(new_probabilities, matrix)

        # Accept the new set of probabilities if its entropy is greater
        if new_value < current_value or np.random.rand() < np.exp((current_value - new_value) / temperature):
            current_probabilities = new_probabilities
            current_value = new_value

        history.append(current_value)

    return history, current_probabilities

# Define simulated annealing for entropy
def simulated_annealing_entropy(initial_probabilities, distribution_type, pareto_constant, num_iterations, initial_temperature=1.0, cooling_rate=0.95):
    current_probabilities = initial_probabilities
    current_entropy = shannon_entropy(current_probabilities)
    entropy_history = [current_entropy]

    for _ in range(num_iterations):
        temperature = initial_temperature * (cooling_rate ** _)

        # Generate a new set of probabilities
        new_probabilities = generate_probability_list(len(current_probabilities), pareto_constant, distribution_type)

        # Evaluate the entropy of the new set of probabilities
        new_entropy = shannon_entropy(new_probabilities)

        # Accept the new set of probabilities if its entropy is greater
        if new_entropy > current_entropy or np.random.rand() < np.exp((new_entropy - current_entropy) / temperature):
            current_probabilities = new_probabilities
            current_entropy = new_entropy

        entropy_history.append(current_entropy)

    return entropy_history, current_probabilities

# Define simulated annealing for energy
def simulated_annealing_free_energy(initial_probabilities, distribution_type, pareto_constant, matrix, num_iterations, initial_temperature=1.0, cooling_rate=0.95):
    current_probabilities = initial_probabilities
    current_value = energy_function(current_probabilities, matrix)
    history = [current_value]

    for _ in range(num_iterations):
        temperature = initial_temperature * (cooling_rate ** _)

        # Generate a new set of probabilities
        new_probabilities = generate_probability_list(len(current_probabilities), pareto_constant, distribution_type)

        # Evaluate the entropy of the new set of probabilities
        new_value = free_energy_function(new_probabilities, matrix, 1)

        # Accept the new set of probabilities if its entropy is greater
        if new_value < current_value:
            current_probabilities = new_probabilities
            current_value = new_value

        history.append(current_value)

    return history, current_probabilities

# Define probability generator
def generate_probability_list(size, pareto_constant, distribution_type='uniform'):
    if distribution_type == 'uniform':
        # Generate a list of random numbers
        probabilities = np.random.rand(size)

    elif distribution_type == 'normal':
        # Sample from a normal distribution
        probabilities = np.random.normal(size=size)

    elif distribution_type == 'poisson':
        # Generate a list of random numbers
        probabilities = np.random.poisson(size=size)

    elif distribution_type == 'chisquare':
        # Sample from a chi-square distribution
        probabilities = np.random.chisquare(df=1, size=size)

    elif distribution_type == 'gamma':
        # Sample from a gamma distribution
        probabilities = np.random.gamma(shape=2, size=size)

    elif distribution_type == 'pareto':
        # Sample from pareto distribution
        probabilities = np.random.pareto(a=2, size=size)

    elif distribution_type == 'lognormal':
        # Sample from lognormal distribution
        probabilities = np.random.lognormal(mean=0, sigma=1, size=size)

    elif distribution_type == 'genpareto':
        # Sample from a generalized Pareto distribution
        probabilities = genpareto.rvs(pareto_constant, size=size)

    else:
        raise ValueError("Invalid distribution_type. Supported types: 'uniform', 'normal', 'poisson', 'chisquare', 'gamma', 'pareto', 'lognormal', 'genpareto'")

    # Ensure all values are positive (abs) and normalize to sum to 1
    probabilities = np.abs(probabilities).astype(float)  # Convert to float
    probabilities /= probabilities.sum()

    return probabilities

# Define functions
def free_energy(matrix, beta):

    # Set initial values
    list_size = len(matrix)
    num_iterations_energy = 100
    initial_probabilities = generate_probability_list(list_size, 'uniform')

    # Minimum internal energy with simulated annealing
    min_energy, p_Umin = simulated_annealing_energy(initial_probabilities, 'uniform', -0.1, matrix, num_iterations_energy)

    # Maximum shannon entropy with simulated annealing
    ####### To approximate the maximum entropy using Simulated Annealing unhash the line below and hash the uniform entropy method ######
    num_iterations_entropy = 10
    max_entropy, p_Smax = simulated_annealing_entropy(initial_probabilities,'uniform', -0.1, num_iterations_entropy)

    # Maximum shannon entropy from uniform distribution
    #n = len(matrix)
    #p_Smax = np.ones(n) / n
    #max_entropy_value = shannon_entropy(p_Smax)
    #max_entropy = [max_entropy_value]*len(min_energy)
    
    U_min = min_energy[-1]
    S_max = max_entropy[-1]
    Free_energy = U_min - beta*S_max

    return U_min, p_Umin, S_max, p_Smax, Free_energy

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

def simulated_annealing(test_matrix, num_runs, num_iterations_energy, num_iterations_entropy):
    # Example for generating energy and entropy data from a given matrix
    list_size = len(test_matrix)
    energy_history = []
    entropy_history = []

    # Run multiple times
    for _ in range(num_runs):
        initial_probabilities = generate_probability_list(list_size)

        # Minimum internal energy with simulated annealing
        min_energy = simulated_annealing_energy(initial_probabilities, test_matrix, num_iterations_energy)
        energy_history.append(min_energy)

        # Maximum shannon entropy with simulated annealing
        max_entropy = simulated_annealing_entropy(initial_probabilities, num_iterations_entropy)
        entropy_history.append(max_entropy)

    # Generate averaged list of entropy_histories
    entropy_history = np.array(entropy_history)
    entropy_history_averaged = np.mean(entropy_history, axis=0)

    # Generate averaged list of energies_optimization_SA
    energy_history = np.array(energy_history)
    energy_history_averaged = np.mean(energy_history, axis=0)
    return entropy_history, entropy_history_averaged, energy_history, energy_history_averaged

# Define a function to obtain a list the simplexes present in the simplicial complex, by counting the complete subgraphs in the connection matrix
def build_clique_complex(correlation_matrix, threshold, max_clique_size):
    n = correlation_matrix.shape[0]
    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            if abs(correlation_matrix[i, j]) > threshold:
                G.add_edge(i, j)

    # Using nx.enumerate_all_cliques in an interactive manner
    seen_cliques = set()
    nodes_list = [set([i]) for i in range(len(correlation_matrix))] # Add nodes otherwise only > 1-simplexes are included in the clique_complex
    all_cliques = list(nx.enumerate_all_cliques(G)) + nodes_list
    for clique in all_cliques:

        if len(clique) > max_clique_size:
            break
        unique_clique = tuple(sorted(clique))
        seen_cliques.add(unique_clique)

    # Building the clique complex
    clique_complex = [set(clique) for clique in seen_cliques]

    # Sort the list of sets based on the length of cliques and sorted vertices within each clique
    clique_complex = sorted(clique_complex, key=lambda x: (len(x), sorted(x)))

    return clique_complex

# Compute the euler characteristic and clique complex
def compute_euler(Mat,cutoff,max_dim):
    eu=len(Mat)
    
    C=build_clique_complex(Mat, cutoff, max_dim) # C is the clique complex ordered by weight and dimension
    for c in C:
        
        d=len(c)-1
        eu+=(-1)**d
    
    clique_complex = list(C)

    return eu, clique_complex

# Compute the free energy indirectly from approximating min_energy - max_entropy
def computing_functionals(matrix, cutoff, max_dim):
    clique_complex = build_clique_complex(matrix, cutoff, max_dim)
    inverse_connectivity_matrix = generate_inverse_connectivity_matrix(clique_complex)[1]
    return free_energy(inverse_connectivity_matrix, 1)

# Compute the free energy directly by approximating min_free_energy
def computing_functionals_direct(matrix, cutoff, max_dim):
    clique_complex = build_clique_complex(matrix, cutoff, max_dim)
    inverse_connectivity_matrix = generate_inverse_connectivity_matrix(clique_complex)[1]
    initial_probabilities = generate_probability_list(len(inverse_connectivity_matrix), 'uniform')
    free_energy_history, f_probabilities = simulated_annealing_free_energy(initial_probabilities, 'uniform', -0.1, inverse_connectivity_matrix, 10, initial_temperature=1.0, cooling_rate=0.95)
    return free_energy_history[-1]

# Compute analytical max entropy and min energy
def analytical_functionals(matrix, cutoff, max_dim):

    # Generate connection matrix and inverse
    euler_characteristic, clique_complex = compute_euler(matrix,cutoff,max_dim)
    matrix, inverse_connectivity_matrix = generate_inverse_connectivity_matrix(clique_complex)

    # Maximum shannon entropy from uniform distribution
    n = len(inverse_connectivity_matrix)
    p_Smax = np.ones(n) / n
    max_entropy_value = shannon_entropy(p_Smax)

    # Minimum internal energy from analytical solution
    min_energy_probabilities = (np.inner(matrix,[1]*len(matrix)))/np.sum(matrix)
    min_energy_value = energy_function(min_energy_probabilities, inverse_connectivity_matrix)

    return max_entropy_value, min_energy_value