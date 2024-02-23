# Define functions to be used in entropy/energy optimisation and free energy calculation

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Define shannon entropy function
def shannon_entropy(probabilities):
    # Remove any zero probabilities to avoid log(0) issues
    probabilities = probabilities[probabilities != 0]
    return -np.sum(probabilities * np.log2(probabilities))

# Define internal energy function
def energy_function(p, A):
    return np.sum(A * np.outer(p, p))

# Define simulated annealing for energy
def simulated_annealing_energy(initial_probabilities, matrix, num_iterations, initial_temperature=1.0, cooling_rate=0.95):
    current_probabilities = initial_probabilities
    current_value = energy_function(current_probabilities, matrix)
    history = [current_value]

    for _ in range(num_iterations):
        temperature = initial_temperature * (cooling_rate ** _)

        # Generate a new set of probabilities
        new_probabilities = generate_probability_list(len(current_probabilities))

        # Evaluate the entropy of the new set of probabilities
        new_value = energy_function(new_probabilities, matrix)

        # Accept the new set of probabilities if its entropy is greater
        if new_value < current_value or np.random.rand() > np.exp((new_value - current_value) / temperature):
            current_probabilities = new_probabilities
            current_value = new_value

        history.append(current_value)

    return history

# Define simulated annealing for entropy
def simulated_annealing_entropy(initial_probabilities, num_iterations, initial_temperature=1.0, cooling_rate=0.95):
    current_probabilities = initial_probabilities
    current_entropy = shannon_entropy(current_probabilities)
    entropy_history = [current_entropy]

    for _ in range(num_iterations):
        temperature = initial_temperature * (cooling_rate ** _)

        # Generate a new set of probabilities
        new_probabilities = generate_probability_list(len(current_probabilities))

        # Evaluate the entropy of the new set of probabilities
        new_entropy = shannon_entropy(new_probabilities)

        # Accept the new set of probabilities if its entropy is greater
        if new_entropy > current_entropy or np.random.rand() < np.exp((new_entropy - current_entropy) / temperature):
            current_probabilities = new_probabilities
            current_entropy = new_entropy

        entropy_history.append(current_entropy)

    return entropy_history

# Define random probability generator
def generate_probability_list(size):
    # Generate a list of random numbers
    random_numbers = np.random.rand(size)

    # Normalize the list to make it a probability distribution
    probabilities = random_numbers / np.sum(random_numbers)

    return probabilities

# Define functions
def free_energy(matrix, beta):

    # Set initial values
    list_size = len(matrix)
    num_iterations_energy = 1000
    initial_probabilities = generate_probability_list(list_size)

    # Minimum internal energy with simulated annealing
    min_energy = simulated_annealing_energy(initial_probabilities, matrix, num_iterations_energy)

    # Maximum shannon entropy from uniform distribution
    n = len(matrix)
    uniform_distribution = np.ones(n) / n
    max_entropy_value = shannon_entropy(uniform_distribution)
    max_entropy = [max_entropy_value]*len(min_energy)

    # Maximum shannon entropy with simulated annealing
    ####### To approximate the maximum entropy using Simulated Annealing unhash the line below and hash the uniform entropy method ######
    #num_iterations_entropy = 100
    #max_entropy = simulated_annealing_entropy(initial_probabilities, num_iterations_entropy)
    
    return min_energy[-1]-beta*max_entropy[-1]

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

    return inverse_connectivity_matrix

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
    for clique in nx.enumerate_all_cliques(G):
        if len(clique) > max_clique_size:
            break
        unique_clique = tuple(sorted(clique))
        seen_cliques.add(unique_clique)

    # Building the clique complex
    clique_complex = [set(clique) for clique in seen_cliques]

    # Sort the list of sets based on the length of cliques and sorted vertices within each clique
    clique_complex = sorted(clique_complex, key=lambda x: (len(x), sorted(x)))

    return clique_complex

def compute_euler(Mat,cutoff,max_dim):
    eu=len(Mat)
    
    C=build_clique_complex(Mat, cutoff, max_dim) # C is the clique complex ordered by weight and dimension
    for c in C:
        
        d=len(c)-1
        eu+=(-1)**d
    
    clique_complex = list(C)

    return [eu, clique_complex]