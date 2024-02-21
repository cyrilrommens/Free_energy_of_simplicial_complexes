# Define functions to be used in entropy/energy optimisation and free energy calculation

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define shannon entropy function
def shannon_entropy(probabilities):
    # Remove any zero probabilities to avoid log(0) issues
    probabilities = probabilities[probabilities != 0]
    return -np.sum(probabilities * np.log(probabilities))

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
    num_iterations_entropy = 100
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

def file_to_matrix(file_name):
    # Import the data of one subject
    df = pd.read_csv(file_name,header=None)

    # Extract the values from the DataFrame
    data_str = df.iloc[:, 0].values

    # Initialize an empty list to store parsed rows
    parsed_data = []

    # Iterate through each row and parse the space-separated values
    for row_str in data_str:
        # Split the string into a list of values
        row_values = [float(val) for val in row_str.split()]
        parsed_data.append(row_values)

    # Convert the list of lists into a numpy array
    M = np.array(parsed_data)

    return M

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

def plot_SA_data(entropy_history, entropy_history_averaged, energy_history, energy_history_averaged):
    # Plotting the optimisation side by side
    plt.figure(figsize=(10, 5))

    # Plot the energy/entropy evolution against the number of Simulated Annealing iterations

    plt.subplot(1, 2, 1)
    for i in range(0, len(energy_history)):
        plt.plot(energy_history[i], color='grey', linestyle='--', linewidth=1)
    if energy_history_averaged != None:
        plt.plot(energy_history_averaged, color='red', linestyle='-', linewidth=1, label = 'Average S for optimised p')
        plt.legend()
    plt.title('Minimum Energy U(p) (Simulated Annealing)')
    plt.xlabel('Iterations')
    plt.ylabel('Energy U(p)')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for i in range(0, len(entropy_history)):
        plt.plot(entropy_history[i], color='grey', linestyle='--', linewidth=1)
    if entropy_history_averaged != None:
        plt.plot(entropy_history_averaged, color='red', linestyle='-', linewidth=1, label = 'Average S for optimised p')
        plt.legend()
    plt.title('Maximum Entropy S(p) (Simulated Annealing)')
    plt.xlabel('Iterations')
    plt.ylabel('Entropy S(p)')
    plt.grid(True)

    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()

def generate_symmetric_binary_matrix(n):
    # Generate a random binary matrix
    random_matrix = np.random.randint(2, size=(n, n))

    # Make the matrix symmetric by taking the upper triangular part and setting it equal to its transpose
    symmetric_matrix = np.triu(random_matrix, 1) + np.triu(random_matrix, 1).T + np.eye(n)

    return symmetric_matrix

