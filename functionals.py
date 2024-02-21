# Define functions to be used in entropy/energy optimisation and free energy calculation

# Import necessary libraries
import numpy as np

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

    # Maximum shannon entropy with simulated annealing
    max_entropy = simulated_annealing_entropy(initial_probabilities, num_iterations_entropy)
    
    return min_energy[-1]-beta*max_entropy[-1]