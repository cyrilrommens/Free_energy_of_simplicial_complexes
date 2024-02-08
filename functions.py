# List of all the necessary functions

# Import libraries
import networkx as nx
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import pandas as pd
import glob
import timeit
import os

# Define function to draw a visual graph from the connection matrix
def plot_graph(correlation_matrix, threshold):
    n = correlation_matrix.shape[0]
    G = nx.Graph()
    
    for i in range(n):
        for j in range(i + 1, n):
            if abs(correlation_matrix[i, j]) > threshold:
                G.add_edge(i, j)

    pos = nx.spring_layout(G)  # You can use a different layout if needed
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700)
    plt.show()

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

# Define a function to generate a connection matrix of the simplexes from the list of cliques
def generate_overlap_matrix(sets_list):
    # Get the set of all unique values in the list of sets
    all_values = sorted(list({value for s in sets_list for value in s}))
    
    # Create a mapping from values to indices
    value_to_index = {value: index for index, value in enumerate(all_values)}
    
    # Initialize the overlap matrix with zeros
    n = len(sets_list)
    overlap_matrix = np.zeros((n, n), dtype=int)
    
    # Set the entries to 1 where values overlap
    for i, s1 in enumerate(sets_list):
        values_s1 = sorted(list(s1))
        indices_s1 = [value_to_index[value] for value in values_s1]
        
        for j, s2 in enumerate(sets_list):
            values_s2 = sorted(list(s2))
            indices_s2 = [value_to_index[value] for value in values_s2]
            
            if any(value in s2 for value in s1):  # Check for overlap
                overlap_matrix[i, j] = 1
    
    return overlap_matrix


# Function to generate the inverse of the matrix given as input
def inverse_matrix_generator(matrix):

    # Find non-zero rows and columns
    non_zero_rows = ~np.all(matrix == 0, axis=1)

    # Store the indices of removed rows. This is the same for the columns since the matrices are symmetric
    removed_rows = np.where(~non_zero_rows)[0].tolist()

    # Extract the non-zero rows and columns
    result_matrix = matrix[non_zero_rows][:, non_zero_rows]

    # Generate inverse matrix
    if np.linalg.det(result_matrix) != 0:
        inverse_matrix_unrounded = np.linalg.inv(result_matrix)
        inverse_matrix = np.round(inverse_matrix_unrounded, decimals=2)
    else:
        inverse_matrix = 0

    return result_matrix, removed_rows, inverse_matrix


# Define function to decompress the compressed array with inserting zeros at the indices of the removed rows
def decompress_row(array_length, index_removed):
    
    # Add +1 to each value in the index_list since the count starts at zero
    index_removed = [x + 1 for x in index_removed]

    # Create a complete array from 1 to 14
    complete_array = np.arange(1, 15)

    # Set every row index that is removed to zero in the complete array
    for index in complete_array:
        if index in index_removed:
            complete_array[index-1] = 0

    array_decompressed = complete_array
    return array_decompressed


# Add a zero column at the desired index
def add_zero_array(original_array, index, axis_choice):
    
    # Create a zero array of fitting shape
    zero_array = np.zeros(original_array.shape[0], dtype=original_array.dtype)

    # Insert the array into the original array
    inserted_array = np.insert(original_array, index, zero_array, axis=axis_choice)

    return inserted_array


# Decompress the matrix to match the shape with the probability matrix L_p for later multiplication
def decompress(matrix, array_length, removed_rows):
    
    # Decompress the 1D array
    decompressed_1D_array = decompress_row(array_length, removed_rows)

    # Insert zero rows/columns at the indices with value zero from the decompressed 1D array
    for index in range(0, len(decompressed_1D_array)):
        if decompressed_1D_array[index] == 0:
            inserted_column_array = add_zero_array(matrix, index, 0) # The column axis is 0
            inserted_row_array = add_zero_array(inserted_column_array, index, 1) # The row axis is 1
            matrix = inserted_row_array
    return matrix


# Compute the internal energy from the inverse connection matrix and the probability matrix as suggested by Knill
def internal_energy(inverse_connection_matrix, probability_matrix):

    # Compute the internal energy
    U = np.sum(inverse_connection_matrix * probability_matrix)
    return U


# Compute the entropy from the connection matrix and the probability matrix as suggested by Knill
def entropy(connection_matrix, probability_matrix):

    # Generate probability matrix specific to the simplicial complex G_i
    specific_probability_matrix = connection_matrix * probability_matrix

    # Extract the diagonal values as a list
    diagonal = np.round(np.diag(specific_probability_matrix).tolist(), decimals=2)

    non_zero_diagonal = [value for value in diagonal if value != 0]

    # Compute the entropy
    S = - np.sum(non_zero_diagonal * np.log(non_zero_diagonal))
    
    return S


# Compute the free energy from the internal energy and the entropy
def free_energy(internal_energy, temperature, entropy):
    return internal_energy - temperature*entropy


# Compute the functionals from the dataset
def functionals(G_matrices, L_matrices, L_p, temperature):
    U_list = []
    S_list = []
    F_list = []
    for matrix in range(0, len(L_matrices)):
        U = internal_energy(L_matrices[matrix], L_p)
        U_list.append(U)
        S = entropy(G_matrices[matrix], L_p)
        S_list.append(S)
        F = free_energy(U, temperature, S)
        F_list.append(F)
    return U_list, S_list, F_list


# flatten a 2D list of sets to a 1D list of sets
def flatten_list(input_list):
    input_list = [item for sublist in input_list for item in sublist]
    input_list = sorted(input_list, key=lambda x: (len(x), x))

    # Remove duplicates
    input_list = list(set(tuple(sorted(s)) for s in input_list))

    # Sort by set size and within each set size by values
    input_list = sorted(input_list, key=lambda x: (len(x), x))

    # Convert back to sets
    result_list = [set(s) for s in input_list]
    return result_list


# Returns a list of sets that are in A but not in B
def sets_difference(A, B):
    return [set_a for set_a in A if set_a not in B]