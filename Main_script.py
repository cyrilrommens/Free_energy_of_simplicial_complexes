########################################################################
########## REAL+PR-TIMESERIES TO FREE ENERGY DATAFRAMES ################
########################################################################
# Record the start time
import time
start_time = time.time()

# Import libraries
import pandas as pd
import numpy as np
import glob
from scipy.optimize import minimize
import infotopo
import warnings
from itertools import tee,combinations
import networkx as nx

# Avoid RunTimeWarning
warnings.filterwarnings("ignore", message="Values in x were outside bounds during a minimize step", category=RuntimeWarning)


######################### DEFINE FUNCTIONS ##############################
# Function to obtain mutual informations per clique from a given timeseries
def obtain_mutual_information(filename, max_d, number_of_variables): 

    # Import time series data
    df = pd.read_csv(filename, sep='\t', header=None)

    # Mask using the 99th and 1th percentile
    stacked_series = df.stack()
    quantile_99 = stacked_series.quantile(0.99)
    quantile_01 = stacked_series.quantile(0.01)
    df[df > quantile_99] = quantile_99
    df[df < quantile_01] = quantile_01

    # Initialize an empty DataFrame to hold discretized values
    discretized_time_series = pd.DataFrame()
    max_BOLD = df.max().max()
    min_BOLD = df.min().min()
    desired_number_of_bins = 16
    stepsize = (max_BOLD-min_BOLD)/desired_number_of_bins
    bin_edges = np.arange(min_BOLD, max_BOLD, stepsize)

    # Iterate over each column of 'df'
    for col in df.columns:
        data = df[col].tolist()
        bin_numbers = np.digitize(data, bin_edges)
        
        # Create a DataFrame with the current column's discretized values
        col_df = pd.DataFrame({col: bin_numbers}, index=df.index)
        
        # Concatenate the new DataFrame along the columns axis
        discretized_time_series = pd.concat([discretized_time_series, col_df], axis=1)

    # Import the information topology functions needed
    #%run infotopo.py
    
    # Settings to use infotopo functions
    dataset = np.array(discretized_time_series).T
    work_on_transpose = False 
    nb_of_values = 16
    deformed_probability_mode = False
    supervised_mode = False
    forward_computation_mode = True
    sampling_mode = 1

    # Call infotopo functions for entropy, mutual information and free energy
    information_topo = infotopo.infotopo(dimension_max = max_d, 
                                dimension_tot = number_of_variables, 
                                sample_size = 2400, 
                                work_on_transpose = work_on_transpose,
                                nb_of_values = nb_of_values, 
                                sampling_mode = sampling_mode, 
                                deformed_probability_mode = deformed_probability_mode,
                                supervised_mode = supervised_mode, 
                                forward_computation_mode = forward_computation_mode)

    Nentropie = information_topo.simplicial_entropies_decomposition(dataset) 
    Ninfomut = information_topo.simplicial_infomut_decomposition(Nentropie)
    Nfree_energy = information_topo.total_correlation_simplicial_lanscape(Nentropie)

    return Nentropie, Ninfomut, Nfree_energy

# Function to split the mutual informations per clique into lists according to their dimension
def infomut_per_dimension(data):

    list_1 = []  # For keys of length 1
    list_2 = []  # For keys of length 2
    list_3 = []  # For keys of length 3

    for key, value in data.items():
        if len(key) == 1:
            list_1.append(value)
        elif len(key) == 2:
            list_2.append(value)
        elif len(key) == 3:
            list_3.append(value)

    return list_1, list_2, list_3

# Function to obtain pruned_clique_complex for given REAL and PHASE RANDOMISED timeseries
def obtain_pruned_CC(filename_REAL, filename_PR, max_d, number_of_variables):
    # Extract patient ID from filename
    identification_code = filename_REAL[-103:-97]

    # Run for max_d=3 and nb_variables=60 takes about 4min
    Real_Ninfomut = obtain_mutual_information(filename_REAL, max_d, number_of_variables)[1]
    Random_Ninfomut = obtain_mutual_information(filename_PR, max_d, number_of_variables)[1]

    # Split the mutual informations per dimension
    Random_I1, Random_I2, Random_I3 = infomut_per_dimension(Random_Ninfomut)

    # Given dictionary
    data = Real_Ninfomut

    # Define common quantiles
    quantile_min = 0.010
    quantile_max = 0.990

    # Initialize a dictionary to store filtered data for different key lengths
    filtered_data = {}

    # Loop through the data
    for key_length in range(1, 4):
        # Calculate quantiles based on the key length
        I_min = np.quantile(locals()[f"Random_I{key_length}"], quantile_min)
        I_max = np.quantile(locals()[f"Random_I{key_length}"], quantile_max)
        
        # Filter data
        mask = {key: I_min <= value <= I_max for key, value in data.items()}
        filtered_data[key_length] = {key: value for key, value in data.items() if mask[key]}

    # Combine filtered dictionaries into one pruned_clique_complex dictionary
    pruned_clique_complex = {key: value for filtered_dict in filtered_data.values() for key, value in filtered_dict.items()}

    # Compute the average free energy component (average mutual information)
    average_free_energy_component = sum(pruned_clique_complex.values())/len(pruned_clique_complex.values())

    # Remove mutual informations from pruned clique complex
    clique_complex = [frozenset(key) for key in pruned_clique_complex if key in pruned_clique_complex]

    return [identification_code, clique_complex, average_free_energy_component]

# Function to count the occurences of specific cliques in all the given clique complexes
def count_occurrences(dicts):
    occurrences = {}
    # Iterate over all dictionaries
    for d in dicts:
        for key in d:
            occurrences[key] = occurrences.get(key, 0) + 1
    return occurrences

# Function to generate a list of clique probabilities from a given set of clique complexes
def generate_clique_probabilities(dict_list):  
      
    # Count occurrences
    occurrences = count_occurrences(dict_list)

    # Divide each value by the sum
    normalized_occurrences = {key: value /sum(occurrences.values()) for key, value in occurrences.items()}
    
    dict_values_list = []

    for i in range(0, len(dict_list)):
        # Initialize an empty list to store values
        values_list = []

        # Iterate over keys of dict1
        for key in dict_list[i]:
            # Check if the key exists in normalized_occurrences
            if key in normalized_occurrences:
                # If the key exists, append its corresponding value to the list
                values_list.append(normalized_occurrences[key])

        dict_values_list.append(np.array(values_list))

    return dict_values_list

# Function to generate Knill's minimum free energy by minimizing using scipy
def complete_f_generator_scipy(clique_complex):
    Q = generate_inverse_connectivity_matrix(clique_complex)[1]

    # Optimization settings
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = [(1e-10, None) for _ in range(len(Q))]
    x0 = np.full(len(Q), 1/len(Q))  # Initial guess

    # Store the latest optimized x0 and all free energies during minimization
    latest_x0 = None
    t = 1
    all_values = []

    # Callback function to collect values during minimization
    def callback(x):
        all_values.append(free_energy_function(x, Q, t))

    #for t in t_values:
    #    result = minimize(objective, x0, args=(Q, t), method='SLSQP', constraints=cons, bounds=bounds)
    #    minimized_values.append(result.fun)
    #    latest_x0 = result.x  # Update the latest optimized x0

    result = minimize(free_energy_function, x0, args=(Q, t), method='SLSQP', constraints=cons, bounds=bounds, callback=callback)
    return [result.fun, result.x]

# Function to obtain the free energies of a given dataset of clique complexes and ID's
def obtain_Knill_free_energy(ID_list, clique_complex_list):
    # Create an empty dataframe
    df_KnillF = pd.DataFrame(columns=['identification_code', 'free_energy_Knill_p', 'min_free_energy_Knill_scipy'])

    # Generate clique probalilities by counting occurences
    clique_probabilities_list = generate_clique_probabilities(clique_complex_list)

    # Generate free energies for the minimizing and fixed probability case
    for i in range(0, len(clique_complex_list)):
        clique_complex = clique_complex_list[i]
        probability = clique_probabilities_list[i]
        inverse_connectivity_matrix = generate_inverse_connectivity_matrix(clique_complex)[1]
        Free_energy = energy_function(probability, inverse_connectivity_matrix)
        Free_energy_min = complete_f_generator_scipy(clique_complex_list[i])
        df_KnillF.loc[len(df_KnillF)] = [ID_list[i], Free_energy, Free_energy_min[0]]

    return df_KnillF

# Build clique complex using gudhi to improve speed over networkx
def build_clique_complex(correlation_matrix, max_clique_size):
    G = nx.from_numpy_array(abs(correlation_matrix))

    Cl = (i for i in nx.find_cliques(G))

    C = (tuple(sorted(c)) for c in Cl)
    C = tee(C, max_clique_size+1)

    cliques = [[] for _ in range(max_clique_size+1)]

    for i in range(max_clique_size+1):
        K = (i for i in set(c for mc in C[i] for c in combinations(mc, i+1)))
        for c in K:
            cliques[i].append(frozenset(c))

    result = []
    for i in range(max_clique_size+1):
        result.extend(sorted(cliques[i], key=lambda x: (len(x), sorted(x))))

    return result

# Genrate the matrix L^-1 as required by Knill
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

# Internal energy function
def energy_function(x, Q):
    return x.T @ Q @ x

# Free energy function
def free_energy_function(x, Q, t):
    entropy_term = - np.sum(x * np.log2(np.maximum(x, 1e-10)))  # Avoid log(0), changed to np.log2 to improve speed.
    return t*(x.T @ Q @ x) - (1-t) * entropy_term

# Print the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Loading functions elapsed time: {elapsed_time:.2f} seconds")

# Record the start time
start_time = time.time()

#################### OBTAIN INFOTOPO FREE ENERGY ########################
# INSERT DESIRED SETTINGS
REST_state = 'REST1' # Choose REST1 or REST2
max_d = 3
number_of_variables = 30

# Path for REST1 real time series
path_REAL = glob.glob(f"TimeSeries_REAL\\{REST_state}\\*.txt")
path_PR = glob.glob(f"TimeSeries_PR\\{REST_state}\\*.txt")

# Create an empty dataframe
df_InfoCoho = pd.DataFrame(columns=['identification_code', 'pruned_clique_complex', 'average_free_energy_component'])

# Loop over the datafiles
for i in range(0, 1): #len(path)):
    filename_REAL = path_REAL[i]
    filename_PR = path_PR[i]
    df_InfoCoho.loc[len(df_InfoCoho)] = obtain_pruned_CC(filename_REAL, filename_PR, max_d, number_of_variables)

# Store dataframe containing patient_ID, pruned_clique_complex and the average_free_energy_component
df_InfoCoho.to_csv(f'InfoCoho_{REST_state}.txt', sep='\t', index=False)

# Print the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"F Infotopo elapsed time: {elapsed_time:.2f} seconds")

# Record the start time
start_time = time.time()

##################### OBTAIN KNILL FREE ENERGY ##########################
# INSERT DESIRED SETTINGS
REST_state = 'REST1'

# Read the CSV file and extract the first and second columns as lists
df_InfoCoho = pd.read_csv(f'InfoCoho_{REST_state}.txt', sep='\t')
ID_list = df_InfoCoho['identification_code'].tolist()
clique_complex_list = df_InfoCoho.iloc[:, 1].apply(eval).tolist()

df_KnillF = obtain_Knill_free_energy(ID_list, clique_complex_list)
df_KnillF.to_csv(f'KnillF_{REST_state}.txt', sep='\t', index=False)

# Print the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"F Knill elapsed time: {elapsed_time:.2f} seconds")

# Record the start time
start_time = time.time()


######################## PRINT RESULTS ##################################
# Print the results
print(" ")
print("Free energy analysis from Information Cohomology:")
print(df_InfoCoho)
print(" ")
print("Free energy analysis from Simplicial Topology:")
print(df_KnillF)
print(" ")

# Print the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Print results elapsed time: {elapsed_time:.2f} seconds")