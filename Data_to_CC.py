#########################################################################
###### Raw + PR timeseries to pruned_clique_complex + H/I/F’s   #########
#########################################################################

# Import libraries
import numpy as np
import pandas as pd

def obtain_mutual_information(filename, max_d, number_of_variables):
    # Example DataFrame
    df = pd.read_csv(filename, sep='\t', header=None)

    # Stack the DataFrame into a Series
    stacked_series = df.stack()

    # Calculate the 99th percentile
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

    dataset = np.array(discretized_time_series).T
        
    # Settings
    work_on_transpose = False
    nb_of_values = 16
    deformed_probability_mode = False
    supervised_mode = False
    forward_computation_mode = True
    sampling_mode = 1

    # Call infotopo function
    information_topo = infotopo(dimension_max = max_d, 
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