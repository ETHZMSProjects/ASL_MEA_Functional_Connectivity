import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from tqdm import tqdm
import json
import os
import networkx as nx
from extract_channels import *

def open_npy(file_dir, file_name):
    file_path = os.path.join(file_dir, file_name)
    arr = np.load(file_path)
    return arr

def process_array_data(data, pval_threshold, sum_axes=-1, normalize=False):
    thresh_data = data < pval_threshold
    if sum_axes != -1:
        thresh_data = np.sum(thresh_data, axis=sum_axes)
        
        if normalize:
            max_count = np.prod([data.shape[ax] for ax in sum_axes])  # Compute max possible count
            thresh_data = thresh_data / max_count  # Normalize by the max count
    return thresh_data

def apply_function_to_arrays(data_dict, func, **func_kwargs):
    """
    Applies a given function to each numpy array in a dictionary and returns the results.
    
    Parameters:
    - data_dict (dict): Dictionary where keys are names and values are numpy arrays.
    - func (callable): Function to apply to each numpy array.
    - **func_kwargs: Additional arguments to pass to func.
    
    Returns:
    - results_dict (dict): Dictionary with the same keys but values being the function results.
    """
    results_dict = {}
    for key, array in data_dict.items():
        try:
            results_dict[key] = func(array, **func_kwargs)
        except Exception as e:
            print(f"Error processing {key}: {e}")
            results_dict[key] = None  # Store None for failed operations
    return results_dict

def apply_slices(arr, slices):
    """
    Applies multiple slices to a NumPy array and extracts scalar if only one element remains.

    Parameters:
    - arr (numpy.ndarray): The array to slice.
    - slices (list of slice or tuple): A list of slices, one for each axis.
      Each element can be:
        - A `slice(start, stop, step)` object
        - A tuple `(start, stop, step)`, which will be converted into a slice
        - `None` to keep the entire axis
    
    Returns:
    - Sliced numpy array OR a scalar if only one element remains.
    """
    slices = [slice(*s) if isinstance(s, tuple) else (s if s is not None else slice(None)) for s in slices]
    result = np.squeeze(arr[tuple(slices)])
    
    # If the result is a single element, extract the scalar
    return result.item() if result.size == 1 else result

def generate_bipartite(data, pre_nodes, post_nodes):
    """
    Generates a NumPy object array of weighted bipartite graphs, ensuring the last 
    two dimensions are (64,64) by reordering axes if necessary. Edge weights correspond 
    to the values in the data array.

    Parameters:
        data (np.ndarray): Input data, where exactly two dimensions must be size 64.
        pre_nodes (list): List of node names for the first set (rows).
        post_nodes (list): List of node names for the second set (cols).

    Returns:
        np.ndarray: NumPy object array of bipartite graphs, maintaining the original 
                    shape except for (64,64).
    """
    # Find all axes that are size 64
    size_64_axes = [i for i, dim in enumerate(data.shape) if dim == 64]
    
    # Ensure there are exactly TWO axes of size 64
    if len(size_64_axes) != 2:
        raise ValueError(f"Expected exactly two dimensions of size 64, but found {len(size_64_axes)} in shape {data.shape}")

    # Move the identified 64x64 axes to the last two positions
    reordered_data = np.moveaxis(data, source=size_64_axes, destination=[-2, -1])

    # Determine the shape of the output (excluding the last two dimensions)
    remaining_shape = reordered_data.shape[:-2]

    # Create an object array to store graphs
    graph_array = np.empty(remaining_shape, dtype=object)

    # Iterate over all indices in the remaining dimensions
    it = np.ndindex(remaining_shape)
    for idx in it:
        sub_data = reordered_data[idx]  # Extract a 64x64 slice

        # Create a weighted bipartite graph
        B = nx.Graph()
        B.add_nodes_from(pre_nodes, bipartite=0)
        B.add_nodes_from(post_nodes, bipartite=1)
        
        # Find nonzero entries (connections)
        rows, cols = np.nonzero(sub_data)
        
        # Add weighted edges
        edges = [(pre_nodes[i], post_nodes[j], sub_data[i, j]) for i, j in zip(rows, cols)]
        B.add_weighted_edges_from(edges)  # Uses 'weight' attribute

        # Store in the array
        graph_array[idx] = B

    return graph_array

def plot_bipartite(bip,from_nodeset,figsize=(12,6),saveFig=False,dpi=300): 
    pos = nx.bipartite_layout(bip, from_nodeset)
    plt.figure(figsize=(12, 6))
    nx.draw(bip, pos, with_labels=True, node_color=['lightblue' if node in from_nodeset else 'lightgreen' for node in bip.nodes()], edge_color='gray', node_size=500, font_size=8)
    plt.title(f"Bipartite Graph with {bip.number_of_edges()} edges")
    if saveFig:
        plt.savefig(saveFig,dpi=300)
    plt.show()