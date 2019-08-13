## functions from Naveen's ipynb


import os, sys, glob
import numpy as np
import scipy as sp
import copy
#import keras
import itertools as it

def pad_to_dimensions(array, dimensions):
    # Return a right-padded N-dimensional array to the specified dimensions
    # If a dimension is None, then padding is not done for that dimensions
    assert type(dimensions) == tuple, 'Dimensions must be a tuple.'
    assert len(array.shape) == len(dimensions), 'Mismatch between array dimension and number of dimensions provided.'
    pad_lengths = [requestDim - arrDim if requestDim != None else arrDim for arrDim, requestDim in zip(array.shape, dimensions)]
    assert all(i>=0 for i in pad_lengths), 'Array larger than provided dimensions.'
    pad_lengths_formatted = [(0, pad_length) for pad_length in pad_lengths]
    return np.pad(array, pad_lengths_formatted, mode='constant', constant_values=0)

def gen_onehot_dict_default(cohort,skipZero):
    # Return an int -> vector map for each unique element in cohort of arrays
    def genVector(entryNum, numEntries):
        onehot_vec = np.zeros((numEntries,))
        onehot_vec[entryNum] = 1
        return onehot_vec
    uniqueEntries = np.unique(np.concatenate([np.unique(array) for array in cohort]))
    if skipZero:
        uniqueEntries = [entry for entry in uniqueEntries if not entry == 0]
    numUniques = len(uniqueEntries)
    return {uniqueEntry: genVector(idx, numUniques) for idx, uniqueEntry in enumerate(uniqueEntries)}

def standardize_array_shapes(arrays):
    # Pad arrays to the same size
    array_shapes = [array.shape for array in arrays]
    dims = tuple([max(lengths) for lengths in zip(*[np.shape(array) for array in arrays])])
    return [pad_to_dimensions(array, dims) for array in arrays]

def make_onehot(cohort, onehot_dict=None, verbose=False, skipZero = True):
    # Makes a cohort of arrays onehot, then pads arrays to the same size.
    # Returns (onehot dictionary, arrays)

    def gen_onehot(array, onehot_dict, skipZero):
        # Should be parallelized for better performance
        extension_length = len(onehot_dict.values()[0])
        new_dims = tuple(list(array.shape) + [extension_length])
        onehot_array = np.zeros(new_dims)
        for index in np.ndindex(array.shape):
            if (not array[index] == 0.0) or not skipZero:
                onehot_index_full = tuple(list(index) + [None]) # Slice whole remaining dimension
                onehot_array[onehot_index_full] = onehot_dict[array[index]]
        return onehot_array

    if onehot_dict == None:
        onehot_dict = gen_onehot_dict_default(cohort, skipZero)
    onehot_cohort = [gen_onehot(array, onehot_dict,skipZero) for array in cohort]
    return onehot_dict, onehot_cohort

def readgraph(fname):
    # Returns a header/IntArray and matrix/IntArray. Assert header and matrix have same length.
    a = open(fname); b = a.readlines(); a.close()
    csv_as_float_2D = [map(float,i.strip().split(',')) for i in b]
    header = np.array(csv_as_float_2D[0])
    matrix = np.array(csv_as_float_2D[1:])
    return {'header':header, 'matrix':matrix}
def generate_graph(vect,n):
    # function to re-pack a
    # vector of bond orders
    # to a symmetric n-n 
    # matrix
    # note the first off-diag
    # has n-1 elements
    matr = np.zeros([n,n])
    start = 0
    end = n-1
    for i in range(0,n-1): 
        #print(i,start,end,np.diag(vect[start:end],i+1).shape)
        matr += np.diag(vect[start:end],i+1)
        start += n-1-i
        end += n-2-i
    matr += np.transpose(matr)
    return(matr)      
        
def generate_vector(mat,n):
    # function to strip
    # n x n symmetric matrix
    # into a vector of unique 
    # elemenets. Inverse of
    # generate_graph
    # not the first off-diag
    # has n-1 elements

    vect = np.zeros(((n*n-n)/2))
    start = 0
    end = n-1
    for i in range(0,n-1): 
        #print(i,start,end,np.diag(vect[start:end],i+1).shape)
        vect[start:end] = np.diag(mat,k=i+1)
        start += n-1-i
        end += n-2-i
    return(vect)

def minimalize_BO_matrix(mat, num_nodes):
    trinds = np.triu_indices(n=num_nodes,k=1)
    this_encode = None
    for types in range(mat.shape[2]):
        this_submat = mat[:,:,types]
        this_minimal = generate_vector(this_submat,num_nodes)
        if this_encode is None:
            this_encode =  this_minimal
        else:
            this_encode =  np.column_stack([this_encode,this_minimal])
    return(this_encode)
    
def standardize_graphs(graphs, max_size=None, verbose=True, skipZero=True):
    # Graphs is an iterable of dictionaries {'header': header, 'matrix': matrix}
    # Take in a N-length atom list, NxN matrix, assert all entries are in one-hot dictionaries
    if max_size == None:
        max_size = max(len(graph['header']) for graph in graphs)
        
    kept_graphs = [graph for graph in graphs if len(graph['header']) <= max_size]
    assert len(kept_graphs) > 0, 'No graphs are smaller than the provided max size'
    headers = [i['header'] for i in kept_graphs]
    matrices = [i['matrix'] for i in kept_graphs]
    headers_padded = standardize_array_shapes(headers)
    matrices_padded = standardize_array_shapes(matrices)
    connectivities = [(i!=0).astype(float) for i in matrices_padded]

    headers_onehot_dict, headers_new = make_onehot(headers_padded,skipZero=skipZero)
    matrices_onehot_dict, matrices_new = make_onehot(matrices_padded,skipZero=skipZero)
    
    outputGraphs = [{'header': header, 'matrix': matrix, 'connectivity': connectivity, 'origHeader': origHeader, 'origMatrix': origMatrix}
                    for header,matrix,connectivity,origHeader,origMatrix
                    in zip(headers_new, matrices_new, connectivities,headers_padded,matrices_padded)]
    
    if verbose:
        print 'Header One-Hot Map:', headers_onehot_dict
        print 'Bond One-Hot Map:', matrices_onehot_dict
        print 'Graph Size:', max_size
        print 'Number of dropped graphs:', len(graphs) - len(kept_graphs)
        
    return headers_onehot_dict, matrices_onehot_dict, outputGraphs
