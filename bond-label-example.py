import os, sys, keras, copy
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils.vis_utils import model_to_dot
from graphTools import *
from layerTools import *


# Read in graphs
graph_files = glob.glob('qm9graph/*.csv')[0:10000]
mygraphs = [readgraph(fname) for fname in graph_files]
headers_onehot_dict, matrices_onehot_dict, mygraphs_standardized = standardize_graphs(mygraphs, max_size=29,skipZero=False)
headers_padded = np.array([i['header'] for i in mygraphs_standardized])
matrices_padded = np.array([i['matrix'] for i in mygraphs_standardized])
connectivities_padded = np.array([i['connectivity'] for i in mygraphs_standardized])
origHeaders_padded = np.array([i['origHeader'] for i in mygraphs_standardized])
origMatrices_padded = np.array([i['origMatrix'] for i in mygraphs_standardized])
print headers_padded.shape
print matrices_padded.shape
print connectivities_padded.shape


gumbel_softmax_mask = lambda x: gumbel_softmax(logits=x,temperature=0.05)

num_input =5 
latent_dim = 50
noise_input = np.random.randn(num_input,latent_dim)
num_nodes = 29

# reduce the BO mat to minimal vectors
bond_types = 4
minimal_mats = None
for m in matrices_padded:
    minimal_m = minimalize_BO_matrix(m,num_nodes)
    if minimal_mats is None:
        minimal_mats = minimal_m.reshape((1,(num_nodes*num_nodes - num_nodes)/2,bond_types))
        print(minimal_mats.shape)
    else:    
        minimal_mats = np.concatenate([minimal_mats,minimal_m.reshape((1,(num_nodes*num_nodes - num_nodes)/2,bond_types))],
                                      axis=0)
                                      
                                      
# define model
from keras.layers import Input, Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU

def deterministic_edge_reconstructor(num_nodes=29,atom_types=6,bond_types=4):
        num_edges = (num_nodes*num_nodes -num_nodes)/2

        inputBOLayer = Input(shape=(num_nodes,atom_types,))
        first_dense = keras.layers.Dense(50, name="first_BO_dense")(inputBOLayer)
        first_leaky = LeakyReLU(alpha=0.1,name="first_BO_LReLU")(first_dense)
        first_dropout = Dropout(rate=0.15,name="first_drop")(first_leaky)

        second_dense = keras.layers.Dense(50, name="second_BO_dense")(first_dropout)
        second_leaky = LeakyReLU(alpha=0.1,name="second_BO_LReLU")(second_dense)
        second_dropout = Dropout(rate=0.15,name="second_drop")(second_leaky)

        
        third_dense = keras.layers.Dense(num_edges*(bond_types)/num_nodes,name="third_BO_dense")(second_dropout)
        third_leaky = LeakyReLU(alpha=0.1,name="third_gen_LReLU")(third_dense)
        bo_reshape = keras.layers.Reshape((num_edges,bond_types),name="reshape")(third_leaky)
        #bo_legalize = keras.layers.Multiply(name='mask apply')[bo_reshape,bo_mask]

        bo_logits = keras.layers.Dense(bond_types, activation='linear',name="linear_logits")(bo_reshape)
        bo_samples = keras.layers.Lambda(gumbel_softmax_mask,name="gumbel_softmax")(bo_logits)

        bo_model = keras.models.Model(inputs=[inputBOLayer],outputs=[bo_samples])
        return(bo_model)
        
        
np.random.seed(3)
# get random partition
msk = np.random.rand(np.shape(headers_padded)[0]) < 0.75
# test and train split
train_matrices_minimal = minimal_mats[msk,:,:]
train_headers_padded = headers_padded[msk,:]
train_connectivities_padded = connectivities_padded[msk,:,:]
test_matrices_minimal_mats = minimal_mats[~msk,:,:]
test_headers_padded = headers_padded[~msk,:]
test_connectivities_padded = connectivities_padded[~msk,:,:]

bo_model_n = deterministic_edge_reconstructor()
bo_model_n.compile(optimizer='adam',loss='categorical_crossentropy',
                 metrics=['categorical_accuracy','categorical_crossentropy'])


                                        
