import os, sys, keras
import numpy as np
import pandas as pd
from graphTools import *
from layerTools import *




# Read in graphs
graph_files = glob.glob('qm9graph/*.csv')[0:1000]
mygraphs = [readgraph(fname) for fname in graph_files]
headers_onehot_dict, matrices_onehot_dict, mygraphs_standardized = standardize_graphs(mygraphs, max_size=29)
headers_padded = np.array([i['header'] for i in mygraphs_standardized])
matrices_padded = np.array([i['matrix'] for i in mygraphs_standardized])
connectivities_padded = np.array([i['connectivity'] for i in mygraphs_standardized])
origHeaders_padded = np.array([i['origHeader'] for i in mygraphs_standardized])
origMatrices_padded = np.array([i['origMatrix'] for i in mygraphs_standardized])
print headers_padded.shape
print matrices_padded.shape
print connectivities_padded.shape

# one gc layer and softmax
inital_atom_layer_1 = generate_gc_atom_layer(num_nodes=29, atom_hidden_length=5, bond_hidden_length=3, hide_atoms=False,\
                      message_dense_resize=5, atom_dense_resize=None,  nonlinear_state_update = False)
dense_read = keras.layers.Dense(32,activation='relu')(inital_atom_layer_1.output)
softmax_ouput = keras.layers.Dense(5,activation='softmax')(dense_read)


final_model = keras.models.Model(inputs=inital_atom_layer_1.inputs, outputs=[softmax_ouput])

np.random.seed(3)
# get random partition
msk = np.random.rand(np.shape(headers_padded)[0]) < 0.75


# test and train split
train_matrices_padded = matrices_padded[msk,:,:]
train_headers_padded = headers_padded[msk,:]
train_connectivities_padded = connectivities_padded[msk,:,:]
test_matrices_padded = matrices_padded[~msk,:,:]
test_headers_padded = headers_padded[~msk,:]
test_connectivities_padded = connectivities_padded[~msk,:,:]



# train
final_model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy','categorical_crossentropy'])

train_res = final_model.fit([train_matrices_padded, 
                             train_headers_padded*0.0, train_connectivities_padded],
                            train_headers_padded, epochs=50, 
                            batch_size=100,
                            validation_split=0.1,
                            verbose=True)
                            
                            
# test predictions
test_predictions = final_model.predict([test_matrices_padded, test_headers_padded, test_connectivities_padded])
test_class_pred = test_predictions.argmax(axis=2)
test_class_pred = test_predictions.argmax(axis=2)
err_rate = 100*np.sum(np.sum(test_errors))/float(test_class_act.shape[0]*test_class_act.shape[1])
print('test error rate is ' + str(err_rate))


