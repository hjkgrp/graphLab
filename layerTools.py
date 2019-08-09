import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # Need if Conda environment isn't resolved properly
import keras
import sklearn.model_selection
import keras.backend as K

def stacker(tensorList):
    # This is a layer.
    # Stack the bond hidden vectors and atom hidden vectors (29x29x(2*size(atom hidden) + size(bond_hidden)))
    bond_hiddens = tensorList[0]
    atom_hiddens = tensorList[1]
    vertical_atom_hiddens = keras.backend.expand_dims(atom_hiddens,axis=2)
    vertical_atom_hiddens_horizontal = keras.backend.repeat_elements(vertical_atom_hiddens,29,2)
    horiz_atom_hiddens = keras.backend.expand_dims(atom_hiddens,axis=1)
    horiz_atom_hiddens_vertical = keras.backend.repeat_elements(horiz_atom_hiddens,29,1)
    
    hidden_vector_matrix = keras.backend.concatenate([vertical_atom_hiddens_horizontal, horiz_atom_hiddens_vertical, bond_hiddens], axis=3)
    return hidden_vector_matrix

def summer(tensorList):
    # This is a layer.
    # Sums the message matrix (29x29x(size(message))) horizontally into 29x(size(message))
    messages = tensorList[0]
    connectivity = tensorList[1]
    # Zero out all unconnected messages
    connectivity_expanded = keras.backend.expand_dims(connectivity, axis=3)
    # Add all messages horizontally (29xsize(message))
    filtered = messages*connectivity_expanded # note: * is element-wise, keras.dot is matrix multiplication
    summed_horiz = keras.backend.sum(filtered, axis=2)
    return summed_horiz

def generate_gc_model(num_nodes=29, atom_hidden_length=6, bond_hidden_length=4, hide_atoms=False,\
                      message_dense_resize=None, atom_dense_resize=None, bond_dense_resize=None, do_readout = False):
    # Generates a graph convolution model based on the provided parameters.
    
    bond_hiddens_input = keras.layers.Input(shape=(num_nodes,num_nodes,bond_hidden_length))
    atom_hiddens_input = keras.layers.Input(shape=(num_nodes,atom_hidden_length))
    connectivity_input = keras.layers.Input(shape=(num_nodes,num_nodes))
    
    # For JP's task
    if hide_atoms:
        atom_hiddens = keras.layers.Lambda(lambda x: x*0.0)(atom_hiddens_input) # For JP's exercise - eliminate atom info.
    else:
        atom_hiddens = atom_hiddens_input
    
    message_stack = keras.layers.Lambda(stacker)([bond_hiddens_input, atom_hiddens])
    
    # Should we dense the total hidden vector?
    if message_dense_resize != None:
        messages = keras.layers.Dense(message_dense_resize, activation='relu')(message_stack)
    else:
        messages = message_stack
    
    message_sum = keras.layers.Lambda(summer)([messages, connectivity_input])
    
    # Should we dense the atom hidden vector?
    if atom_dense_resize != None:
        message_interpret = keras.layers.Dense(atom_dense_resize, activation='relu')(message_sum)
    else:
        message_interpret = message_sum
    
    message_to_onehot = keras.layers.Dense(atom_hidden_length, activation='softmax')(message_interpret)
    
    if do_readout:
        message_to_readout__ = keras.layers.Lambda(lambda x: K.sum(x, axis=1))(message_to_onehot)
        message_to_readout_ = keras.layers.Dense(30)(message_to_readout__)
        message_to_readout = keras.layers.Dense(1, activation='sigmoid')(message_to_readout_)
        model = keras.models.Model(inputs=[bond_hiddens_input, atom_hiddens_input, connectivity_input], outputs=message_to_readout)
        return model
    else:
        if bond_dense_resize == None:
            connectivity_output = keras.layers.Lambda(lambda x: K.identity(x))(connectivity_input)
            bond_hiddens_output = keras.layers.Lambda(lambda x: K.identity(x))(bond_hiddens_input)
        else:
            bond_hiddens_output_ = keras.layers.Dense(bond_dense_resize)(messages)
            bond_hiddens_output = keras.layers.Dense(bond_hidden_length,activation='softmax')(bond_hiddens_output_)
            connectivity_output = keras.layers.Lambda(lambda x: K.sum(x[:,:,:,1:], axis=3))(bond_hiddens_output)
        model = keras.models.Model(inputs=[bond_hiddens_input, atom_hiddens_input, connectivity_input], outputs=[bond_hiddens_output, message_to_onehot, connectivity_output])
        return model
