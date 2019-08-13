import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # Need if Conda environment isn't resolved properly
import keras
import sklearn.model_selection
import keras.backend as K
import tensorflow as tf

def draw_model(model):
    IPython.display.SVG(model_to_dot(model,show_shapes=True).create(prog='dot', format='svg'))

def sample_gumbel(shape, eps=1e-20): 
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature): 
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    k = tf.shape(logits)[-1]
    #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y

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

def generate_gc_atom_layer(num_nodes=29, atom_hidden_length=5, bond_hidden_length=3, hide_atoms=False,\
                      message_dense_resize=None, atom_dense_resize=None,
                      nonlinear_state_update = False,layer_number=None):
                          
    if layer_number is None:
        layer_number = "_1"
    else:
        layer_number = "_"+str(layer_number)
        
    # Generates a graph convolution model based on the provided parameters.
    bond_hiddens_input = keras.layers.Input(shape=(num_nodes,num_nodes,bond_hidden_length),
    name="bond_input" +layer_number)
    atom_hiddens_input = keras.layers.Input(shape=(num_nodes,atom_hidden_length),
    name="atom_input" +layer_number)
    connectivity_input = keras.layers.Input(shape=(num_nodes,num_nodes),
    name="connectivity_input" +layer_number)
    
    # For JP's task
    if hide_atoms:
        atom_hiddens = keras.layers.Lambda(lambda x: x*0.0)(atom_hiddens_input) # For JP's exercise - eliminate atom info.
    else:
        atom_hiddens = atom_hiddens_input
    
    message_stack = keras.layers.Lambda(stacker, name="stacker" +layer_number)([bond_hiddens_input, atom_hiddens])
    
    # Should we dense the total hidden vector?
    if message_dense_resize != None:
        messages = keras.layers.Dense(message_dense_resize, activation='relu',
        name="message_dense" +layer_number)(message_stack)
    else:
        messages = message_stack
    
    message_sum = keras.layers.Lambda(summer,
    name="message_sum" +layer_number)([messages, connectivity_input])
    
    # Should we dense the atom hidden vector?
    if atom_dense_resize != None:
        message_interpret = keras.layers.Dense(atom_dense_resize, activation='relu',
        name="interpret_dense" +layer_number)(message_sum)
    else:
        message_interpret = message_sum
        
    combined_message_state = keras.layers.Concatenate(axis=2, name="combine_message" +layer_number)([atom_hiddens_input, message_interpret])
    print('combined shape is ' +str(combined_message_state.shape))


    if  nonlinear_state_update:
        message_to_out = keras.layers.Dense(atom_hidden_length, activation='relu',name="nonlinear_combine" +layer_number)(combined_message_state)

    else:
        message_to_out = keras.layers.Dense(atom_hidden_length, activation='linear',name="linear_combine" +layer_number)(combined_message_state)
    print('bond_hiddens_input shape is ' +str(bond_hiddens_input.shape))
    print('atom_hiddens_input shape is ' +str(atom_hiddens_input.shape))
    print('connectivity_input shape is ' +str(connectivity_input.shape))
    print('message_to_out shape is ' +str(message_to_out.shape))
    
    model = keras.models.Model(inputs=[bond_hiddens_input, atom_hiddens_input, connectivity_input], outputs=[message_to_out])
    return model
    
    
