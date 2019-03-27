import os, sys, glob
import numpy as np
import scipy as sp
import copy
import keras
from graphClasses import *

## layers import
from keras.layers import SimpleRNN, LSTM, Input, Dense, RNN, SimpleRNNCell
from keras.models import Model


## load graphs
graphlist = glob.glob('qm9graph/*.csv')
graphs = []
for g in graphlist:
    g1 = simpleGraph()
    g1.graphLoader(g)
    graphs.append(g1)

## run info
ind = 14
maxdim = 15


## regular/flattened X
xx = np.zeros((len(graphs),105))
yy = np.zeros((len(graphs),15))
for ii, gr in enumerate(graphs):
    xx[ii,:] = np.reshape(gr.edges,(1,105))
    yy[ii,:] = np.reshape(gr.ats,(1,15))


## let's try a simply fully connected network
## to predict input attributes
maxdim = 15
inputs = Input(shape=(105,))

dense1 = Dense(units=100,activation='relu')(inputs)
dense2 = Dense(units=100,activation='relu')(dense1)
output = Dense(units=15,activation='linear')(dense2)

model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam',loss='mean_squared_error')


model.fit(xx,yy,epochs=20)





## sequence X values
xxx = np.reshape(xx,(len(graphs),105,1))

## form the y values as catageorical
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
ohe_encoder = OneHotEncoder()
yyy= np.zeros([len(graphs),15,6])
yyyy = np.reshape(yy,(len(graphs),15,1))
for i, y in  enumerate(yy):
    for j,e in enumerate(y):
        if int(e) == 0:
            yyy[i,j,0] = 1
        elif int(e) == 1:
            yyy[i,j,1] = 1
        elif int(e) == 6:
            yyy[i,j,2] = 1
        elif int(e) == 7:
            yyy[i,j,3] = 1
        elif int(e) == 8:
            yyy[i,j,4] = 1
        elif int(e) == 9:
            yyy[i,j,5] = 1
            
            
## let's a sequence classifier to 
## assign atom types based on input
maxdim = 15
inputs = Input(shape=(15,1))
LSTMstack1 = LSTM(units=100,return_sequences=True)(inputs)
LSTMstack2 = LSTM(units=100,return_sequences=True)(LSTMstack1)
output = Dense(units=6,activation='softmax')(LSTMstack2)
LSTM_classifier_model = Model(inputs=inputs, outputs=output)
LSTM_classifier_model.compile(optimizer='adam',loss='categorical_crossentropy')

LSTM_classifier_model.fit(x=yyyy,y=yyy,epochs=5,batch_size=2)
