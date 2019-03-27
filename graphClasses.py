import itertools as it
import numpy as np
import scipy as sp
import copy
class simpleGraph:
    ## custom class for graph manipulation
    def __init__(self,maxdim = 15):
        ## max possible size of graph
        # I think a max graph size is
        # inevitable for a Markov-process 
        # graph generation scheme, which is 
        # where I think this will go
        
        ## number of verticies
        self.nvert = maxdim
        
        ## number of edges
        self.nedge = int(sp.special.comb(maxdim,2))
        
        ## attribued verticies list
        self.ats = np.zeros((maxdim))
        
        ## attributed edge list
        self.edges = np.zeros((self.nedge))
    
        ## unattributed edge list
        self.uedges = np.zeros((self.nedge))
        
        ## holders for BO mat and AM
        self.bomat =  np.zeros((self.nvert,self.nvert))
        self.am =  np.zeros((self.nvert,self.nvert))
        
        ## holders for neighbors 
        self.neighbor1 = []
        self.neighbor2 = []
        
        ## set internal counter 
        self.__intcount__ = 0
        self.__indlist__ = range(maxdim)

    def graphLoader(self,record):
        ## function to read in from csv
        with open(record,'r') as f:
            for ii, line in enumerate(f.readlines()):
                if ii == 0:
                    self.ats = np.array([float(i) for i in line.split(",")])
                    self.nvert = len(self.ats)
                    self.bomat = np.zeros((self.nvert,self.nvert))
                else:
                    self.bomat[ii-1,:] =  np.array([float(i) for i in line.split(",")])
         
            
        ## now get the raw AM
        self.am = copy.copy(self.bomat)
        self.am[self.am > 1] = 1

        ## now do ordered attribute read for BO/AM matrices
        for ii, c in enumerate(it.combinations(range(0,self.nvert),2)):
            self.edges[ii] = self.bomat[c]
            self.uedges[ii] = self.am[c]
        ## bulid out connectivtiy mape
        self.__buildNeighborhoods()   
        
    def getVertexAts(self,ind):
        return(self.ats[ind])
    
    def getNeighbors1(self,ind):
        return(list(it.compress(self.__indlist__,self.neighbor1[ind])))
            
    def getNeighbors2(self,ind):
        return(list(it.compress(self.__indlist__,self.neighbor2[ind])))

    def __buildNeighborhoods(self):    
        ## get inds of 1-neighbors 
        # from basic graph ops
        amsq = np.matmul(self.am,self.am) 
        
        amsq -= np.diag(np.diag(amsq))
        
    
        ## populate neighborhoods
        self.neighbor1 = self.am>0
        self.neighbor2 = amsq>0
