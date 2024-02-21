#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import networkx as nx
import gudhi as gd


# In[2]:
 
#from FRC_filtration_commented_Jonatas import *


# In[8]:


#%load_ext cython


# In[18]:


#%%cython

#import networkx as nx
#import gudhi as gd
#import numpy as np
#from itertools import combinations

def cliques_gudhi_matrix_comb(distance_matrix,f,d=2):
  
    
    rips_complex = gd.RipsComplex(distance_matrix=distance_matrix, max_edge_length=f)
    Rips_simplex_tree_sample = rips_complex.create_simplex_tree(max_dimension = d)
    rips_generator = Rips_simplex_tree_sample.get_filtration()
    
    for i in rips_generator:
        c,w=i
        if w>0.:
            yield c
    
    #return (i[0] for i in rips_generator if i[1]>0. )

def compute_euler(Mat,cutoff,max_dim):
    eu=len(Mat)
    
    C=cliques_gudhi_matrix_comb(Mat,cutoff,max_dim) # C is the clique complex ordered by weight and dimension
    for c in C:
        
        d=len(c)-1
        eu+=(-1)**d
        
    return eu, C

def shannon_entropy(probabilities):
    """Calculate the Shannon entropy of a probability distribution."""
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def compute_shannon_entropy(M,cutoff,max_dim):
    
    epsilon=1e-20
    
    N={d:{} for d in range(1,max_dim+1)}
    
    G=nx.Graph(M)
    Neigh={i:set(G.neighbors(i)) for i in G.nodes()}
    
    C=cliques_gudhi_matrix_comb(M,cutoff,max_dim)
    eu=len(M)
    for c in C:
        
        d=len(c)-1
        eu+=(-1)**d
        #print(d)
        boundary=[b for b in combinations(c,d)]
        n=sum([len(set.intersection(*[Neigh[j] for j in l])) for l in boundary])-1
        try:
            N[d][n]+=n
        except:
            N[d][n]=0
            N[d][n]+=n
            
  
    for d in N.keys():
        norm=sum(N[d].values())
        N[d]=[N[d][i]/norm for i in N[d].keys() if N[d][i]>=epsilon]
     
    E={d:shannon_entropy(N[d]) for d in N.keys()}
    E["euler"]=eu
     
    return E
        
    

