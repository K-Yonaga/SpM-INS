"""
This module contains the function for calculating the coefficents of 
total square variation.
"""
import numpy as np

def get_tsv_array(size, Qsize, Esize):
    """Function for the TSV coeffcients
  
    To describe the inelastic neutron scattering with a linear measurement
    problem, we represent the measurement data with vectors.  
 
    Args:
        size (int) : size of vectorized data
        Qsize (int) : size of Q-axis
        Esize (int) : size of E-axis

    Returns:
        tsvQ, tsvE (ndarray, float) : array of TSV coefficients
    """
    tsvQ = np.zeros((size, size))
    tsvE = np.zeros((size, size))
    
    for i in range(size-Esize):
        tsvQ[i][i] = 1.0 
        tsvQ[i][i+Esize] = -1.0
            
    for iQ in range(Qsize):
        for iE in range(Esize-1):
            i = iE + Esize*iQ
            tsvE[i][i] = 1.0
            tsvE[i][i+1] = -1.0
            
    tsvQ = np.dot(tsvQ.T, tsvQ)
    tsvE = np.dot(tsvE.T, tsvE)
            
    return tsvQ, tsvE