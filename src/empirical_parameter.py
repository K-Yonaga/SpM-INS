"""
This module contains the functions for calculating the empirical
hyperparameters in L1-TSV.
"""
import numpy as np

def get_empirical_tsv_parameter(measurement_matrix, weight, eta=0.5):
    """Function for calculating empirical hyperparameter
 
    Args:
        measurement_matrix (ndarray, float) : measurement matrix
        weight (ndarray, float) : weight for measurment

    Returns:
        l2 (float) : hyperparameters for TSV
    """
    tmp = np.dot(measurement_matrix.T, np.dot(np.diag(weight), measurement_matrix))
    l2 = np.max(np.abs(tmp))**eta
    return l2