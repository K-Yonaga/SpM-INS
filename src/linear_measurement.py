"""
This module contains the functions for linear measurement.
"""
import math
import numpy as np

def get_measurement_matrix(
    dQ, dE, Q_grid, E_grid, intensity, spectrum):
    """Function for calculating measurement matrix
 
    Linear measurement problem in inelastic neutron scattering (INS) is
    defined as
                      I = R*S + noise,
    where I is a measurement intensity, R is a measurement matrix, S is 
    a original spectrum, respectively. This function calculates 
    the measurement matrix R.

    In INS, the measurement matrix is given by
                R = 4*log(2)*Q_grid*E_grid / pi*dQ*dE,
    Q_grid (E_grid) is the grid size in the Q- (E-) direction. dQ and dE
    correspond to the resolutions in Q- and E-directions, respectively.

    Args:
        dQ, dE (float) : resolutions
        Q_grid, E_grid (float) : grid sizes
        intensity, spectrum (Spectrum) : vectorized spectra

    Returns:
        measurement_matrix (ndarray, float) : measurement matrix
    """
    # scale factor
    ln_fac = 4.0 * math.log(2.0)
    normal_fac = Q_grid*E_grid*ln_fac / (math.pi*dQ*dE)

    size_intensity = intensity.Qmesh.size
    Qmesh_intensity = intensity.Qmesh
    Emesh_intensity = intensity.Emesh

    size_spectrum = spectrum.Qmesh.size
    Qmesh_spectrum = spectrum.Qmesh
    Emesh_spectrum = spectrum.Emesh

    # calculate measurment matrix
    measurement_matrix = np.zeros((size_intensity, size_spectrum))
    for i in range(size_intensity):
        for j in range(size_spectrum):
            Qi = Qmesh_intensity[i]
            Ei = Emesh_intensity[i]
            Qj = Qmesh_spectrum[j]
            Ej = Emesh_spectrum[j]
            measurement_matrix[i][j] = normal_fac \
                                       * np.exp(-1.0*ln_fac*(Qi-Qj)**2/dQ**2) \
                                       * np.exp(-1.0*ln_fac*(Ei-Ej)**2/dE**2)

    return measurement_matrix


def get_weight(intensity, weight_type="snr"):
    """Function for weight array
  
    Args:
        intensity (Spectrum) : intensity data
        weight_type (str) : weight type (default:snr)

    Returns:
        weight (ndarray, float) : weight array
    """
    if weight_type == "ones":
        weight = np.ones((intensity.size))

    elif weight_type == "snr":
        weight = np.zeros((intensity.size))
        for i in range(intensity.size):
            if intensity.value[i] != 0.0:
                weight[i] = np.abs(intensity.value[i]/intensity.error[i])
            elif intensity.value[i] == 0.0:
                weight[i] = 1.0
                
    return weight