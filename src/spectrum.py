"""
This module contains the manager class for spectrum.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Spectrum: 
    """Class for Spectrum
  
    To describe the inelastic neutron scattering with a linear measurement
    problem, we represent measured and estimated spectra with vectors.  
 
    Args:
        size (int) : size of vectorized spectrum
        Qsize, Esize (int) : sizes for Q and E directions
        Qmesh, Emesh (ndarray, int) : meshgrid
        value (ndarray, float) : value of data
        error (ndarray, float) : error of data
    """
  
    def __init__(
        self, 
        size=None, 
        Qsize=None, 
        Esize=None, 
        Qmesh=np.empty(0), 
        Emesh=np.empty(0), 
        value=np.empty(0), 
        error=np.empty(0),
        zero_thres=10e-2):
        self._size = size
        self._Qsize = Qsize
        self._Esize = Esize
        self._Qmesh = Qmesh
        self._Emesh = Emesh
        self._value = value
        self._error = error
        self._zero_thres = zero_thres

    @property
    def size(self):
        if np.any(self._value<0):
            return self._value[np.where(self._value>=0.0)].size
        else:
            return self._value.size

    @property
    def Qsize(self):
        if np.any(self._value<0):
            return len(np.unique(self._Qmesh[np.where(self._value>=0.0)]))
        else:
            return len(np.unique(self._Qmesh))

    @property
    def Esize(self):
        if np.any(self._value<0):
            return len(np.unique(self._Emesh[np.where(self._value>=0.0)]))
        else:
            return len(np.unique(self._Emesh))

    @property 
    def Qmesh(self):
        if np.any(self._value<0):
            return self._Qmesh[np.where(self._value>=0.0)]
        else:
            return self._Qmesh

    @property
    def Emesh(self):
        if np.any(self._value<0):
            return self._Emesh[np.where(self._value>=0.0)]
        else:
            return self._Emesh
    
    @property 
    def value(self):
        if np.any(self._value<0):
            return self._value[np.where(self._value>=0.0)]
        else:
            return self._value

    @property
    def error(self):
        if np.any(self._value<0):
            return self._error[np.where(self._value>=0.0)]
        else:
            return self._error

    @property 
    def Qaxis(self):
        if np.any(self._value<0):
            return np.unique(self._Qmesh[np.where(self._value>=0.0)])
        else:
            return np.unique(self._Qmesh)

    @property 
    def Eaxis(self):
        if np.any(self._value<0):
            return np.unique(self._Emesh[np.where(self._value>=0.0)])
        else:
            return np.unique(self._Emesh)
    
    @size.setter
    def size(self, new_size):
        self._size = new_size

    @Qmesh.setter
    def Qmesh(self, new_Qmesh):
        self._Qmesh = new_Qmesh

    @Emesh.setter
    def Emesh(self, new_Emesh):
        self._Emesh = new_Emesh
        
    @value.setter
    def value(self, new_value):
        self._value = new_value

    @error.setter
    def error(self, new_error):
        self._error = new_error

    def set_parameters(self, Q_min, Q_max, Q_grid, E_min, E_max, E_grid):
        """Method for reading iexy file
        """
        Qarray = np.arange(Q_min, Q_max+Q_grid, Q_grid)
        Earray = np.arange(E_min, E_max+E_grid, E_grid)
        self._Qsize = len(Qarray)
        self._Esize = len(Earray)
        self._size = self._Qsize * self._Esize
        
        Etemp, Qtemp = np.meshgrid(Earray, Qarray)
        self._Qmesh = Qtemp.ravel()
        self._Emesh = Etemp.ravel()

        self._value = np.zeros(self._size)
        self._error = np.zeros(self._size)

    def read_iexy(
        self, Q_min, Q_max, E_min, E_max, value_min, value_max, 
        file_path, normalization=False):
        """Method for reading iexy file

        Args:
            file_path (str): path to iexy file.
            normalization (bool) : if True, self.value will be normalized.
        """
        file = np.loadtxt(file_path)
        self._value = np.zeros(len(file))
        self._error = np.zeros(len(file))
        self._Qmesh = np.zeros(len(file))
        self._Emesh = np.zeros(len(file))

        counter = 0
        for i in range(len(file)):
            value, error, Q, E = file[i]
            if (value_min <= value and value <= value_max and 
                Q_min <= Q and Q <= Q_max and E_min <= E and E <= E_max):
                if value <= 0.0 and np.abs(value)<self._zero_thres:
                        self._value[counter] = 0.0
                        self._error[counter] = 0.0
                else:
                    self._value[counter] = value
                    self._error[counter] = error
                        
                self._Qmesh[counter] = Q
                self._Emesh[counter] = E
                counter += 1

        self._size = counter
        self._Qmesh = np.resize(self._Qmesh, self._size)
        self._Emesh = np.resize(self._Emesh, self._size)
        self._value = np.resize(self._value, self._size)
        self._error = np.resize(self._error, self._size)
        
        if normalization == True:
            max_val = np.max(np.abs(self._value))
            self._value /= max_val

    def plot(self, blank=True):
        X, Y = np.meshgrid(np.unique(self._Qmesh), np.unique(self._Emesh))
        plot_size = np.unique(self._Qmesh).size*np.unique(self._Emesh).size

        # Check size
        if self._value.size == plot_size:
            temp = self._value.reshape(X.shape, order='F')

        elif self._value.size < plot_size:
            # Padding missing values with -1
            Q = np.unique(self._Qmesh)
            E = np.unique(self._Emesh)

            list_data = []
            for i in range(self.size):
                list_data.append((self._Qmesh[i], self._Emesh[i]))

            temp = np.full((len(E), len(Q)), -1.0)
            for j, ev in enumerate(E):
                for i, qv in enumerate(Q):
                    try:
                        idx = list_data.index((qv, ev))
                        temp[j, i] = self._value[idx]
                    except:
                        pass
                    
        else:
             raise ValueError('Check data size')

        if blank == True:
            temp =  np.ma.masked_where(temp<0, temp)
        plt.pcolormesh(X, Y, temp, cmap=cm.gnuplot, vmin=0.0)

        plt.colorbar()
        plt.axis('tight')
        plt.xlabel('$\it{Q}$', fontsize='20')
        plt.ylabel('$\it{E}$', fontsize='20')
        plt.show()