"""
This module contains the admm solver.
"""
import numpy as np

class ADMMSolver:
    """Class for ADMM solver
   
   ADMM explores the optimal solution of the follwoing l1-type cost
   function: 
                         E(x) = Ax + l1*|x|
   In addition, we consider 
       ・Tostal Square Variation : |tsvQx|^2 + |tsvQx|^2
       ・x > 0 (definition of spectrum)

    Args:
        p1, p2 (float) : parameters for ADMM algorithm
        eps (float) : convergence criteria
        max_itr (ndarray, int) : maximum number of iterations
        status (bool) : if true, ADMM is convergence.
    """
    def __init__(self, p1=0.1, p2=0.1, max_itr=3000, eps=10.0**(-5)):
        self._p1 = p1
        self._p2 = p2
        self._eps = eps
        self._max_itr = max_itr
        self._status = False
    
    @property
    def status(self):
        return self._status
    
    @property
    def p1(self):
        return self._p1

    def _softth(self, v, thres):
        """Method for soft threshoding 

        Args:
            v (ndarray, float) : vector
            thres (float) : threshold value

        Returns:
            result (ndarray, float) : soft-thresholded vector 
        """
        result = np.zeros(len(v))
        for i in range(len(v)):
            sign = 0.0
            if v[i] >= 0.0:
                sign = 1.0
            else:
                sign = -1.0
            result[i] = sign * max(0.0, abs(v[i])-thres)
        return result
    
    def _maxvec(self, v, thres):
        """Method for maxv vector

        result[i] = max(v[i], thres)

        Args:
            v (ndarray, float) : vector
            thres (float) : threshold value

        Returns:
            result (ndarray, float) : max vector
        """
        result = np.zeros(len(v))
        for i in range(len(v)):
            result[i] = max(v[i], thres)
        return result
    
    def fit(self, y, A, weight, tsv1, tsv2, l1, l2):
        """Method for running ADMM

        Args:
            y (ndarray, float) : measurement vector
            A (ndarray, float) : measurement matrix
            weight (ndarray, float) : weight array
            tsv1, tsv2 (ndarray, float) : TSV array
            l1 (float) : coefficient for l1-norm
            l2 (float) : coefficient for TSV term

        Returns:
            x (ndarray, float) : original signal
        """
        # initialization
        status = False
        xsize = len(A[0])
        x = np.dot(A.T, y)
        h1, h2 = np.zeros(xsize), np.zeros(xsize)
        z1, z2 = np.zeros(xsize), np.zeros(xsize)

        # Transpose(A) times y
        tvec = np.dot(A.T, weight*y)
        
        # inverse matrix in ADMM
        tmp = np.dot(A.T, np.dot(np.diag(weight), A))
        tmp = np.dot(A.T, np.dot(np.diag(weight), A)) \
              + (self._p1+self._p2)*np.identity(A[0].size) \
              + l2*tsv1 + l2*tsv2
        invmat = np.linalg.inv(tmp)
        
        # iteration 
        for itr in range(self._max_itr):
            xold = x
            z1old, z2old = z1, z2
            
            # update x
            tmp = tvec + self._p1*z1-h1 + self._p2*z2-h2
            x = np.dot(invmat, tmp)
                 
            # update z1 and h1
            z1 = self._softth(x+(1.0/self._p1)*h1, l1/self._p1)
            h1 = h1 + self._p1*(x-z1)
                   
            # update z2 and h2
            z2 = self._maxvec(x+(1.0/self._p2)*h2, 0.0)
            h2 = h2 + self._p2*(x-z2)
            
            # residual
            res1 = np.linalg.norm(x-z1, ord=2)
            res2 = np.linalg.norm(x-z2, ord=2)
            
            # convergence check
            if res1 < self._eps and res2 < self._eps:
                self._status = True
                break
        
        # check x>0
        if np.any(x<0.0):
            x_negative = x[np.where(x<0.0)]
            if np.any(np.abs(x_negative)>10e-5):
                raise ValueError("Violation of Constraint X>0")
            else:
                x[np.where(x<0.0)] = 0.0

        return x