"""
This module contains the cross-validation manager.
"""
import joblib
import numpy as np
import pandas as pd
from scipy import stats
import itertools as itr
from sklearn.model_selection import KFold

class CrossValidationManager:
    """Class for cross validation
   
    Args:
        y (Spectrum) : measurement data
        A (ndarray, float) : measurement matrix
        weight (ndarray, float) : weight array
        tsv1, tsv2 (ndarray, float) : coefficients of total square variation
        solver (class) : solver class, e.g. ADMMSolver
        num_folds (int) : number of folds in CV
        num_jobs (int) : number of parallel running jobs
        _l1, _l2 (float) : hyperparameters
        self._score (DataFrame) : score of cross validation
    """
    def __init__(
        self,  
        y, 
        A, 
        weight, 
        tsv1, 
        tsv2, 
        solver, 
        num_folds=10,
        num_jobs=-1):
        self._y = y
        self._A = A
        self._weight = weight
        self._tsv1 = tsv1
        self._tsv2 = tsv2
        self._l1 = None
        self._l2 = None
        self._solver = solver
        self._num_folds = num_folds
        self._num_jobs = num_jobs
        
    @property
    def score(self):
        return self._score

    @property
    def l1(self):
        return self._l1

    @property
    def l2(self):
        return self._l2

    def _get_cv_error(self, l1, l2, train, test):
        """Method for calculating CV error

        Args:
            l1, l2 (float) : hyperparameters
            train (ndarray) : training set indices
            test (ndarray) : testing set indices

        Returns:
            cv_error (float) : mean square error with weight
        """
        y_train, y_test = self._y[train], self._y[test]
        A_train, A_test = self._A[train, :], self._A[test, :]
        w_train, w_test = self._weight[train], self._weight[test],
                    
        # training 
        x = self._solver.fit(
            y_train, A_train, w_train, self._tsv1, self._tsv2, l1, l2)
    
        # test (computing cv_error)
        temp = y_test - np.dot(A_test, x)
        cv_error = 0.5 * np.sum(w_test*temp**2) / len(y_test)
        return cv_error

    def _get_loocv_error(self, l1, l2):
        """Method for calculating approximated LOOCV error

        Args:
            l1, l2 (float) : hyperparameters

        Returns:
            cv_error (float) : mean square error with weight
        """
        # Solve optimization problem
        x = self._solver.fit(self._y, self._A, self._weight, self._tsv1, self._tsv2, l1, l2)
        
        # active index
        thres = l1 / self._solver.p1
        act = np.where(np.abs(x)>=thres)[0].tolist()
        
        # approximated loocv errors
        alpha = len(self._y) / len(x)
        rho = len(act) / len(x)
        mse = 0.5 * self._weight * (self._y-np.dot(self._A,x))**2
        errors = (alpha/(alpha-rho))**2 * mse
        return errors
    
    def run_kfold_cv(self, l1, l2):
        """Method for running kfold CV

        Args:
            l1, l2 (float) : hyperparameters

        Returns:
            mean of CV error (float)
            std of CV error (float)
        """
        kf = KFold(n_splits=self._num_folds, shuffle=True)
        cv_errors = joblib.Parallel(self._num_jobs)(
            joblib.delayed(self._get_cv_error)(l1, l2, train, test) for train, test in kf.split(self._y))
        
        return np.mean(cv_errors), stats.sem(cv_errors)

    def run_loocv(self, l1, l2):
        """Method for running approximated LOOCV

        Args:
            l1, l2 (float) : hyperparameters

        Returns:
            mean of CV error (float)
            std of CV error (float)
        """
        cv_errors = self._get_loocv_error(l1, l2)
        return np.mean(cv_errors), stats.sem(cv_errors)

    def run_grid_search(self, l1_list, l2_list, cv_type='Kfold'):
        """Method for running grid search with kfold CV

        Args:
            l1_list, l2_list (list) : lists for hayper-parameters
            cv_type (str) : type of cross validation (Kfold or LOOCV)

        Note:
            Calculation results are stored in self._score.
        """
        self._score = pd.DataFrame(
            columns=['l1', 'l2', 'mean error','standard error'])
        if cv_type == 'Kfold':
            _run_cv = self.run_kfold_cv
        elif cv_type == 'LOOCV':
            _run_cv = self.run_loocv
        else:
            raise ValueError('cv_type must be "Kfold" or "LOOCV"')
        
        for k, (l1,l2) in enumerate(itr.product(l1_list, l2_list)):
            cv_error_mean, cv_error_sem = _run_cv(l1, l2)
            self._score.loc[k] = [l1, l2, cv_error_mean, cv_error_sem]

    def _get_lowest_error_parameters(self):
        """Method for selecting hyperparameters

        Reteruns:
            l1_lowest, l2_lowest (float) : 
                hyperparameters that minimize CV error
        """
        lowest_index = np.argmin(self._score['mean error'].values)
        l1_lowest = self._score.iloc[lowest_index]['l1']
        l2_lowest = self._score.iloc[lowest_index]['l2']
        return l1_lowest, l2_lowest

    def _get_one_standard_error_parameters(self):
        """Method for selecting hyperparameters

        Reteruns:
            l1_one_standard, l2_one_standard (float) : 
                hyperparameters based on one-standard-error
        """
        lowest_index = np.argmin(self._score['mean error'].values)
        lowest_error = self._score.iloc[lowest_index]['mean error']
        lowest_sem = self._score.iloc[lowest_index]['standard error']
        one_standard_error = lowest_error + lowest_sem

        temp = self._score[self._score['mean error']<=one_standard_error]
        index = np.argmax(temp['mean error'].values)
        l1_one_standard = temp.iloc[index]['l1']
        l2_one_standard = temp.iloc[index]['l2']
        
        return l1_one_standard, l2_one_standard

    def get_best_solution(self, selection_mode='Lowest Error'):
        """Method for the fitting result with the best hyperparameters

        Args:
            selection_mode (str) : 
                'Lowest Error' or 'One Standard Error'

        Reteruns:
            fitting result (ndarray, float)
        """
        if selection_mode == 'Lowest Error':
            self._l1, self._l2 = self._get_lowest_error_parameters()
        elif selection_mode == 'One Standard Error':
            self._l1, self._l2 = self._get_one_standard_error_parameters()
        else:
            raise ValueError('Not supported')

        return self._solver.fit(
            self._y, 
            self._A, 
            self._weight, 
            self._tsv1, 
            self._tsv2, 
            self._l1,
            self._l2)