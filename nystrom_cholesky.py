import numpy as np
from scipy.linalg import svd
from scipy.linalg import cholesky

class NystromCholesky:

    def __init__(self, kernel, num_sample) -> None:
        self.kernel = kernel
        self.num_sample = num_sample
        self.basis_ = None
        self.normalization_ = None
        self._trained_flag = False
        self._prev_transform = None

        print("We use the Nyström Approximation WITH Incomplete Cholesky Decomposition")

    def fit(self, X, kern_el_dict=None, gram_matrix=None):
        """ This function randomly chooses landmark points and constructs the Nystrom embedding."""
        assert self.num_sample <= len(X), "Cannot sample more than supplied. {} <= {}".format(self.num_sample, len(X)) 
        X = np.array(X)

        # Compute the incomplete Cholesky decomposition
        covariance_matrix = np.cov(X)  # Compute the covariance matrix
        covariance_matrix += np.eye(covariance_matrix.shape[0]) * 0.01
        cholesky_decomp = cholesky(covariance_matrix, lower=True, overwrite_a=True)

        # Generate a set of indices based on the Cholesky decomposition
        selected_indices = np.argsort(np.sum(cholesky_decomp**2, axis=0))[:self.num_sample]

        # Select the corresponding basis using the selected indices
        self.basis_indxs_ = selected_indices # Instead of the random selection, that has been done before, we now select based on the incomplete CHolesky decomposition
        self.basis_ = X[selected_indices,:]

        if gram_matrix is None:
            basis_kernel = self.kernel(self.basis_, self.basis_, kern_el_dict=kern_el_dict)
        else:
            basis_kernel = gram_matrix[self.basis_indxs_]
            basis_kernel = basis_kernel[:, self.basis_indxs_]

        U, S, V = svd(basis_kernel)
        S = np.maximum(S, 1e-12)
        self.normalization_ = np.dot(U / np.sqrt(S), V)
        
        self._trained_flag = True
        return self

    def transform(self, X, kern_el_dict=None, gram_matrix=None):
        assert self._trained_flag, "Must train prior to transformation."

        if gram_matrix is None:
            embedded = self.kernel(X, self.basis_, kern_el_dict=kern_el_dict)
        else:
            selected_indices = np.arange(self.num_sample)  # Use all landmark indices
            embedded = gram_matrix[:, selected_indices]

        return np.dot(embedded, self.normalization_.T)

    def fit_transform(self, X, kern_el_dict=None, gram_matrix=None):
        self.fit(X, kern_el_dict=kern_el_dict, gram_matrix=gram_matrix)
        self._prev_transform = self.transform(X, kern_el_dict=kern_el_dict, gram_matrix=gram_matrix)
        return self._prev_transform