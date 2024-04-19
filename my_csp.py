from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import eigh
import numpy as np

# The two methods needed for the pipeline are fit and transform
class CSP(TransformerMixin, BaseEstimator):
    """
    Common Spatial Patterns (CSP) algorithm. 
    This algorithm is used to extract features from EEG signals.
    The algorithm is based on the class CSP from mne.decoding library.
    """
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.filters_ = None

    def fit(self, X, y):
        # Check all the inputs
        if not isinstance(X, np.ndarray):
            raise ValueError(f"X should be of type ndarray (got {type(X)}).")
        if y is not None and len(X) != len(y) or len(y) < 1:
                raise ValueError("X and y must have the same length.")
        if X.ndim < 3:
            raise ValueError("X must have at least 3 dimensions.")
        
        # Check that the number of classes is greater than 2
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        if n_classes != 2:
            raise ValueError("n_classes must be equal to 2.")
        
        # Calculate the covariance matrix for each class
        covs = []
        for unique_classe in unique_classes:
            x_class = X[y == unique_classe]
            _, n_channels, _ = x_class.shape

            x_class = np.transpose(x_class, [1, 0, 2])
            x_class = x_class.reshape(n_channels, -1)
            cov = np.cov(x_class)

            covs.append(cov)
        covs = np.stack(covs)

        eigen_values, eigen_vectors = eigh(covs[0], covs.sum(0))
        # Sort eigen values in descending order
        ix = np.argsort(np.abs(eigen_values - 0.5))[::-1]

        # Reorgonaize the eigen vectors columns to palce the biggest eigen values first
        eigen_vectors = eigen_vectors[:, ix]
        self.filters_ = eigen_vectors.T

        return self
    
    def transform(self, X):
        # Check the input
        if not isinstance(X, np.ndarray):
            raise ValueError(f"X should be of type ndarray (got {type(X)}).")
        
        if self.filters_ is None:
            raise RuntimeError("Please fit the model first.")
        
        pick_filters = self.filters_[: self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

        # compute features (mean band power)
        X = (X**2).mean(axis=2)
        return np.log(X)