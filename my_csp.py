from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import eigh

# The two methods needed for the pipeline are fit and transform
class CSP(TransformerMixin, BaseEstimator):
    def __init__(self) -> None:
        pass

    def fit(self, X, y):
        # covs = [np.cov(X[y==i].T) for i in np.unique(y)]
        # eigen_values, eigen_vectors = eigh(covs[0], covs.sum(0))
        return self
    
    def transform(self, X):
        return X