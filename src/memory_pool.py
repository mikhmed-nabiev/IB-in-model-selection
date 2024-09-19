import numpy as np
from copy import deepcopy

class Pool:
    """Memory pool for models to access."""

    def __init__(self, nscalars : int, nvectors : int, nfeatures):
        """
        Initiazlize pool of scalars and vectors.

        Args:
          nscalars: Total of scalar 
          nvectors: Total of vector
          nfeatures: Dimensionality of vectors

        TODO: add matrices 
        """
        self._nscalars = nscalars
        self._nvectors = nvectors
        self._nfeatures = nfeatures
        self._scalars = np.zeros(nscalars)
        self._vectors = np.zeros(shape=(nvectors, nfeatures))
            
    # Setting variables
    def set_zeros(self):
        self._scalars = np.zeros(self._nscalars)
        self._vecotrs = np.zeros(shape=(self._nvectors, self._nfeatures))

    def set_scalar(self, index: int, value: int):
        self._scalars[index] = value 
        
    def set_scalars(self, values: np.array):
        assert values.shape == self._scalars.shape
        self._scalars = deepcopy(values)
        
    def set_vector(self, index: int, value: np.array):
        self._vectors[index] = deepcopy(value)
        
    def set_vectors(self, values: np.array):
        assert values.shape == self._vectors.shape
        self._vectors = deepcopy(values)
        
    # Getting variables
    def get_scalar(self, index: int):
        return self._scalars[index]
    
    def get_scalars(self, indices: list = None):
        if indices is None:
            return self._scalars
        return self._scalars[indices]
    
    def get_vector(self, index: int):
        return self._vectors[index]
    
    def get_vectors(self, indices: list = None):
        if indices is None:
            return self.vectors 
        return self._vectors[indices]

    # Pretty print pool state
    def __repr__(self):
        return f"Pool(\nscalars={self._scalars}, \nvectors=\n{self._vectors}\n)"