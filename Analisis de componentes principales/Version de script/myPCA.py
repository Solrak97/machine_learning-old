from cmath import sqrt
import numpy as np
import numpy.linalg as alg


class My_PCA:
    def __init__(self, n_components = -1):
        self.n_components = n_components
        self.matrix = []
        self.inertia = []
        self.points = []
        self.eigenvalues = []
        self.eigenvectors = []
    pass

    def fit(self, X):
        redux = self.center_and_scale(X)
        corr = self.corr_matrix(redux)
        self.eigenvalues, self.eigenvectors = self.eigensomething(corr)
        self.matrix = self.pca_matrix(self.eigenvectors)
        self.remove_components()
        self.inertia = self.calculate_inertia(self.eigenvalues)
        self.points = self.calculate_points(self.eigenvalues)
        pass


    def transform(self, X):
        return np.matmul(X, self.matrix)


    def center_and_scale(self, X):
        _X = X      

        mmean = np.mean(_X, axis=0)
        mstd = np.std(_X, axis=0)

        _X = (_X - mmean) / mstd
        return _X


    def corr_matrix(self, X):
        n, m = X.shape
        return (1/n) * np.matmul(X.transpose(), X)

    
    def eigensomething(self, R):
        w, v = alg.eigh(R)
        sort_index = np.argsort(abs(w))[::-1]
        sorted_eigenvals = w[sort_index]
        sorted_eigenvecs = v[:, sort_index]
        
        return (sorted_eigenvals, sorted_eigenvecs)


    def pca_matrix(self, vectors):
        return np.array(vectors)


    def remove_components(self):
        self.matrix = np.delete(self.matrix, np.s_[self.n_components - 1 : -1], axis = 1)
        self.eigenvalues = self.eigenvalues[0:self.n_components]
        pass

    def calculate_inertia(self, eigenvalues):
        n, m = self.matrix.shape
        return eigenvalues / m

    
    def calculate_points(self, eigenvalues):
        w0 = eigenvalues[0]
        w1 = eigenvalues[1]
        v0 = self.matrix[:,0]
        v1 = self.matrix[:,1]
        
        return ((v0 * sqrt(w0)).real, (v1 * sqrt(w1)).real)