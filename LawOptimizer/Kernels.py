# %%
import numpy as np
# import config
from GPy.kern.src.rbf import RBF
from GPy.kern import Kern
from GPy.core.parameterization.param import  Param
from paramz.transformations import Logexp

# %%
class CoulombKernel(Kern):
    ''' Wrapper od GPy kernel to be used with eigenvalues Coulomb descriptors.
        input_dim: dimensionality of input data (not of the descriptors)
        GPy_kern: GPy kernel used with the descriptors.
        domain: list of indeces of all the molecules available.   
        
    '''

    def __init__(self, input_dim, GPy_kern=None, domain=None, seed = None, active_dims=None, useGPU=False, *args, **options):
        super().__init__(input_dim, active_dims, useGPU=useGPU, name="Coulomb", *args, **options)

        self._name = "CoulombKernel"
        self.kernel = GPy_kern
        self.domain =domain
        self.variance=self.kernel.variance
        self.lengthscale=self.kernel.lengthscale
        self.link_parameters(self.variance, self.lengthscale)
        if options: 
            self.options = options
        else:
            self.options = {'permutation':"eigenspectrum", 'flatten':False}

    
    def look_up(self, X):
        ''' Utility  method to look-up for descriptor from index.
            useful when the serch space is the space of the indexes
            X : 2D array of indexes'''
        idxs = X[:,1].tolist()
        concs = X[:,0].reshape(-1,1)
        D = [self.domain[int(j)] for j in idxs]
        D = np.vstack(D)
        mols = np.hstack([concs, D])
        return mols
    

    def K(self, X1, X2=None):

        if X2 is None:
            X2 = X1
        descr_1 = self.look_up(X1)
        descr_2 = self.look_up(X2)
        return self.kernel.K(descr_1, X2=descr_2)

    def Kdiag(self,X):
        descr = self.look_up(X)
        return self.kernel.Kdiag(descr)


    def update_gradients_full(self, dL_dK, X1, X2=None):
        X2=X1 if X2 is None else X2
        descr_1 = self.look_up(X1)
        descr_2 = self.look_up(X2)

        self.kernel.update_gradients_full(dL_dK, descr_1, X2=descr_2)
    
    def to_dict(self):
        pass


    @staticmethod
    def from_dict(input_dict):
        # return super().from_dict(input_dict)
        pass


