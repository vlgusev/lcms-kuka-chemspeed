# %%
import numpy as np
# import config
from GPy.kern.src.rbf import RBF
from GPy.kern import Kern
from GPy.core.parameterization.param import  Param
from paramz.transformations import Logexp
from dscribe.kernels import REMatchKernel as Rematch

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


class GPyREMatchKern(Kern):
    '''
        domain is a dictionary {idx:molecule_dict}, where molecule_dict is an ase.Atoms dicctionary
        from which it is pssible to build descriptors using DScribe.
        domain is retrieved at the beginning of the experiment via read_config(). 

    '''


    def __init__(self, input_dim, domain=None, seed=None, active_dims=None, useGPU=False, 
                 alpha=0.1, threshold=1e-6, metric="rbf", gamma=0.5, degree=3, 
                 coef0=1, kernel_params=None, normalize_kernel=True):
    
        super(GPyREMatchKern, self).__init__(input_dim, active_dims, name = 'Rematch')

        self.kernel=Rematch(alpha=alpha, threshold=threshold, metric=metric,
                        gamma=gamma, degree=degree, coef0=coef0, kernel_params=kernel_params,
                        normalize_kernel=normalize_kernel)

        self._name = "Rematch"
        self.metric = metric
        self.domain = domain
        self.gamma = Param("gamma", self.kernel.gamma, Logexp())
        self.alpha = Param("alpha", self.kernel.alpha,Logexp())
        self.degree = degree
        self.coef0 = 1
        self.kernel_params=kernel_params
        self.normalize_kernel=normalize_kernel
        self.threshold = threshold
        self.link_parameters(self.gamma, self.alpha)

    def look_up(self, X, n=1):
        ''' Converts indexes into descriptors.
            X: 2D array containing the indexes (int)
            descriptors: the full set of descriptors, from which to 
                         search by index.
            returns D: list of descriptors'''
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
        return self.kernel.create(descr_1, y=descr_2)

    def Kdiag(self, X):

        ''' '''
        if self.kernel.normalize_kernel:
            Kdiag = np.ones(len(X))
        else:
            Kdiag = np.zeros(len(X))
            D =  self.look_up(X)
            for j, d in enumerate(D):
                Kdiag[j] = self.kernel.create([D[j]], y=[D[j]])
        return Kdiag

    def update_gradients_full(self, dL_dK, X1, X2=None):
        pass

    def to_dict(self):
        # return super().to_dict()
        pass
        
    @staticmethod
    def from_dict(input_dict):
        pass
              

