# %%

import numpy as np
import copy
from scipy.linalg import cho_factor, cho_solve
from sklearn.preprocessing import normalize, minmax_scale
from GPyOpt.models import GPModel
from GPyOpt.acquisitions.EI import AcquisitionEI as EI
from GPy.models import GPRegression
from GPyOpt.core.task.space import Design_space
import Kernels

# %%
class LAW_acq(object):
    def __init__(self, model, weight_func, v_lim=10, normalize_values=False):
        self.normalize = normalize_values
        self.v_lim = v_lim
        self.weight_func = weight_func
        self.model = model

    def compute_variance(self, X, x, noise=None):
        """This function will be in the LAW_acq class:
        var(x_j) = K(x_j, x_j) - K(x_j, X)(K(X,X) - sigma^2I)^(-1)K(X, x_j)
        var(x_j) has shape len(x_j) X len((x_j))"""

        kernel = self.model.model.kern
        if noise is None:
            noise = self.model.model.parameters[1].param_array
        # if len(x==0):
        # print('shape X at compute_variance: ', x.shape)
            # return -999
    
        Kxx = kernel.K(x, x)
        KxX = kernel.K(x, X)
        KXX = kernel.K(X, X)
        W = KXX + noise * np.eye(len(X))
        W_inv = cho_solve(cho_factor(W), np.eye(len(W)))
        temp = (KxX.dot(W_inv)).dot(KxX.T)

        return Kxx - temp

    def compute_value(self, AF, x, X=None):
        # print("AF.shape: ", AF.shape)
        
        var_size = self.v_lim
        if X is None:
            X = self.model.model.X
        Gauss_noise = self.model.model.parameters[1].param_array
        N = len(x) // var_size
        p = len(x) % var_size
        # if self.compute_variance(X, np.atleast_2d(x))==-999:
        #     return -999
        if N == 0:
            var = np.diag(self.compute_variance(X, np.atleast_2d(x), noise=Gauss_noise))
            var = var.reshape(-1, 1)
        else:
            var = np.zeros((len(x), 1))
            for j in range(N):
                cov = self.compute_variance(
                    X, np.atleast_2d(x[j * var_size : (j + 1) * var_size]), noise=Gauss_noise
                )
                var[j * var_size : (j + 1) * var_size] = np.diag(cov).reshape(-1, 1)
            if p !=0:
                cov = self.compute_variance(
                    X, np.atleast_2d(x[N * var_size : N * var_size + p]), noise=Gauss_noise
                )
                var[N * var_size : N * var_size + p] = np.diag(cov).reshape(-1, 1)
        if self.normalize:
            norm_var = minmax_scale(var)
            norm_acq_value = minmax_scale(AF)
            return norm_var * (self.weight_func(norm_acq_value)) ** 2
        self.var = var
        return var * (self.weight_func(AF)) ** 2





class LAW_BOptimizer(object):
    def __init__(
        self,
        batch_size,
        search_domain,
        acquisition,
        objective=None,
        # n_jobs=1,
        costs=None,
        verbose=False,
    ):
        """ acquisition_function: the GPyOpt acqusition function chosen
            objective: the function to maximize in order to sample points for the batch
            search_domain: 2D array of all the available combinations [concentration, molecular_id]
            acquisition: A GPyOpt acquisition function
            Costs: dictionsry {molecular_id: cost(£/gram)}
        """

        self.objective = objective
        self.acquisition = acquisition
        self.model = acquisition.model
        self.batch_size = batch_size
        self.search_domain = search_domain
        self.costs = costs
        self.verbose = verbose

    def compute_batch(self, verbose=True, X_testing=None):
        # print(len(self.search_domain))
        L_obj_val = []
        AF = self.acquisition._compute_acq(self.search_domain)
        if self.costs is not None:
            Costs = [self.costs[int(jj)]*ii for (ii,jj) in self.search_domain.tolist()]
            Costs = np.array(Costs).reshape(-1,1)            
            AF /= Costs

        idx = np.argmax(AF)
        X_af =np.atleast_2d(self.search_domain[idx])
        to_remove = np.where((self.search_domain==X_af).all(axis=1))
        self.search_domain=np.delete(self.search_domain, to_remove, axis=0)

        X_batch = X_af
        for j in range(1, self.batch_size):
            # print(len(self.search_domain))

            AF = np.delete(AF, idx).reshape(-1,1)
            try:
                LAW = self.objective.compute_value(AF, self.search_domain, X=X_batch) 
            except:
                print('compute_batch stopped at step: ', j)
                print('X_batch shape: ', X_batch.shape)
            idx = np.argmax(LAW)
            new_sample = np.atleast_2d(self.search_domain[idx])
            to_remove = np.where((self.search_domain==new_sample).all(axis=1))
            self.search_domain=np.delete(self.search_domain, to_remove, axis=0)
            L_max = LAW[idx]

            if self.verbose:
                print("New sample: {}, max LAW:{}".format(new_sample, L_max))
            X_batch = np.vstack([X_batch, new_sample])
            if X_testing is not None:
                L_obj_val.append(self.objective.compute_value(AF, self.search_domain, X=X_testing))
            # print('X_batch shape: ', X_batch.shape)
        return X_batch, L_obj_val
    
    def update_gpmodel(self, X_new, Y_new):
        
        X=self.acquisition.model.model.X.copy()
        Y=self.acquisition.model.model.Y.copy()
        X1, Y1 = np.vstack([X,X_new]), np.vstack([Y,Y_new])
        self.acquisition.model.model.set_XY(X1,Y1)
        self.acquisition.model.model.optimize()






    @staticmethod
    def model_from_dict(dict_model):
        ''' Builds model from loaded file '''

        # gp_dict = dict_model.pop('gp_model')
        gp_dict = dict_model['gp_model']
        gp_reg_model = gp_dict.pop('model')
        gp_model = GPModel(**gp_dict)
        gp_model.model = gp_reg_model
        # optimizer = LAW_BOptimizer(**dict_model)
        # return  optimizer
        return gp_model

    def create_model_dict(self):
        '''Creates a dictionary to be saved'''
        D = self.__dict__.copy()
        model = D.pop('model')
        gm_dict = model.__dict__.copy()
        kernel = gm_dict.pop('kernel')
        del D['objective']
        del D['acquisition']
        # D.update({'gp_model':gm_dict, 'kernel':kernel})
        D.update({'gp_model':gm_dict})
        return D



# %%
