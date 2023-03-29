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
        Kxx = kernel.K(x, x)
        KxX = kernel.K(x, X)
        KXX = kernel.K(X, X)
        W = KXX + noise * np.eye(len(X))
        W_inv = cho_solve(cho_factor(W), np.eye(len(W)))
        temp = (KxX.dot(W_inv)).dot(KxX.T)
        return Kxx - temp

    def compute_value(self, AF, x, X=None):
        print("AF.shape: ", AF.shape)
        
        var_size = self.v_lim
        if X is None:
            X = self.model.model.X
        Gauss_noise = self.model.model.parameters[1].param_array
        N = len(x) // var_size
        p = len(x) % var_size
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
        Costs=None,
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
        self.costs = Costs
        self.verbose = verbose

    def compute_batch(self, verbose=True, X_testing=None):
        L_obj_val = []

        AF = self.acquisition._compute_acq(self.search_domain)
        if self.costs is not None:
            Costs = [self.costs[int(jj)]*ii for (ii,jj) in self.search_domain.tolist()]
            Costs = np.array(Costs).reshape(-1,1)            
            AF /= Costs
        #TO DO: Costs array from search_domain---> elementwise AF = AF/Costs
        # Costs is an array given to this function: default: Costs=None
        # Costs is attribute of the experiment like descriptors


        idx = np.argmax(AF)
        X_af =np.atleast_2d(self.search_domain[idx])
        # to_remove = np.where(self.search_domain==X_af)[0]
        to_remove = np.where((self.search_domain==X_af).all(axis=1))
        self.search_domain=np.delete(self.search_domain, to_remove, axis=0)

        X_batch = X_af
        for j in range(1, self.batch_size):
            AF = np.delete(AF, idx).reshape(-1,1)
            LAW = self.objective.compute_value(AF, self.search_domain, X=X_batch) 
            idx = np.argmax(LAW)
            new_sample = np.atleast_2d(self.search_domain[idx])
            # to_remove = np.where(self.search_domain==new_sample)[0]
            to_remove = np.where((self.search_domain==new_sample).all(axis=1))

            self.search_domain=np.delete(self.search_domain, to_remove, axis=0)
            L_max = LAW[idx]

            if verbose:
                print("New sample: {}, max LAW:{}".format(new_sample, L_max))
            X_batch = np.vstack([X_batch, new_sample])
            if X_testing is not None:
                L_obj_val.append(self.objective.compute_value(AF, self.search_domain, X=X_testing))
        return X_batch, L_obj_val

    @staticmethod
    def model_from_dict(dict_model):
        ''' Builds model from loaded file '''
        
        gpmodel = dict_model.pop('gp_model')
        gp_reg_model = gpmodel.pop('model')
        # gp_reg_model = GPy.models.GPRegression(**gp_reg_dict)
        GPModel = GPModel(**gpmodel)
        GPModel.model = gp_reg_model
        optimizer = LAW_BOptimizer(**dict_model)
        return  optimizer

    def create_dict(self):
        '''Creates a dictionary to be saved'''
        D ={k: self.__dict__[k]
            for k in ['batch_size', 'acquisition_name', 'law_params', 'kernel', 'optimize_restarts']
            }
        # --clean up: removal the kernel from dictionary necessary to 
        # --rebuild the optimizer from the dictionary D
        gm_dict = self.model.__dict__
        if 'kernel'in gm_dict.keys():
            del gm_dict ['kernel']
        D.update({'gp_model':gm_dict})
        return D




class LAW_BOptimizer_old(object):

    def  __init__(
                    self,
                    domain,
                    # acquisition_name,
                    batch_size,
                    law_params,
                    kernel, 
                    optimize_restarts,
                    weigth_function =  None
                    ):
        self.batch_size = batch_size
        # self.acquisition_name = acquisition_name
        space = Design_space(domain)

        if self.acquisition_name == "EI":
            AF = EI(self.model, space, optimizer=None)
        else:
            raise NotImplementedError("Only EI acquisition function implemented at the moment")
        self.acquisition = AF
        self.law_params = law_params
        self.kernel = kernel
        self.optimize_restarts = optimize_restarts

    def create_model(self, X, Y,  **constraints_dict):
        ''' Creates model and sets ups self.model=model'''
        kernel = self.kernel.copy()
        noise_var = np.std(Y)*0.001
        gp_model = GPModel(kernel, optimize_restarts=self.optimize_restarts)
        gp_model.model = GPRegression(X, Y, kernel=kernel, noise_var=noise_var)
        # gp_model.model = GPRegression(X, Y, kernel=kernel)
        self.model=gp_model
        self.apply_model_constraints(constraints_dict)
        gp_model.model.optimize()
        # self.model=gp_model

    @staticmethod
    def model_from_dict(dict_model):
        ''' Builds model from loaded file '''
        
        gpmodel = dict_model.pop('gp_model')
        gp_reg_model = gpmodel.pop('model')
        # gp_reg_model = GPy.models.GPRegression(**gp_reg_dict)
        GPModel = GPModel(**gpmodel)
        GPModel.model = gp_reg_model
        optimizer = LAW_BOptimizer(**dict_model)
        return  optimizer

    def create_dict(self):
        '''Creates a dictionary to be saved'''
        # D ={k: self.__dict__[k]
        #     for k in ['batch_size', 'acquisition_name', 'law_params', 'kernel', 'optimize_restarts']
        #     }
        D =self.__dict__
        # --clean up: removal the kernel from dictionary necessary to 
        # --rebuild the optimizer from the dictionary D
        gm_dict = self.model.__dict__
        if 'kernel'in gm_dict.keys():
            del gm_dict ['kernel']
        D.update({'gp_model':gm_dict})
        return D

    def apply_model_constraints(self, constraints_dict):
        '''  '''
        if self.model is None:
            raise  Exception("Error: optimizer has no model")
        for k in constraints_dict.keys():
            for j, c in constraints_dict[k].items():
                if isinstance(c, float):
                    self.model.model.parameters[int(k)].parameters[int(j)].constrain_fixed(c)
                elif isinstance(c, list):
                    self.model.model.parameters[int(k)].parameters[int(j)].constrain_bounded(*c)


    def update_model(self, X_new, Y_new):

        X= np.vstack([self.model.model.X, X_new ])
        Y = np.vstack([self.model.model.Y, Y_new])
        self.model.model.set_XY(X, Y)
        self.model.model.optimize()



    def compute_batch(self, verbose=True, X_testing=None):
        L_obj_val = []

        AF = self.acquisition._compute_acq(self.search_domain)
        idx = np.argmax(AF)
        X_af =np.atleast_2d(self.search_domain[idx])
        to_remove = np.where(self.search_domain==X_af)[0]
        self.search_domain=np.delete(self.search_domain, to_remove, axis=0)

        X_batch = X_af
        for j in range(1, self.batch_size):
            AF = np.delete(AF, idx).reshape(-1,1)
            LAW = self.objective.compute_value(AF, self.search_domain, X=X_batch) 
            idx = np.argmax(LAW)
            new_sample = np.atleast_2d(self.search_domain[idx])
            to_remove = np.where(self.search_domain==new_sample)[0]
            self.search_domain=np.delete(self.search_domain, to_remove, axis=0)
            L_max = LAW[idx]

            if verbose:
                print("New sample: {}, max LAW:{}".format(new_sample, L_max))
            X_batch = np.vstack([X_batch, new_sample])
            if X_testing is not None:
                L_obj_val.append(self.objective.compute_value(AF, self.search_domain, X=X_testing))
        return X_batch, L_obj_val







    def create_evaluator(self, domain,  weight_func=None):
        '''
            Creates the evaluator and sets it as attribute of the LAW_BOptimizer 
        '''
        space = Design_space(domain)

        ## -- Create Acquisition function --

        if self.acquisition_name == "EI":
            from GPyOpt.acquisitions.EI import AcquisitionEI as EI
            AF = EI(self.model, space, optimizer=None)
        else:
            raise NotImplementedError("Only EI acquisition function implemented at the moment")
        
        ## -- Create the wirght function for LAW2ORDER --

        v_lim = self.law_params['LAW_params']['var_size']
        c = self.law_params['LAW_params']['c_value']
        b = self.law_params['LAW_params']['b_value']
        if weight_func is None:
            weight_func = lambda x: c +b*x
        LAW_func  = LAW_acq(self.model, weight_func, v_lim=v_lim, normalize_values=False)
        X_domain  = domain['domain']
        evaluator = LAW_Evaluator(self.batch_size, X_domain, AF, objective=LAW_func, verbose = True)
        self.evaluator = evaluator 

    def suggest_batch(self, X_new, Y_new,  **constraints_dict):

        if self.model is None:
            model = self.create_model( X_new, Y_new,  **constraints_dict)
            self.model = model
            X_all, Y_all  = X_new, Y_new
        else:
            X = self.model.model.X.copy()
            Y = self.model.model.Y.copy()
            X_all = np.vstack([X, X_new])
            Y_all = np.vstack([Y, Y_new])
        self.model.updateModel(X_all=X_all, Y_all=Y_all,
                               X_new=None, Y_new=None
                               )
        X_batch =self.evaluator.compute_batch()[0]
        return X_batch


# %%
if __name__ == "__main__":
    from GPy.kern import RBF


    law_params = {
                "var_size":10,
                "c_value": 1,
                "b_value": 10
                }
    
    model_constraints = {
                        "0":{"0":0.5,"1":0.1}, 
                        "1": {"0":[1e-6, 10]}
                        }
    GPy_kernel = RBF(input_dim=2, ARD=True)
    kernel = Kernels.CoulombKernel(input_dim=2, GPy_kern=GPy_kernel, seed=123)


    bopt = LAW_BOptimizer(
                    acquisition_name="EI",
                    batch_size=3,
                    weigth_function=None,
                    law_params=law_params,
                    kernel=kernel,
                    optimize_restarts=1
                    )


# %%
