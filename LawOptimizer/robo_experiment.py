
# %%
import numpy as np
from copy import copy 
from itertools import product
import os 
import json
import pandas as pd
from time import sleep
from Kernels import *
import LAW_2_ORDER as LAW
import GPyOpt
import GPy


# %%

# class Chem_GPModel(GPyOpt.models.GPModel):

#     def __init__(self, kernel, noise_var=None, exact_feval=False, optimizer='bfgs', max_iters=1000, optimize_restarts=5, 
#                  sparse=False, num_inducing=10, verbose=False, ARD=False, mean_function=None, Gauss_noise_lim = None):
#         super().__init__(kernel=kernel, noise_var=noise_var, exact_feval=exact_feval, optimizer=optimizer, 
#                          max_iters=max_iters, optimize_restarts=optimize_restarts, sparse=sparse, 
#                          num_inducing=num_inducing, verbose=verbose, ARD=ARD, mean_function=mean_function)
#         self.Gauss_noise_lim =  Gauss_noise_lim
        

    # def _create_model(self, X, Y):

    #     kern = self.kernel
    #     self.kernel = None
    #     noise_lim  = self.Gauss_noise_lim
    #     if not self.sparse:
    #         self.model = GPy.models.GPRegression(X, Y, kernel=kern, mean_function=self.mean_function)
    #     else:
    #         self.model = GPy.models.SparseGPRegression(X, Y, kernel=kern, num_inducing=self.num_inducing, mean_function=self.mean_function)
    #     if self.Gauss_noise_lim is not None and len(noise_lim) ==1:
    #         self.model.Gaussian_noise.variance.constrain_fixed(noise_lim, warning=False)
    #     elif self.Gauss_noise_lim is not None and len(noise_lim) ==2:
    #         self.model.Gaussian_noise.variance.constrain_bounded(*noise_lim, warning=False)






#%%

class Experiment(object):


    def __init__(self,
                root_path = "C:\\ACL_UoL\\LawOptimiser\\",
                settings_file = "expsettings.json",
                descr_path = "descriptors\\descriptors_{}.npy",
                exp_res_path = "experiments",
                exp_res_file_start = "PFAS_Dyes-res-",
                batch_file_start = "PFAS_Dyes-batch-"                
                 
                 ):
        self.root_path = root_path
        self.exp_res_path = os.path.join(root_path, exp_res_path)
        self.exp_res_file_start = exp_res_file_start
        self.batch_file_start = batch_file_start
        self.num_batch = 0
        self.runnning = False
        self.acquisition_name = None
        self.kernel_name = None
        self.kern_params = None
        self.batch_size = None
        self.law_params = None
        self.create_directories()
        self.X = None
        self.Y = None 
        if settings_file is  None:
            raise
        else:
            self.settings_file =  os.path.join(self.root_path, settings_file) 

    def create_directories(self, 
                           dirs = ["exp_results", "descriptors", "sugg_batches"],
                           names =["exp_path", "descr_path", "save_batch_path"]
                           ):
        '''
            Creates directores if they do not exist, and set their absoulte 
            paths as an attribute of the experiment
        '''
        for d, n in zip(dirs, names):
            try:
                os.makedirs(os.path.join(self.root_path, d), exist_ok=True)
                setattr(self, n, self.root_path + d)
            except:
                raise

    def apply_settings(self):
        
        '''
            Sets the following attributes of the experiment:
            config_path: the address of the file with the coordinates and molecular indexes
            descr_path: address of the file with the descriptors
            descriptors: the descriptors of all the molecules of the search space
        '''

        with open(self.settings_file, "r") as f:
            settings = json.load(f) 

        #  -- Setting batch size, problem type n_starts 
        self.__dict__.update(settings['exp_settings'])

        # -- Setting the  config_path and the descr_path
        dataset_name = settings['dataset']
        descript_type = settings["descr_params"]["descript_type"]
        chem_config_path = "./{}.xyz".format(dataset_name)
        self.config_path = chem_config_path
        descr_file_name = "{}.npy".format(dataset_name)
        descr_path = os.path.join(self.descr_path, descr_file_name)
        # descr_path = os.path.join(self.root_path, descr_file_name)
        self.descr_path = descr_path
        
        # -- Setting all the possible conpounds of the experiment
        self.set_compounds(settings)  

        # -- Setting the parameters for the BO method
        self.law_params = settings['LAW_params']
        self.acquisition_name = settings['bo_params']['Acquisition']
        self.kernel_name = settings['bo_params']['kern_name']
        self.kern_params = settings['bo_params']['kern_params']
        self.model_constraints = settings['bo_params']['model_constraints'] 

        return settings
        

    def set_compounds(self,settings):
        ''' 
            Sets the descriptors  which will be used to create the optimizer.
            Looks for a file containing the lsit of descriptors. 
            Loads the file if it exixst, otherwise creates the descriptors 
            and saves them, using DScribe. 
            Gets a dictionary chem_formula:mol_idx: will be used to read exp data 
            and convert formulas to mol idx

        '''
        descr_params = settings["descr_params"]
        if os.path.isfile(self.descr_path):
            print("Loading descriptors")
            descriptors = np.load(
                                self.descr_path, 
                                allow_pickle=True
                                ).item()
            self.descriptors =  descriptors
            self.space_size = len(descriptors)
            
        else:
            print('Error: No descriptor file found') 
            return


    def get_inputs(self, skiprows=0):

        exp_files = [f for f in os.listdir(self.exp_res_path) if f.startswith(self.exp_res_file_start)]
        new_exp_files=[f for f in exp_files
                       if int(f.rstrip(".run").split("-")[-1]) == self.num_batch]

        if len(new_exp_files) == 0:
            return

        elif  new_exp_files:
            X, Y  = copy(self.X), copy(self.Y)
            
            exp_file = os.path.join(self.exp_res_path, new_exp_files[0]) 
            new_data = pd.read_csv(exp_file, skiprows=0)
            new_data.drop(["SampleIndex"], axis=1, inplace=True)
            Y_new = new_data["PeakArea"].values
            Y_new = Y_new.reshape(-1,1)
            X_data=new_data.drop("PeakArea", axis=1).values
            X_data=X_data[:,:-1]

            # -- conversion data into x = np.array([concentration, mol_idx])
            idxs = [np.where(x > 0)[0] for x in list( X_data)]
            idxs = np.vstack(idxs)
            concs =  X_data[ X_data > 0].reshape(-1,1)
            X_new = np.hstack([concs, idxs])
        if X is not None:
            X=np.vstack([X,  X_new])
            Y=np.vstack([Y, Y_new])
        else:
            X =  X_new
            Y = Y_new
        return X, Y


    def create_kernel(self):
        ''' '''
        descr_dim = len(self.descriptors[0]) # dimensionality of the descriptors
        descr_dim = (self.descriptors[0]).shape[1] # dimensionality of the descriptors
        
        print('descriptors dimension:', descr_dim)
        # --  Creating the kernel 
        if self.kernel_name == "CoulombKernel":
            GPy_kernel = RBF(input_dim=descr_dim, ARD=True)
            kernel = CoulombKernel(input_dim=NDIMS, GPy_kern=GPy_kernel, domain=self.descriptors)
        else:
            raise ValueError("Only CoulombKernel is implemented at the moment")
        return kernel
    
    def create_model(self, kernel, X, Y, optimize_restarts=5):
        ''' '''
        kernel = kernel.copy()
        noise_var = np.std(Y)*0.001
        gp_model = LAW.GPModel(kernel, optimize_restarts=optimize_restarts, ARD=True)
        gp_model.model = LAW.GPRegression(X, Y, kernel=kernel, noise_var=noise_var)
        constraints = self.model_constraints
        for k in  constraints.keys():
            for j, c in  constraints[k].items():
                if isinstance(c, float):
                    gp_model.model.parameters[int(k)].parameters[int(j)].constrain_fixed(c)
                elif isinstance(c, list):
                    gp_model.model.parameters[int(k)].parameters[int(j)].constrain_bounded(*c)
        return gp_model
    
    def create_LAW_optimizer(self, search_domain, domain,X_new, Y_new, costs=None, stored_data = None):
        ''' Creates the LAW_optimizer.
            To be used when the experiment has not yet started, and no optimizer 
            model has been saved  yet. 
            search_domain: 2D array with all the input points.
        '''


        v_lim = self.law_params['var_size']
        b, c = self.law_params['b_value'], self.law_params['c_value']
        weight_function = lambda x:c + b*x

        if stored_data is None:
            kernel = self.create_kernel()
            gp_model =self.create_model(kernel.copy(), X_new, Y_new)
        else:
            gp_model= LAW.LAW_BOptimizer.model_from_dict(data)
        gp_model.model.optimize()
        space = LAW.Design_space(domain)
        acq_class = eval("LAW." + exp.acquisition_name)
        AF = acq_class(gp_model, space, optimizer=None)

        LAW_func = LAW.LAW_acq(
                                model=gp_model,
                                v_lim=v_lim,
                                weight_func=weight_function
                            )
        optimizer =LAW.LAW_BOptimizer(
                                    
                                batch_size = self.batch_size,
                                search_domain = search_domain,
                                acquisition = AF,
                                objective=LAW_func,
                                costs=costs,
                                verbose=False,
                                ) 
        return optimizer

    def create_initial_batch(self, search_domain):
        ''' Creates initial suggested experiment to run an saves it to file'''

        ii=np.random.choice(range(len(search_domain)), self.batch_size, replace=False)
        selected_exp = np.array(search_domain)[ii]
        self.num_batch +=1
        self.save_batch(selected_exp)
        return selected_exp


    def  suggest_batch(self, optimizer, X_testing=None):
        '''
            Runs optimizer.run_bo
            saves the new file with the last suggested batch.
            Deletes file with last model.
            Saves new model.
        '''
        self.runnning = True
        batch = optimizer.compute_batch(X_testing=X_testing)
        out = batch[0] if X_testing is None else batch
        self.num_batch +=1
        self.runnning = False
        # self.space_size -= self.batch_size
        return out
    
    def last_batch(self):
        pass

    def save_batch(self, X_batch):
        ''' 
            Saves the suggested batch to file.
        '''
        print('batch num: ',self.num_batch)
        f_name = self.batch_file_start + "{num}.run".format(num=self.num_batch)
        save_path =os.path.join(self.exp_res_path, f_name)   
        columns = ['SampleIndex']
        columns.extend(self.compounds)  
        columns.append('Water')
        # X_out = np.zeros((len(X_batch), len(columns)))
        X_out = np.zeros((self.batch_size, len(columns)))
        for j, x in enumerate(list(X_batch)):
            ii = int(x[1]); value= x[0]
            X_out[j,ii]=value
            X_out[j,-1]=1-value
        sample_idxs = (np.arange(1,17) + self.batch_size*(self.num_batch))
        X_out[:,0]=sample_idxs
        np.savetxt(save_path, X_out, fmt='%.3f', delimiter=',' ,header= ",".join(columns), comments='')



# %%

if __name__ == "__main__":

    root_path = "./"
    NDIMS = 2
    SLEEP_TIME = 5
    COUNT = 0

    exp = Experiment(
                    root_path = root_path,
                    settings_file = os.path.join(root_path, "expsettings.json"),
                    descr_path = os.path.join(root_path, "descriptors/descriptors_{}.npy"),
                    exp_res_path = os.path.join(root_path, "experiments"),
                    )
    optimizer_path = os.path.join(root_path, "optimizer.npy")
    if os.path.exists(optimizer_path):
        os.remove(optimizer_path)
    exp.apply_settings()
    # root_path = exp.root_path
    root_dir = exp.root_path
 

    mol_idxs = list(exp.descriptors.keys())
    domain = [{'name':'concentration', 'type':'discrete', 'domain':0.1*np.arange(1,11), 'dimensionality':1},
              {'name':'mol_id', 'type':'discrete', 'domain':mol_idxs, 'dimensionality':1}]

    search_domain = list(product([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], mol_idxs))
    # search_domain = list(product([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 
    #                                1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 
    #                                2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8], mol_idxs))

    DF_costs = pd.read_csv(os.path.join(root_path,'costs_compounds.csv'))
    costs = dict(zip(DF_costs['idx'].values.tolist(), 
                      DF_costs['costs'].values.tolist()))
    exp.compounds=DF_costs['names'].values.tolist()

    X_init = exp.create_initial_batch(search_domain)

    # -- remove initial inputs from the search space
    to_remove = list(map(tuple, X_init.tolist()))
    search_domain_init=list(set(search_domain) - set(to_remove))
    X_domain_init = np.array(search_domain_init)
    exp.space_size = len(X_domain_init)

    while exp.space_size > 0:
        while exp.runnning == True:
            sleep(SLEEP_TIME)
            print('optimizer running')
            continue
        data = exp.get_inputs()
        if data is None:
            sleep(SLEEP_TIME)
            continue
        else:
            X_new, Y_new = data
        # -- if no experiment has just started (COUNT = 0) there is no optimizer: create one
        if COUNT == 0:
            optimizer =  exp.create_LAW_optimizer(X_domain_init, domain, X_new, Y_new, costs=costs)

        # -- If there is an optimizer model saved, the optimizer created from file  
        if "optimizer.npy" in os.listdir(root_path):
            data = np.load(os.path.join(root_path,'optimizer.npy'), allow_pickle=True).item()
            search_domain=data['search_domain']
            costs = data['costs']
            optimizer = exp.create_LAW_optimizer(search_domain, domain, X_new, Y_new, costs=costs, stored_data=data)

        # -- If there is no optimizer model file, all is created from scratch: 
        else:
            optimizer = exp.create_LAW_optimizer(X_domain_init, domain, X_new, Y_new, costs=costs)
        
        optimizer.update_gpmodel(X_new, Y_new)

        # -- Get and save the batch
        if exp.space_size >= exp.batch_size:
            X_batch = exp.suggest_batch(optimizer)
        else:
            print('LAST BATCH, EXPERIMET OVER')
            X_batch = optimizer.search_domain
            m,n = X_batch.shape
            X_batch = np.vstack([X_batch, np.zeros((exp.batch_size-m,n))]) 
        optimizer.update_gpmodel(X_new, Y_new)
        exp.save_batch(X_batch)
        model_dict= optimizer.create_model_dict()
        exp.space_size -= exp.batch_size
        np.save(os.path.join(root_path,'optimizer'), model_dict)
        COUNT +=1






# %%
