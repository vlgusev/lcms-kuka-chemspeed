
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
from Utils import *



# %%

class Experiment(object):


    def __init__(self,
                #  weigth_function_type,
                 root_path = "C:\\ACL_UoL\\LawOptimiser\\",
                 settings_file = "\\expsettings.json",
                 descr_path = "./descriptors/descriptors_{}.npy",
                 exp_res_path = "C:\\ACL_UoL\\LawOptimiser\\experiments"
                 
                 ):
        self.root_path = root_path
        self.exp_res_path = exp_res_path
        # self.descr_path = descr_path
        self.num_batch = 0
        self.runnning = False
        self.acquisition_name = None
        self.kernel_name = None
        self.kern_params = None
        self.batch_size = None
        # self.weigth_function_type = weigth_function_type
        self.law_params = None
        self.create_directories()
        self.X = None
        self.Y = None 
        if settings_file is  None:
            raise
        else:
            self.settings_file = root_path + settings_file 

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
        # descr_file_name = "{}_{}.npy".format(descript_type, dataset_name)
        descr_file_name = "{}.npy".format(dataset_name)
        descr_path = os.path.join(self.descr_path, descr_file_name)
        # descr_path = os.path.join(self.descr_path, dataset_name)
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
        # if os.path.isfile(descr_path):
        if os.path.isfile(self.descr_path):
            print("Loading descriptors")
            descriptors = np.load(
                                self.descr_path, 
                                allow_pickle=True
                                ).item()
            self.descriptors =  descriptors
            
        else:
            print('Error: No descriptor file found') 
            return
            # all_spieces, nmax, descriptors = read_config(descr_params, file_path = self.config_path)
            # np.save(self.descr_path, descriptors)
        # self.descriptors =  descriptors


    def get_inputs(self, skiprows=0):

        exp_files = list(filter(lambda x:"Dyes-res" in x, os.listdir(self.exp_res_path)))
        new_exp_files=[f for f in exp_files
                       if int(f.rstrip(".run").split("-")[-1]) > self.num_batch]
        if len(new_exp_files) == 0:
            return

        elif  new_exp_files:
            X, Y  = copy(self.X), copy(self.Y)
            
            exp_file = os.path.join(self.exp_res_path, new_exp_files[0]) 
            new_data = pd.read_csv(exp_file, skiprows=skiprows)
            new_data.drop(["SampleIndex"], axis=1, inplace=True)
            Y_new = new_data["PeakArea"].values
            Y_new = Y_new.reshape(-1,1)
            X_data=new_data.drop("PeakArea", axis=1).values

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
        # Y=Y.reshape(-1,1)
        # if  X_new.ndim == 1:
        #     X_new.reshape(-1,1)
        return X, Y


    def create_kernel(self):
        ''' '''
        descr_dim = len(self.descriptors[0]) # dimensionality of the descriptors
        
        # --  Creating the kernel 
        if self.kernel_name == "CoulombKernel":
            GPy_kernel = RBF(input_dim=descr_dim, ARD=True)
            # GPy_kernel.lengthscale.constrain_bounded(1e-09, 1e+09)
            kernel = CoulombKernel(input_dim=ndims, GPy_kern=GPy_kernel.copy(), domain=self.descriptors)
            # print(kernel.kernel)
        elif self.kernel_name ==  "REMatch":
            kernel = GPyREMatchKern(domain=domain)
            if self.kern_params is not None:
                for k, p in  self.kern_params.items():
                    setattr(kernel,k,p)
        return kernel
    
    def create_model(self, kernel, X, Y, optimize_restarts=5):
        ''' '''
        kernel = kernel.copy()
        noise_var = np.std(Y)*0.001
        gp_model = LAW.GPModel(kernel, optimize_restarts=optimize_restarts)
        gp_model.model = LAW.GPRegression(X, Y, kernel=kernel, noise_var=noise_var)
        # gp_model.model = GPRegression(X, Y, kernel=kernel)
        # self.model=gp_model

        # --  settin the gp model contraints
        constraints = self.model_constraints
        for k in  constraints.keys():
            for j, c in  constraints[k].items():
                if isinstance(c, float):
                    gp_model.model.parameters[int(k)].parameters[int(j)].constrain_fixed(c)
                elif isinstance(c, list):
                    gp_model.model.parameters[int(k)].parameters[int(j)].constrain_bounded(*c)
        return gp_model
    
    def create_LAW_optimizer(self, search_domain, domain):
        ''' Creates the LAW_optimizer.
            To be used when the experiment has not yet started, and no optimizer 
            model has been saved  yet. 
            search_domain: 2D array with all the input points.
        '''


        v_lim = self.law_params['var_size']
        b, c = self.law_params['b_value'], self.law_params['c_value']
        weight_function = lambda x:c + b*x
        kernel = self.create_kernel()
        gp_model =self.create_model(kernel.copy(), X_new, Y_new)
        # AF = self.create_acquisition()
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
                                # n_jobs=1,
                                verbose=False,
                                )
        return optimizer

    # def create_optimizer(self, ndims, optimize_restarts, domain=None, weigth_function=None):


    #     '''
    #         Creates an optimizer from scratch if there is no previous model.
    #         Loads a file and creates a model if there is a model saved. 
    #     '''
    #     # TO DO: add function to create models from file to Utils module 
    #     descr_dim = len(self.descriptors[0]) # dimensionality of the descriptors
        
    #     # --  Creating the kernel 
    #     if self.kernel_name == "CoulombKernel":
    #         GPy_kernel = RBF(input_dim=descr_dim, ARD=True)
    #         GPy_kernel.lengthscale.constrain_bounded(1e-09, 1e+09)

    #         print('The RBF kernel is:', GPy_kernel)
    #         # kernel = CoulombKernel(input_dim=ndims, GPy_kern=GPy_kernel, domain=domain)
    #         kernel = CoulombKernel(input_dim=ndims, GPy_kern=GPy_kernel.copy(), domain=self.descriptors)
    #         print(kernel.kernel)
    #     elif self.kernel_name ==  "REMatch":
    #         kernel = GPyREMatchKern(domain=domain)
    #         if self.kern_params is not None:
    #             for k, p in  self.kern_params.items():
    #                 setattr(kernel,k,p)

              
    #     bopt = LAW.LAW_BOptimizer(
    #                             domain = domain,
    #                             acquisition_name = self.acquisition_name,
    #                             batch_size = self.batch_size,
    #                             law_params = self.law_params,
    #                             kernel = kernel,
    #                             optimize_restarts=optimize_restarts,
    #                             weigth_function = weigth_function
    #     )
    #     X=self.X; Y=self.Y
    #     return bopt


    def  suggest_batch(self, optimizer, X_testing=None):
        '''
            Runs optimizer.run_bo
            saves the new file with the last suggested batch.
            Deletes file with last model.
            Saves new model.
        '''
        # update batch_file_name with updated num_batch
        # batch = bopt.run_opt
        # save batch
        self.runnning = True
        batch = optimizer.compute_batch(X_testing=X_testing)
        out = batch[0] if X_testing is None else batch
        self.num_batch +=1
        self.runnning = False
        return out

    def save_batch(self):

        
        f_name = "PFAS_Dyes-{num:04}.run".format(num=self.num_batch)
        save_path = os.path.join(exp.exp_res_path, f_name)
        X_out = np.zeros((len(X_batch), len(columns)-1))
        for j, x in enumerate(list(X_batch)):
            ii = int(x[1]); value= x[0]
            X_out[j,ii]=value
        # df = pd.DataFrame(columns=columns, data=X_batch)
        np.savetxt(save_path, X_out, fmt='%.3f', delimiter=',' ,header= ",".join(columns), comments='')





    def run(self):

        while True:
            n+=1

            # check for new experimental data
            data = self.get_inputs()
            if data is None:
                continue
            else: 
                X_new, Y_new = data 
            print('New data available')
            self.num_batch +=1
            # update the optimizer with new data to add to the dataset
            # Opt.X_new, Opt.Y_new = X_new, Y_new

            # write the new data to file.  
            # For  the moment this file is not used. 
            # Could be used by optimizer to load the data to add to its model 
            # new_data.to_csv("./new_data.csv")

            if self.runnning==True:
                continue

            X_batch = opt.run_bo()
            opt.runnning=False

            # save batch to file for the robot
            out_data=pd.DataFrame(columns = 'mol_idxs', data=X_batch)


# %%
# descr_path = "./descriptors/Test.npy" 
# domain_dict = np.load(descr_path,allow_pickle=True).item()
# domain_dict = {
#                 'DMSO': 0,
#                 'MeCN': 1,
#                 'water': 2,
#                 'Acetone': 3,
#                 'EtOAc': 4,
#                 'K2HPO4': 5,
#                 'Na2HPO4': 6,
#                 'K3PO4': 7,
#                 'K2CO3': 8,
#                 'Cs2CO3': 9
#                 }

if __name__ == "__main__":


    exp = Experiment(
                    root_path = "./",
                    settings_file = "expsettings.json",
                    exp_res_path = "./experiments/"
                    )
    exp.apply_settings()
    # ndims = len(exp.descriptors)
    ndims = 2
    mol_idxs = list(exp.descriptors.keys())
    domain = [{'name':'concentration', 'type':'discrete', 'domain':0.1*np.arange(1,11), 'dimensionality':1},
              {'name':'mol_id', 'type':'discrete', 'domain':mol_idxs, 'dimensionality':1}]

    search_domain = list(product((0.1*np.arange(1,11)).tolist(), mol_idxs ))
## TO DO load D_costs
    X_new, Y_new=exp.get_inputs()

    # -- remove initial inputs from the search space
    to_remove = list(map(tuple, X_new.tolist()))
    search_domain_init=list(set(search_domain) - set(to_remove))
    X_domain_init = np.array(search_domain_init)

    bopt =  exp.create_LAW_optimizer( X_domain_init, domain)
    # kernel = exp.create_kernel()
    # gp_model = exp.create_model(kernel, X_new, Y_new)

    # space = LAW.Design_space(domain)
    # acq_class = eval("LAW." + exp.acquisition_name)
    # AF = acq_class(gp_model, space, optimizer=None)

    # bopt = exp.create_optimizer(ndims=ndims,
    #                 domain = domain,
    #                 # acquisition_name =  exp.acquisition_name,
    #                 # batch_size = exp.batch_size,
    #                 # law_params = exp.law_params,
    #                 # kernel = kernel, 
    #                 optimize_restarts=5
    #                 )
        # batch=bopt.compute_batch()

# %%
sleep_time = 5
while True:

    data = exp.get_inputs()
    if data is None:
        sleep(sleep_time)
        continue
    else:
        X_new, Y_new = data

    # -- wait that the optimizer has finished to computing the last batch 
    while exp.runnning == True:
        sleep(sleep_time)

    # -- Check if there is an optimizer model saved
    # TO DO
    # -- If there is no optimizer, create one from scratch: 
    # optimizer = exp.create_LAW_optimizer( X_domain_init, domain) 
    
    # -- Get the batch
    X_batch = exp.suggest_batch()
    exp.save_batch()
    opt_dict = bopt.create_dict()
    np.save('./optimizer', opt_dict)
    # -- save the batch 



# %%
# exp = Experiment(
#                 root_path = "./",
#                 settings_file = "expsettings.json",
#                 exp_res_path = "./experiments/"
#                 )
# exp.apply_settings()


# descriptors = exp.descriptors
# ndims = len(descriptors[0])
# GPy_kernel = RBF(input_dim=ndims, ARD=True)
# kernel = CoulombKernel(input_dim=1, GPy_kern=GPy_kernel.copy(), domain=descriptors)

# %%

#################################  TO SAVE #################################
# D ={k: bopt.__dict__[k]
# for k in ['batch_size', 'acquisition_name', 'law_params', 'kernel', 'optimize_restarts']
# }
# gm_dict = bopt.model.__dict__
# if 'kernel'in gm_dict.keys():
#     del gm_dict ['kernel']
# D.update({'gp_model':gm_dict})
# np.save('./Test_dict.npy', D)
# #################################  LOAD AND CREATE #################################

# data= np.load('./Test_dict.npy',allow_pickle=True).item()
# gpmodel = data.pop('gp_model')
# gp_reg_model = gpmodel.pop('model')
# # gp_reg_model = GPy.models.GPRegression(**gp_reg_dict)
# from GPyOpt.models import GPModel
# GPModel = GPModel(**gpmodel)
# GPModel.model = gp_reg_model
# new_opt = LAW.LAW_BOptimizer(**data)
# # new_opt.model=GPModel