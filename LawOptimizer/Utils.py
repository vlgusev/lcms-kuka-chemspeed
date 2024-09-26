
# %%
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcCoulombMat
from sklearn.preprocessing import normalize
from ase.atoms import Atoms
from dscribe.descriptors import SOAP, CoulombMatrix
from ase.data.pubchem import pubchem_atoms_search

# %%

def create_descriptors(all_spieces, nmax, descr_params, molecules):
    '''
        Creates descriptors from list of ase.Atoms.
        all_spieces: all the types of atoms present in the whole dataset. 
                     It is necessary for SOAP descriptors 
        nmax: max number of atom of the largest molecule in the  dataset.
              It is necessary for Coulomb descriptors
        descr_params: building parameters specific of the descriptors. It
                      is given in the experimental settings file
        molecules: dictionary {mol_idx:Atoms}
    '''
    
    descr_type = descr_params.pop('descript_type')
    desc_class = eval(descr_type)
    if descr_type == "Coulomb":
        descr_params.update({'nmax':nmax})
    elif descr_type == "SOAP":
        descr_params.update({'species':all_spieces})
    
    desc = desc_class( **descr_params)

    descriptors = {}
    for k, m in molecules.items():

        d = desc.create(m)
        if np.ndim(d)==1:
            d = d.reshape(-1,1)
        d = normalize(d)
        descriptors[k] = d
    return descriptors




def read_config_to_dic(file_path = "./optimizer_config.txt", save_folder=None):

    ############ --- WILL BE A METHOD OF THE EXPERIMENT CLASS --- ############
    '''
    Reads the config file which contains the atomic positions of atoms.
    Creates a dictionary D1={formula:idx}, D2={idx:descriptor} ]}
    Saves D1 and D2 in separate npy files in the current folder
    Returns D1, D2, all_species
    '''

    f = open(file_path,'r')
    k = 0
    nmax=0
    mols= {}
    all_spieces =  set()
    formulas_to_idx = {}
    # mol_idxs = []
    to_skip = 1

    while line := f.readline():

        n = int(line) # first line is the Number of atoms 
        for _ in range(to_skip):
            f.readline()
        s = []
        pos = []
        formula = ''
        for i in range(n+4):
            line = f.readline().split()
            if i>=n:
                continue
            s += line[0]
            pos.append([float(x) for x in line[1:-1]])
            formula += line[0]
        formulas_to_idx[formula]=k
        all_spieces = all_spieces.union(set(s))
        mols[k] = (Atoms(''.join(s),positions=pos)).todict()
        nmax = max(nmax,len(formula))
        mol_data={'all_spieces':all_spieces, 'nmax':nmax, 'molecules': mols}
        k +=1
    ## call create_desc(all_spieces, descr_params)
    return mol_data

def read_config(descr_params, file_path = "./optimizer_config.txt", save_folder=None):

    ############ --- WILL BE A METHOD OF THE EXPERIMENT CLASS --- ############
    '''
    Reads the config file which contains the atomic positions of atoms.
    Creates a dictionary D1={formula:idx}, D2={idx:descriptor} ]}
    Saves D1 and D2 in separate npy files in the current folder
    Returns D1, D2, all_species
    '''

    f = open(file_path,'r')
    nmax=0
    mols= {}
    all_spieces =  set()

    while line := f.readline():
        k = int(line.split(':')[-1])
        line=f.readline()
        n = int(line) # first line is the Number of atoms 
        s = []
        pos = []
        for i in range(n):
            line = f.readline().split()
            s += line[0]
            pos.append([float(x) for x in line[1:]])
        f.readline()
        all_spieces = all_spieces.union(set(s))
        mols[k] = (Atoms(''.join(s),positions=pos))
        nmax = max(nmax,n)

    ## -- creation of the descriptors
    descriptors = create_descriptors(all_spieces, nmax, descr_params, mols) 
    return all_spieces, nmax, descriptors

def cut_file(fname, fout, idx_list):

    f = open(fname, 'r')
    info=[]
    while line:=f.readline():
        temp_L=[]
        idx =  int(line.rstrip('\n').split(':')[-1])
        if idx >=len(idx_list):
            break
    
        temp_L.append(line)
        line = f.readline()
        temp_L.append(line)
        n_atoms = int(line.rstrip('\n'))
        for n in range(n_atoms + 1):
            temp_L.append(f.readline())
        if idx in idx_list:
            info.extend(temp_L)
    with open(fout,'w') as fo:
        for l in info:
            fo.write(l)

# %%
if __name__ == "__main__":
    ## --Transform the original xyz file to another format
    info=[]
    file_name = "/home/simona/Documents/ro_bo/qm9.xyz"
    out_file= "/home/simona/Documents/ro_bo/qm9_Test.xyz"
    f = open(file_name,'r')
    f_out=open(out_file, 'w')
    to_skip=1

    k=0
    while line := f.readline():
        line=line.rstrip('\n')
        n = int(line) # first line is the Number of atoms na
        info.append(line)
        print('mol_idx:{}'.format(k), file=f_out)
        print(line, file=f_out)
        for _ in range(to_skip):
            f.readline()
        for _ in range(n):
            line = f.readline().rstrip('\n').split()[: -1]
            line = "\t ".join(line)
            info.append(line)
            print(line, file=f_out)
        for _ in range(4):
            f.readline()
        info.append('')
        print('', file=f_out)
        k +=1
    f_out.close()
    f.close()

