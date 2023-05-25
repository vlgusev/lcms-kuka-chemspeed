
# %%
import numpy as np
import pandas as pd 
import json
import pprint as pp
from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit.Chem.rdMolDescriptors import CalcCoulombMat
from sklearn.preprocessing import normalize
from ase.atoms import Atoms
from dscribe.descriptors import SOAP, CoulombMatrix
from ase.data.pubchem import pubchem_atoms_search

# %%

names = [
    'Rose bengal', 
    'Erythrosin B',
    'Eosin B',
    'Eosin Y',
    'Coumarin 6',
    'Methylene Blue',
    'Bromocresol green',
    "1-1'-Dibenzyl-4-4'-bipyridinium dichloride",
    'Disperse Red 19',
    "2-4-6-Triphenylpyrylium tetrafluoroborate",
    'Solvent Green 3',
    'Riboflavin',
    'Vitamin B12'
    ] 
D_cost =  {
    'Rose bengal':94.20, 
    'Coumarin 6':47.8, 
    'Erythrosin B':12.44,
    'Eosin B':4.56,
    'Eosin Y':5.96, 
    'Methylene Blue':3.11,
    'Bromocresol green':20.64,
    "1-1'-Dibenzyl-4-4'-bipyridinium dichloride":62.2,
    'Disperse Red 19':16.26,
    "2-4-6-Triphenylpyrylium tetrafluoroborate":14.24,
    'Solvent Green 3':0.94,
    'Riboflavin':1.06,
    'Vitamin B12':79.8
    }

D_SMILEs =  {
    'Erythrosin B': 'C1=CC=C2C(=C1)C(=O)OC23C4=CC(=C(C(=C4OC5=C(C(=C(C=C35)I)O)I)I)O)I',
    'Eosin B':'C1=CC=C(C(=C1)C2=C3C=C(C(=O)C(=C3OC4=C(C(=C(C=C24)[N+](=O)[O-])[O-])Br)Br)[N+](=O)[O-])C(=O)[O-]',
    'Eosin Y':"C1=CC=C2C(=C1)C(=O)OC23C4=CC(=C(C(=C4OC5=C(C(=C(C=C35)Br)O)Br)Br)O)Br", 
    'Coumarin 6': 'CCN(CC)C1=CC2=C(C=C1)C=C(C(=O)O2)C3=NC4=CC=CC=C4S3',
    'Methylene Blue':'CN(C)C1=CC2=C(C=C1)N=C3C=CC(=[N+](C)C)C=C3S2.[Cl-]',
    'Bromocresol green':'CC1=C(C(=C(C=C1C2(C3=CC=CC=C3S(=O)(=O)O2)C4=CC(=C(C(=C4C)Br)O)Br)Br)O)Br',
    "1-1'-Dibenzyl-4-4'-bipyridinium dichloride": 'C1=CC=C(C=C1)C[N+]2=CC=C(C=C2)C3=CC=[N+](C=C3)CC4=CC=CC=C4',
    'Rose bengal':'C1=C2C(=C3C=C(C(=O)C(=C3OC2=C(C(=C1[131I])O)[131I])[131I])[131I])C4=C(C(=C(C(=C4Cl)Cl)Cl)Cl)C(=O)O', 
    'Disperse Red 19':'C1=CC(=CC=C1N=NC2=CC=C(C=C2)[N+](=O)[O-])N(CCO)CCO',
    "2-4-6-Triphenylpyrylium tetrafluoroborate":'[B-](F)(F)(F)F.C1=CC=C(C=C1)C2=CC(=[O+]C(=C2)C3=CC=CC=C3)C4=CC=CC=C4', 
    'Solvent Green 3':'CC1=CC=C(C=C1)NC2=C3C(=C(C=C2)NC4=CC=C(C=C4)C)C(=O)C5=CC=CC=C5C3=O',
    'Riboflavin':'CC1=CC2=C(C=C1C)N(C3=NC(=O)NC(=O)C3=N2)CC(C(C(CO)O)O)O',
    'Vitamin B12':'CC1=CC2=C(C=C1C)N(C=N2)[C@@H]3[C@@H]([C@@H]([C@H](O3)CO)OP(=O)([O-])O[C@H](C)CNC(=O)CC[C@@]4([C@H]([C@@H]5[C@]6([C@@]([C@@H](/C(=C(/C7=N/C(=C\C8=N/C(=C(\C4=N5)/C)/[C@H](C8(C)C)CCC(=O)N)/[C@H]([C@]7(C)CC(=O)N)CCC(=O)N)\C)/[N-]6)CCC(=O)N)(C)CC(=O)N)C)CC(=O)N)C)O.[C-]#N.[Co+3]'
    }

################################### SAVE DICTIONARY WITH SMILES ###################################

with open("./Dyes_DSMILEs.json", "w") as f:
    f.write(json.dumps(D_SMILEs, indent=4))


dd = {'names':names, 'costs':[D_cost[n] for n in names]}
data=pd.DataFrame.from_dict(dd)
data.insert(0, 'idx', data.index.values)
data.to_csv('/home/simona/Documents/lcms-kuka-chemspeed/LawOptimizer/costs_compounds.csv', index = False)

# %%
################################### LOAD DICTIONARY WITH SMILES ###################################


with open("./Dyes_DSMILEs.json", "r") as f:
    ref_data = json.load(f) 

# %%
################################### CALCULATING ATOMIC POSITIONS FROM SMILES TO COULOMB ###################################
out_file = "/home/simona/Documents/lcms-kuka-chemspeed/LawOptimizer/descriptors/Dyes.xyz"
f = open(out_file, 'w')
L_CM = {}; L_mols = []
for j, (k,s) in enumerate(ref_data.items()):
    # print(k, file=f)

    try:
        mol = Chem.MolFromSmiles(s, sanitize=True)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)
        n = mol.GetNumAtoms()
        print(n,  file=f)

        for i, atom in  enumerate(mol.GetAtoms()):
            a = atom.GetSymbol()
            positions = mol.GetConformer().GetAtomPosition(i)
            x, y, z =  positions.x, positions.y, positions.z
            text= "{a}\t{x:.8f}\t{y:.8f}\t{z:.8f}".format(a=a, x=x, y=y, z=z)
            print(text,  file=f)
        print('',  file=f)
    except:
        print(k)

f.close()
    

# %%
