


# %%
import numpy as np
import pandas as pd
from ase.atoms import Atoms
from dscribe.descriptors import CoulombMatrix
from sklearn.preprocessing import normalize

file_path = '/home/simona/Documents/lcms-kuka-chemspeed/LawOptimizer/descriptors/Dyes.xyz'
# file_path = '/home/simona/Documents/lcms-kuka-chemspeed/LawOptimizer/descriptors/OldDyes.xyz'


# %%

f = open(file_path,'r')

k = 0
nmax=0
mols = {}
coulomb_matrices = {}
while line := f.readline():
    n = int(line)
    # props = [float(x) for x in f.readline().split()]
    s = []
    pos = []
    # homos = []
    # lumos =[]
    for i in range(n):
       line = f.readline().split()
       s += line[0]
       pos.append([float(x) for x in line[1:]])
    #    homos.append(props[4])
    #    lumos.append(props[7])
    #    homo = props[4]
    #    lumo = props[7]
    f.readline()
    mols[k] = Atoms(''.join(s),positions=pos)
    # energies.append([k,homo, lumo])
    k += 1
    nmax = max(nmax,n)
print("read {} molecules".format(k))
cm = CoulombMatrix(n_atoms_max=nmax, permutation='eigenspectrum')
coulomb_matrices = cm.create([mols[x] for x in mols], n_jobs=2)
# coulomb_matrices = cm.create([mol for mol in mols.values()], n_jobs=2)
coulomb_matrices = list(map(lambda x:normalize(np.atleast_2d(x)),coulomb_matrices ))
print(coulomb_matrices)

#%%

coulomb_descriptors = dict(enumerate(coulomb_matrices)) 

# # energies = np.array(energies)
# Energies = pd.DataFrame(data=energies, columns=['mol', 'homo', 'lumo'])
# Energies.to_csv("/home/simona/Documents/PermCombBO/gdb7-13/Energies.csv")
np.save("/home/simona/Documents/lcms-kuka-chemspeed/LawOptimizer/descriptors/Dyes.npy", coulomb_descriptors)

# %%
