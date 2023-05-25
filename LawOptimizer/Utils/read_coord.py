


# %%
import numpy as np
import pandas as pd
from ase.atoms import Atoms
from dscribe.descriptors import CoulombMatrix
from sklearn.preprocessing import normalize

file_path = '../descriptors/Dyes.xyz'


def create_Coulombs_from_coords(data_path, save_path):
    pass


# %%

f = open(file_path,'r')

k = 0
nmax=0
mols = {}
coulomb_matrices = {}
while line := f.readline():
    n = int(line)
    s = []
    pos = []
    for i in range(n):
       line = f.readline().split()
       s += line[0]
       pos.append([float(x) for x in line[1:]])
    f.readline()
    mols[k] = Atoms(''.join(s),positions=pos)
    k += 1
    nmax = max(nmax,n)
# print("read {} molecules".format(k))
cm = CoulombMatrix(n_atoms_max=nmax, permutation='eigenspectrum')
coulomb_matrices = cm.create([mols[x] for x in mols], n_jobs=2)
coulomb_matrices = list(map(lambda x:normalize(np.atleast_2d(x)),coulomb_matrices ))
print(coulomb_matrices)
coulomb_descriptors = dict(enumerate(coulomb_matrices)) 

np.save("../descriptors/Dyes.npy", coulomb_descriptors)

# %%
