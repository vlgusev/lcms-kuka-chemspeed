
#%%
import numpy as np
import os
import pandas as pd

#%%
batch_path = "/home/simona/Documents/lcms-kuka-chemspeed/LawOptimizer/experiments/"
batch_filename_start = "PFAS_Dyes-batch-"
batch_files = [f for f in os.listdir(batch_path) if f.startswith(batch_filename_start)]
num_batch = [f.rstrip('.run').split('-')[-1] for f in batch_files] 
num_batch = sorted(list(map(int, num_batch)))
last_batch = np.max(num_batch)
current_batch_file = os.path.join(batch_path, batch_filename_start +'{}.run'.format(last_batch))

DF=pd.read_csv(current_batch_file)
# columns=DF.columns
# m = len(columns)
m = DF.shape[1]
Peaks = np.random.uniform(15000, 70000, size=16)
DF.insert(m, 'PeakArea',Peaks)
save_path = batch_path + 'PFAS_Dyes-res-{}.run'.format(last_batch)
np.savetxt(save_path, DF, fmt='%.3f', delimiter=',' ,header= ",".join(DF.columns), comments='')
 # %%
# batch_size=16
# # num_batch=1
# columns = ['SampleIndex']
# columns.extend(exp.compounds)  
# columns.extend(['Water', 'PeakArea'])
# ii=np.random.choice(range(len(search_domain)), batch_size, replace=False)
# selected_exp = np.array(search_domain)[ii]
# X_out = np.zeros((len(selected_exp), len(columns)  -1 ))
# for j, x in enumerate(list(selected_exp)):
#     ii = int(x[1]); value= x[0]
#     X_out[j,ii]=value
#     X_out[j,-2]=1-value
#     peak=np.random.uniform(15000, 70000)
#     X_out[j,-1]=round(peak, 3)
# sample_idxs = (np.arange(1,17) + batch_size*(num_batch-1)).reshape(-1,1)
# X_out = np.hstack([sample_idxs, X_out])

# X_out
# np.savetxt(save_path, X_out, fmt='%.3f', delimiter=',' ,header= ",".join(columns), comments='')