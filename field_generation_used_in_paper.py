import matplotlib.pyplot as plt
import os
import logging, sys, os
logger = logging.getLogger('21cmFAST')
logger.setLevel(logging.INFO)
import py21cmfast as p21c        ### used version: 3.3.1
from py21cmfast import plotting
from py21cmfast import cache_tools
import os
import numpy as np
import sys
import pandas as pn


file_name = 'test.csv'  # or 'training.csv'; 'validation.csv'
folder_name = './raw_data'

### reading simulation specifications from provided tables
sim_specs = pn.read_csv('test.csv',dtype=str) 
tvm  = sim_specs['ION_Tvir_MIN'] 
ax   = sim_specs['X_RAY_SPEC_INDEX']
xh   = sim_specs['HII_EFF_FACTOR']
seed = sim_specs['random_seed']
nit  = len(sim_specs)


if not os.path.exists(folder_name):
    os.mkdir(folder_name)
if not os.path.exists(folder_name+'/'+file_name[:-4]):
    os.mkdir(folder_name+'/'+file_name[:-4])

for i in range(nit):

    
    pth = './tmp_21cm'
    if not os.path.exists(pth):
        os.mkdir(pth)

    p21c.config['direc'] = pth
    cache_tools.clear_cache(direc=pth)

    new_coeval15 = p21c.run_coeval(
        redshift = [15],
        user_params = {"DIM":1024,"HII_DIM": 20, "BOX_LEN": 80, "USE_INTERPOLATION_TABLES": True,"N_THREADS":50},
        astro_params = p21c.AstroParams({"ION_Tvir_MIN":float(tvm[i]),"X_RAY_SPEC_INDEX": float(ax[i]),"HII_EFF_FACTOR":float(xh[i])}),
        random_seed=int(seed[i]),
        flag_options={"USE_TS_FLUCT": True}
    )

    np.save(f'{folder_name}/{file_name[:-4]}/delta_{seed[i]}_Tvm_{tvm[i]}_ax_{ax[i]}_xh_{xh[i]}.npy',new_coeval15[0].density)
    np.save(f'{folder_name}/{file_name[:-4]}/ts_{seed[i]}_Tvm_{tvm[i]}_ax_{ax[i]}_xh_{xh[i]}.npy',new_coeval15[0].Ts_box)
    np.save(f'{folder_name}/{file_name[:-4]}/tb_{seed[i]}_Tvm_{tvm[i]}_ax_{ax[i]}_xh_{xh[i]}.npy',new_coeval15[0].brightness_temp)
    np.save(f'{folder_name}/{file_name[:-4]}/xh_{seed[i]}_Tvm_{tvm[i]}_ax_{ax[i]}_xh_{xh[i]}.npy',new_coeval15[0].xH_box)
    
### remember to renormalize and debias mean before the training!!!