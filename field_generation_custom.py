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


stvm = '4.70'          #used values: 4.67, 4.69, 4.71, 4.73 training and validation; 4.70 test
sax  = '1.000'        #used values: 0.935, 1.065 training and validation; 1.000 test
sxh  = '30.0'         #used values: 28.0, 32.0 training and validation; 30.0 test
nit  =10              #number of realisation    


tvm  =float(stvm)
ax   =float(sax)
xh   =float(sxh)


for ij in range(nit):
    seed = int(np.random.uniform(0,100000))
    pth = './tmp_21cm'
    if not os.path.exists(pth):
        os.mkdir(pth)

    p21c.config['direc'] = pth
    cache_tools.clear_cache(direc=pth)

    new_coeval15 = p21c.run_coeval(
        redshift = [15],
        user_params = {"DIM":1024,"HII_DIM": 256, "BOX_LEN": 1024, "USE_INTERPOLATION_TABLES": True,"N_THREADS":50},
        astro_params = p21c.AstroParams({"ION_Tvir_MIN":tvm,"X_RAY_SPEC_INDEX": ax,"HII_EFF_FACTOR":xh}),
        random_seed=seed,
        flag_options={"USE_TS_FLUCT": True}
    )

    np.save(f'./raw_output/delta_{seed}_Tvm_{stvm}_ax_{sax}_xh_{sxh}.npy',new_coeval15[0].density)
    np.save(f'./raw_output/ts_{seed}_Tvm_{stvm}_ax_{sax}_xh_{sxh}.npy',new_coeval15[0].Ts_box)
    np.save(f'./raw_output/tb_{seed}_Tvm_{stvm}_ax_{sax}_xh_{sxh}.npy',new_coeval15[0].brightness_temp)
    np.save(f'./raw_output/xh_{seed}_Tvm_{stvm}_ax_{sax}_xh_{sxh}.npy',new_coeval15[0].xH_box)
    
### remember to renormalize and debias mean before the training!!!