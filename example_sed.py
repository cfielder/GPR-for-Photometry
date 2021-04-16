import numpy as np
import pandas as pd
from pathlib import Path

from sed_functions import *

#Load in your calculations, e.g.:
arrays = Path.home() / "NewDocuments" / "Arrays" / "Paper3"

samples_Mr = np.load(arrays/"samples_Mr.npy")
pred_Mr = np.load(arrays/"pred_Mr.npy")
std_Mr = np.load(arrays/"std_Mr.npy")
bias_means_Mr = np.load(arrays / "bias_means_cmodel_M_r_.npy")
bias_sigmas_Mr = np.load(arrays / "bias_sigmas_cmodel_M_r_.npy")

samples_fuvmr = np.load(arrays/"samples_fuvmr.npy")
pred_fuvmr = np.load(arrays/"pred_fuvmr.npy")
std_fuvmr = np.load(arrays/"std_fuvmr.npy")
bias_means_FUVmr = np.load(arrays / "bias_means_fabs_minus_r.npy")
bias_sigmas_FUVmr = np.load(arrays / "bias_sigmas_fabs_minus_r.npy")

###############################
#Do SED calcs here
###############################
filters = np.array([0.155,0.622])#In micrometer
c = 3e8 #m/s
nu = c/(filters*1e-6) #In Hz

L_r = 10.**((np.mean(pred_Mr)-np.mean(bias_means_Mr)-34.04)/(-2.5)) #In W/Hz
L_r_bias = 10.**((np.mean(pred_Mr)-34.04)/(-2.5))

frac_fuv = frac_flux(np.mean(pred_fuvmr)-np.mean(bias_means_FUVmr))
L_fuv = lum(frac_fuv,L_r)
frac_fuv_bias = frac_flux(np.mean(pred_fuvmr))
L_fuv_bias = lum(frac_fuv_bias,L_r_bias)

Ls = np.array([L_fuv,L_r])
Lnu = Ls*nu
Ls_bias = np.array([L_fuv_bias,L_r_bias])
Lnu_bias = Ls_bias*nu

fuv_err = calc_non_r_error(np.mean(pred_fuvmr)-np.mean(bias_means_FUVmr),np.sqrt(np.std(samples_fuvmr)**2+np.mean(bias_sigmas_FUVmr)**2),nu[0],L_r)

r_err = calc_r_error(np.mean(pred_Mr)-np.mean(bias_means_Mr),np.sqrt(np.std(samples_Mr)**2+np.mean(bias_sigmas_Mr)**2),nu[4])
nuLnu_errors = np.array([fuv_err,r_err])

fuv_err_bias = calc_non_r_error(np.mean(pred_fuvmr),np.std(samples_fuvmr),nu[0],L_r_bias)
r_err_bias = calc_r_error(np.mean(pred_Mr),np.std(samples_Mr),nu[4])

bias_nuLnu_errors = np.array([fuv_err_bias,r_err_bias])

filter_names=np.array(["FUV","r"])

d = {'filter_value':filters,'nu':nu,'Lnu':Ls,'nu_Lnu':Lnu,"error":nuLnu_errors,'nu_Lnu_bias':Lnu_bias,"bias_error":bias_nuLnu_errors}
sed_df = pd.DataFrame(data=d,index=filter_names)
sed_df.to_pickle("SED_calculations.pkl")