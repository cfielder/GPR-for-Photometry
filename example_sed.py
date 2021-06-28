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

#Calculate fraction flux and luminosity
log_L_r = log_lum_r((np.mean(pred_Mr)-np.mean(bias_means_Mr))) #In W/Hz

log_frac_fuv = log_frac_flux(np.mean(pred_fuvmr)-np.mean(bias_means_FUVmr))
log_L_fuv = log_lum_non_r(log_frac_fuv,log_L_r)

#Convert to array and make nuLnu
log_Ls = np.array([log_L_fuv,log_L_r])
log_nuLnu = log_Ls + np.log10(nu)

#Calculate errors
r_err_mag = np.sqrt(np.std(samples_Mr)**2+eddbias_error(bias_sigmas_Mr)**2)
log_L_r_err = 0.4*r_err_mag #In W/Hz

FUVmr_err_mag = np.sqrt(np.std(samples_fuvmr)**2+eddbias_error(bias_sigmas_FUVmr)**2)
log_L_fuv_err = calc_log_non_r_error(FUVmr_err_mag,r_err_mag)

#Put into arrays
log_nuLnu_errors = np.array([log_L_fuv_err,log_L_r_err])
mag_errors = np.array([FUVmr_err_mag,r_err_mag])

filter_names=np.array(["FUV","r"])

#Make the dataframe
d = {'filter_value':filters,'nu':nu,'log_Lnu':log_Ls,'log_nu_Lnu':log_nuLnu,"error_nuLnu":log_nuLnu_errors,
     "mag_errors":mag_errors}
sed_df = pd.DataFrame(data=d,index=filter_names)
sed_df.to_pickle("SED_calculations.pkl")