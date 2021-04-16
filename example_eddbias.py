import numpy as np
import pandas as pd
from pathlib import Path
from calc_gp_eddbias import calc_eddbias

catalogs = Path.home() / "Catalogs"
cross_matched_catalog = pd.read_pickle(catalogs / "Cross_Matched_Catalog.pkl")

galaxies = cross_matched_catalog[["logmass", "sfr","AB_EXP"]].copy(deep=True)
galaxies_sigma = cross_matched_catalog[["sigma_logmass","sigma_sfr","expABErr_r"]].copy(deep=True)
galaxies_sigma.rename(columns={"sigma_logmass":"logmass","sigma_sfr":"sfr","expABErr_r":"AB_EXP"},inplace=True)


mw_mstar = [10.75, 0.1]
mw_mean_sfr = 1.65
mw_sigma_sfr = 0.19
mw_sigma_log_sfr = (np.log10(mw_mean_sfr+mw_sigma_sfr)-np.log10(mw_mean_sfr-mw_sigma_sfr))/2.
mw_sfr = [np.log10(mw_mean_sfr), mw_sigma_sfr]
mw_axis = [0.9,0.1]
mw = pd.DataFrame({"logmass": mw_mstar,
                   "sfr": mw_sfr,
                   "AB_EXP":mw_axis,},
                  index=["mean", "sigma"])
#Nominal MW values array
mw_array = np.array([mw_mstar[0],mw_sfr[0],mw_axis[0]]).reshape(1,-1)

predictor_gmr = cross_matched_catalog["model_M_g"] - cross_matched_catalog["model_M_r"]
bias_means, bias_sigmas = calc_eddbias(
                mw,
                mw_array,
                predictor_gmr,
                galaxies,
                galaxies_sigma
)
np.save("bias_means_gmr",bias_means)
np.save("bias_sigmas_gmr",bias_sigmas)






