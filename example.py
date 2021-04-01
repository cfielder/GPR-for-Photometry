import numpy as np
import pandas as pd
from pathlib import Path

from mw_gp import single_predictor_mw_gp

catalogs = Path.home() / "Catalogs"
cross_matched_catalog = pd.read_pickle(catalogs / "Cross_Matched_Catalog.pkl")

galaxies = cross_matched_catalog[["logmass", "sfr","AB_EXP"]].copy(deep=True)

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

predictor_gmr = cross_matched_catalog["model_M_g"] - cross_matched_catalog["model_M_r"]
samples_gmr,pred_gmr,std_gmr,mw_realizations_gmr = single_predictor_mw_gp(
                                                                          galaxies,
                                                                          mw,
                                                                          predictor_gmr,
                                                                          return_mw_realizations=True,
                                                                          )
np.save("samples_gmr", samples_gmr)
np.save("pred_gmr", pred_gmr)
np.save("std_gmr", std_gmr)
np.save("mw_realizations_gmr", mw_realizations_gmr)