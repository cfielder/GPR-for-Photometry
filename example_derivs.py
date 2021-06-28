import numpy as np
import pandas as pd
from pathlib import Path
from gp_derivs import derivatives

catalogs = Path.home() / "Catalogs"
cross_matched_catalog = pd.read_pickle(catalogs / "Cross_Matched_Catalog.pkl")

norm = 5*np.log10(0.7)

galaxies = cross_matched_catalog[["logmass", "sfr", "AB_EXP","Rd","B_T_r","bar_probability"]].copy(deep=True)
mw_mstar = [10.75, 0.1]
mw_mean_sfr = 1.65
mw_sigma_sfr = 0.19
mw_sigma_log_sfr = (np.log10(mw_mean_sfr+mw_sigma_sfr)-np.log10(mw_mean_sfr-mw_sigma_sfr))/2.
mw_sfr = [np.log10(mw_mean_sfr), mw_sigma_sfr]
mw_axis = [0.9,0.1]
mw_rd = [2.48,((2.48 + 0.15) - (2.48 - 0.15)) / 2.0]
mw_b2tr = [0.16, 0.03]
mw_bar = [0.45,0.15]
mw = pd.DataFrame({"logmass": mw_mstar, "sfr": mw_sfr,"AB_EXP":mw_axis,"Rd":mw_rd,"B_T_r":mw_b2tr,"bar_probability":mw_bar}, index=["mean", "sigma"])

predictor_gmr = cross_matched_catalog.gmr
deriv_gmr = pd.DataFrame(derivatives(galaxies,mw,predictor_gmr)).transpose().rename(index={0:"g-r"})

frames = [deriv_gmr]
deriv_df = pd.concat(frames)
deriv_df.to_pickle("deriv_df.pkl")