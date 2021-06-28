import numpy as np
#Calculate fractional flux density
def log_frac_flux(color):
    log_f = color/-2.5
    return log_f
#Calculate luminosity
def log_lum_r(Mr):
    log_L = (Mr - 34.04)/(-2.5)
    return log_L
def log_lum_non_r(log_fractional_flux, log_L_r):
    log_L = log_fractional_flux + log_L_r
    return log_L
def calc_log_non_r_error(sigma_color,sigma_mr):
    sigma = 0.4*np.sqrt(sigma_color**2+sigma_mr**2)
    return sigma

def eddbias_error(sigma):
    error = np.sqrt(np.sum(np.square(sigma))) / len(sigma)
    return error