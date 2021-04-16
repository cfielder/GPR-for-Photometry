import numpy as np
#Calculate fractional flux density
def frac_flux(color):
    f = 10.**(color/-2.5)
    return f
#Calculate luminosity
def lum(fractional_flux, L_weight):
    L = fractional_flux * L_weight
    return L
def se(predicted,n):
    se = np.std(predicted)/np.sqrt(n)
    return se
def calc_r_error(Mr,sigma_Mr,nu):
    partial_deriv = -3.80431e13*nu*np.exp(-0.921034*Mr)
    sigma = np.sqrt(partial_deriv**2*(sigma_Mr)**2)
    return sigma
def calc_non_r_error(color,sigma_color,nu,L_r):
    partial_deriv = -0.921304*nu*L_r*np.exp(-0.921034*color)
    sigma = np.sqrt(partial_deriv**2*sigma_color**2)
    return sigma