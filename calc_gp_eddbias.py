import numpy as np
import scipy
from gp_eddbias import single_predictor_mw_gp_prediction

# Define the function of fit and covariance
def func(x, a, b, c):
    return a * x ** 2 + b * x + c

def calc_eddbias(
        mw,
        mw_array,
        predictor,
        galaxies,
        galaxies_sigma,
        n_trials = 25,
        **kwargs
):
    """
    A wrapper around the gp_eddbias function in order to calculate Eddington bias for a sample.

        Args:
            mw (dataframe): A dataframe that contains the measured mean and standard deviation
                of the Milky Way. This should include all parameters with which to use as to
                determine the target value.
            mw_array (array): An array of just the nominal Milky Way measured values.
            predictor (array): An array that contains the real values from the
                catalog with which to train the gaussian process on. This will be the parameter
                that the model predicts form.
            galaxies (dataframe): The subset of the main catalog that contains columns of
                the properties of interest. This is the x-training set.
            galaxies_sigma (dataframe): The subset of the main catalog that
                contains the errors of the columns of interest (e.g. the errors of the galaxies
                df).
            n_trials (int): Default 25. The number of times used to calculate the the Eddington bias
                sample. This number must be sufficiently large to not be effected by statistical fluctuation.
            **kwargs: Keyword arguments to be passed into gp_eddbias()

        Returns:
            bias_means (array): the bias mean resulting from each trial, subtract the average of this value
                off of your predicted photometric property
            bias_sigmas (array): the errors of the bias means resulting from each trial
        """

    #Needed arrays
    prediction_sets = np.zeros((n_trials, 5))
    std_sets = np.zeros((n_trials, 5))
    bias_means = np.zeros((n_trials))
    bias_sigmas = np.zeros((n_trials))
    delta_mean = np.zeros((n_trials,4))
    delta_sigma = np.zeros((n_trials,4))
    xx = np.arange(2, 6, 1) #Fitting array

    for j in range(n_trials):
        for d in range(5):
            prediction_sets[j, d], std_sets[j, d] = single_predictor_mw_gp_prediction(
                galaxies=galaxies,
                mw=mw,
                predictor=predictor,
                mw_draw=mw_array,
                sigma_galaxies=galaxies_sigma,
                noise_level=d,
                get_kernel_result=True,
                **kwargs
            )
            # This call yields a single prediction based on the mw nominal parameters
            # Calculate deltas
            if d != 0:
                delta_mean[j,d-1] = prediction_sets[j,d] - prediction_sets[j,d-1]
                delta_sigma[j,d-1] = np.sqrt(std_sets[j,d]**2 + std_sets[j,d-1]**2)
        # Do the fit and calc bias mean/sigma
        coeffs, covar = scipy.optimize.curve_fit(func,xx,delta_mean[j,:],sigma=delta_sigma[j,:])
        bias_means[j] = np.sum(coeffs)
        bias_sigmas[j] = np.sqrt(np.sum(covar))
    #return prediction_sets,bias_means,bias_sigmas,std_sets,delta_mean,delta_sigma -> other values are for testing
    return bias_means, bias_sigmas