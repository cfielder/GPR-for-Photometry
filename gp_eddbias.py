import numpy as np
import sklearn.gaussian_process as gp
import pandas as pd
import random

def single_predictor_mw_gp_prediction(
        galaxies,
        mw,
        predictor,
        mw_draw,
        sigma_galaxies = None,
        rs = random.randint(1,1e4),
        noise_level = 0,
        downsample_n = 1500,
        sigma_cutoff = 12,
        restrict_sigma = True,
        downsample = True,
        get_kernel_result = False,
):
    """Predicts a sample of Milky Way values from the given galaxy sample based
    upon the MW PDF and trained on the desired prediction parameter. 

    Args:
        galaxies (dataframe): The subset of the main catalog that contains columns of
            the properties of interest. This is the x-training set.
        mw (dataframe): A dataframe that contains the measured mean and standard deviation
            of the Milky Way. This should include all parameters with which to use as to
            determine the target value.
        predictor (array): An array that contains the real values from the 
            catalog with which to train the gaussian process on. This will be the parameter
            that the model predicts form.
        sigma_galaxies (dataframe): Default None. The subset of the main catalog that 
            contains the errors of the columns of interest (e.g. the errors of the galaxies 
            df). This is used in the Eddington bias calculation and is optional input.
        rs (int): The random state for the downsample. Default is a random integer between
            1 and 1e5. Can be defined if a certain random state is necessary.
        noise_level (int): The noise level used in the Eddington bias calculation.
            This value is typically between 1 and 5. Default is 0 or no noise added.
        downsample_n (int): Default 1000. The size of the downsample, only used if 
            downsample is set to True. 
        sample_n (int): Default 1000. Size of the prediction that you want from the the GP.
        sigma_cutoff (int/float): Default 10. The level of sigma with which to restrict
            the training sample on. Only used if restrict_sigma is True.
        restrict_sigma (bool): Default True. This applies a sigma boundary from the MW 
            properties. The training sample is cutoff to be within this limit.
        downsample (bool): Default True. If your catalog size is sufficiently large
            it is necessary to downsample or the gaussian process will run out of memory.
            If your sample size is smaller than 1000 before this step when it is set to 
            True you will recieve an error.
        get_kernel_result (bool): Default False. If set to true will return the optimized
            values of the kernel.

    Returns:
        A single prediction and standard deviation (floats) based on the actual MW values.
            

    """
    n_params = int(len(list(mw)))
    #Initiate the kernel used for the GP
    initial_scale = np.ones(n_params)
    kernel = gp.kernels.RBF(length_scale=initial_scale)+gp.kernels.WhiteKernel(noise_level=1,noise_level_bounds=(1e-5,1e5))
    model = gp.GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10,normalize_y=True)

    #Calculate Eddington bias if a sigma dataframe is passed
    if sigma_galaxies is not None and noise_level != 0:
        noise = pd.DataFrame(0,index=galaxies.index,columns=galaxies.columns)
        for column in galaxies:
            column_noise = np.random.standard_normal(size=galaxies.shape[0]) * sigma_galaxies[column] * noise_level
            noise[column] = column_noise
        noisy_galaxies = galaxies + noise

    #Sigma restriction for largest samples
    if restrict_sigma:
        conditional = np.logical_and(
            galaxies < (mw.loc["mean"]+sigma_cutoff*mw.loc["sigma"]),
            galaxies > (mw.loc["mean"]-sigma_cutoff*mw.loc["sigma"]))
        flat_conditional = conditional.all(axis = 'columns')

        galaxies = galaxies.where(conditional).dropna(axis="index",how="any")
        predictor = predictor.where(flat_conditional).dropna(axis="index",how="any")

        if sigma_galaxies is not None and noise_level != 0:
            noisy_galaxies = noisy_galaxies.where(conditional).dropna(axis="index",how="any")
    #Downsampling for large samples
    if downsample:
        galaxies = galaxies.sample(n=downsample_n,random_state=rs,axis="index")
        predictor = predictor.sample(n=downsample_n,random_state=rs,axis="index")
        if sigma_galaxies is not None and noise_level != 0:
            noisy_galaxies = noisy_galaxies.sample(n=downsample_n,random_state=rs,axis="index")
    #Fit the GP with the training set
    if sigma_galaxies is not None and noise_level != 0:
        print("Calculating with Eddington bias.")
        model.fit(noisy_galaxies.values,predictor)
    else:
        print("Calculating with no bias.")
        model.fit(galaxies,predictor)
    
    if get_kernel_result == True:
        print(model.kernel_)

    #Predict for the MW draw input
    prediction, std = model.predict(mw_draw, return_std=True)
    return prediction,std



