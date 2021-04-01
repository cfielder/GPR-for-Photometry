import numpy as np
import sklearn.gaussian_process as gp
import random
import pickle

def single_predictor_mw_gp(
        galaxies,
        mw,
        predictor,
        sigma_cutoff = 12,
        downsample_n = 2000,
        sample_n = 1000,
        n = 1000,
        rs = random.randint(1,1e5),
        restrict_sigma = True,
        downsample = True,
        sample_mw_pdf = True,
        prediction = True,
        return_mw_realizations = False,
        return_samples = True,
        save_training = False,
        save_model = False,
        get_kernel_result = False,
):
    """Predicts a sample of Milky Way values from the given galaxy sample based
    upon the MW PDF and trained on the desired prediction parameter. 

    Args:
        galaxies (dataframe): The subset of the main catalog that contains columns of
            the properties of interest. This is the x-training set.
        predictor (array): An array that contains the real values from the 
            catalog with which to train the gaussian process on. This will be the parameter
            that the model predicts form.
        mw (dataframe): A dataframe that contains the measured mean and standard
            deviation of the Milky Way. This should include all parameters with which to
            use to determine the target value. This MUST be passed in to restrict by a
            sigma_cutoff within the Milky Way.
        sigma_cutoff (int/float): Default 12. The level of sigma with which to restrict
            the training sample on. Only used if restrict_sigma is True.
        downsample_n (int): Default 1500. The size of the downsample, only used if
            downsample is set to True. 
        sample_n (int): Default 1000. The number of draws to take from the MW fiducial PDF.
        n (int): Default 1000. Size of the sample that you want from the the GP.
        rs (int): The random state for the downsample. Default is a random integer between
            1 and 1e5. Can be defined if a certain random state is necessary.
        restrict_sigma (bool): Default True. This applies a sigma boundary from the MW 
            properties. The training sample is cutoff to be within this limit.
        downsample (bool): Default True. If your catalog size is sufficiently large
            it is necessary to downsample or the gaussian process will run out of memory.
            If your sample size is smaller than 1000 before this step when it is set to 
            True you will recieve an error.
        sample_mw_pdf (bool) : Default True. If set to false draws will NOT be done from the Milky Way
            PDF.
        save_training (bool): Default False. If set to True will save the training set used to train
            the GPR
        save_model (bool): Default False. If set to True the GPR fit will be saved.
        prediction (bool): Default True. The gaussian process regression will return a full
            sample. If set to True the gaussian process regression will instead return just the
            prediction value and standard deviation.
        get_kernel_result (bool): Default False. If set to true will return the optimized
            values of the kernel.
        return_mw_realizations (bool): Default False. If set to true this function will also return
            the random draws of the Milky Way fiducial pdf used to determine the sample or prediction.

    Returns:
        A predicted sample that is (sample_n x 1000) in size if prediction is set to False (default).
        A predicted value and standard deviation (single floats) if prediction is set to True.
            

    """
    n_params = int(len(list(mw)))
    print("You are using {} parameters.".format(n_params))

    #Initiate the kernel used for the GP
    initial_scale = np.ones(n_params)
    kernel = gp.kernels.RBF(length_scale=initial_scale)+gp.kernels.WhiteKernel(noise_level=1,noise_level_bounds=(1e-5,1e5))
    model = gp.GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10,normalize_y=True)

    #Sigma restriction for largest samples
    if restrict_sigma:
        temp_mw = mw.replace(0,10000) #this is used when running at a nominal value with no variance but still restricting sigma
        galaxies, temp_mw = galaxies.align(temp_mw, axis=1, copy=False)
        conditional = np.logical_and(
            galaxies < (temp_mw.loc["mean"]+sigma_cutoff*temp_mw.loc["sigma"]),
            galaxies > (temp_mw.loc["mean"]-sigma_cutoff*temp_mw.loc["sigma"]))
        flat_conditional = conditional.all(axis = 'columns')
        galaxies = galaxies.where(conditional).dropna(axis="index",how="any")
        predictor = predictor.where(flat_conditional).dropna(axis="index",how="any")

    #Downsampling for large samples
    if downsample:
        galaxies = galaxies.sample(n=downsample_n,random_state=rs,axis="index")
        predictor = predictor.sample(n=downsample_n,random_state=rs,axis="index")

    if save_training == True:
        galaxies.to_pickle("galaxy_training_set")
        predictor.to_pickle("predictor_training_set")

    #Fit the GP with the training set
    model.fit(galaxies,predictor)

    if save_model == True:
        filename = 'gpr_model.sav'
        pickle.dump(model, open(filename, 'wb'))
    
    if get_kernel_result == True:
        print(model.kernel_)

    #Select random MW point
    if sample_mw_pdf:
        mw_realizations = np.zeros((sample_n,n_params))
        for i in range(sample_n):
            single_realization = np.zeros((n_params))
            for j,column in enumerate(mw.columns):
                if column == "logmass" or column=="sfr" or column=="Rd" or column=="B_T_r":
                    single_realization[j] = np.random.normal(loc=mw.loc["mean",column],scale=mw.loc["sigma",column])
                if column == "AB_EXP":
                    if (mw[column] == 0).any() == True:
                        single_realization[j] = mw.loc["mean", column]
                    else:
                        single_realization[j] = np.random.uniform(low=mw.loc["mean", column]-mw.loc["sigma", column],high=1.0)
                if column=="bar_probability":
                    if (mw[column] == 0).any() == True:
                        single_realization[j] = mw.loc["mean", column]
                    else:
                        single_realization[j] = np.random.uniform(low=mw.loc["mean", column] - mw.loc["sigma", column],
                                                              high=mw.loc["mean", column] + mw.loc["sigma", column])
            mw_realizations[i,:] = single_realization
        samples = model.sample_y(mw_realizations, n_samples=n)
    else:
        mw_means = mw.loc["mean"].values.reshape(1,-1)
        if return_samples:
            samples = model.sample_y(mw_means,n_samples=n)

    if prediction == False:
        if return_mw_realizations:
            if return_samples:
                return samples,mw_realizations
            else:
                return mw_realizations
        if not return_mw_realizations:
            if return_samples:
                return samples
    else:
        if sample_mw_pdf:
            pred, std = model.predict(mw_realizations,return_std=True)
            if return_mw_realizations and not return_samples:
                return pred,std,mw_realizations
            if not return_mw_realizations and return_samples:
                return samples,pred,std
            if return_mw_realizations and return_samples:
                return samples,pred,std,mw_realizations
            if not return_mw_realizations and not return_samples:
                return pred, std
        if not sample_mw_pdf:
            pred, std = model.predict(mw_means, return_std=True)
            if return_samples:
                return samples, pred, std
            else:
                return pred,std
