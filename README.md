# GPR-for-Photometry
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)
![GitHub last commit](https://img.shields.io/github/last-commit/cfielder/GPR-for-Photometry.svg)

This hosts the essential GPR code for predicting photometry for the Milky Way, utilised in Fielder et al. 2021.

## Installing

### Directly from Repository

`git clone https://github.com/cfielder/MW_Morphology`

## Usage

This code has been sepcifically built to work around a cleaned sample. For example in Licquia et al. 2015
the sample contains all objects in SDSS-III DR8 and MPA-JHU of which a significant portion was then discarded 
due to various flags. Of these a volume-limited sample is then selected. Fielder et al. 2021 utilises cross matches to 
this sample. Those catalogs are provided in this repository: https://github.com/cfielder/Catalogs

If you use this code for your work, we ask that you give proper credit to Fielder et al. 2021.

## Description of Scripts

Gaussian Process Regression (GPR) is a machine learning approach to performing a fit. GPR is a statistical technique that 
leverages information from both local information and global trends. In our application, the GPR uses a wide variety of 
galaxies to capture information from global trends between galaxy structural properties and galaxy photometric properties.
This allows us to predict an SED derived from photometric prediction based on the Milky Way's measured parameters.

We also provide some code for determining systematics - specifically Eddington bias. For those interested in obtaining
k-corrections please refer to Fielder et al. 2021.

For ease of use of all of these scripts, we will provide an example for a basic photometric prediction. We also provide 
some functions for those interested in constructing SEDs.

### Step 1:
**Understand how mw_gp.py works**
  - This entire function is built around the scikit-learn implementation of a Gaussian process regression algorithm. For more details
    please refer to the documentation before proceeding: 
    https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
  - `single_predictor_mw_gp()` is a function that you can call when you want to run a GPR fit. At it's core this function 
    takes in (1) a catalog consisting of the galaxy properties that you wish to train the model with, (2) a dataframe of the 
    mean Milky Way (or galaxy of your interest) properties and their corresponding errors, and (3) an array containing the target 
    values of the galaxies used to train the GPR (`predictor`). (3) is the property that you wish to predict for your galaxy. 
    Therefore if (1) has the stellar mass and star formation rates of your galaxy sample, (2) will contain the mass and star 
    formation rate of your galaxy of interest, and then (3) will contain the photometric property you wish to predict, such as 
    (g-r) restframe color. 
    **NOTE** that the column names in (1) MUST match the column names in (2).
  - This algorithm is adapted for the method presented in Fielder et al. 2021. After the GPR has been defined on line 75-77, we
    restric the sample by a given sigma and downsample it. This is necessary such that the computer does not run out of memory 
    due to how the GPR scales.
  - For use of various checks and conviences we include options for saving the training model, the fit GPR model, and retrievng the
    kernel result.
  - Once the GPR has been fit we allow for two approaches to handling the Milky Way parameters depending on what the user wishes
    to do (lines 110-132). Either draw from a fiducial PDF of the Milky Way properties `sample_n` number of times, or query the fit 
    at the mean Milky Way parameters.
    If you choose to perform draws from the fiducial PDFs of the properties you have the option of returning a 2D array of those draws.
  - There are many options of things one can get out of a GPR fit:
    (i) A mean predicted value. In our example this would be a prediction for (g-r) for the Milky Way.
    (ii) A sample of possible values (default 1000 for `n`) of (g-r) for the Milky Way.
    (iii) The standard deviation of the mean prediction (i.e. the error of the prediction).
    Once can choose to obtain just (ii), just (i) and (iii), or all three.

### Step 2:
**Perform your photometric predictions** 
  - In `example.py` we provide basic sample code to perform a photometric prediction for Milky Way restframe (g-r) color using 
    three parameters: stellar mass, star formation rate, and axis ratio.
  - Line 7-8 reads in the catalog which contains all of your galaxies. 
  - Line 10 makes the equivalent of (1) in Step 1, or the `galaxies`read into `single_predictor_mw_gp()`. 
  - Lines 12-21 create the equivalent of (2) in Step 1, or the `mw` read into `single_predictor_mw_gp()`. This is the dataframe of 
    the Milky Way properties.
  - Line 23 creates the equivalent of (3) in Step 1, or the `predictor` read into `single_predictor_mw_gp()`.
  - Line 24-29 calls the function `single_predictor_mw_gp()`.
  - The final lines then save the output as numpy arrays.
    
One could stop here if systematics are not a concern. However, if they are we provide `gp_eddbias.py` and `example_eddbias.py` for 
those interested in addressing Eddington bias. If not please procees to Step 4.

### Step 3 (optional):
**Perform Eddington bias calculations**
  - In `example_eddbias.py` we provide basid sample code to perform an Eddington bias calculation for (g-r) color using three
    parameters: stellar mass, star formation rate, and axis ratio.
  - This code is very similar to `example.py`. Here the function `calc_gp_eddbias.py` is called which has a function that uses `gp_eddbias.py`
    in order to calculate Eddington bias for a given photometric band. 
  - The saved output can be subtracted off of the mean photometric prediction in order to have an Eddington bias corrected prediction for 
    your photometric band.
  
### Step 4:
**Estimate an SED**
  - The code in `example_sed.py` provides sample calculations to produce an SED. We provide SDSS r band and GALEX FUV but this method can be
    extracted to more bands.
  - This code calls in conviennce functions in `sed_functions.py`. These calculation fractional flux which is then used to calculate luminosity, and 
    respective errors.
  - These calculations are then saved as a dataframe. To plot an SED like that presented in Fielder et al. 2021 one would simply do the following:
    ```
    plt.errorbar(sed_df["filter_value"],
                  sed_df["nu_Lnu"],
                  yerr=sed_df["error"],
                  markeredgecolor="black",
                  markerfacecolor="white", 
                  ecolor="black",
                  fmt='o')
    ```

## Authors

* **Catherine Fielder** - *Code constriction* 

With additional assistance from Brett Andrews and Jeff Newman.

