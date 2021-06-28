import numpy as np
import pandas as pd

from mw_gp import gp_model_only
from three_point_derivative import deriv

def derivatives(
        galaxies,
        mw,
        predictor,
        n_trainings = 10,
        **kwargs
):
    """Calculate the derivative of photometric properties w.r.t. Mass, SFR, B/T, Rd, etc by offsetting
        the MW's values by sigma/10 in both directions, giving 3 point curve.
        We assume Eddington bias does not affect these derivatives.
        This runs the GP at the MW values and offset MW values.

        Args:
            galaxies (dataframe): The subset of the main catalog that contains columns of
                the properties of interest. This is the x-training set.
            mw (dataframe): A dataframe that contains the measured mean and standard
                deviation of the Milky Way. This should include all parameters with which to
                use to determine the target value. This MUST be passed in to restrict by a
                sigma_cutoff within the Milky Way.
            predictor (array): An array that contains the real values from the
            catalog with which to train the gaussian process on. This will be the parameter
            that the model predicts form.
            predictor_name (string): A string of the predictor name to make indexing easy with your
            dataframe.

        Returns:
            derivatives of the galaxy properties with respect to the predictor
    """
    mw_offset = pd.DataFrame(index=["-delta", "0", "delta"], columns=mw.columns)
    mw_offset = mw_offset.fillna(0)

    temp_derivs = pd.DataFrame(index=[np.arange(0,10,1)],columns=mw.columns)
    temp_derivs = temp_derivs.fillna(0)

    for column in mw:
        mw_offset.loc["-delta", column] = -mw.at["sigma", column] / 10.0
        mw_offset.loc["delta", column] = mw.at["sigma", column] / 10.0

    #Offset the properties, one at a time
    for i in range(n_trainings):
        print(i)
        model = gp_model_only(galaxies, mw, predictor,get_kernel_result=True,**kwargs)
        for col, col_val in enumerate(mw_offset.columns):
            predicted_prop = np.zeros((3))
            for row, row_val in enumerate(mw_offset.index):
                new_mw = mw.copy(deep=True)
                new_mw.loc["mean", col_val] = mw.loc["mean",col_val] + mw_offset.loc[row_val,col_val]
                pred = model.predict(new_mw.loc["mean"].values.reshape(1, -1))
                predicted_prop[row] = pred
            dy = deriv(mw_offset[col_val].values + mw.at["mean", col_val], predicted_prop)
            temp_derivs.loc[i,col_val] = dy[1]
    prop_derivs = temp_derivs.mean(axis=0)
    return prop_derivs

