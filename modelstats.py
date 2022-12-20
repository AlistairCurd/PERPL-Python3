"""
modelstats.py

Library of statistical functions used to compare models.

Created on Mon Jul 22 11:16:00 2019

Alistair Curd
University of Leeds
30 July 2018

Software Engineering practices applied

Joanna Leng (an EPSRC funded Research Software Engineering Fellow (EP/R025819/1)
University of Leeds
January 2019

---
Copyright 2018 Peckham Lab

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at
http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import numpy as np


def aic_from_least_sqr_fit(expt, mod, popt, fitlength=400.):
    """Calculate AICc for a model fit to data using (non-linear) least
    squares. AICc is corrected Akaike information criterion, Model
    Selection and Inference, Burnham & Anderson, 1998, pub.
    Springer-Verlag, p51).
    Specific to data evaluated
    Args:
        expt:       Experimental pairwise distance distribution, evaluated
                        at n + 0.5 nm, when n is an integer.
        model:      Parametric model distribution.
        popt:       Optimise model parameters, using least squares fit.
        fitlength:  Maximum distance included in the fit.
    Returns:
        ssr:        Sum of Squared Residuals is the sum of the squares of
                        residuals (deviations predicted from actual empirical
                        values of data). It is a measure of the discrepancy
                        between the data and an estimation model. A small RSS
                        indicates a tight fit of the model to the data. It is
                        used as an optimality criterion in parameter selection
                        and model selection.
        aicc:       AICc is an AIC with a correction for small sample sizes where
                        the AIC (Akaike Information Criterion) is an estimator
                        of the relative quality of statistical models for a
                        given set of data
    """
    k = float(len(popt) + 1)  # No. free parameters,
                              # including var. of residuals
                              # for least squares fit.
    ssr = np.sum((mod(np.arange(fitlength) + 0.5, *popt) -
                  expt) ** 2)
    aic = fitlength * np.log(ssr / fitlength) + 2 * k
    aicc = aic + 2 * k * (k + 1) / (fitlength - k - 1)
    # aiccorr = aic + 2 * k * (k + 1) / (len(axpoints) - k - 1)
    # print('SSR =', ssr)
    # print('AIC =', aic)
    # print('AICc =', aicc)

    return ssr, aicc


def akaike_weights(aic_values):
    """Calculate weights for likelihood of models from Akaike information
    criteria. From Burnham & Anderson (1998).
    Args:
        aic_values: numpy array of AIC values from which to calculate weights.
    Returns:
        weights:    numpy array of weights.
    """
    # Get differences of each AIC from the min value
    deltas = aic_values - np.min(aic_values)

    weights = np.exp(-0.5 * deltas) / np.sum(np.exp(-0.5 * deltas))
    return weights
