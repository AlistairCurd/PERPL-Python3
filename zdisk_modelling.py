"""
zdisk_modelling.py

Functions to fit models of relative positions in Z-disk data to
relative positions among localisation microscopy data.

Created on Wed Dec 12 14:28:49 2018

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
import pandas as pd
from scipy.optimize import curve_fit
import linearrepeatmodels as linmods
import models_2d_distances_normalised as mods2d
from modelling_general import ModelWithFitSettings


def read_relpos_from_pickles(input_files):
    """Prepare array of relative positions, for use in analysis and figures,
    using data in pickle files.

    Args:
        input_files (list):
            The list of paths from which the relative position data should be
            read.
            If only one input file, use as a list of length 1, i.e. in [].

    Returns:
        relpos (numpy-like array):
            The list of 2D or 3D relative positions, aggregated over all
            input files.
    """
    # Use Affimer data pickles by default
    relpos = pd.read_pickle(input_files[0])
    if len(input_files) > 1:
        for i in range(1, len(input_files)):
            relpos = relpos.append(pd.read_pickle(input_files[i]))

    return relpos


def getaxialseparations_no_smoothing(relpos,
                                     max_distance=100.,
                                     transverse_limit=10.):
    """Get axial separations between localisations, also within a limited
    transverse distance of one another.

    Args:
        relpos:
            Relative positions containing .axial and .transverse
            distance (e.g. pandas dataframe).
        max_distance:
            The maximum axial separation upto which data will be
            modelled.
        transverselimit:
            The maximum distance across the transverse plane
            between localisations involved.
    Returns:
        sorted_axpoints:
            The axial separations as desired, sorted from lowest to
            highest.
    """
    # Make all axial separations positive.
    relpos.axial = abs(relpos.axial)

    # Select desired data.
    axpoints = relpos.axial[(relpos.axial < max_distance) &
                            (relpos.transverse < transverse_limit)]
    sorted_axpoints = np.sort(axpoints)
    return sorted_axpoints


def getaxialseparations_with_smoothing(relpos,
                                       fitlength=100.,
                                       locprec=3.1, # AF647 default
                                       transverse_limit=10.,
                                       smoothing=False):
    """Get axial separations between localisations, also within a limited
    transverse distance of one another.

    Args:
        relpos:
            Relative positions containing .axial and .transverse
            distance (e.g. pandas dataframe).
        fitlength:
            The maximum axial separation upto which data will be
            modelled.
        locprec:
            Average localisation precision of the localisations involved.
        transverselimit:
            The maximum distance across the transverse plane
            between localisations involved.
    Returns:
        axpoints:
            The axial separations as desired, sorted from lowest to
            highest.
    """
    # Make all axial separations positive.
    relpos.axial = abs(relpos.axial)

    smoothing = np.sqrt(2) * locprec

    # Select desired data.
    # The smoothing is used like this so that points beyond those not
    # included would only have 1% effect on the values at fitlength
    axpoints = relpos.axial[(relpos.axial < (fitlength + smoothing * 3)) &
                            (relpos.transverse < transverse_limit)]
    sorted_axpoints = np.sort(axpoints)
    return sorted_axpoints


def get_transverse_separations(relpos,
                               max_distance=100.,
                               axial_limit=10.):
    """Get axial separations between localisations, also within a limited
    transverse distance of one another.

    Args:
        relpos:
            Relative positions containing .axial and .transverse
            distance (e.g. pandas dataframe).
        max_distance:
            The maximum transverse separation upto which data will be
            analysed.
        axial_limit:
            The maximum distance along the cell-axis
            between localisations involved.
    Returns:
        sorted_trans_points:
            The axial separations as desired, sorted from lowest to
            highest.
    """
    # Select desired data.
    trans_points = relpos.transverse[(relpos.transverse < max_distance)
                                      & (relpos.axial < axial_limit)
                                      & (relpos.axial > -axial_limit)]
    sorted_trans_points = np.sort(trans_points)
    return sorted_trans_points


def remove_duplicates(sortedrelpos):
    """Slice relative positions so that duplicates are removed. This can be
    necessary when absolute amplitude of the
    histograms/density is important.
    """
    slicedrelpos = sortedrelpos[::2]
    return slicedrelpos


def fitmodel_to_hist(x,
                     experimentaldist,
                     model=linmods.linrepplusreps5fixedpeakratio,
                     initial_params=None,
                     param_bounds=None):
    """Fit model to distance histogram. Designed to allow user editting of
    parameter guesses and bounds for models of RPDs for linear repeating
    models, as in linearrepeatmodels.py.

    Args:
        x:
            Distance values at which the experimental distribution of
            distances and the model are evaluated.
        experimentaldist:
            The values of the experimental distribution of distances. Can be
            e.g. histogram or KDE.
        model:
            The model for the RPD.
        fitlength:
            The maximum distance that may be included in the fit.
    """
    popt, pcov = curve_fit(
            model, x, experimentaldist,
            p0=initial_params,
            bounds=param_bounds
            )
    # plt.plot(x, model(x, *popt))
    perr = np.sqrt(np.diag(pcov))
    params = np.column_stack((popt, perr))
    print(params)
    k = float(len(popt) + 1)  # No. free parameters,
                              # including var. of residuals
                              # for least squares fit.

    ssr = np.sum((model(x, *popt)
                  - experimentaldist) ** 2)
    aic = len(x) * np.log(ssr / len(x)) + 2 * k
    aiccorr = aic + 2 * k * (k + 1) / (len(x) - k - 1)

    print('SSR =', ssr)
    print('AIC =', aic)
    print('AICcorr =', aiccorr)

    return popt, pcov, perr


def create_default_fitting_params_dicts():
    """Create lists for initial guesses and bounds for parameters during
    fitting. Sets the defaults for scipy.optimise.curve_fit

    Returns:
        lower_bound_dict (dict):
            Dictionary of default lower bound options for parameter values.
        upper_bound_dict (dict):
            Dictionary of default upper bound options for parameter values.
        initial_params_dict (dict):
            Dictionary of default initial parameter value options.
    """
    lower_bound_dict = {
            'repeat_distance': 0,
            'repeat_broadening': 0,
            'first_peak_offset': 0,
            'amp_peak_1': 0,
            'amp_peak_2': 0,
            'amp_peak_3': 0,
            'amp_peak_4': 0,
            'amp_peak_5': 0,
            'amp_peak_6': 0,
            'loc_prec_sd': 0,
            'loc_prec_amp': 0,
            'bg_slope': -100,
            'bg_offset': 0,
            }

    upper_bound_dict = {
            'repeat_distance': 50,
            'repeat_broadening': 20,
            'first_peak_offset': 30,
            'amp_peak_1': 1000,
            'amp_peak_2': 1000,
            'amp_peak_3': 1000,
            'amp_peak_4': 1000,
            'amp_peak_5': 10000,
            'amp_peak_6': 1000,
            'loc_prec_sd': 20,
            'loc_prec_amp': 1000,
            'bg_slope': 100,
            'bg_offset': 100,
            }

    initial_params_dict = {
            'repeat_distance': 20,
            'repeat_broadening': 5,
            'first_peak_offset': 0,
            'amp_peak_1': 1,
            'amp_peak_2': 1,
            'amp_peak_3': 1,
            'amp_peak_4': 1,
            'amp_peak_5': 1,
            'amp_peak_6': 1,
            'loc_prec_sd': 3,
            'loc_prec_amp': 1,
            'bg_slope': -0.2,
            'bg_offset': 20,
            }

    return lower_bound_dict, upper_bound_dict, initial_params_dict


def set_up_model_3_peaks_fixed_ratio_with_fit_settings():
    """Set up the RPD model with fitting settings.
    The fitting settings are to pass to scipy's
    curve_fit, and the vector-input version of the model is for
    differentiation and error propagation with numdifftools.

    Args:
        None

    Returns:
        A ModelWithFitSettings object containing:
            model_rpd (function name):
                Relative position density as a function of separation
                between localisations.
            initial_params (list):
                Starting guesses for the parameter values by
                scipy.optimize.curve_fit
            lower_bounds (list), upper_bounds (list):
                The bounds on allowable parameter values as
                scipy.optimize.curve_fit runs.
    """
    # Generate ModelWithFitSettings object, conatining a model_rpd
    model_with_fit_settings = (
        ModelWithFitSettings(model_rpd=linmods.linrepplusreps3fixedpeakratio)
        )

    # Add fitting parameters to ModelWithFitSettings object
    (lower_bound_dict,
     upper_bound_dict,
     initial_params_dict) = create_default_fitting_params_dicts()

    # Can optionally modify these dictionaries here:

    initial_params = [initial_params_dict['repeat_distance'],
                      initial_params_dict['repeat_broadening'],
                      initial_params_dict['amp_peak_1'],
                      initial_params_dict['loc_prec_sd'],
                      initial_params_dict['loc_prec_amp'],
                      initial_params_dict['bg_slope'],
                      initial_params_dict['bg_offset']
                      ]

    lower_bounds = [lower_bound_dict['repeat_distance'],
                    lower_bound_dict['repeat_broadening'],
                    lower_bound_dict['amp_peak_1'],
                    lower_bound_dict['loc_prec_sd'],
                    lower_bound_dict['loc_prec_amp'],
                    lower_bound_dict['bg_slope'],
                    lower_bound_dict['bg_offset']
                    ]

    upper_bounds = [upper_bound_dict['repeat_distance'],
                    upper_bound_dict['repeat_broadening'],
                    upper_bound_dict['amp_peak_1'],
                    upper_bound_dict['loc_prec_sd'],
                    upper_bound_dict['loc_prec_amp'],
                    upper_bound_dict['bg_slope'],
                    upper_bound_dict['bg_offset']
                    ]

    bounds = (lower_bounds, upper_bounds)

    model_with_fit_settings.initial_params = (
        initial_params
        )
    model_with_fit_settings.param_bounds = (
        bounds
        )
    model_with_fit_settings.vector_input_model = (
        linmods.linrepplusreps3fixedpeakratiovectorinput
        )

    return model_with_fit_settings


def set_up_model_4_peaks_fixed_ratio_with_fit_settings():
    """Set up the RPD model with fitting settings.
    The fitting settings are to pass to scipy's
    curve_fit, and the vector-input version of the model is for
    differentiation and error propagation with numdifftools.

    Args:
        None

    Returns:
        A ModelWithFitSettings object containing:
            model_rpd (function name):
                Relative position density as a function of separation
                between localisations.
            initial_params (list):
                Starting guesses for the parameter values by
                scipy.optimize.curve_fit
            lower_bounds (list), upper_bounds (list):
                The bounds on allowable parameter values as
                scipy.optimize.curve_fit runs.
    """
    # Generate ModelWithFitSettings object, conatining a model_rpd
    model_with_fit_settings = (
        ModelWithFitSettings(model_rpd=linmods.linrepplusreps4fixedpeakratio)
        )

    # Add fitting parameters to ModelWithFitSettings object
    (lower_bound_dict,
     upper_bound_dict,
     initial_params_dict) = create_default_fitting_params_dicts()

    # Can optionally modify these dictionaries here:

    initial_params = [initial_params_dict['repeat_distance'],
                      initial_params_dict['repeat_broadening'],
                      initial_params_dict['amp_peak_1'],
                      initial_params_dict['loc_prec_sd'],
                      initial_params_dict['loc_prec_amp'],
                      initial_params_dict['bg_slope'],
                      initial_params_dict['bg_offset']
                      ]

    lower_bounds = [lower_bound_dict['repeat_distance'],
                    lower_bound_dict['repeat_broadening'],
                    lower_bound_dict['amp_peak_1'],
                    lower_bound_dict['loc_prec_sd'],
                    lower_bound_dict['loc_prec_amp'],
                    lower_bound_dict['bg_slope'],
                    lower_bound_dict['bg_offset']
                    ]

    upper_bounds = [upper_bound_dict['repeat_distance'],
                    upper_bound_dict['repeat_broadening'],
                    upper_bound_dict['amp_peak_1'],
                    upper_bound_dict['loc_prec_sd'],
                    upper_bound_dict['loc_prec_amp'],
                    upper_bound_dict['bg_slope'],
                    upper_bound_dict['bg_offset']
                    ]

    bounds = (lower_bounds, upper_bounds)

    model_with_fit_settings.initial_params = (
        initial_params
        )
    model_with_fit_settings.param_bounds = (
        bounds
        )
    model_with_fit_settings.vector_input_model = (
        linmods.linrepplusreps4fixedpeakratiovectorinput
        )

    return model_with_fit_settings


def set_up_model_5_peaks_fixed_ratio_with_fit_settings():
    """Set up the RPD model with fitting settings.
    The fitting settings are to pass to scipy's
    curve_fit, and the vector-input version of the model is for
    differentiation and error propagation with numdifftools.

    Args:
        None

    Returns:
        A ModelWithFitSettings object containing:
            model_rpd (function name):
                Relative position density as a function of separation
                between localisations.
            initial_params (list):
                Starting guesses for the parameter values by
                scipy.optimize.curve_fit
            lower_bounds (list), upper_bounds (list):
                The bounds on allowable parameter values as
                scipy.optimize.curve_fit runs.
    """
    # Generate ModelWithFitSettings object, conatining a model_rpd
    model_with_fit_settings = (
        ModelWithFitSettings(model_rpd=linmods.linrepplusreps5fixedpeakratio)
        )

    # Add fitting parameters to ModelWithFitSettings object
    (lower_bound_dict,
     upper_bound_dict,
     initial_params_dict) = create_default_fitting_params_dicts()

    # Can optionally modify these dictionaries here:

    initial_params = [initial_params_dict['repeat_distance'],
                      initial_params_dict['repeat_broadening'],
                      initial_params_dict['amp_peak_1'],
                      initial_params_dict['loc_prec_sd'],
                      initial_params_dict['loc_prec_amp'],
                      initial_params_dict['bg_slope'],
                      initial_params_dict['bg_offset']
                      ]

    lower_bounds = [lower_bound_dict['repeat_distance'],
                    lower_bound_dict['repeat_broadening'],
                    lower_bound_dict['amp_peak_1'],
                    lower_bound_dict['loc_prec_sd'],
                    lower_bound_dict['loc_prec_amp'],
                    lower_bound_dict['bg_slope'],
                    lower_bound_dict['bg_offset']
                    ]

    upper_bounds = [upper_bound_dict['repeat_distance'],
                    upper_bound_dict['repeat_broadening'],
                    upper_bound_dict['amp_peak_1'],
                    upper_bound_dict['loc_prec_sd'],
                    upper_bound_dict['loc_prec_amp'],
                    upper_bound_dict['bg_slope'],
                    upper_bound_dict['bg_offset']
                    ]

    bounds = (lower_bounds, upper_bounds)

    model_with_fit_settings.initial_params = (
        initial_params
        )
    model_with_fit_settings.param_bounds = (
        bounds
        )
    model_with_fit_settings.vector_input_model = (
        linmods.linrepplusreps5fixedpeakratiovectorinput
        )

    return model_with_fit_settings


def set_up_model_5_peaks_fixed_ratio_no_replocs_with_fit_settings():
    """Set up the RPD model with fitting settings.
    
    This model has five peaks on a linear repeats, with linear decreasing
    amplitude, above a linear sloping backgroung, with no component for
    repeated localisations.
    
    The fitting settings are to pass to scipy's
    curve_fit, and the vector-input version of the model is for
    differentiation and error propagation with numdifftools.

    Args:
        None

    Returns:
        A ModelWithFitSettings object containing:
            model_rpd (function name):
                Relative position density as a function of separation
                between localisations.
            initial_params (list):
                Starting guesses for the parameter values by
                scipy.optimize.curve_fit
            lower_bounds (list), upper_bounds (list):
                The bounds on allowable parameter values as
                scipy.optimize.curve_fit runs.
    """
    # Generate ModelWithFitSettings object, conatining a model_rpd
    model_with_fit_settings = (
        ModelWithFitSettings(model_rpd=linmods.linrepnoreps5_fixedpeakratio)
        )

    # Add fitting parameters to ModelWithFitSettings object
    (lower_bound_dict,
     upper_bound_dict,
     initial_params_dict) = create_default_fitting_params_dicts()

    # Can optionally modify these dictionaries here:

    initial_params = [initial_params_dict['repeat_distance'],
                      initial_params_dict['repeat_broadening'],
                      initial_params_dict['amp_peak_1'],
                      initial_params_dict['bg_slope'],
                      initial_params_dict['bg_offset']
                      ]

    lower_bounds = [lower_bound_dict['repeat_distance'],
                    lower_bound_dict['repeat_broadening'],
                    lower_bound_dict['amp_peak_1'],
                    lower_bound_dict['bg_slope'],
                    lower_bound_dict['bg_offset']
                    ]

    upper_bounds = [upper_bound_dict['repeat_distance'],
                    upper_bound_dict['repeat_broadening'],
                    upper_bound_dict['amp_peak_1'],
                    upper_bound_dict['bg_slope'],
                    upper_bound_dict['bg_offset']
                    ]

    bounds = (lower_bounds, upper_bounds)

    model_with_fit_settings.initial_params = (
        initial_params
        )
    model_with_fit_settings.param_bounds = (
        bounds
        )
    model_with_fit_settings.vector_input_model = (
        linmods.linrepnoreps5_fixedpeakratio_vectorinput
        )

    return model_with_fit_settings


def set_up_model_5_variable_peaks_with_fit_settings():
    """Set up the RPD model with fitting settings.
    The fitting settings are to pass to scipy's
    curve_fit, and the vector-input version of the model is for
    differentiation and error propagation with numdifftools.

    Args:
        None

    Returns:
        A ModelWithFitSettings object containing:
            model_rpd (function name):
                Relative position density as a function of separation
                between localisations.
            initial_params (list):
                Starting guesses for the parameter values by
                scipy.optimize.curve_fit
            lower_bounds (list), upper_bounds (list):
                The bounds on allowable parameter values as
                scipy.optimize.curve_fit runs.
    """
    # Generate ModelWithFitSettings object, conatining a model_rpd
    model_with_fit_settings = (
        ModelWithFitSettings(model_rpd=linmods.linrepnoreps5_bg_non_negative)
        )

    # Add fitting parameters to ModelWithFitSettings object
    (lower_bound_dict,
     upper_bound_dict,
     initial_params_dict) = create_default_fitting_params_dicts()

    # Can optionally modify these dictionaries here:

    initial_params = [initial_params_dict['repeat_distance'],
                      initial_params_dict['repeat_broadening'],
                      initial_params_dict['amp_peak_1'],
                      initial_params_dict['amp_peak_2'],
                      initial_params_dict['amp_peak_3'],
                      initial_params_dict['amp_peak_4'],
                      initial_params_dict['amp_peak_5'],
                      initial_params_dict['bg_slope'],
                      initial_params_dict['bg_offset']
                      ]

    lower_bounds = [lower_bound_dict['repeat_distance'],
                    lower_bound_dict['repeat_broadening'],
                    lower_bound_dict['amp_peak_1'],
                    lower_bound_dict['amp_peak_2'],
                    lower_bound_dict['amp_peak_3'],
                    lower_bound_dict['amp_peak_4'],
                    lower_bound_dict['amp_peak_5'],
                    lower_bound_dict['bg_slope'],
                    lower_bound_dict['bg_offset']
                    ]

    upper_bounds = [upper_bound_dict['repeat_distance'],
                    upper_bound_dict['repeat_broadening'],
                    upper_bound_dict['amp_peak_1'],
                    upper_bound_dict['amp_peak_2'],
                    upper_bound_dict['amp_peak_3'],
                    upper_bound_dict['amp_peak_4'],
                    upper_bound_dict['amp_peak_5'],
                    upper_bound_dict['bg_slope'],
                    upper_bound_dict['bg_offset']
                    ]

    bounds = (lower_bounds, upper_bounds)

    model_with_fit_settings.initial_params = (
        initial_params
        )
    model_with_fit_settings.param_bounds = (
        bounds
        )
    model_with_fit_settings.vector_input_model = (
        linmods.linrepnoreps5_bg_linear_vectorinput
        )
        # linmods.linrepnoreps5_bg_zero_vectorinput

    return model_with_fit_settings


def set_up_model_5_variable_peaks_bg_flat_with_fit_settings():
    """Set up the RPD model with fitting settings.

    This model has five peaks on a linear repeat with independent
    amplitudes. With a flat background level.

    The fitting settings are to pass to scipy's
    curve_fit, and the vector-input version of the model is for
    differentiation and error propagation with numdifftools.

    Args:
        None

    Returns:
        A ModelWithFitSettings object containing:
            model_rpd (function name):
                Relative position density as a function of separation
                between localisations.
            initial_params (list):
                Starting guesses for the parameter values by
                scipy.optimize.curve_fit
            lower_bounds (list), upper_bounds (list):
                The bounds on allowable parameter values as
                scipy.optimize.curve_fit runs.
    """
    # Generate ModelWithFitSettings object, conatining a model_rpd
    model_with_fit_settings = (
        ModelWithFitSettings(model_rpd=linmods.linrepnoreps5_bg_flat)
        )

    # Add fitting parameters to ModelWithFitSettings object
    (lower_bound_dict,
     upper_bound_dict,
     initial_params_dict) = create_default_fitting_params_dicts()

    # Can optionally modify these dictionaries here:

    initial_params = [initial_params_dict['repeat_distance'],
                      initial_params_dict['repeat_broadening'],
                      initial_params_dict['amp_peak_1'],
                      initial_params_dict['amp_peak_2'],
                      initial_params_dict['amp_peak_3'],
                      initial_params_dict['amp_peak_4'],
                      initial_params_dict['amp_peak_5'],
                      initial_params_dict['bg_offset']
                      ]

    lower_bounds = [lower_bound_dict['repeat_distance'],
                    lower_bound_dict['repeat_broadening'],
                    lower_bound_dict['amp_peak_1'],
                    lower_bound_dict['amp_peak_2'],
                    lower_bound_dict['amp_peak_3'],
                    lower_bound_dict['amp_peak_4'],
                    lower_bound_dict['amp_peak_5'],
                    lower_bound_dict['bg_offset']
                    ]

    upper_bounds = [upper_bound_dict['repeat_distance'],
                    upper_bound_dict['repeat_broadening'],
                    upper_bound_dict['amp_peak_1'],
                    upper_bound_dict['amp_peak_2'],
                    upper_bound_dict['amp_peak_3'],
                    upper_bound_dict['amp_peak_4'],
                    upper_bound_dict['amp_peak_5'],
                    upper_bound_dict['bg_offset']
                    ]

    bounds = (lower_bounds, upper_bounds)

    model_with_fit_settings.initial_params = (
        initial_params
        )
    model_with_fit_settings.param_bounds = (
        bounds
        )
    model_with_fit_settings.vector_input_model = (
        linmods.linrepnoreps5_bg_flat_vectorinput
        )

    return model_with_fit_settings


def set_up_model_5_variable_peaks_with_replocs_bg_flat_with_fit_settings():
    """Set up the RPD model with fitting settings.

    This model has five peaks on a linear repeat with independent
    amplitudes. With a flat background level. With repeated
    localisations.

    The fitting settings are to pass to scipy's
    curve_fit, and the vector-input version of the model is for
    differentiation and error propagation with numdifftools.

    Args:
        None

    Returns:
        A ModelWithFitSettings object containing:
            model_rpd (function name):
                Relative position density as a function of separation
                between localisations.
            initial_params (list):
                Starting guesses for the parameter values by
                scipy.optimize.curve_fit
            lower_bounds (list), upper_bounds (list):
                The bounds on allowable parameter values as
                scipy.optimize.curve_fit runs.
    """
    # Generate ModelWithFitSettings object, conatining a model_rpd
    model_with_fit_settings = (
        ModelWithFitSettings(model_rpd=linmods.linrepplusreps5_bg_flat)
        )

    # Add fitting parameters to ModelWithFitSettings object
    (lower_bound_dict,
     upper_bound_dict,
     initial_params_dict) = create_default_fitting_params_dicts()

    # Can optionally modify these dictionaries here:

    initial_params = [initial_params_dict['repeat_distance'],
                      initial_params_dict['repeat_broadening'],
                      initial_params_dict['amp_peak_1'],
                      initial_params_dict['amp_peak_2'],
                      initial_params_dict['amp_peak_3'],
                      initial_params_dict['amp_peak_4'],
                      initial_params_dict['amp_peak_5'],
                      initial_params_dict['loc_prec_sd'],
                      initial_params_dict['loc_prec_amp'],
                      initial_params_dict['bg_offset']
                      ]

    lower_bounds = [lower_bound_dict['repeat_distance'],
                    lower_bound_dict['repeat_broadening'],
                    lower_bound_dict['amp_peak_1'],
                    lower_bound_dict['amp_peak_2'],
                    lower_bound_dict['amp_peak_3'],
                    lower_bound_dict['amp_peak_4'],
                    lower_bound_dict['amp_peak_5'],
                    lower_bound_dict['loc_prec_sd'],
                    lower_bound_dict['loc_prec_amp'],
                    lower_bound_dict['bg_offset']
                    ]

    upper_bounds = [upper_bound_dict['repeat_distance'],
                    upper_bound_dict['repeat_broadening'],
                    upper_bound_dict['amp_peak_1'],
                    upper_bound_dict['amp_peak_2'],
                    upper_bound_dict['amp_peak_3'],
                    upper_bound_dict['amp_peak_4'],
                    upper_bound_dict['amp_peak_5'],
                    50,# upper_bound_dict['loc_prec_sd'],
                    upper_bound_dict['loc_prec_amp'],
                    upper_bound_dict['bg_offset']
                    ]

    bounds = (lower_bounds, upper_bounds)

    model_with_fit_settings.initial_params = (
        initial_params
        )
    model_with_fit_settings.param_bounds = (
        bounds
        )
    model_with_fit_settings.vector_input_model = (
        linmods.linrepplusreps5_bg_flat_vectorinput
        )

    return model_with_fit_settings


def set_up_model_5_variable_peaks_after_offset_with_fit_settings():
    """Set up the RPD model with fitting settings.
    The fitting settings are to pass to scipy's
    curve_fit, and the vector-input version of the model is for
    differentiation and error propagation with numdifftools.

    Args:
        None

    Returns:
        A ModelWithFitSettings object containing:
            model_rpd (function name):
                Relative position density as a function of separation
                between localisations.
            initial_params (list):
                Starting guesses for the parameter values by
                scipy.optimize.curve_fit
            lower_bounds (list), upper_bounds (list):
                The bounds on allowable parameter values as
                scipy.optimize.curve_fit runs.
    """
    # Generate ModelWithFitSettings object, conatining a model_rpd
    model_with_fit_settings = (
        ModelWithFitSettings(model_rpd=linmods.lin_repeat_after_offset_5)
        )

    # Add fitting parameters to ModelWithFitSettings object
    (lower_bound_dict,
     upper_bound_dict,
     initial_params_dict) = create_default_fitting_params_dicts()

    # Can optionally modify these dictionaries here:

    initial_params = [initial_params_dict['repeat_distance'],
                      initial_params_dict['repeat_broadening'],
                      10.,#initial_params_dict['first_peak_offset'],
                      initial_params_dict['amp_peak_1'],
                      initial_params_dict['amp_peak_2'],
                      initial_params_dict['amp_peak_3'],
                      initial_params_dict['amp_peak_4'],
                      initial_params_dict['amp_peak_5'],
                      initial_params_dict['amp_peak_6'],
                      initial_params_dict['bg_slope'],
                      initial_params_dict['bg_offset']
                      ]

    lower_bounds = [lower_bound_dict['repeat_distance'],
                    lower_bound_dict['repeat_broadening'],
                    lower_bound_dict['first_peak_offset'],
                    lower_bound_dict['amp_peak_1'],
                    lower_bound_dict['amp_peak_2'],
                    lower_bound_dict['amp_peak_3'],
                    lower_bound_dict['amp_peak_4'],
                    lower_bound_dict['amp_peak_5'],
                    lower_bound_dict['amp_peak_6'],
                    lower_bound_dict['bg_slope'],
                    lower_bound_dict['bg_offset']
                    ]

    upper_bounds = [upper_bound_dict['repeat_distance'],
                    upper_bound_dict['repeat_broadening'],
                    upper_bound_dict['first_peak_offset'],
                    upper_bound_dict['amp_peak_1'],
                    upper_bound_dict['amp_peak_2'],
                    upper_bound_dict['amp_peak_3'],
                    upper_bound_dict['amp_peak_4'],
                    upper_bound_dict['amp_peak_5'],
                    upper_bound_dict['amp_peak_6'],
                    upper_bound_dict['bg_slope'],
                    upper_bound_dict['bg_offset']
                    ]

    bounds = (lower_bounds, upper_bounds)

    model_with_fit_settings.initial_params = (
        initial_params
        )
    model_with_fit_settings.param_bounds = (
        bounds
        )
    model_with_fit_settings.vector_input_model = (
        linmods.lin_repeat_after_offset_5_vectorargs
        )

    return model_with_fit_settings


def set_up_model_5_variable_peaks_after_offset_flat_bg_with_fit_settings():
    """Set up the RPD model with fitting settings.
    
    This model has a term for repeated localisations, followed by peaks at
    offset + n * repeat distance, where is 1-5. With a flat background term.

    The fitting settings are to pass to scipy's
    curve_fit, and the vector-input version of the model is for
    differentiation and error propagation with numdifftools.

    Args:
        None

    Returns:
        A ModelWithFitSettings object containing:
            model_rpd (function name):
                Relative position density as a function of separation
                between localisations.
            initial_params (list):
                Starting guesses for the parameter values by
                scipy.optimize.curve_fit
            lower_bounds (list), upper_bounds (list):
                The bounds on allowable parameter values as
                scipy.optimize.curve_fit runs.
    """
    # Generate ModelWithFitSettings object, conatining a model_rpd
    model_with_fit_settings = (
        ModelWithFitSettings(model_rpd=linmods.lin_repeat_after_offset_5_flat_bg_replocs)
        )

    # Add fitting parameters to ModelWithFitSettings object
    (lower_bound_dict,
     upper_bound_dict,
     initial_params_dict) = create_default_fitting_params_dicts()

    # Can optionally modify these dictionaries here:

    initial_params = [initial_params_dict['repeat_distance'],
                      initial_params_dict['repeat_broadening'],
                      initial_params_dict['first_peak_offset'],
                      initial_params_dict['amp_peak_1'],
                      initial_params_dict['amp_peak_2'],
                      initial_params_dict['amp_peak_3'],
                      initial_params_dict['amp_peak_4'],
                      initial_params_dict['amp_peak_5'],
                      initial_params_dict['amp_peak_6'],
                      initial_params_dict['loc_prec_sd'],
                      initial_params_dict['loc_prec_amp'],
                      initial_params_dict['bg_offset']
                      ]

    lower_bounds = [lower_bound_dict['repeat_distance'],
                    lower_bound_dict['repeat_broadening'],
                    lower_bound_dict['first_peak_offset'],
                    lower_bound_dict['amp_peak_1'],
                    lower_bound_dict['amp_peak_2'],
                    lower_bound_dict['amp_peak_3'],
                    lower_bound_dict['amp_peak_4'],
                    lower_bound_dict['amp_peak_5'],
                    lower_bound_dict['amp_peak_6'],
                    lower_bound_dict['loc_prec_sd'],
                    lower_bound_dict['loc_prec_amp'],
                    lower_bound_dict['bg_offset']
                    ]

    upper_bounds = [upper_bound_dict['repeat_distance'],
                    upper_bound_dict['repeat_broadening'],
                    10.,#upper_bound_dict['first_peak_offset'],
                    upper_bound_dict['amp_peak_1'],
                    upper_bound_dict['amp_peak_2'],
                    upper_bound_dict['amp_peak_3'],
                    upper_bound_dict['amp_peak_4'],
                    1000.,#upper_bound_dict['amp_peak_5'],
                    10000.,#upper_bound_dict['amp_peak_6'],
                    upper_bound_dict['loc_prec_sd'],
                    upper_bound_dict['loc_prec_amp'],
                    upper_bound_dict['bg_offset']
                    ]

    bounds = (lower_bounds, upper_bounds)

    model_with_fit_settings.initial_params = (
        initial_params
        )
    model_with_fit_settings.param_bounds = (
        bounds
        )
    model_with_fit_settings.vector_input_model = (
        linmods.lin_repeat_after_offset_5_flat_bg_replocs_vectorargs
        )

    return model_with_fit_settings


def set_up_model_4_variable_peaks_with_fit_settings():
    """Set up the RPD model with fitting settings.
    The fitting settings are to pass to scipy's
    curve_fit, and the vector-input version of the model is for
    differentiation and error propagation with numdifftools.

    Args:
        None

    Returns:
        A ModelWithFitSettings object containing:
            model_rpd (function name):
                Relative position density as a function of separation
                between localisations.
            initial_params (list):
                Starting guesses for the parameter values by
                scipy.optimize.curve_fit
            lower_bounds (list), upper_bounds (list):
                The bounds on allowable parameter values as
                scipy.optimize.curve_fit runs.
    """
    # Generate ModelWithFitSettings object, conatining a model_rpd
    model_with_fit_settings = (
        ModelWithFitSettings(model_rpd=linmods.linrepnoreps4_bg_non_negative)
        )

    # Add fitting parameters to ModelWithFitSettings object
    (lower_bound_dict,
     upper_bound_dict,
     initial_params_dict) = create_default_fitting_params_dicts()

    # Can optionally modify these dictionaries here:

    initial_params = [initial_params_dict['repeat_distance'],
                      initial_params_dict['repeat_broadening'],
                      initial_params_dict['amp_peak_1'],
                      initial_params_dict['amp_peak_2'],
                      initial_params_dict['amp_peak_3'],
                      initial_params_dict['amp_peak_4'],
                      initial_params_dict['bg_slope'],
                      initial_params_dict['bg_offset']
                      ]

    lower_bounds = [lower_bound_dict['repeat_distance'],
                    lower_bound_dict['repeat_broadening'],
                    lower_bound_dict['amp_peak_1'],
                    lower_bound_dict['amp_peak_2'],
                    lower_bound_dict['amp_peak_3'],
                    lower_bound_dict['amp_peak_4'],
                    lower_bound_dict['bg_slope'],
                    lower_bound_dict['bg_offset']
                    ]

    upper_bounds = [upper_bound_dict['repeat_distance'],
                    upper_bound_dict['repeat_broadening'],
                    upper_bound_dict['amp_peak_1'],
                    upper_bound_dict['amp_peak_2'],
                    upper_bound_dict['amp_peak_3'],
                    upper_bound_dict['amp_peak_4'],
                    upper_bound_dict['bg_slope'],
                    upper_bound_dict['bg_offset']
                    ]

    bounds = (lower_bounds, upper_bounds)

    model_with_fit_settings.initial_params = (
        initial_params
        )
    model_with_fit_settings.param_bounds = (
        bounds
        )
    model_with_fit_settings.vector_input_model = (
        [linmods.linrepnoreps4_bg_linear_vectorinput,
         linmods.linrepnoreps4_bg_zero_vectorinput
         ]
        )

    return model_with_fit_settings


def set_up_model_4_variable_peaks_after_offset_with_fit_settings():
    """Set up the RPD model with fitting settings.
    The fitting settings are to pass to scipy's
    curve_fit, and the vector-input version of the model is for
    differentiation and error propagation with numdifftools.

    Args:
        None

    Returns:
        A ModelWithFitSettings object containing:
            model_rpd (function name):
                Relative position density as a function of separation
                between localisations.
            initial_params (list):
                Starting guesses for the parameter values by
                scipy.optimize.curve_fit
            lower_bounds (list), upper_bounds (list):
                The bounds on allowable parameter values as
                scipy.optimize.curve_fit runs.
    """
    # Generate ModelWithFitSettings object, conatining a model_rpd
    model_with_fit_settings = (
        ModelWithFitSettings(model_rpd=linmods.lin_repeat_after_offset_4)
        )

    # Add fitting parameters to ModelWithFitSettings object
    (lower_bound_dict,
     upper_bound_dict,
     initial_params_dict) = create_default_fitting_params_dicts()

    # Can optionally modify these dictionaries here:

    initial_params = [initial_params_dict['repeat_distance'],
                      initial_params_dict['repeat_broadening'],
                      initial_params_dict['first_peak_offset'],
                      initial_params_dict['amp_peak_1'],
                      initial_params_dict['amp_peak_2'],
                      initial_params_dict['amp_peak_3'],
                      initial_params_dict['amp_peak_4'],
                      initial_params_dict['bg_slope'],
                      initial_params_dict['bg_offset']
                      ]

    lower_bounds = [lower_bound_dict['repeat_distance'],
                    lower_bound_dict['repeat_broadening'],
                    lower_bound_dict['first_peak_offset'],
                    lower_bound_dict['amp_peak_1'],
                    lower_bound_dict['amp_peak_2'],
                    lower_bound_dict['amp_peak_3'],
                    lower_bound_dict['amp_peak_4'],
                    lower_bound_dict['bg_slope'],
                    lower_bound_dict['bg_offset']
                    ]

    upper_bounds = [upper_bound_dict['repeat_distance'],
                    upper_bound_dict['repeat_broadening'],
                    upper_bound_dict['first_peak_offset'],
                    upper_bound_dict['amp_peak_1'],
                    upper_bound_dict['amp_peak_2'],
                    upper_bound_dict['amp_peak_3'],
                    upper_bound_dict['amp_peak_4'],
                    upper_bound_dict['bg_slope'],
                    upper_bound_dict['bg_offset']
                    ]

    bounds = (lower_bounds, upper_bounds)

    model_with_fit_settings.initial_params = (
        initial_params
        )
    model_with_fit_settings.param_bounds = (
        bounds
        )
    model_with_fit_settings.vector_input_model = (
        linmods.lin_repeat_after_offset_4_vectorargs
        )

    return model_with_fit_settings


def set_up_model_linear_fit_with_fit_settings():
    """Set up the RPD model with fitting settings.
    The fitting settings are to pass to scipy's
    curve_fit, and the vector-input version of the model is for
    differentiation and error propagation with numdifftools.

    Args:
        None

    Returns:
        A ModelWithFitSettings object containing:
            model_rpd (function name):
                Relative position density as a function of separation
                between localisations.
            initial_params (list):
                Starting guesses for the parameter values by
                scipy.optimize.curve_fit
            lower_bounds (list), upper_bounds (list):
                The bounds on allowable parameter values as
                scipy.optimize.curve_fit runs.
    """
    # Generate ModelWithFitSettings object, conatining a model_rpd
    model_with_fit_settings = (
        ModelWithFitSettings(model_rpd=linmods.linear_fit)
        )

    # Add fitting parameters to ModelWithFitSettings object
    (lower_bound_dict,
     upper_bound_dict,
     initial_params_dict) = create_default_fitting_params_dicts()

    # Can optionally modify these dictionaries here:

    initial_params = [initial_params_dict['bg_slope'],
                      initial_params_dict['bg_offset']
                      ]

    lower_bounds = [lower_bound_dict['bg_slope'],
                    lower_bound_dict['bg_offset']
                    ]

    upper_bounds = [upper_bound_dict['bg_slope'],
                    upper_bound_dict['bg_offset']
                    ]

    bounds = (lower_bounds, upper_bounds)

    model_with_fit_settings.initial_params = (
        initial_params
        )
    model_with_fit_settings.param_bounds = (
        bounds
        )
    model_with_fit_settings.vector_input_model = (
        linmods.linear_fit_vector_args
        )

    return model_with_fit_settings


def set_up_model_linear_fit_plusreplocs_with_fit_settings():
    """Set up the RPD model with fitting settings.
    The fitting settings are to pass to scipy's
    curve_fit, and the vector-input version of the model is for
    differentiation and error propagation with numdifftools.

    Args:
        None

    Returns:
        A ModelWithFitSettings object containing:
            model_rpd (function name):
                Relative position density as a function of separation
                between localisations.
            initial_params (list):
                Starting guesses for the parameter values by
                scipy.optimize.curve_fit
            lower_bounds (list), upper_bounds (list):
                The bounds on allowable parameter values as
                scipy.optimize.curve_fit runs.
    """
    # Generate ModelWithFitSettings object, conatining a model_rpd
    model_with_fit_settings = (
        ModelWithFitSettings(model_rpd=linmods.justreplocs)
        )

    # Add fitting parameters to ModelWithFitSettings object
    (lower_bound_dict,
     upper_bound_dict,
     initial_params_dict) = create_default_fitting_params_dicts()

    # Can optionally modify these dictionaries here:

    initial_params = [initial_params_dict['loc_prec_sd'],
                      initial_params_dict['loc_prec_amp'],
                      initial_params_dict['bg_slope'],
                      initial_params_dict['bg_offset']
                      ]

    lower_bounds = [lower_bound_dict['loc_prec_sd'],
                    lower_bound_dict['loc_prec_amp'], 
                    lower_bound_dict['bg_slope'],
                    lower_bound_dict['bg_offset']
                    ]

    upper_bounds = [50,# upper_bound_dict['loc_prec_sd'],
                    upper_bound_dict['loc_prec_amp'],
                    upper_bound_dict['bg_slope'],
                    upper_bound_dict['bg_offset']
                    ]

    bounds = (lower_bounds, upper_bounds)

    model_with_fit_settings.initial_params = (
        initial_params
        )
    model_with_fit_settings.param_bounds = (
        bounds
        )
    model_with_fit_settings.vector_input_model = (
        linmods.justreplocs_vectorinput
        )

    return model_with_fit_settings


def set_up_model_onepeak_with_fit_settings():
    """Set up the RPD model with fitting settings.

    This model has one peak representing a characteristic
    distance, and no repeated localisations.

    The fitting settings are to pass to scipy's
    curve_fit, and the vector-input version of the model is for
    differentiation and error propagation with numdifftools.

    Args:
        None

    Returns:
        A ModelWithFitSettings object containing:
            model_rpd (function name):
                Relative position density as a function of separation
                between localisations.
            initial_params (list):
                Starting guesses for the parameter values by
                scipy.optimize.curve_fit
            lower_bounds (list), upper_bounds (list):
                The bounds on allowable parameter values as
                scipy.optimize.curve_fit runs.
    """
    # Generate ModelWithFitSettings object, conatining a model_rpd
    model_with_fit_settings = (
        ModelWithFitSettings(model_rpd=linmods.onepeaknoreps)
        )

    # Add fitting parameters to ModelWithFitSettings object
    (lower_bound_dict,
     upper_bound_dict,
     initial_params_dict) = create_default_fitting_params_dicts()

    # Can optionally modify these dictionaries here:

    initial_params = [initial_params_dict['repeat_distance'],
                      initial_params_dict['repeat_broadening'],
                      initial_params_dict['amp_peak_1'],
                      initial_params_dict['bg_slope'],
                      initial_params_dict['bg_offset']
                      ]

    lower_bounds = [lower_bound_dict['repeat_distance'],
                    lower_bound_dict['repeat_broadening'],
                    lower_bound_dict['amp_peak_1'],
                    lower_bound_dict['bg_slope'],
                    lower_bound_dict['bg_offset']
                    ]

    upper_bounds = [upper_bound_dict['repeat_distance'],
                    upper_bound_dict['repeat_broadening'],
                    upper_bound_dict['amp_peak_1'],
                    upper_bound_dict['bg_slope'],
                    upper_bound_dict['bg_offset']
                    ]

    bounds = (lower_bounds, upper_bounds)

    model_with_fit_settings.initial_params = (
        initial_params
        )
    model_with_fit_settings.param_bounds = (
        bounds
        )
    model_with_fit_settings.vector_input_model = (
        linmods.onepeaknoreps_vectorinput
        )

    return model_with_fit_settings


def set_up_model_onepeak_plus_replocs_with_fit_settings():
    """Set up the RPD model with fitting settings.

    This model has one peak representing a characteristic
    distance, as well as repeated localisations.

    The fitting settings are to pass to scipy's
    curve_fit, and the vector-input version of the model is for
    differentiation and error propagation with numdifftools.

    Args:
        None

    Returns:
        A ModelWithFitSettings object containing:
            model_rpd (function name):
                Relative position density as a function of separation
                between localisations.
            initial_params (list):
                Starting guesses for the parameter values by
                scipy.optimize.curve_fit
            lower_bounds (list), upper_bounds (list):
                The bounds on allowable parameter values as
                scipy.optimize.curve_fit runs.
    """
    # Generate ModelWithFitSettings object, conatining a model_rpd
    model_with_fit_settings = (
        ModelWithFitSettings(model_rpd=linmods.onepeakplusreps)
        )

    # Add fitting parameters to ModelWithFitSettings object
    (lower_bound_dict,
     upper_bound_dict,
     initial_params_dict) = create_default_fitting_params_dicts()

    # Can optionally modify these dictionaries here:

    initial_params = [initial_params_dict['repeat_distance'],
                      initial_params_dict['repeat_broadening'],
                      initial_params_dict['amp_peak_1'],
                      initial_params_dict['loc_prec_sd'],
                      initial_params_dict['loc_prec_amp'],
                      initial_params_dict['bg_slope'],
                      initial_params_dict['bg_offset']
                      ]

    lower_bounds = [lower_bound_dict['repeat_distance'],
                    lower_bound_dict['repeat_broadening'],
                    lower_bound_dict['amp_peak_1'],
                    lower_bound_dict['loc_prec_sd'],
                    lower_bound_dict['loc_prec_amp'],
                    lower_bound_dict['bg_slope'],
                    lower_bound_dict['bg_offset']
                    ]

    upper_bounds = [upper_bound_dict['repeat_distance'],
                    upper_bound_dict['repeat_broadening'],
                    upper_bound_dict['amp_peak_1'],
                    50,# upper_bound_dict['loc_prec_sd'],
                    upper_bound_dict['loc_prec_amp'],
                    upper_bound_dict['bg_slope'],
                    upper_bound_dict['bg_offset']
                    ]

    bounds = (lower_bounds, upper_bounds)

    model_with_fit_settings.initial_params = (
        initial_params
        )
    model_with_fit_settings.param_bounds = (
        bounds
        )
    model_with_fit_settings.vector_input_model = (
        linmods.onepeakplusreps_vectorinput
        )

    return model_with_fit_settings


def set_up_model_onepeak_plus_replocs_flat_bg_with_fit_settings():
    """Set up the RPD model with fitting settings.

    This model has one peak representing a characteristic
    distance, as well as repeated localisations. Only uniform background level.

    The fitting settings are to pass to scipy's
    curve_fit, and the vector-input version of the model is for
    differentiation and error propagation with numdifftools.

    Args:
        None

    Returns:
        A ModelWithFitSettings object containing:
            model_rpd (function name):
                Relative position density as a function of separation
                between localisations.
            initial_params (list):
                Starting guesses for the parameter values by
                scipy.optimize.curve_fit
            lower_bounds (list), upper_bounds (list):
                The bounds on allowable parameter values as
                scipy.optimize.curve_fit runs.
    """
    # Generate ModelWithFitSettings object, conatining a model_rpd
    model_with_fit_settings = (
        ModelWithFitSettings(model_rpd=linmods.onepeakplusreps_flat_bg)
        )

    # Add fitting parameters to ModelWithFitSettings object
    (lower_bound_dict,
     upper_bound_dict,
     initial_params_dict) = create_default_fitting_params_dicts()

    # Can optionally modify these dictionaries here:

    initial_params = [15.,#initial_params_dict['repeat_distance'],
                      initial_params_dict['repeat_broadening'],
                      initial_params_dict['amp_peak_1'],
                      initial_params_dict['loc_prec_sd'],
                      initial_params_dict['loc_prec_amp'],
                      initial_params_dict['bg_offset']
                      ]

    lower_bounds = [lower_bound_dict['repeat_distance'],
                    lower_bound_dict['repeat_broadening'],
                    lower_bound_dict['amp_peak_1'],
                    lower_bound_dict['loc_prec_sd'],
                    lower_bound_dict['loc_prec_amp'],
                    lower_bound_dict['bg_offset']
                    ]

    upper_bounds = [upper_bound_dict['repeat_distance'],
                    upper_bound_dict['repeat_broadening'],
                    upper_bound_dict['amp_peak_1'],
                    100,# upper_bound_dict['loc_prec_sd'],
                    upper_bound_dict['loc_prec_amp'],
                    upper_bound_dict['bg_offset']
                    ]

    bounds = (lower_bounds, upper_bounds)

    model_with_fit_settings.initial_params = (
        initial_params
        )
    model_with_fit_settings.param_bounds = (
        bounds
        )
    model_with_fit_settings.vector_input_model = (
        linmods.onepeakplusreps_flat_bg_vectorinput
        )

    return model_with_fit_settings


def set_up_model_2d_onepeak_plus_replocs_flat_bg_with_fit_settings():
    """Set up the RPD model with fitting settings.

    This model has one peak representing a characteristic
    distance, as well as repeated localisations. Only uniform background level.

    The fitting settings are to pass to scipy's
    curve_fit, and the vector-input version of the model is for
    differentiation and error propagation with numdifftools.

    Args:
        None

    Returns:
        A ModelWithFitSettings object containing:
            model_rpd (function name):
                Relative position density as a function of separation
                between localisations.
            initial_params (list):
                Starting guesses for the parameter values by
                scipy.optimize.curve_fit
            lower_bounds (list), upper_bounds (list):
                The bounds on allowable parameter values as
                scipy.optimize.curve_fit runs.
    """
    # Generate ModelWithFitSettings object, conatining a model_rpd
    model_with_fit_settings = (
        ModelWithFitSettings(model_rpd=mods2d.onepeakplusreps_normalised_flat_bg)
        )

    # Add fitting parameters to ModelWithFitSettings object
    (lower_bound_dict,
     upper_bound_dict,
     initial_params_dict) = create_default_fitting_params_dicts()

    # Can optionally modify these dictionaries here:

    initial_params = [initial_params_dict['repeat_distance'],
                      initial_params_dict['repeat_broadening'],
                      initial_params_dict['amp_peak_1'],
                      initial_params_dict['loc_prec_sd'],
                      initial_params_dict['loc_prec_amp'],
                      initial_params_dict['bg_offset']
                      ]

    lower_bounds = [lower_bound_dict['repeat_distance'],
                    lower_bound_dict['repeat_broadening'],
                    lower_bound_dict['amp_peak_1'],
                    lower_bound_dict['loc_prec_sd'],
                    lower_bound_dict['loc_prec_amp'],
                    lower_bound_dict['bg_offset']
                    ]

    upper_bounds = [upper_bound_dict['repeat_distance'],
                    upper_bound_dict['repeat_broadening'],
                    upper_bound_dict['amp_peak_1'],
                    upper_bound_dict['loc_prec_sd'],
                    upper_bound_dict['loc_prec_amp'],
                    upper_bound_dict['bg_offset']
                    ]

    bounds = (lower_bounds, upper_bounds)

    model_with_fit_settings.initial_params = (
        initial_params
        )
    model_with_fit_settings.param_bounds = (
        bounds
        )
    model_with_fit_settings.vector_input_model = (
        mods2d.onepeakplusreps_normalised_flat_bg_vectorinput
        )

    return model_with_fit_settings


def set_up_model_onepeak_plus_replocs_no_bg_with_fit_settings():
    """Set up the RPD model with fitting settings.

    This model has one peak representing a characteristic
    distance, as well as repeated localisations.

    The fitting settings are to pass to scipy's
    curve_fit, and the vector-input version of the model is for
    differentiation and error propagation with numdifftools.

    Args:
        None

    Returns:
        A ModelWithFitSettings object containing:
            model_rpd (function name):
                Relative position density as a function of separation
                between localisations.
            initial_params (list):
                Starting guesses for the parameter values by
                scipy.optimize.curve_fit
            lower_bounds (list), upper_bounds (list):
                The bounds on allowable parameter values as
                scipy.optimize.curve_fit runs.
    """
    # Generate ModelWithFitSettings object, conatining a model_rpd
    model_with_fit_settings = (
        ModelWithFitSettings(model_rpd=linmods.onepeakplusreps_no_bg)
        )

    # Add fitting parameters to ModelWithFitSettings object
    (lower_bound_dict,
     upper_bound_dict,
     initial_params_dict) = create_default_fitting_params_dicts()

    # Can optionally modify these dictionaries here:

    initial_params = [initial_params_dict['repeat_distance'],
                      initial_params_dict['repeat_broadening'],
                      initial_params_dict['amp_peak_1'],
                      initial_params_dict['loc_prec_sd'],
                      initial_params_dict['loc_prec_amp']
                      ]

    lower_bounds = [lower_bound_dict['repeat_distance'],
                    lower_bound_dict['repeat_broadening'],
                    lower_bound_dict['amp_peak_1'],
                    lower_bound_dict['loc_prec_sd'],
                    lower_bound_dict['loc_prec_amp']
                    ]

    upper_bounds = [upper_bound_dict['repeat_distance'],
                    upper_bound_dict['repeat_broadening'],
                    upper_bound_dict['amp_peak_1'],
                    50,# upper_bound_dict['loc_prec_sd'],
                    upper_bound_dict['loc_prec_amp']
                    ]

    bounds = (lower_bounds, upper_bounds)

    model_with_fit_settings.initial_params = (
        initial_params
        )
    model_with_fit_settings.param_bounds = (
        bounds
        )
    model_with_fit_settings.vector_input_model = (
        linmods.onepeakplusreps_no_bg_vectorinput
        )

    return model_with_fit_settings


def set_up_model_2d_twopeaks_flat_bg_with_fit_settings():
    """Set up the RPD model with fitting settings.

    This model has two peaks representing characteristic
    distances. Only uniform background level.

    The fitting settings are to pass to scipy's
    curve_fit, and the vector-input version of the model is for
    differentiation and error propagation with numdifftools.

    Args:
        None

    Returns:
        A ModelWithFitSettings object containing:
            model_rpd (function name):
                Relative position density as a function of separation
                between localisations.
            initial_params (list):
                Starting guesses for the parameter values by
                scipy.optimize.curve_fit
            lower_bounds (list), upper_bounds (list):
                The bounds on allowable parameter values as
                scipy.optimize.curve_fit runs.
    """
    # Generate ModelWithFitSettings object, conatining a model_rpd
    model_with_fit_settings = (
        ModelWithFitSettings(model_rpd=mods2d.twopeaks_normalised_flat_bg)
        )

    # Add fitting parameters to ModelWithFitSettings object
    (lower_bound_dict,
     upper_bound_dict,
     initial_params_dict) = create_default_fitting_params_dicts()

    # Can optionally modify these dictionaries here:

    initial_params = [initial_params_dict['repeat_distance'],
                      initial_params_dict['repeat_distance'] * 2.,   
                      initial_params_dict['repeat_broadening'],
                      initial_params_dict['amp_peak_1'],
                      initial_params_dict['amp_peak_2'],                      
                      initial_params_dict['bg_offset']
                      ]

    lower_bounds = [lower_bound_dict['repeat_distance'],
                    lower_bound_dict['repeat_distance'],
                    lower_bound_dict['repeat_broadening'],
                    lower_bound_dict['amp_peak_1'],
                    lower_bound_dict['amp_peak_2'],
                    lower_bound_dict['bg_offset']
                    ]

    upper_bounds = [upper_bound_dict['repeat_distance'],
                    50.,#upper_bound_dict['repeat_distance'],    
                    upper_bound_dict['repeat_broadening'],
                    upper_bound_dict['amp_peak_1'],
                    upper_bound_dict['amp_peak_2'],
                    upper_bound_dict['bg_offset']
                    ]

    bounds = (lower_bounds, upper_bounds)

    model_with_fit_settings.initial_params = (
        initial_params
        )
    model_with_fit_settings.param_bounds = (
        bounds
        )
    model_with_fit_settings.vector_input_model = (
        mods2d.twopeaks_normalised_flat_bg_vectorinput
        )

    return model_with_fit_settings
