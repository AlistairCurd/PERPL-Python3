"""
dna_paint_data_fitting.py

Library of functions for fitting model relative position distributions (RPDs)
to relative positions obtained from localisations appearing to be arranged
on a relatively simple geometric pattern, i.e. polyhedra.

Created on Mon Jul 29 15:41:31 2019

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


from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import perpl.modelling.polyhedramodelling as poly
import perpl.statistics.modelstats as stats
import perpl.modelling.modelling_general as models
from perpl.modelling.modelling_general import ModelWithFitSettings
from perpl.modelling.modelling_general import stdev_of_model


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
    lower_bound_dict = {'side_1': 0,
                        'side_2': 0,
                        'side_3': 0,
                        'loc_prec_sd': 0,
                        'loc_prec_amp': 0,
                        'vertices_amp': 0,
                        'vertices_sd': 0,
                        'substruct_amp': 0,
                        'substruct_sd': 0,
                        'square_grid_spacing': 0,
                        'square_grid_amp': 0,
                        'square_grid_sd': 0,
                        'bg_slope': 0,
                        }

    upper_bound_dict = {'side_1': 200,
                        'side_2': 200,
                        'side_3': 200,
                        'loc_prec_sd': 100,
                        'loc_prec_amp': 100,
                        'vertices_amp': 100,
                        'vertices_sd': 100,
                        'substruct_amp': 100,
                        'substruct_sd': 100,
                        'square_grid_spacing': 400,
                        'square_grid_amp': 1000,
                        'square_grid_sd': 100,
                        'bg_slope': 0.1,
                        }

    initial_params_dict = {'side_1': 100,
                           'side_2': 100,
                           'side_3': 100,
                           'loc_prec_sd': 10,
                           'loc_prec_amp': 10,
                           'vertices_amp': 10,
                           'vertices_sd': 10,
                           'substruct_amp': 10,
                           'substruct_sd': 10,
                           'square_grid_spacing': 200,
                           'square_grid_amp': 100,
                           'square_grid_sd': 10,
                           'bg_slope': 0.001,
                           }
    return lower_bound_dict, upper_bound_dict, initial_params_dict


def set_up_tri_prism_on_grid_1_length_2disobg_substruct_with_fit_info():
    """Set up the RPD model with fitting settings,
    for a rotationally symmetric model with spread due to repeated
    localisations and spread to unresolvable substructure.
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
    model_with_info = (
        ModelWithFitSettings(
            model_rpd=poly.tri_prism_on_grid_1_length_2disobg_substruct_rpd
            )
        )

    # Add fitting parameters to ModelWithFitSettings object
    (lower_bound_dict,
     upper_bound_dict,
     initial_params_dict) = create_default_fitting_params_dicts()

    # Can optionally modify these dictionaries here:
    
    initial_params = [initial_params_dict['side_1'],
                      initial_params_dict['loc_prec_amp'],
                      initial_params_dict['loc_prec_sd'],
                      initial_params_dict['vertices_amp'],
                      initial_params_dict['vertices_sd'],
                      initial_params_dict['substruct_amp'],
                      initial_params_dict['substruct_sd'],
                      initial_params_dict['square_grid_spacing'],
                      initial_params_dict['square_grid_amp'],
                      initial_params_dict['square_grid_sd'],
                      initial_params_dict['bg_slope']
                      ]

    lower_bounds = [lower_bound_dict['side_1'],
                    lower_bound_dict['loc_prec_amp'],
                    lower_bound_dict['loc_prec_sd'],
                    lower_bound_dict['vertices_amp'],
                    lower_bound_dict['vertices_sd'],
                    lower_bound_dict['substruct_amp'],
                    lower_bound_dict['substruct_sd'],
                    lower_bound_dict['square_grid_spacing'],
                    lower_bound_dict['square_grid_amp'],
                    lower_bound_dict['square_grid_sd'],
                    lower_bound_dict['bg_slope']
                    ]


    upper_bounds = [upper_bound_dict['side_1'],
                    upper_bound_dict['loc_prec_amp'],
                    upper_bound_dict['loc_prec_sd'],
                    upper_bound_dict['vertices_amp'],
                    upper_bound_dict['vertices_sd'],
                    upper_bound_dict['substruct_amp'],
                    upper_bound_dict['substruct_sd'],
                    upper_bound_dict['square_grid_spacing'],
                    upper_bound_dict['square_grid_amp'],
                    upper_bound_dict['square_grid_sd'],
                    upper_bound_dict['bg_slope']
                    ]

    bounds = (lower_bounds, upper_bounds)

    model_with_info.initial_params = (
        initial_params
        )
    model_with_info.param_bounds = (
        bounds
        )
    model_with_info.vector_input_model = (
        poly.tri_prism_on_grid_1_length_2disobg_substruct_rpd_vectorargs
        )

    return model_with_info


def set_up_tri_prism_on_grid_1_length_substruct_with_fit_info():
    """Set up the RPD model with fitting settings,
    for a rotationally symmetric model with spread due to repeated
    localisations and spread to unresolvable substructure.
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
    model_with_info = (
        ModelWithFitSettings(
            model_rpd=poly.tri_prism_on_grid_1_length_substructure_rpd
            )
        )

    # Add fitting parameters to ModelWithFitSettings object
    (lower_bound_dict,
     upper_bound_dict,
     initial_params_dict) = create_default_fitting_params_dicts()

    # Can optionally modify these dictionaries here:
    
    initial_params = [initial_params_dict['side_1'],
                      initial_params_dict['loc_prec_amp'],
                      initial_params_dict['loc_prec_sd'],
                      initial_params_dict['vertices_amp'],
                      initial_params_dict['vertices_sd'],
                      initial_params_dict['substruct_amp'],
                      initial_params_dict['substruct_sd'],
                      initial_params_dict['square_grid_spacing'],
                      initial_params_dict['square_grid_amp'],
                      initial_params_dict['square_grid_sd'],
                      ]

    lower_bounds = [lower_bound_dict['side_1'],
                    lower_bound_dict['loc_prec_amp'],
                    lower_bound_dict['loc_prec_sd'],
                    lower_bound_dict['vertices_amp'],
                    lower_bound_dict['vertices_sd'],
                    lower_bound_dict['substruct_amp'],
                    lower_bound_dict['substruct_sd'],
                    lower_bound_dict['square_grid_spacing'],
                    lower_bound_dict['square_grid_amp'],
                    lower_bound_dict['square_grid_sd'],
                    ]


    upper_bounds = [upper_bound_dict['side_1'],
                    upper_bound_dict['loc_prec_amp'],
                    upper_bound_dict['loc_prec_sd'],
                    upper_bound_dict['vertices_amp'],
                    upper_bound_dict['vertices_sd'],
                    upper_bound_dict['substruct_amp'],
                    upper_bound_dict['substruct_sd'],
                    upper_bound_dict['square_grid_spacing'],
                    upper_bound_dict['square_grid_amp'],
                    upper_bound_dict['square_grid_sd'],
                    ]

    bounds = (lower_bounds, upper_bounds)

    model_with_info.initial_params = (
        initial_params
        )
    model_with_info.param_bounds = (
        bounds
        )
    #model_with_info.vector_input_model = (
    #    poly.tri_prism_on_grid_1_length_2disobg_substruct_rpd_vectorargs
    #    )

    return model_with_info


def set_up_tri_prism_on_grid_1_length_3disobg_substruct_with_fit_info():
    """Set up the RPD model with fitting settings,
    for a rotationally symmetric model with spread due to repeated
    localisations and spread to unresolvable substructure.
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
    model_with_info = (
        ModelWithFitSettings(
            model_rpd=poly.tri_prism_on_grid_1_length_3disobg_substruct_rpd
            )
        )

    # Add fitting parameters to ModelWithFitSettings object
    (lower_bound_dict,
     upper_bound_dict,
     initial_params_dict) = create_default_fitting_params_dicts()

    # Can optionally modify these dictionaries here:
    
    initial_params = [initial_params_dict['side_1'],
                      initial_params_dict['loc_prec_amp'],
                      initial_params_dict['loc_prec_sd'],
                      initial_params_dict['vertices_amp'],
                      initial_params_dict['vertices_sd'],
                      initial_params_dict['substruct_amp'],
                      initial_params_dict['substruct_sd'],
                      initial_params_dict['square_grid_spacing'],
                      initial_params_dict['square_grid_amp'],
                      initial_params_dict['square_grid_sd'],
                      initial_params_dict['bg_slope'] # Use this as the bg_scale
                      ]

    lower_bounds = [lower_bound_dict['side_1'],
                    lower_bound_dict['loc_prec_amp'],
                    lower_bound_dict['loc_prec_sd'],
                    lower_bound_dict['vertices_amp'],
                    lower_bound_dict['vertices_sd'],
                    lower_bound_dict['substruct_amp'],
                    lower_bound_dict['substruct_sd'],
                    #lower_bound_dict['square_grid_spacing'],
                    200.,
                    lower_bound_dict['square_grid_amp'],
                    lower_bound_dict['square_grid_sd'],
                    lower_bound_dict['bg_slope'] # Use this as the bg_scale
                    ]


    upper_bounds = [upper_bound_dict['side_1'],
                    upper_bound_dict['loc_prec_amp'],
                    upper_bound_dict['loc_prec_sd'],
                    upper_bound_dict['vertices_amp'],
                    upper_bound_dict['vertices_sd'],
                    upper_bound_dict['substruct_amp'],
                    upper_bound_dict['substruct_sd'],
                    upper_bound_dict['square_grid_spacing'],
                    upper_bound_dict['square_grid_amp'],
                    upper_bound_dict['square_grid_sd'],
                    upper_bound_dict['bg_slope'] # Use this as the bg_scale
                    ]

    bounds = (lower_bounds, upper_bounds)

    model_with_info.initial_params = (
        initial_params
        )
    model_with_info.param_bounds = (
        bounds
        )
    #model_with_info.vector_input_model = (
    #    poly.tri_prism_on_grid_1_length_2disobg_substruct_rpd_vectorargs
    #    )

    return model_with_info


def set_up_tri_prism_1_length_3disobg_substruct_with_fit_info():
    """Set up the RPD model with fitting settings,
    for a rotationally symmetric model with spread due to repeated
    localisations and spread to unresolvable substructure.
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
    model_with_info = (
        ModelWithFitSettings(
            model_rpd=poly.tri_prism_1_length_3disobg_substruct_rpd
            )
        )

    # Add fitting parameters to ModelWithFitSettings object
    (lower_bound_dict,
     upper_bound_dict,
     initial_params_dict) = create_default_fitting_params_dicts()

    # Can optionally modify these dictionaries here:
    
    initial_params = [initial_params_dict['side_1'],
                      initial_params_dict['loc_prec_amp'],
                      initial_params_dict['loc_prec_sd'],
                      initial_params_dict['vertices_amp'],
                      initial_params_dict['vertices_sd'],
                      initial_params_dict['substruct_amp'],
                      initial_params_dict['substruct_sd'],
                      initial_params_dict['bg_slope'] # Use this as the bg_scale
                      ]

    lower_bounds = [lower_bound_dict['side_1'],
                    lower_bound_dict['loc_prec_amp'],
                    lower_bound_dict['loc_prec_sd'],
                    lower_bound_dict['vertices_amp'],
                    lower_bound_dict['vertices_sd'],
                    lower_bound_dict['substruct_amp'],
                    lower_bound_dict['substruct_sd'],
                    lower_bound_dict['bg_slope'] # Use this as the bg_scale
                    ]


    upper_bounds = [upper_bound_dict['side_1'],
                    upper_bound_dict['loc_prec_amp'],
                    upper_bound_dict['loc_prec_sd'],
                    upper_bound_dict['vertices_amp'],
                    upper_bound_dict['vertices_sd'],
                    upper_bound_dict['substruct_amp'],
                    upper_bound_dict['substruct_sd'],
                    upper_bound_dict['bg_slope'] # Use this as the bg_scale
                    ]

    bounds = (lower_bounds, upper_bounds)

    model_with_info.initial_params = (
        initial_params
        )
    model_with_info.param_bounds = (
        bounds
        )
    #model_with_info.vector_input_model = (
    #    poly.tri_prism_on_grid_1_length_2disobg_substruct_rpd_vectorargs
    #    )

    return model_with_info


def set_up_model_tri_prism_on_grid_2disobg_substructure_with_fit_info():
    """Set up the RPD model with fitting settings,
    for a rotationally symmetric model with spread due to repeated
    localisations and spread to unresolvable substructure.
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
    model_with_info = (
        ModelWithFitSettings(
            model_rpd=poly.tri_prism_on_grid_2disobg_substructure_rpd
            )
        )

    # Add fitting parameters to ModelWithFitSettings object
    (lower_bound_dict,
     upper_bound_dict,
     initial_params_dict) = create_default_fitting_params_dicts()

    # Can optionally modify these dictionaries here:
    
    initial_params = [initial_params_dict['side_1'],
                      initial_params_dict['side_2'],
                      initial_params_dict['loc_prec_amp'],
                      initial_params_dict['loc_prec_sd'],
                      initial_params_dict['vertices_amp'],
                      initial_params_dict['vertices_sd'],
                      initial_params_dict['substruct_amp'],
                      initial_params_dict['substruct_sd'],
                      initial_params_dict['square_grid_spacing'],
                      initial_params_dict['square_grid_amp'],
                      initial_params_dict['square_grid_sd'],
                      initial_params_dict['bg_slope']
                      ]

    lower_bounds = [lower_bound_dict['side_1'],
                    lower_bound_dict['side_2'],
                    lower_bound_dict['loc_prec_amp'],
                    lower_bound_dict['loc_prec_sd'],
                    lower_bound_dict['vertices_amp'],
                    lower_bound_dict['vertices_sd'],
                    lower_bound_dict['substruct_amp'],
                    lower_bound_dict['substruct_sd'],
                    lower_bound_dict['square_grid_spacing'],
                    lower_bound_dict['square_grid_amp'],
                    lower_bound_dict['square_grid_sd'],
                    lower_bound_dict['bg_slope']
                    ]


    upper_bounds = [upper_bound_dict['side_1'],
                    upper_bound_dict['side_2'],
                    upper_bound_dict['loc_prec_amp'],
                    upper_bound_dict['loc_prec_sd'],
                    upper_bound_dict['vertices_amp'],
                    upper_bound_dict['vertices_sd'],
                    upper_bound_dict['substruct_amp'],
                    upper_bound_dict['substruct_sd'],
                    upper_bound_dict['square_grid_spacing'],
                    upper_bound_dict['square_grid_amp'],
                    upper_bound_dict['square_grid_sd'],
                    upper_bound_dict['bg_slope']
                    ]

    bounds = (lower_bounds, upper_bounds)

    model_with_info.initial_params = (
        initial_params
        )
    model_with_info.param_bounds = (
        bounds
        )
    model_with_info.vector_input_model = (
        poly.tri_prism_on_grid_2disobg_substructure_rpd_vectorargs
        )

    return model_with_info


def set_up_model_cuboid_on_grid_2disobg_substructure_with_fit_info():
    """Set up the RPD model with fitting settings,
    for a rotationally symmetric model with spread due to repeated
    localisations and spread to unresolvable substructure.
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
    model_with_info = (
        ModelWithFitSettings(
            model_rpd=poly.cuboid_on_grid_2disobg_substructure_rpd
            )
        )

    # Add fitting parameters to ModelWithFitSettings object
    (lower_bound_dict,
     upper_bound_dict,
     initial_params_dict) = create_default_fitting_params_dicts()

    # Can optionally modify these dictionaries here:
    
    initial_params = [initial_params_dict['side_1'],
                      initial_params_dict['side_2'],
                      initial_params_dict['side_3'],
                      initial_params_dict['loc_prec_amp'],
                      initial_params_dict['loc_prec_sd'],
                      initial_params_dict['vertices_amp'],
                      initial_params_dict['vertices_sd'],
                      initial_params_dict['substruct_amp'],
                      initial_params_dict['substruct_sd'],
                      initial_params_dict['square_grid_spacing'],
                      initial_params_dict['square_grid_amp'],
                      initial_params_dict['square_grid_sd'],
                      initial_params_dict['bg_slope']
                      ]

    lower_bounds = [lower_bound_dict['side_1'],
                    lower_bound_dict['side_2'],
                    lower_bound_dict['side_3'],
                    lower_bound_dict['loc_prec_amp'],
                    lower_bound_dict['loc_prec_sd'],
                    lower_bound_dict['vertices_amp'],
                    lower_bound_dict['vertices_sd'],
                    lower_bound_dict['substruct_amp'],
                    lower_bound_dict['substruct_sd'],
                    lower_bound_dict['square_grid_spacing'],
                    lower_bound_dict['square_grid_amp'],
                    lower_bound_dict['square_grid_sd'],
                    lower_bound_dict['bg_slope']
                    ]


    upper_bounds = [upper_bound_dict['side_1'],
                    upper_bound_dict['side_2'],
                    upper_bound_dict['side_3'],
                    upper_bound_dict['loc_prec_amp'],
                    upper_bound_dict['loc_prec_sd'],
                    upper_bound_dict['vertices_amp'],
                    upper_bound_dict['vertices_sd'],
                    upper_bound_dict['substruct_amp'],
                    upper_bound_dict['substruct_sd'],
                    upper_bound_dict['square_grid_spacing'],
                    upper_bound_dict['square_grid_amp'],
                    upper_bound_dict['square_grid_sd'],
                    upper_bound_dict['bg_slope']
                    ]

    bounds = (lower_bounds, upper_bounds)

    model_with_info.initial_params = (
        initial_params
        )
    model_with_info.param_bounds = (
        bounds
        )
    #    model_with_info.vector_input_model = (
    #        poly.tri_prism_on_grid_2disobg_substructure_rpd_vectorargs
    #        )

    return model_with_info


def set_up_model_tri_pyramid_on_grid_2disobg_substructure_with_fit_info():
    """Set up the RPD model with fitting settings,
    for a rotationally symmetric model with spread due to repeated
    localisations and spread to unresolvable substructure.
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
    model_with_info = (
        ModelWithFitSettings(
            model_rpd=poly.tri_pyramid_on_grid_2disobg_substructure_rpd
            )
        )

    # Add fitting parameters to ModelWithFitSettings object
    (lower_bound_dict,
     upper_bound_dict,
     initial_params_dict) = create_default_fitting_params_dicts()

    # Can optionally modify these dictionaries here:
    
    initial_params = [initial_params_dict['side_1'],
                      initial_params_dict['side_2'],
                      initial_params_dict['loc_prec_amp'],
                      initial_params_dict['loc_prec_sd'],
                      initial_params_dict['vertices_amp'],
                      initial_params_dict['vertices_sd'],
                      initial_params_dict['substruct_amp'],
                      initial_params_dict['substruct_sd'],
                      initial_params_dict['square_grid_spacing'],
                      initial_params_dict['square_grid_amp'],
                      initial_params_dict['square_grid_sd'],
                      initial_params_dict['bg_slope']
                      ]

    lower_bounds = [lower_bound_dict['side_1'],
                    lower_bound_dict['side_2'],
                    lower_bound_dict['loc_prec_amp'],
                    lower_bound_dict['loc_prec_sd'],
                    lower_bound_dict['vertices_amp'],
                    lower_bound_dict['vertices_sd'],
                    lower_bound_dict['substruct_amp'],
                    lower_bound_dict['substruct_sd'],
                    lower_bound_dict['square_grid_spacing'],
                    lower_bound_dict['square_grid_amp'],
                    lower_bound_dict['square_grid_sd'],
                    lower_bound_dict['bg_slope']
                    ]


    upper_bounds = [upper_bound_dict['side_1'],
                    upper_bound_dict['side_2'],
                    upper_bound_dict['loc_prec_amp'],
                    upper_bound_dict['loc_prec_sd'],
                    upper_bound_dict['vertices_amp'],
                    upper_bound_dict['vertices_sd'],
                    upper_bound_dict['substruct_amp'],
                    upper_bound_dict['substruct_sd'],
                    upper_bound_dict['square_grid_spacing'],
                    upper_bound_dict['square_grid_amp'],
                    upper_bound_dict['square_grid_sd'],
                    upper_bound_dict['bg_slope']
                    ]

    bounds = (lower_bounds, upper_bounds)

    model_with_info.initial_params = (
        initial_params
        )
    model_with_info.param_bounds = (
        bounds
        )
    #    model_with_info.vector_input_model = (
    #        poly.tri_prism_on_grid_2disobg_substructure_rpd_vectorargs
    #        )

    return model_with_info

    
def fitmodel_to_hist(
        distancehist,
        model=poly.tri_prism_on_grid_1_length_2disobg_substruct_rpd,
        fitlength=400.):
    """Fit model RPD to distance histogram. Designed to allow user editting of
    parameter guesses and bounds for models of RPDs for simple polyhedra,
    as in polyhedramodelling.py.

    Args:
        distancehist (numpy array):
            An array of Euclidean distances between localisations.
        model:
            The model RPD to be fitted.
        fitlength:
            The maximum distance between localisations used for the fit.

    Returns:
        popt:
        perr
    """
    params_optimised, params_covar = curve_fit(
        model, np.arange(fitlength) + 0.5, distancehist,
        p0=(100., 100., #100.,  # a, b, c
            10., 10.,  # locamp, locprec
            10., 10.,  # structamp, spread
            10., 10.,  # substructamp, substructspread
            200., 100., 10.,  # gridspace, gridamp, gridspread
            0.001),  # bgslope

        bounds=(
            [0., 0., #0., # a, b, c
             0., 0.,  # locamp, locprec
             0., 0.,  # structamp, spread
             0., 0.,  # substructamp, substructspread
             0., 0., 0.,  # gridspace, gridamp, gridspread
             0.],  # bgslope

            [200., 200., #200., # a, b, c
             100., 100.,  # locamp, locprec
             100., 100.,  # structamp, spread
             100., 100.,  # substructamp, substructspread
             400., 1000., 100.,  # gridspace, gridamp, gridspread
             0.1])  # bgslope
        )
    # plt.plot(np.arange(fitlength) + 0.5,
    #          model(np.arange(fitlength) + 0.5, *params_optimised))
    params_1sd_error = np.sqrt(np.diag(params_covar))
    params_table = np.column_stack((params_optimised, params_1sd_error))
    print(params_table)

    # No. free parameters, including var. of residuals for least squares fit.
    k = float(len(params_optimised) + 1)

    # Calculate AICc
    ssr = np.sum((model(np.arange(fitlength) + 0.5, *params_optimised) -
                  distancehist) ** 2)
    aic = fitlength * np.log(ssr / fitlength) + 2 * k
    aiccorr = aic + 2 * k * (k + 1) / (fitlength - k - 1)

    print('SSR =', ssr)
    print('AIC =', aic)
    print('AICcorr =', aiccorr)

    return params_optimised, params_covar, params_1sd_error


def plot_xyz_distance_histogram(distances, fitlength, color='gray'):
    """Plot histogram of experimental distances, with 1 nm bins.
    Scales counts, so that mean = 1, to suit scipy.optimize.curve_fit

    Args:
        distances (numpy array-like):
            Set of distances (nm) to plot.
        fitlength (float):
            The distance upto which the histogram will be calculated.
        color (string):
            Colour for the matplotlib histogram.
    Returns:
        hist_values (numpy array):
            Histogram bin values.
        bin_edges (numpy array):
            Histogram bin edge positions. 
    """
    # Histogram figure with 1-nm bins
    histfig = plt.figure()
    histaxes = histfig.add_subplot(111)
    hist_values, bin_edges = histaxes.hist(
        distances,
        bins=np.arange(fitlength + 1),
        weights=np.repeat(float(fitlength) / len(distances),
                          len(distances)
                          ),
        color=color, alpha=0.5
        )[0:2] # 2 not required
    histaxes.set_xlim([0, fitlength])
    histaxes.set_title('Histogram')
    histaxes.set_xlabel(r'$\Delta$XYZ (nm)')
    histaxes.set_ylabel('Counts (scaled: mean = 1)')

    return hist_values, bin_edges


def plot_distance_hist_and_fit(distances,
                               fitlength,
                               params_optimised,
                               params_covar,
                               model_with_info,
                               plot_95ci=False,
                               color='xkcd:red'):
    # Use only distances within fitlength
    distances = distances[distances <= fitlength]

    fig = plt.figure()
    axes = plt.subplot(111)

    histogram_output = axes.hist(distances,
                                 bins=np.arange(fitlength + 1),
                                 weights=np.repeat(float(fitlength)
                                                   / len(distances),
                                                   len(distances)
                                                   ),
                                 color='gray', alpha=0.5
                                 )
    bin_edges = histogram_output[1]
    bin_centres = (bin_edges[0:(len(bin_edges) - 1)] + bin_edges[1:]) / 2

    axes.plot(distances,
              model_with_info.model_rpd(distances, *params_optimised),
              color=color,
              lw=0.75
              )
    axes.set_xlim([0, fitlength])
    axes.set_xlabel(r'$\Delta$XYZ (nm)')
    axes.set_ylabel('Counts (scaled: mean = 1)')
    axes.set_title('Model: ' +model_with_info.model_rpd.__name__)

    # Get 1 SD uncertainty on model result from uncertainty on parameters
    # and plot 95% CI.
    if plot_95ci is True:
        stdev = stdev_of_model(bin_centres,
                               params_optimised,
                               params_covar,
                               model_with_info.vector_input_model
                               )

        axes.fill_between(bin_centres,
                        model_with_info.model_rpd(bin_centres,
                                                    *params_optimised)
                        - stdev * 1.96,
                        model_with_info.model_rpd(bin_centres,
                                                    *params_optimised)
                        + stdev * 1.96,
                        facecolor=color,
                        alpha=0.25
                        )

    return fig, axes


def plot_model_components_tri_prism(fitlength,
                                    side_length,
                                    locamp, locprec,
                                    structamp, spread,
                                    substructamp, substructspread,
                                    gridspace, gridamp, gridspread,
                                    bgslope):
    """
    fitlength: distance upto to which the model fit was performed
    side_length: side length of the triangular prism (all sides equal)
    locamp: Amplitude of sinlge molecule localisation precision component
    locprec: Average single molecule localisation precision
    structamp: Amplitude of components reflecting the structural features
        of the complex.
    spread: Spread owing to unresolvable complexity or inhomogeneity between
        complexes.
    substructamp: Amplitude of component reflecting unresolvable substructure
        at one vertex.
    substructspread: Spread in localisations at one vertex, as a result of
        unresolvable substructure there.
    gridspace: The spacing of a square grid the complexes are found on.
    gridamp: Amplitude of components reflecting the nieghbouring complexes at
        nearby grid points.
    gridspread: Spread owing to different orientation at different grid points.
    bgslope: Approximate background to 2D (relatively flat); this is the
        linear slope.
    """
    distance_values = np.arange(0, fitlength + 1, 1)

    fig = plt.figure()
    axes = plt.subplot(111)
    axes.set_xlim([0, fitlength])
    axes.set_xlabel(r'$\Delta$XYZ (nm)')
    axes.set_ylabel('Counts (scaled: mean = 1)')
    axes.set_title('Model: Triangular pyramid, equal side lengths')

    # Background term
    bg_term = distance_values * bgslope 
    plt.plot(distance_values, bg_term)

    # Repeated localisations term
    rep_locs_term = (locamp
                     * models.pairwise_correlation_3d(distance_values,
                                                      0.,
                                                      np.sqrt(2) * locprec
                                                      )
                     )
    plt.plot(distance_values, rep_locs_term)

    # Plot substructure term
    substructure_term = (substructamp
                         * models.pairwise_correlation_3d(distance_values,
                                                          0.,
                                                          np.sqrt(2) * locprec
                                                          )
                         )
    plt.plot(distance_values, substructure_term)

    # Triangular prism peaks:
    # Set up triangular prism
    verts = poly.tri_prism_vertices(side_length, side_length)
    relpos = poly.get_1d_relpos_no_filter(verts)
    # Get unique distances
    xyz_distances = np.sqrt(relpos[:, 0] ** 2 + relpos[:, 1] ** 2 + relpos[:, 2] ** 2)
    xyz_distances = np.unique(xyz_distances)
    # Two peaks:
    # 3 x shorter distance x 6 vertices
    tri_prism_peak_1 = (18 * structamp
                        * models.pairwise_correlation_3d(distance_values,
                                                         xyz_distances[0],
                                                         spread
                                                         )
                        )
    plt.plot(distance_values, tri_prism_peak_1)

    # 2 x longer distance x 6 vertices
    tri_prism_peak_2 = (12 * structamp
                        * models.pairwise_correlation_3d(distance_values,
                                                         xyz_distances[1],
                                                         spread
                                                         )
                        )
    plt.plot(distance_values, tri_prism_peak_2)

    # Plot square grid componenet
    square_grid_component = (
        gridamp * models.pairwise_correlation_2d(distance_values,
                                                 gridspace,
                                                 gridspread
                                                 )
        + gridamp * models.pairwise_correlation_2d(distance_values,
                                                   gridspace * np.sqrt(2),
                                                   gridspread
                                                   )
        )
    plt.plot(distance_values, square_grid_component)

    # Plot total model
    total_model = (bg_term
                   + rep_locs_term
                   + substructure_term
                   + tri_prism_peak_1
                   + tri_prism_peak_2
                   + square_grid_component
                   )
    plt.plot(distance_values, total_model, color='xkcd:red')


def dna_origami_fig_plot(
        relpos,
        model_with_info=set_up_tri_prism_on_grid_1_length_2disobg_substruct_with_fit_info(),
        fitlength=250):
    """Creates a histogram of the DNA origami data and plots a fitted model,
    with a 95% confidence interval.

    Args:
        relpos (numpy array):
            Array of 3D relative positions between localisations with
            shape (N, 3), where N is the number of relative positions.
        model:
            The fitted model
        fitlength:
            The maximum distance between localisations used for the fit.

    Returns:
        Nothing
    """
    # Set up experimental data histogram and plot
    #    plt.figure()
    #    axes = plt.subplot(111)
    #    distance_hist = make_xyz_histogram_nm(relpos,
    #                                         fitlength=fitlength,
    #                                         axes=axes)[0]

    xyz_distances = np.sqrt(relpos[:, 0] ** 2
                            + relpos[:, 1] ** 2
                            + relpos[:, 2] ** 2)
    xyz_distances = xyz_distances[xyz_distances < fitlength]
    bin_edges = np.arange(fitlength + 1)

    distance_hist, bin_edges = np.histogram(
        xyz_distances,
        weights=np.repeat(float(fitlength) / len(xyz_distances),
                          len(xyz_distances)
                          ),
        bins=bin_edges
        )
    width = 1.
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig_histogram = plt.figure(num=None,
                               figsize=(10, 8),
                               dpi=100,
                               facecolor='w',
                               edgecolor='k')

    axes = fig_histogram.add_subplot(111)
    axes.bar(bin_centres,
             distance_hist,
             align='center', width=width,
             color='lightblue', alpha=0.5
             )
    axes.set_xlabel('Separation (nm)')
    axes.set_ylabel('Counts (normalised)')

    # Fit model to experimental distance histogram
    (params_optimised,
     params_covar,
     params_1sd_error) = models.fit_model_to_experiment(
                             distance_hist,
                             model_with_info.model_rpd,
                             model_with_info.initial_params,
                             model_with_info.param_bounds,
                             fitlength=fitlength
                             )

    # Plot fitted model
    # x_values = np.arange(fitlength + 0.5)
    axes.plot(bin_centres,
              model_with_info.model_rpd(bin_centres, *params_optimised),
              color='xkcd:red', lw=0.5
              )

    # Plot 95% confidence interval
    stdev = stdev_of_model(bin_centres,
                           params_optimised,
                           params_covar,
                           model_with_info.vector_input_model
                           )
    axes.fill_between(bin_centres,
                      model_with_info.model_rpd(bin_centres, *params_optimised)
                      - stdev * 1.96,
                      model_with_info.model_rpd(bin_centres, *params_optimised)
                      + stdev * 1.96,
                      color='xkcd:red', alpha=0.25
                      )


def model_akaike_weights():
    """Calculate and tabulate Akaike weights, based on results recorded
    for model fits.
    Args:
        Nothing
    Returns:
        Table of models, AICc and Akaike weights.
    """
    models = ['tri_prism_on_grid_1_length_2disobg_substruct',
              'tri_prism_on_grid_2disobg_substructure_rpd',
              'tri_pyramid_on_grid_2disobg_substructure_rpd',
              'cuboid_on_grid_2disobg_substructure_rpd'
              ]
    aiccs = [-1748.52577,
             -1746.32688,
             -1477.75366,
             -1438.53290
             ]
    weights = stats.akaike_weights(aiccs)

    print('Model\t\tAICc\t\tAkaike weight')
    for i in range(len(models)):
        print('{0}\t\t{1:.2f} \t{2:.2}'.format(models[i],
                                               aiccs[i],
                                               weights[i]
                                               )
              )


def plot_three_models(relpos_xyz, fitlength):
    """Creates a histogram of distances in the DNA origami data and plots
    three fitted models

    Args:
        relpos (numpy array):
            Array of 3D relative positions between localisations with
            shape (N, 3), where N is the number of relative positions.
        fitlength:
            The maximum distance between localisations used for the fit.

    Returns:
        Nothing
    """
    #    relpos_xyz = np.loadtxt('C:/Temp/DNA-PAINTdata/'
    #                            + 'dataset2_DNA-PAINT_xyz_1in4_3555locs_PERPL-'
    #                            + 'relposns_400.0filter.csv', delimiter=',')
    xyz_distances = np.sqrt(relpos_xyz[:, 0] ** 2
                            + relpos_xyz[:, 1] ** 2
                            + relpos_xyz[:, 2] ** 2)
    xyz_distances = xyz_distances[xyz_distances < fitlength]
    bin_edges = np.arange(fitlength + 1)

    distance_hist, bin_edges = np.histogram(
        xyz_distances,
        weights=np.repeat(float(fitlength) / len(xyz_distances),
                          len(xyz_distances)
                          ),
        bins=bin_edges
        )
    width = 1.
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig_histogram = plt.figure(num=None,
                               figsize=(10, 8),
                               dpi=100,
                               facecolor='w',
                               edgecolor='k')

    axes = fig_histogram.add_subplot(111)
    axes.bar(bin_centres,
             distance_hist,
             align='center', width=width,
             color='gray', alpha=0.5
             )
    axes.set_xlabel('Separation (nm)')
    axes.set_ylabel('Counts (normalised)')
    
    for model_with_info in [
        set_up_tri_prism_on_grid_1_length_2disobg_substruct_with_fit_info(),
        set_up_model_cuboid_on_grid_2disobg_substructure_with_fit_info(),
        set_up_model_tri_pyramid_on_grid_2disobg_substructure_with_fit_info(),
        ]: 
        (params_optimised,
         params_covar,
         params_1sd_error) = models.fit_model_to_experiment(
                                 distance_hist,
                                 model_with_info.model_rpd,
                                 model_with_info.initial_params,
                                 model_with_info.param_bounds,
                                 fitlength=fitlength
                                 )
        print('Model: ' + model_with_info.model_rpd.__name__)
        print('Parameters / 1SD:')
        print(np.column_stack((params_optimised, params_1sd_error)))
        aicc = stats.aic_from_least_sqr_fit(distance_hist,
                                            model_with_info.model_rpd,
                                            params_optimised,
                                            fitlength=fitlength)[1]
        print('AICc: ' + repr(aicc))
        print('')

        axes.plot(bin_centres,
                  model_with_info.model_rpd(bin_centres, *params_optimised)
                  #lw=0.5
                  )
    