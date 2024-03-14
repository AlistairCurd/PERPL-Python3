"""
centriole_analysis.py

Created on Tue Sep 25 12:00:00 2019

Functions for analysing segmented top-view centriolar protein localisations.

Alistair Curd
University of Leeds
16 September 2019

---
Copyright 2019 Peckham Lab

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at
http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import time
from scipy.ndimage.filters import gaussian_filter
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import perpl.modelling.modelling_general as models
from perpl.modelling.modelling_general import ModelWithFitSettings
from relative_positions import getdistances
from perpl.modelling.background_models import pair_correlation_disk


def get_input_data(infile='S:/Peckham/Bioimaging2/Alistair/Centriole-EPFL'
                          '/forAlistair_310718/Cep152_A647_Top.mat'
                   ):
    """Import and extract a list of dSTORM data on centrioles
    from Christian Sieben's MATLAB file.

    Args:
        infile (string):
            The path to the data to be extracted.

    Returns:
        simul (list of numpy arrays):
            A list of arrays of dSTORM localisations, one element per
            centriole.
            For each centriole, columns [0] and [1] of the array
            are (x, y) for each localisation; column [3] is estimated
            localisation precision in nm.
    """
    # Import centriole localisations
    epfl_matlab_centrioles = loadmat(infile)
    epfl_matlab_centrioles = epfl_matlab_centrioles['Predicted_Top'][:, 0]

    return epfl_matlab_centrioles


def reconstruct_centrioles(centriole_localisation_data,
                           index=None,
                           plot=True,
                           binsize_nm=5.,
                           smoothing_nm=5.,
                           color='inferno'):
    """Produces centriole reconstructions from localisation data and plots
    if desired.

    Args:
        centriole_localisation_data (list of numpy arrays):
            A list of arrays of dSTORM localisations, one element per
            centriole.
            For each centriole, columns [0] and [1] of the array
            are (x, y) for each localisation.
        index (list):
            If desired, the inidices of selected centrioles in the list.
            If not specified, a reconstruction is produced for all centrioles.
        plot (Boolean):
            Choose whether plots will be produced of each centriole.
        binsize_nm (float):
            The pixel size in the reconstructions, in nm.
        smoothing_nm (float):
            The kernal for Gaussian smoothing of the histogram, in nm.

    Returns:
        indices (list of integers):
            Indices of the chosen centrioles for reconstruction.
        list_of_reconstructions (list of 2D numpy arrays):
            Reconstructions of the centrioles, one centriole per list
            element.
    """
    # Choose all centrioles if none weere specified
    if index is None:
        index = range(len(centriole_localisation_data))

    # Make and store reconstructions in list
    indices = []
    list_of_reconstructions = []

    for i in index:
        # Get x and y data for the localisations
        locs_x = centriole_localisation_data[i][:, 0]
        locs_y = centriole_localisation_data[i][:, 1]

        # Make reconstruction.
        # Extend plot by 4 * Gaussian smoothing kernel beyond extreme
        # coordinates.
        bin_edges_x = np.arange(np.min(locs_x) - smoothing_nm * 4.,
                                np.max(locs_x) - smoothing_nm * 4.,
                                binsize_nm)
        bin_edges_y = np.arange(np.min(locs_y) - smoothing_nm * 4.,
                                np.max(locs_y) - smoothing_nm * 4.,
                                binsize_nm)
        reconstruction = np.histogram2d(locs_x, locs_y,
                                        bins=(bin_edges_x, bin_edges_y))[0]
        # Do smoothing by one pixel
        reconstruction = reconstruction.astype(float)
        smoothing_in_pixels = smoothing_nm / binsize_nm
        reconstruction = gaussian_filter(reconstruction,
                                         sigma=smoothing_in_pixels)

        # Append centriole number and reconstruction to list
        indices.append(i)
        list_of_reconstructions.append(reconstruction)

        # Plot if desired
        if plot is True:
            plt.matshow(reconstruction, cmap=color)

    return indices, list_of_reconstructions


# getdistances took 139 s to run centriole [0] (filterdist=1000.)
def get_xy_distances(localisations_per_centriole,
                     loc_precision_filter=10.,
                     filter_distance=1000.,
                     ):
    """Get the XY between localisations in each centriole,
    filtered by localisation precision.

    Args:
        centriole_localisations (list of numpy arrays):
            A list of arrays of dSTORM localisations, one element per
            centriole.
            For each centriole, columns [0] and [1] of the array
            are (x, y) for each localisation; column [3] is estimated
            localisation precision in nm.
        loc_precision_filter (float):
            Upper limit on estimated localisation precision. Localisations
            with an estimate above this value will not be included.
        filter_distance (float):
            Required by relative_positions. Set to 1000 by default to
            comfortably include all distances in each centriole.

    Returns:
        distances_xy_per_centriole (list of numpy arrays):
            List containing the XY-distances found within each
            centriole, one centriole per list item.
    """
    distances_xy_per_centriole = []
    start_time = time.time()
    for i, centriole in enumerate(localisations_per_centriole):
        # Filter imprecise localisations, and use only X and Y
        filtered_locs = centriole[centriole[:, 3]
                                  < loc_precision_filter][:, 0:2]
        print('Analysing '
              + repr(len(filtered_locs))
              + ' localisations in centriole '
              + repr(i + 1) + '.')
        # Get relative positions, using a filter distance large enough
        # to include the whole centriole
        rel_pos = getdistances(filtered_locs, filterdist=filter_distance)
        time_elapsed = time.time() - start_time
        print('Found '
              + repr(len(rel_pos))
              + ' relative positions for centriole '
              + repr(i + 1) + ' of ' + repr(len(localisations_per_centriole))
              + ',\n'
              'after ' + repr(time_elapsed) + ' s.',
              flush=True)
        # Get distances across XY and append to the per-centriole list
        distances_xy = np.sqrt(rel_pos[:, 0] ** 2 + rel_pos[:, 1] ** 2)
        distances_xy_per_centriole.append(distances_xy)
    return distances_xy_per_centriole


def make_distance_histogram(distance_values,
                            max_distance,
                            normalise=True):
    """Calculate distance histogram bin values, in bins of 1 nm,
    upto a maximum distance. Bin values will be normalised so that
    their mean is 1.

    Args:
        distance_values (numpy array):
            Distances between localisations in a centriole
        max_distance (float):
            Upper limit on distnaces included in the histogram.
        normalise (bool):
            Decide whether to normalise the histograms so that the mean
            bin value is 1 (good for scipy's curve_fit on individual
            centriole data).

    Returns:
        hist (numpy array):
            Distance histogram bin values, normalised so that the mean is 1.
        bin_edges (numpy array):
            Distance histogram bin edge locations.
    """
    bin_edges = np.arange(max_distance + 1)

    if normalise is True:
        hist, bin_edges = np.histogram(
            distance_values,
            bins=bin_edges,
            weights=np.repeat(float(max_distance) / len(distance_values),
                              len(distance_values)
                              )
            )

    else:
        hist, bin_edges = np.histogram(distance_values,
                                       bins=bin_edges
                                       )

    return hist, bin_edges


def make_distance_histograms_per_centriole(distances_per_centriole,
                                           max_distance,
                                           normalise=True):
    """Make a distance histogram for each centriole.

    Args:
        distances per_centriole (list of numpy arrays):
            List containing the XY-distances found within each
            centriole, one centriole per list item.
        max_distance (float):
            Upper limit on distaces included in the histograms.
        normalise (bool):
            Decide whether to normalise the histograms so that the mean
            bin value is 1 (good for scipy's curve_fit on individual
            centriole data).

    Returns:
        dist_hist_list (2D numpy array):
            A list of distance histogram bin values; each row contains
            the distance histogram for one centriole.
    """
    dist_hist_list = []
    for distances in distances_per_centriole:
        hist_values, bin_edges = make_distance_histogram(distances,
                                                         max_distance,
                                                         normalise)
        dist_hist_list.append(hist_values)
    return dist_hist_list, bin_edges


def plot_distance_histogram(histogram_values, bin_edges):
    """Plot distance histogram.

    Args:
        histogram_values (numpy array):
            Histogram bin values.
        bin_edges (numpy array):
            Histogram bin edge locations.

    Returns:
        axes:
            Matplotlib Axes instance containing the histogram plot.
    """
    # Set up the plotting area
    fig_histogram = plt.figure(num=None,
                               figsize=(10, 8), dpi=100,
                               facecolor='w', edgecolor='k'
                               )
    axes = fig_histogram.add_subplot(111)

    # Set up and create the histogram plot
    center = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = 1.0
    axes.bar(center, histogram_values, align='center', width=width,
             alpha=0.5, color='lightblue'
             )

    axes.set_xlabel('XY-separation (nm)')
    axes.set_ylabel('Counts (normalised)')

    return fig_histogram, axes


def plot_95_ci(axes,
               x_values,
               model,
               params_optimised,
               params_covar,
               vector_input_model
               ):
    """Plot 95% confidence interval for a fitted model.

    Args:
        axes:
            matplotlib Axes on which to plot the CI.
        x_values (numpy array):
            x-values at which to calculate and plot the CI.
        model (function name):
            The parametric model optimised to fit experimental data.
        params_optimised (numpy array):
            The optimised parameters for the fitted model.
        params_covar (numpy array):
            The covariance matrix for the fitted parameters.
        vector_input_model (function name):
            The version of the parametric model with input required to be a
            vector of the parameter values, for differentiation and error
            propagation with numdifftools.

    Returns:
        stdev (numpy array):
            Standard deviation of the model estimate at each x_value.
    """
    # Get SD of model at the x-values
    stdev = models.stdev_of_model(x_values,
                                  params_optimised,
                                  params_covar,
                                  vector_input_model
                                  )

    # Plot 95% CI
    axes.fill_between(x_values,
                      model(x_values, *params_optimised) - stdev * 1.96,
                      model(x_values, *params_optimised) + stdev * 1.96,
                      color='xkcd:red', alpha=0.25
                      )

    return stdev


def plot_95_ci_internal_bg(axes,
                           x_values,
                           model,
                           params_optimised,
                           params_covar,
                           bg_dia,
                           vector_input_model_below_bg_dia,
                           vector_input_model_above_bg_dia
                           ):
    """Plot 95% confidence interval for a fitted model with internal
    background. This requires splitting into below and above the diameter
    of the background disk

    Args:
        axes:
            matplotlib Axes on which to plot the CI.
        x_values (numpy array):
            x-values at which to calculate and plot the CI.
        model (function name):
            The parametric model optimised to fit experimental data.
        params_optimised (numpy array):
            The optimised parameters for the fitted model.
        params_covar (numpy array):
            The covariance matrix for the fitted parameters.
        bg_dia (float):
            The diameter of the background disk of localisations.
        vector_input_model_below_bg_dia (function name):
            The version of the parametric model for separations less than
            the diameter of the background disk, with input required to be a
            vector of the parameter values, for differentiation and error
            propagation with numdifftools.
        vector_input_model_above_bg_dia (function name):
            The version of the parametric model for separations greater than
            the diameter of the background disk, with input required to be a
            vector of the parameter values, for differentiation and error
            propagation with numdifftools.

    Returns:
        stdev (numpy array):
            Standard deviation of the model estimate at each x_value.
    """
    # Get SD of model at the x-values less than bg_dia
    stdev_lower = models.stdev_of_model(x_values[x_values < bg_dia],
                                        params_optimised,
                                        params_covar,
                                        vector_input_model_below_bg_dia
                                        )

    # Get SD of model at the x-values less than bg_dia
    stdev_higher = models.stdev_of_model(x_values[x_values > bg_dia],
                                         params_optimised[0:-2],
                                         params_covar[0:-2, 0:-2],
                                         vector_input_model_above_bg_dia
                                         )

    stdev = np.append(stdev_lower, stdev_higher)

    # Plot 95% CI
    axes.fill_between(x_values,
                      model(x_values, *params_optimised) - stdev * 1.96,
                      model(x_values, *params_optimised) + stdev * 1.96,
                      color='xkcd:red', alpha=0.25
                      )

    return stdev


def centriole_model_xy_distances_9fold_variable_vertices(
        separation_values,
        diameter,
        vertssd,
        vertamp1, vertamp2, vertamp3, vertamp4,
        replocssd, replocsamp,
        substructsd, substructamp
        ):
    """Parametric model for XY distances between localisations of a centriolar
    protein.

    Args:
        separation_values (numpy array):
            Distances at which density values of the model will be obtained.
        diameter (float):
            Diameter of the circle containing the vertices of the polygon.
        vertssd (float):
            Broadening of the peaks located at
            distances between the vertices.
        vertamp1, vertamp2, vertamp3, vertamp4 (float):
            Amplitudes of the contributions of the different inter-vertex
            distances.
        replocssd (float):
            Spread representing localisation precision for repeated
            localisations of the same fluorescent molecule.
        replocsamp (float):
            Amplitude of the contribution of repeated localisations
            of the same fluorescent molecule.
        substructsd (float):
            Spread of a contribution resulting from unresolvable
            substructure, or mislocalisations resulting from
            a combination of simultaneous nearby emitters.
        substructamp (float):
            Amplitude of the contribution of unresolvable
            substructure, or mislocalisations resulting from
            a combination of simultaneous nearby emitters.

    Returns:
        rpd (numpy array):
           The relative position density given by the model
           at distances r.
    """
    # Set non-fittable symmetry order
    sym_order = 9

    # Prepare amplitudes of the different inter-vertex distances
    # for easy use.
    vertices_contributions = [vertamp1, vertamp2, vertamp3, vertamp4]

    # Calculate the inter-vertex distances
    vertices = models.generate_polygon_points(sym_order, diameter)

    filter_distance = (2 * diameter)
    # Need to get unsorted relative positions, to find unique distances
    # at the start of the output list from getdistance(),
    # so use sort_and_halve=False.
    relative_positions = getdistances(vertices,
                                      filter_distance,
                                      verbose=False,
                                      sort_and_halve=False)
    xy_separations = np.sqrt(relative_positions[:, 0] ** 2
                             + relative_positions[:, 1] ** 2)
    unique_xy_seps = xy_separations[0:np.floor(sym_order / 2).astype(int)]

    # Include the contributions from the inter-vertex distance in the RPD.
    rpd = separation_values * 0.
    for i, distance in enumerate(unique_xy_seps):
        rpd = (rpd
               + vertices_contributions[i]
               * models.pairwise_correlation_2d(separation_values,
                                                distance,
                                                vertssd
                                                )
               )

    # Add pair correlation distribution for repeated localisations.
    rpd = rpd + (replocsamp
                 * models.pairwise_correlation_2d(separation_values,
                                                  0.,
                                                  np.sqrt(2) * replocssd
                                                  )
                 )

    # Add pair correlation distribution for unresolvable substructure/
    # mislocalisations of simultaneous nearby emitters.
    rpd = rpd + (substructamp
                 * models.pairwise_correlation_2d(separation_values,
                                                  0.,
                                                  np.sqrt(2) * substructsd
                                                  )
                 )

    return rpd


def centriole_model_xy_distances_9fold_variable_vertices_vectorargs(
        input_vector):
    """Function to calculate the values given by
    centriole_model_xy_distances_9fold_variable_vertices, but using a
    vector input for the parameters, so that the numdifftools package can be
    used to calculate partial derivatives for correct error propagation in the
    model.

    Args:
        input_vector (list or numpy array):
            A concatenation of:
                1. A distance at which density values of the model will be
                obtained (numpy array)

                2. The parameters used by
                centriole_model_xy_distances_9fold_variable_vertices
    Returns:
        rpd (numpy array):
            The relative position density given by the model at the input
            distances (called separation_values_1d).
    """
    (separation_values,
     diameter,
     vertssd,
     vertamp1, vertamp2, vertamp3, vertamp4,
     replocssd, replocsamp,
     substructsd, substructamp) = input_vector

    rpd = centriole_model_xy_distances_9fold_variable_vertices(
        separation_values,
        diameter,
        vertssd,
        vertamp1, vertamp2, vertamp3, vertamp4,
        replocssd, replocsamp,
        substructsd, substructamp
        )

    return rpd


def centriole_model_xy_distances_9fold_variable_vertices_internal_bg(
        separation_values,
        diameter,
        vertssd,
        vertamp1, vertamp2, vertamp3, vertamp4,
        replocssd, replocsamp,
        substructsd, substructamp,
        bg_dia, bg_amp
        ):
    """Parametric model for XY distances between localisations of a centriolar
    protein. Background is modelled as resulting from a disk with isotropic
    localisations (this provides a reasonable fit to the distance
    histograms for the less-ordered centriole images).

    Args:
        separation_values (numpy array):
            Distances at which density values of the model will be obtained.
        diameter (float):
            Diameter of the circle containing the vertices of the polygon.
        vertssd (float):
            Broadening of the peaks located at
            distances between the vertices.
        vertamp1, vertamp2, vertamp3, vertamp4 (float):
            Amplitudes of the contributions of the different inter-vertex
            distances.
        replocssd (float):
            Spread representing localisation precision for repeated
            localisations of the same fluorescent molecule.
        replocsamp (float):
            Amplitude of the contribution of repeated localisations
            of the same fluorescent molecule.
        substructsd (float):
            Spread of a contribution resulting from unresolvable
            substructure, or mislocalisations resulting from
            a combination of simultaneous nearby emitters.
        substructamp (float):
            Amplitude of the contribution of unresolvable
            substructure, or mislocalisations resulting from
            a combination of simultaneous nearby emitters.
        bg_dia (float):
            Diameter of the disk containing the modelled background
            localisations.
        bg_amp (float):
            Amplitude of the background component.

    Returns:
        rpd (numpy array):
           The relative position density given by the model
           at distances r.
    """
    # Set non-fittable symmetry order
    sym_order = 9

    # Prepare amplitudes of the different inter-vertex distances
    # for easy use.
    vertices_contributions = [vertamp1, vertamp2, vertamp3, vertamp4]

    # Calculate the inter-vertex distances
    vertices = models.generate_polygon_points(sym_order, diameter)

    filter_distance = (2 * diameter)
    # Need to get unsorted relative positions, to find unique distances
    # at the start of the output list from getdistance(),
    # so use sort_and_halve=False.
    relative_positions = getdistances(vertices,
                                      filter_distance,
                                      verbose=False,
                                      sort_and_halve=False)
    xy_separations = np.sqrt(relative_positions[:, 0] ** 2
                             + relative_positions[:, 1] ** 2)
    unique_xy_seps = xy_separations[0:np.floor(sym_order / 2).astype(int)]

    # Include the contributions from the inter-vertex distance in the RPD.
    rpd = separation_values * 0.
    for i, distance in enumerate(unique_xy_seps):
        rpd = (rpd
               + vertices_contributions[i]
               * models.pairwise_correlation_2d(separation_values,
                                                distance,
                                                vertssd
                                                )
               )

    # Add pair correlation distribution for repeated localisations.
    rpd = rpd + (replocsamp
                 * models.pairwise_correlation_2d(separation_values,
                                                  0.,
                                                  np.sqrt(2) * replocssd
                                                  )
                 )

    # Add pair correlation distribution for unresolvable substructure/
    # mislocalisations of simultaneous nearby emitters.
    rpd = rpd + (substructamp
                 * models.pairwise_correlation_2d(separation_values,
                                                  0.,
                                                  np.sqrt(2) * substructsd
                                                  )
                 )

    # Add background within disk
    separation_values_below_bg_dia = separation_values[separation_values
                                                       < bg_dia]
    background = bg_amp * pair_correlation_disk(separation_values_below_bg_dia,
                                                radius=bg_dia/2.
                                                )

    rpd[0:len(background)] = rpd[0:len(background)] + background

    return rpd


def centriole_model_xy_distances_9fold_variable_vertices_internal_bg_vectorargs(
        input_vector):
    """Function to calculate the values given by
    centriole_model_xy_distances_9fold_variable_vertices, but using a
    vector input for the parameters, so that the numdifftools package can be
    used to calculate partial derivatives for correct error propagation in the
    model.

    Args:
        input_vector (list or numpy array):
            A concatenation of:
                1. A distance at which density values of the model will be
                obtained (numpy array)

                2. The parameters used by
                centriole_model_xy_distances_9fold_variable_vertices
    Returns:
        rpd (numpy array):
            The relative position density given by the model at the input
            distances (called separation_values_1d).
    """
    (separation_values,
     diameter,
     vertssd,
     vertamp1, vertamp2, vertamp3, vertamp4,
     replocssd, replocsamp,
     substructsd, substructamp,
     bg_dia, bg_amp) = input_vector

    rpd = centriole_model_xy_distances_9fold_variable_vertices(
        separation_values,
        diameter,
        vertssd,
        vertamp1, vertamp2, vertamp3, vertamp4,
        replocssd, replocsamp,
        substructsd, substructamp,
        )
    # Need to add background below bg_diameter separately for differentiation
    rpd = rpd + bg_amp * pair_correlation_disk(separation_values,
                                               radius=bg_dia/2.
                                               )

    return rpd


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
    # Lower bounds (switched off)
    lower_bound_dict = {'diameter': 0,
                        'vertssd': 0,
                        'vertsamp': 0,
                        'vertamp1': 0,
                        'vertamp2': 0,
                        'vertamp3': 0,
                        'vertamp4': 0,
                        'replocssd': 0,
                        'replocsamp': 0,
                        'substructsd': 0,
                        'substructamp': 0,
                        'bg_dia': 0,
                        'bg_amp': 0
                        }

    # Upper bounds (switched off)
    upper_bound_dict = {'diameter': np.inf,
                        'vertssd': np.inf,
                        'vertsamp': np.inf,
                        'vertamp1': np.inf,
                        'vertamp2': np.inf,
                        'vertamp3': np.inf,
                        'vertamp4': np.inf,
                        'replocssd': np.inf,
                        'replocsamp': np.inf,
                        'substructsd': np.inf,
                        'substructamp': np.inf,
                        'bg_dia': np.inf,
                        'bg_amp': np.inf
                        }

    # Initial guesses, set to default value of 1
    initial_params_dict = {'diameter': 1.,
                           'vertssd': 1.,
                           'vertsamp': 1.,
                           'vertamp1': 1.,
                           'vertamp2': 1.,
                           'vertamp3': 1.,
                           'vertamp4': 1.,
                           'replocssd': 1.,
                           'replocsamp': 1.,
                           'substructsd': 1.,
                           'substructamp': 1.,
                           'bg_dia': 1.,
                           'bg_amp': 1.
                           }

    return lower_bound_dict, upper_bound_dict, initial_params_dict


def set_up_variable_vertices_model_9fold_no_bg_with_fit_settings():
    """Set up the RPD model with its fitting settings to pass to scipy's
    curve_fit, and the vector-input version of it for differentiation with
    numdifftools.
    """
    variable_vertices_model_9fold_no_bg_with_fit_settings = (
        ModelWithFitSettings(
            model_rpd=centriole_model_xy_distances_9fold_variable_vertices
            )
        )

    (lower_bound_dict,
     upper_bound_dict,
     initial_params_dict) = create_default_fitting_params_dicts()

    # Can optionally modify these dictionaries here.

    initial_params = [initial_params_dict['diameter'],
                      initial_params_dict['vertssd'],
                      initial_params_dict['vertamp1'],
                      initial_params_dict['vertamp2'],
                      initial_params_dict['vertamp3'],
                      initial_params_dict['vertamp4'],
                      initial_params_dict['replocssd'],
                      initial_params_dict['replocsamp'],
                      initial_params_dict['substructsd'],
                      initial_params_dict['substructamp'],
                      ]

    lower_bounds = [lower_bound_dict['diameter'],
                    lower_bound_dict['vertssd'],
                    lower_bound_dict['vertamp1'],
                    lower_bound_dict['vertamp2'],
                    lower_bound_dict['vertamp3'],
                    lower_bound_dict['vertamp4'],
                    lower_bound_dict['replocssd'],
                    lower_bound_dict['replocsamp'],
                    lower_bound_dict['substructsd'],
                    lower_bound_dict['substructamp'],
                    ]

    upper_bounds = [upper_bound_dict['diameter'],
                    upper_bound_dict['vertssd'],
                    upper_bound_dict['vertamp1'],
                    upper_bound_dict['vertamp2'],
                    upper_bound_dict['vertamp3'],
                    upper_bound_dict['vertamp4'],
                    upper_bound_dict['replocssd'],
                    upper_bound_dict['replocsamp'],
                    upper_bound_dict['substructsd'],
                    upper_bound_dict['substructamp'],
                    ]

    bounds = (lower_bounds, upper_bounds)

    variable_vertices_model_9fold_no_bg_with_fit_settings.initial_params = (
        initial_params
        )
    variable_vertices_model_9fold_no_bg_with_fit_settings.param_bounds = (
        bounds
        )
    variable_vertices_model_9fold_no_bg_with_fit_settings.vector_input_model = (
        centriole_model_xy_distances_9fold_variable_vertices_vectorargs
        )
    return variable_vertices_model_9fold_no_bg_with_fit_settings


def set_up_variable_vertices_model_9fold_internal_bg_with_fit_settings():
    """Set up the RPD model with its fitting settings to pass to scipy's
    curve_fit, and the vector-input version of it for differentiation with
    numdifftools.
    """
    variable_vertices_model_9fold_internal_bg_with_fit_settings = (
        ModelWithFitSettings(
            model_rpd=centriole_model_xy_distances_9fold_variable_vertices_internal_bg
            )
        )

    (lower_bound_dict,
     upper_bound_dict,
     initial_params_dict) = create_default_fitting_params_dicts()

    # Can optionally modify these dictionaries here:

    initial_params = [initial_params_dict['diameter'],
                      initial_params_dict['vertssd'],
                      initial_params_dict['vertamp1'],
                      initial_params_dict['vertamp2'],
                      initial_params_dict['vertamp3'],
                      initial_params_dict['vertsamp'],
                      initial_params_dict['replocssd'],
                      initial_params_dict['replocsamp'],
                      initial_params_dict['substructsd'],
                      initial_params_dict['substructamp'],
                      initial_params_dict['bg_dia'],
                      initial_params_dict['bg_amp']
                      ]

    lower_bounds = [lower_bound_dict['diameter'],
                    lower_bound_dict['vertssd'],
                    lower_bound_dict['vertamp1'],
                    lower_bound_dict['vertamp2'],
                    lower_bound_dict['vertamp3'],
                    lower_bound_dict['vertamp4'],
                    lower_bound_dict['replocssd'],
                    lower_bound_dict['replocsamp'],
                    lower_bound_dict['substructsd'],
                    lower_bound_dict['substructamp'],
                    lower_bound_dict['bg_dia'],
                    lower_bound_dict['bg_amp']
                    ]

    upper_bounds = [upper_bound_dict['diameter'],
                    upper_bound_dict['vertssd'],
                    upper_bound_dict['vertamp1'],
                    upper_bound_dict['vertamp2'],
                    upper_bound_dict['vertamp3'],
                    upper_bound_dict['vertamp4'],
                    upper_bound_dict['replocssd'],
                    upper_bound_dict['replocsamp'],
                    upper_bound_dict['substructsd'],
                    upper_bound_dict['substructamp'],
                    upper_bound_dict['bg_dia'],
                    upper_bound_dict['bg_amp']
                    ]

    bounds = (lower_bounds, upper_bounds)

    variable_vertices_model_9fold_internal_bg_with_fit_settings.initial_params = (
        initial_params
        )
    variable_vertices_model_9fold_internal_bg_with_fit_settings.param_bounds = (
        bounds
        )
    variable_vertices_model_9fold_internal_bg_with_fit_settings.vector_input_model = (
        centriole_model_xy_distances_9fold_variable_vertices_internal_bg_vectorargs
        )
    return variable_vertices_model_9fold_internal_bg_with_fit_settings


def sum_distance_histograms(distance_histogram_list, normalise=True):
    """Add together distance histograms from a list, to aggregate results over
    multiple centrioles.

    Args:
        distance_histograms_list (2D numpy array):
            List of distance histogram bin values. One distance histogram
            per row.
        normalise (bool):
            Decide whether to normalise the histograms so that the mean
            bin value is 1 (good for scipy's curve_fit on individual
            centriole data).

    Returns:
        result (numpy array):
            Sum of the distance histograms.
    """
    result = 0
    for one_histogram in distance_histogram_list:
        result = result + one_histogram

    # Normalise
    if normalise is True:
        result = result / (np.sum(result) / len(result))

    return result


def make_fit_results_table_variable_vertices_model_9fold_no_bg(
        distance_histograms_list,
        model_with_info
        ):
    """Make tables of fitted parameter estimates and uncertainties, and
    their ratios.

    Args:
        distance_histograms_list (list of numpy arrays (e.g. 2D numpy array)):
            List with each element being the distance histogram for one
            centriole image. We assume the distance histograms to have the
            same length, and the bin centres to be at distances of
            (integer + 0.5) nm
        model_with_info (ModelWithFitSettings object):
            The model to be fitted, with initial parameter guesses and
            bounds on allowable parameter values.

    Returns:
        estimated_params_df (pandas dataframe):
            Dataframe containing the index and fitted parameter
            estimates for each distance histogram in distance_histograms_list.
        error_on_params_df (pandas dataframe):
            Dataframe containing the index and errors on the fitted
            parameters for each distance histogram in distance_histograms_list.
        ratios (pandas dataframe):
            Dataframe containing the index and error/estimate ratios for
            each parameter for each distance histogram in
            distance_histograms_list.
    """
    # Set up dataframes for fit results
    estimated_params_df = pd.DataFrame(columns=['centriole_number',
                                                'diameter',
                                                'vertices_broadening',
                                                'peak1_amp',
                                                'peak2_amp',
                                                'peak3_amp',
                                                'peak4_amp',
                                                'rept_locs_sd',
                                                'rept_locs_amp',
                                                'substruct_sd',
                                                'substruct_amp'
                                                ]
                                       )

    error_on_params_df = pd.DataFrame(columns=['centriole_number',
                                               'diameter',
                                               'vertices_broadening',
                                               'peak1_amp',
                                               'peak2_amp',
                                               'peak3_amp',
                                               'peak4_amp',
                                               'rept_locs_sd',
                                               'rept_locs_amp',
                                               'substruct_sd',
                                               'substruct_amp'
                                               ]
                                      )

    # Populate the tables
    max_fit_distance = len(distance_histograms_list[0])

    for i, histogram in enumerate(distance_histograms_list):
        # Do the fit
        (params_optimised,
         params_covar,
         params_1sd_error) = models.fit_model_to_experiment(
             distance_histograms_list[i],
             model_with_info.model_rpd,
             model_with_info.initial_params,
             model_with_info.param_bounds,
             fitlength=max_fit_distance
             )

        # Add a row to the table, including centriole number
        # Make centriole numbers integers
        table_row_estimates = np.append(i, params_optimised)
        table_row_errors = np.append(i, params_1sd_error)

        estimated_params_df.loc[i] = table_row_estimates
        estimated_params_df['centriole_number'] = (
            estimated_params_df['centriole_number'].astype(int))
        error_on_params_df.loc[i] = table_row_errors
        error_on_params_df['centriole_number'] = (
            error_on_params_df['centriole_number'].astype(int))

    # Make table of ratios of error / estimate
    ratios = error_on_params_df / estimated_params_df
    ratios['centriole_number'] = estimated_params_df['centriole_number']

    return estimated_params_df, error_on_params_df, ratios


def select_centrioles_by_error_ratio_threshold_on_peaks(
        ratios_df, ratio_threshold=0.1, n_peaks_required_below_threshold=2):
    """Choose centrioles based on the number of peaks corresponding to
    inter-vertex distances) in the fitted model that have error/estimate
    ratio below a threshold.

    Args:
        ratios_df (pandas dataframe):
            Dataframe containing the index and error/estimate ratios for
            each parameter for at least one distance histogram.
        ratio_threshold (float):
            Ratio of error/estimate for peak amplitudes, above which
            centrioles will be excluded from the selection.
        n_peaks_required_below_threshold (int):
            The number of peaks required to be below the error/estimate
            threshold, for the distance histogram to be selected.

    Returns:
        selected_centriole_numbers (list of integers):
            List of indices for centrioles with the required
            error/estimate ratios for the peak amplitudes corresponding
            to inter-vertex distances.
    """
    # Set < or >= threshold to binary values
    ratios_below_threshold_df = ratios_df < ratio_threshold
    ratios_below_threshold_df = ratios_below_threshold_df.astype(int)

    # Calculate the number of peaks below the ratio threshold per row
    n_peaks_below_threshold = (ratios_below_threshold_df.peak1_amp +
                               ratios_below_threshold_df.peak2_amp +
                               ratios_below_threshold_df.peak3_amp +
                               ratios_below_threshold_df.peak4_amp
                               )

    # Select the rows with >= the required number of peaks below the
    # ratio threshold
    selected_below_threshold = (
        ratios_df.loc[n_peaks_below_threshold
                      >= n_peaks_required_below_threshold
                      ]
        )

    # Get the centriole numbers
    selected_centriole_numbers = list(selected_below_threshold.centriole_number)

    return selected_centriole_numbers


def main():
    """Do stuff."""
    outfile_root = 'C:\\Temp\\Centriole\\analysis_results'

    # Get centriole localisations to analyse
    locs_per_centriole = get_input_data()

    # Set filter to remove imprecise localisations
    max_localisation_precision = 5

    outfile_root = (outfile_root
                    + '_' + repr(max_localisation_precision) + 'nm_precision'
                    )

    # Get a list of XY-distances between localisations within each centriole,
    # with one list item per centriole.
    distances_xy_per_centriole = get_xy_distances(locs_per_centriole,
                                                  max_localisation_precision
                                                  )

    outfile_relpos = outfile_root + '_distances_xy.npy'
    np.save(outfile_relpos, distances_xy_per_centriole)

    # Set upper limit on distances in histograms and fits
    max_fit_distance = 500

    # Make list of distance histograms, with each list item being the
    # distance histogram for one centriole.
    # Mean bin value is 1.
    (distance_histograms_list
     ) = make_distance_histograms_per_centriole(distances_xy_per_centriole,
                                                max_fit_distance
                                                )[0]

    outfile_dist_hist_list = (outfile_root
                              + '_distance_histograms_normalised.npy'
                              )
    np.save(outfile_dist_hist_list, distance_histograms_list)

    # Make list of distance histograms, with each list item being the
    # distance histogram for one centriole.
    # Bin values not normalised.
    (distance_histograms_list
     ) = make_distance_histograms_per_centriole(distances_xy_per_centriole,
                                                max_fit_distance,
                                                normalise=False
                                                )[0]

    outfile_dist_hist_list = (outfile_root
                              + '_distance_histograms_not_normalised.npy'
                              )
    np.save(outfile_dist_hist_list, distance_histograms_list)

    # Choose model
    # model_with_info = set_up_variable_vertices_model_9fold_internal_bg_with_fit_settings()
    model_with_info = set_up_variable_vertices_model_9fold_no_bg_with_fit_settings()

    # Could modify the fit parameter settings here

    model_with_info.initial_params = [# diameter,
                                      # vertssd,
                                      # vertsamp
                                      # vertamp1, vertamp2, vertamp3, vertamp4,
                                      # replocssd, replocsamp,
                                      # substructsd, substructamp,
                                      # bg_dia, bg_amp
                                      300.0,
                                      35.0,
                                      #100.,
                                      100., 100., 100.0, 100.0,
                                      10.0, 10.0,
                                      10., 10.]
                                      # 300., 10.]

    model_with_info.param_bounds = (0, [# diameter,
                                        # vertssd,
                                        # vertsamp
                                        # vertamp1, vertamp2, vertamp3, vertamp4,
                                        # replocssd, replocsamp,
                                        # substructsd, substructamp,
                                        # bg_dia, bg_amp
                                        400.0,
                                        1000.0,
                                        #500.,
                                        500., 500., 500.0, 500.0,
                                        50.0, 500.0,
                                        50., 500.]
                                        # 1000., 100.]
                                    )

    # Make dataframes containing fitted parameters, their 1 SD uncertainties,
    # and the ratios between them.
    # Use the normalised histograms.
    (estimated_params_df,
     error_on_params_df,
     error_to_estimate_ratios_df
     ) = (make_fit_results_table_variable_vertices_model_9fold_no_bg(
          distance_histograms_list, model_with_info)
          )
#     estimated_params_df.to_csv(
#         'C:/Temp/Centriole/5nm_precision_distance_histograms_normalised_parameter_estimates.csv',
#         index=False
#         )

    for i in [0, 26]:
        # Fit model to data
        (params_optimised,
         params_covar,
         params_1sd_error) = models.fit_model_to_experiment(
             distance_histograms_list[i],
             model_with_info.model_rpd,
             model_with_info.initial_params,
             model_with_info.param_bounds,
             fitlength=max_fit_distance
             )

        print('Optimised, SD error')
        print('___________________')
        print(np.column_stack((params_optimised, params_1sd_error)))

        # Plot distance histogram
        fig, axes = plot_distance_histogram(
            distance_histograms_list[i],
            bin_edges=np.arange(max_fit_distance + 1)
            )

        # Plot curve
        x_values = np.arange(max_fit_distance) + 0.5
        fitted_curve_values = model_with_info.model_rpd(x_values,
                                                        *params_optimised
                                                        )
        axes.plot(x_values, fitted_curve_values, color='xkcd:red', lw=0.5)

        # Plot 95% CI
        # For model with internal background
        # plot_95_ci_internal_bg(axes,
        #                       x_values,
        #                       model_with_info.model_rpd,
        #                       params_optimised,
        #                       params_covar,
        #                       params_optimised[-2],
        #                       model_with_info.vector_input_model)

        # For model without internal background
        stdev = plot_95_ci(axes,
                           x_values,
                           model_with_info.model_rpd,
                           params_optimised,
                           params_covar,
                           model_with_info.vector_input_model)
        axes.set_title('Centriole ' + repr(i) +
                       ' (0 to' + repr(len(distance_histograms_list) - 1) +
                       ')'
                       )

    # Sum the distance histograms
    # Use not normalised ones
    # Optionally select centrioles according to error to estimate ratios
    # of peak amplitudes
    chosen_centrioles = select_centrioles_by_error_ratio_threshold_on_peaks(
        error_to_estimate_ratios_df)
    distance_histograms_list = distance_histograms_list[chosen_centrioles]

    summed_histograms = sum_distance_histograms(distance_histograms_list,
                                                normalise=True)

    (params_optimised,
     params_covar,
     params_1sd_error) = models.fit_model_to_experiment(
         summed_histograms,
         model_with_info.model_rpd,
         model_with_info.initial_params,
         model_with_info.param_bounds,
         fitlength=max_fit_distance
         )

    print('Optimised, SD error')
    print('___________________')
    print(np.column_stack((params_optimised, params_1sd_error)))

    # Plot distance histogram
    fig, axes = plot_distance_histogram(
        summed_histograms,
        bin_edges=np.arange(max_fit_distance + 1)
        )

    # Plot curve
    x_values = np.arange(max_fit_distance) + 0.5
    fitted_curve_values = model_with_info.model_rpd(x_values,
                                                    *params_optimised
                                                    )
    axes.plot(x_values, fitted_curve_values, color='xkcd:red', lw=0.5)

    stdev = plot_95_ci(axes,
                       x_values,
                       model_with_info.model_rpd,
                       params_optimised,
                       params_covar,
                       model_with_info.vector_input_model)
    axes.set_title('All centrioles')


# if __name__ == '__main__':
#     main()
# start_time = time.time()  # Start timing it.
