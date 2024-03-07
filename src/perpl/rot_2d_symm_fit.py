"""
Created on Wed Oct 2 14:50:19 2019

Functions for creating models with rotational symmetry and fitting.

Alistair Curd
University of Leeds
2 October 2019

Software Engineering practices applied

Joanna Leng (an EPSRC funded Research Software Engineering Fellow (EP/R025819/1)
University of Leeds
October 2019

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

import os
import sys
import argparse
import datetime
import timeit
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np
import matplotlib.pyplot as plt

from perpl.background_models import zero_to_constant_gradient
import perpl.modelling_general as models
from perpl.modelling_general import ModelWithFitSettings
from perpl.modelling_general import stdev_of_model
import perpl.modelstats as stats
from perpl.relative_positions import getdistances
from perpl import utils, plotting, reports


class Number:
    """Class object providing only a number, to be passed to rotational
    symmetry models to provide the order of symmetry, without being fitted.
    """
    def __init__(self, number):
        self.number = number


def get_inputs(info):
    """Creates a file browser to reads the filename and then reads the other
       inputs as text from the command line. Puts inputs into the doctionary.
       These are:
           in_file_and_path (string): The input filename and path.
           filterdist (int): The distance within which relative positions were calculated.
           verbose (Boolean): If True prints outputs to screen as program executes.
    Args:
        info (dict): A python dictionary containing a collection of useful parameters
            such as the filenames and paths.

    Returns:
        Nothing
    """
    #root = Tk()
    Tk().withdraw()
    print('\n\nPlease select input file containing relative positions to assess '
          'for rotational symmetry (.csv or .txt with comma delimiters).')
    print('The file should contain an array with one relative position '
          'vector per row. '
          'The first two columns of the array (e.g. X-separation, Y-separation) '
          'will be used for the investigation.\n')

    infile = askopenfilename()
    #root.destroy()

    print("The file you selected is: ", infile)

    info['in_file_and_path'] = infile



    # Set longest distance used between localisations when producing
    # distance histograms and fitting.
    try:
        fitlength = int(input('What is the maximum distance between '
                              'localisations that you would like to use in the '
                              'analysis (nm)? '))
    except ValueError:
        print('This must be an integer.\n')
        sys.exit("The filter distance must be an integer.\n")

    info['filter_dist'] = fitlength

    print('\nDo you want updates printed to the screen as the analysis '
          'progresses?')


    silent = False
    answer = input('yes/no \n').lower()
    if answer.startswith('y'):
        silent = True

    info['verbose'] = silent


def rot_sym_only(separation_values,
                 diameter,
                 broadening,
                 amplitude):
    """Parametric model for distances between localisations on vertices
    of a polygon (order of symmetry = number of vertices). The value of
    the model at a distance is termed the relative position density (RPD)
    at that distance. Broadening is modelled assuming Gaussian imprecision on
    the positions of the vertices.

    Calls sym_order.number from outside, so that the degree of symmetry os
    not fitted when this function is passed to scipy.optimize.curve_fit.

    Args:
        separation_values (numpy array):
            Distances at which density values of the model will be obtained.
        diameter (float):
            Diameter of the circle containing the vertices of the polygon.
        broadening (float):
            Broadening of the peaks located at distances between the vertices.
        amplitude (float):
            Amplitude of the contribution of one inter-vertex distance.

    Returns:
        rpd (numpy array):
            The relative position density given by the model at the
            separation_values.
    """
    vertices = models.generate_polygon_points(sym_order.number,
                                              diameter)
    filter_distance = (2 * diameter)
    # getdistances includes removel of duplicates 27/11/2019
    relative_positions = getdistances(vertices, filter_distance)
    xy_separations = np.sqrt(relative_positions[:, 0] ** 2
                             + relative_positions[:, 1] ** 2)

    # Initialise array for set of density values.
    rpd = separation_values * 0.

    # Add 2D pair correlations at the distances between vertices.
    for distance in xy_separations:
        rpd = (rpd
               + (amplitude * models.pairwise_correlation_2d(separation_values,
                                                             distance,
                                                             broadening)
                  )
               )

    return rpd


def rot_sym_with_replocs_and_substructure_isotropic_bg(
        r, dia,
        vertssd, vertsamp,
        replocssd, replocsamp,
        substructsd, substructamp,
        bggrad, info):
    """Parametric model for distances between localisations on vertices
    of a polygon (order of symmetry = number of vertices). The value of
    the model at a distance is termed the relative position density (RPD)
    at that distance. Includes the effect of localisation precision and
    unresolvable substructure at the polygon vertices.
    Args:
        r:      Distances at which density values of the model
                    will be obtained.
        dia:    Diameter of the circle containing the vertices of the polygon.
        vertssd: Broadening of the peaks located at
                    distances between the vertices.
        vertsamp:  Amplitude of the contribution of one inter-vertex distance.
        replocssd: Spread representing localisation precision for repeated
                    localisations of the same fluorescent molecule.
        replocsamp: Amplitude of the contribution of repeated localisations
                        of the same fluorescent molecule.
        substructsd: Spread of a contribution resulting from unresolvable
                        substructure, or mislocalisations resulting from
                        a combination of simultaneous nearby emitters.
        substructamp: Amplitude of the contribution of unresolvable
                        substructure, or mislocalisations resulting from
                        a combination of simultaneous nearby emitters.
        bggrad: Gradient of an isotropic (linearly increasing) background term.
    Returns:
        rpd:    The relative position density given by the model
                    at distances r.
    """
    # Get RPD arising from rotationally symmetric structure with broadening
    # only from imprecision on single vertex points.
    rpd = rot_sym_only(r, dia, vertssd, vertsamp)

    # Add 2D isotropic background
    background = r * bggrad
    rpd = rpd + background
    # Add pair correlation distribution for repeated localisations.
    rpd = rpd + replocsamp * models.pairwise_correlation_2d(r, 0., np.sqrt(2) * replocssd)

    # Add pair correlation distribution for unresolvable substructure/
    # mislocalisations of simultaneous nearby emitters.
    rpd = rpd + substructamp * models.pairwise_correlation_2d(r, 0.,
                                                              np.sqrt(2) * substructsd)

    return rpd


def rot_sym_replocs_substructure_isotropic_bg_with_onset(
        r, dia,
        vertssd, vertsamp,
        replocssd, replocsamp,
        substructsd, substructamp,
        bggrad, bgonset):
    """Parametric model for distances between localisations on vertices
    of a polygon (order of symmetry = number of vertices). The value of
    the model at a distance is termed the relative position density (RPD)
    at that distance. Includes the effect of localisation precision and
    unresolvable substructure at the polygon vertices.

    In this model, we take account of the fact that rotationally
    symmetric structures may be unlikely to be found within one another:
    we allow the isotropic (linearly increasing) background term to
    remain zero until an onset distance (bgonset) is reached.

    Args:
        r (numpy array):
            Distances at which density values of the model will be obtained.
        dia (float):
            Diameter of the circle containing the vertices of the polygon.
        vertsd (float):
            Broadening of the peaks located at
            distances between the vertices.
        vertsamp (float):
            Amplitude of the contribution of one inter-vertex distance.
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
        bggrad (float):
            Gradient of an isotropic (linearly increasing) background term.
        bgonset (float):
            Onset distance for linearly increasing background term,
            since rotationally symmetric structures may exclude
            one another.
    Returns:
        rpd (numpy array):
            The relative position density given by the model
            at distances r.
    """
    # GETDISTANCES IS CALLED BY relative_positions.py.
    # There is an option to set verbose=True if output to screen if desired.

    # Get RPD arising from rotationally symmetric structure with broadening
    # only from imprecision on single vertex points.
    rpd = rot_sym_only(r, dia, vertssd, vertsamp)

    # Isotropic 2D background after an onset distance.
    # Background is zero before the onset distance.
    background = r * bggrad - bggrad * bgonset
    background[background < 0] = 0
    rpd = rpd + background

    # Add pair correlation distribution for repeated localisations.
    rpd = rpd + (replocsamp
                 * models.pairwise_correlation_2d(r,
                                                  0.,
                                                  np.sqrt(2) * replocssd
                                                  )
                 )

    # Add pair correlation distribution for unresolvable substructure/
    # mislocalisations of simultaneous nearby emitters.
    rpd = rpd + (substructamp
                 * models.pairwise_correlation_2d(r,
                                                  0.,
                                                  np.sqrt(2) * substructsd
                                                  )
                 )

    return rpd


def rot_sym_with_replocs_and_substructure_isotropic_bg_with_onset_vectorargs(
        input_vector):
    """Function to calculate the values given by
    rot_sym_with_replocs_and_substructure_isotropic_bg_with_onset, but using a
    vector input for the parameters, so that the numdifftools package can be
    used to calculate partial derivatives for correct error propagation in the
    model.

    Args:
        input_vector (list or numpy array):
            A concatenation of:
                1. A distance at which density values of the model will be
                obtained (numpy array)

                2. The parameters used by
                rot_sym_with_replocs_and_substructure_isotropic_bg_with_onset.
    Returns:
        rpd (numpy array):
            The relative position density given by the model at the input
            distances (called separation_values_1d).
    """
    (separation_values_1d,
     dia,
     vertssd, vertsamp,
     replocssd, replocsamp,
     substructsd, substructamp,
     bggrad, bgonset) = input_vector

    rpd = rot_sym_replocs_substructure_isotropic_bg_with_onset(
        separation_values_1d,
        dia,
        vertssd, vertsamp,
        replocssd, replocsamp,
        substructsd, substructamp,
        bggrad, bgonset)

    return rpd


def rot_sym_with_replocs_and_substructure_isotropic_bg_with_smooth_onset(
        separation_values,
        dia,
        vertssd, vertsamp,
        replocssd, replocsamp,
        substructsd, substructamp,
        bggrad, bgonset, bgvariation):
    """Parametric model for distances between localisations on vertices
    of a polygon (order of symmetry = number of vertices). The value of
    the model at a distance is termed the relative position density (RPD)
    at that distance. Includes the effect of localisation precision and
    unresolvable substructure at the polygon vertices.

    In this model, we take account of the fact that rotationally
    symmetric structures may be unlikely to be found within one another,
    with a differentiable background term varying from
    zero_to_constant_gradient.

    Args:
        r (numpy array):
            Distances at which density values of the model will be obtained.
        dia (float):
            Diameter of the circle containing the vertices of the polygon.
        vertsd (float):
            Broadening of the peaks located at
                    distances between the vertices.
        vertsamp (float):
            Amplitude of the contribution of one inter-vertex distance.
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
        bggrad (float):
            The gradient of the background term  at large separation_values.
        bgonset (float):
            A characteristic value of the background term, between tending to
            zero and tending a constant gradient.
        bgvariation:
            Determines the amount of variation on the background term
            in the intermediate range.

    Returns:
        rpd (numpy array):
            The relative position density given by the model
                    at distances r.
    """
    # GETDISTANCES IS CALLED BY relative_positions.py.
    # There is an option to set verbose=True if output to screen if desired.

    # Get RPD arising from rotationally symmetric structure with broadening
    # only from imprecision on single vertex points.
    rpd = rot_sym_only(separation_values, dia, vertssd, vertsamp)

    # Add 2D isotropic background with onset distance.
    background = zero_to_constant_gradient(separation_values,
                                           bggrad, bgonset, bgvariation)

    rpd = rpd + background
    # Add pair correlation distribution for repeated localisations.
    rpd = rpd + (replocsamp
                 * models.pairwise_correlation_2d(separation_values,
                                                  0.,
                                                  np.sqrt(2) * replocssd)
                 )

    # Add pair correlation distribution for unresolvable substructure/
    # mislocalisations of simultaneous nearby emitters.
    rpd = rpd + (substructamp
                 * models.pairwise_correlation_2d(separation_values,
                                                  0.,
                                                  np.sqrt(2) * substructsd)
                 )

    return rpd


def test_plot_withvectorinput(relpos,
                              model_with_info,
                              fitlength=200.):
    """Testing vector-input version of parametric model function."""
    plt.figure()
    axes = plt.subplot(111)
    xy_histogram = models.make_xy_histogram_nm(relpos, fitlength=fitlength,
                                               axes=axes)[0]
    sym_order.number = 8
    (params_optimised,
     params_covar,
     params_1sd_error) = models.fit_model_to_experiment(xy_histogram,
                                                        model_with_info.model_rpd,
                                                        model_with_info.initial_params,
                                                        model_with_info.param_bounds,
                                                        fitlength=fitlength)

    x_values = np.arange(fitlength) + 0.5

    vector_input = np.concatenate(x_values, params_optimised)
    rpd = rot_sym_with_replocs_and_substructure_isotropic_bg_with_onset_vectorargs(vector_input)

    axes.plot(x_values, rpd)


def rotsym_withreplocs_nosubstructure_isotropicbgwithonset(
        r, dia,
        vertssd, vertsamp,
        replocssd, replocsamp,
        bggrad, bgonset):
    """Parametric model for distances between localisations on vertices
    of a polygon (order of symmetry = number of vertices). The value of
    the model at a distance is termed the relative position density (RPD)
    at that distance. Includes the effect of localisation precision and
    unresolvable substructure at the polygon vertices.

    In this model, we take account of the fact that rotationally
    symmetric structures may be unlikely to be found within one another:
    we allow the isotropic (linearly increasing) background term to
    remain zero until an onset distance (bgonset) is reached.

    Args:
        r:      Distances at which density values of the model
                    will be obtained.
        dia:    Diameter of the circle containing the vertices of the polygon.
        vertsd: Broadening of the peaks located at
                    distances between the vertices.
        vertsamp:  Amplitude of the contribution of one inter-vertex distance.
        replocssd: Spread representing localisation precision for repeated
                    localisations of the same fluorescent molecule.
        replocsamp: Amplitude of the contribution of repeated localisations
                        of the same fluorescent molecule.
        bggrad: Gradient of an isotropic (linearly increasing) background term.
        bgonset: Onset distance for linearly increasing background term,
                    since rotationally symmetric structures may exclude
                    one another.
    Returns:
        rpd:    The relative position density given by the model
                    at distances r.
    """
    # Get positions of vertices, their relative positions
    # and distances between them.
    verts = models.generate_polygon_points(sym_order.number, dia)
    relpos = getdistances(verts, filterdist=(2 * dia))
    dists = np.sqrt(relpos[:, 0] ** 2 + relpos[:, 1] ** 2)
    # Select unduplicated distances between the vertices.
    dists = dists[0:(sym_order.number - 1)]
    # sigma = np.array([sigma0, sigma1, sigma2, sigma3])
    # Initialise array for set of density values.
    rpd = r * 0.
    # Add 2D pair correlations at the distances between vertices.
    for dist in dists:
        rpd = rpd + vertsamp * models.pairwise_correlation_2d(r, dist, vertssd)
    # Add 2D isotropic background with onset distance.
    background = r * bggrad - bggrad * bgonset
    background[background < 0] = 0
    rpd = rpd + background
    # Add pair correlation distribution for repeated localisations.
    rpd = rpd + replocsamp * models.pairwise_correlation_2d(r, 0., np.sqrt(2) * replocssd)

    return rpd


def model_replocs_substruct_no_bg(
        separation_values,
        diameter,
        vertices_broadening, vertices_amplitude,
        rept_locs_broadening, rept_locs_amplitude,
        substructure_broadening, substructure_amplitude):
    """Parametric model for distances between localisations on vertices
    of a polygon (order of symmetry = number of vertices). The value of
    the model at a distance is termed the relative position density (RPD)
    at that distance. Includes the effect of localisation precision and
    unresolvable substructure at the polygon vertices.

    No background is included, so that error propagation can be calculated on
    the portion of the rpd before the hard onset distance of the background.

    Args:
        separation_values (numpy array):
            Distances at which density values of the model will be obtained.
        diameter (float):
            Diameter of the circle containing the vertices of the polygon.
        vertices_broadening (float):
            Broadening of the peaks located at
            distances between the vertices.
        vertices_amplitude (float):
            Amplitude of the contribution of one inter-vertex distance.
        rept_locs_broadening (float):
            Spread representing localisation precision (SD) for repeated
            localisations of the same fluorescent molecule.
        rept_locs_amplitude (float):
            Amplitude of the contribution of repeated localisations
            of the same fluorescent molecule.
        substructure_broadening (float):
            Spread (SD) of a contribution resulting from unresolvable
            substructure, or mislocalisations resulting from
            a combination of simultaneous nearby emitters.
        substructure_amplitude (float):
            Amplitude of the contribution of unresolvable
            substructure, or mislocalisations resulting from
            a combination of simultaneous nearby emitters.

    Returns:
        rpd (numpy array):
            The relative position density given by the model
            at the separation_values.
    """
    # Get RPD arising from rotationally symmetric structure with broadening
    # only from imprecision on single vertex points.
    rpd = rot_sym_only(separation_values,
                       diameter,
                       vertices_broadening,
                       vertices_amplitude
                       )

    # Add pair correlation distribution for repeated localisations.
    rpd = rpd + (rept_locs_amplitude
                 * models.pairwise_correlation_2d(separation_values,
                                                  0.,
                                                  np.sqrt(2) * rept_locs_broadening)
                 )

    # Add pair correlation distribution for unresolvable substructure/
    # mislocalisations of simultaneous nearby emitters.
    rpd = rpd + (substructure_amplitude
                 * models.pairwise_correlation_2d(separation_values,
                                                  0.,
                                                  np.sqrt(2) * substructure_broadening)
                 )

    return rpd


def model_replocs_substruct_no_bg_vectorargs(input_vector):
    """Function to calculate the values given by
    rot_sym_with_replocs_and_substructure_no_bg, but using a
    vector input for the parameters, so that the numdifftools package can be
    used to calculate partial derivatives for correct error propagation in the
    model.

    Args:
        input_vector (list or numpy array):
            A concatenation of:
                1. Distances at which density values of the model will be
                obtained (numpy array)

                2. The parameters used by
                rot_sym_with_replocs_and_substructure_no_bg.
    Returns:
        rpd (numpy array):
            The relative position density given by the model at the input
            distances (called separation_values).
    """
    (separation_values,
     diameter,
     vertices_broadening, vertices_amplitude,
     rept_locs_broadening, rept_locs_amplitude,
     substructure_broadening, substructure_amplitude) = input_vector

    rpd = model_replocs_substruct_no_bg(separation_values,
                                        diameter,
                                        vertices_broadening,
                                        vertices_amplitude,
                                        rept_locs_broadening,
                                        rept_locs_amplitude,
                                        substructure_broadening,
                                        substructure_amplitude)

    return rpd


def rot_sym_with_replocs_and_substructure_isotropic_bg_after_onset(
        separation_values,
        diameter,
        vertices_broadening, vertices_amplitude,
        rept_locs_broadening, rept_locs_amplitude,
        substructure_broadening, substructure_amplitude,
        bg_grad, bg_onset):
    """Parametric model for distances between localisations on vertices
    of a polygon (order of symmetry = number of vertices). The value of
    the model at a distance is termed the relative position density (RPD)
    at that distance. Includes the effect of localisation precision and
    unresolvable substructure at the polygon vertices.

    THIS FUNCTION IS FOR USE AFTER THE ONSET OF A CONSTANT BACKGROUND, AND WILL
    GIVE NEGATIVE VALUES BEFORE THE BACKGROUND ONSET DISTANCE.

    A linear (isotropic 2D) background term is used.

    Args:
        separation_values (numpy array):
            Distances at which density values of the model will be obtained.
        diameter (float):
            Diameter of the circle containing the vertices of the polygon.
        vertices_broadening (float):
            Broadening of the peaks located at
            distances between the vertices.
        vertices_amplitude (float):
            Amplitude of the contribution of one inter-vertex distance.
        rept_locs_broadening (float):
            Spread representing localisation precision (SD) for repeated
            localisations of the same fluorescent molecule.
        rept_locs_amplitude (float):
            Amplitude of the contribution of repeated localisations
            of the same fluorescent molecule.
        substructure_broadening (float):
            Spread (SD) of a contribution resulting from unresolvable
            substructure, or mislocalisations resulting from
            a combination of simultaneous nearby emitters.
        substructure_amplitude (float):
            Amplitude of the contribution of unresolvable
            substructure, or mislocalisations resulting from
            a combination of simultaneous nearby emitters.
        bggrad (float):
            Gradient of an isotropic (linearly increasing) background term.
        bgonset (float):
            Onset distance for linearly increasing background term,
            since rotationally symmetric structures may exclude
            one another.

    Returns:
        rpd (numpy array):
            The relative position density given by the model
            at the separation_values.
    """
    # Get RPD arising from rotationally symmetric structure with broadening
    # only from imprecision on single vertex points.
    rpd = rot_sym_only(separation_values,
                       diameter,
                       vertices_broadening,
                       vertices_amplitude
                       )

    # Add pair correlation distribution for repeated localisations.
    rpd = rpd + (rept_locs_amplitude
                 * models.pairwise_correlation_2d(separation_values,
                                                  0.,
                                                  np.sqrt(2) * rept_locs_broadening)
                 )

    # Add pair correlation distribution for unresolvable substructure/
    # mislocalisations of simultaneous nearby emitters.
    rpd = rpd + (substructure_amplitude
                 * models.pairwise_correlation_2d(separation_values,
                                                  0.,
                                                  np.sqrt(2) * substructure_broadening)
                 )

    background = separation_values * bg_grad - bg_grad * bg_onset

    rpd = rpd + background

    return rpd


def model_linear_bg_after_onset_vectorargs(input_vector):
    """Function to calculate the values given by
    rot_sym_with_replocs_and_substructure_isotropic_bg_after_onset, but using a
    vector input for the parameters, so that the numdifftools package can be
    used to calculate partial derivatives for correct error propagation in the
    model.

    Args:
        input_vector (list or numpy array):
            A concatenation of:
                1.  A distance at which density values of the model will be
                obtained (numpy array)

                2. The parameters used by
                rot_sym_with_replocs_and_substructure_isotropic_bg_after_onset.
    Returns:
        rpd (numpy array):
            The relative position density given by the model at the input
            distances (called separation_values_1d).
    """
    (separation_values,
     diameter,
     vertices_broadening, vertices_amplitude,
     rept_locs_broadening, rept_locs_amplitude,
     substructure_broadening, substructure_amplitude,
     bg_grad, bg_onset) = input_vector

    rpd = rot_sym_with_replocs_and_substructure_isotropic_bg_after_onset(separation_values,
                                                                         diameter,
                                                                         vertices_broadening,
                                                                         vertices_amplitude,
                                                                         rept_locs_broadening,
                                                                         rept_locs_amplitude,
                                                                         substructure_broadening,
                                                                         substructure_amplitude,
                                                                         bg_grad,
                                                                         bg_onset)

    return rpd


def model_variable_vertices_replocs_substructure_no_bg(
        separation_values,
        diameter,
        vertssd,
        vertamp1, vertamp2, vertamp3, vertamp4,
        replocssd, replocsamp,
        substructsd, substructamp
        ):
    """Parametric model for distances between localisations on vertices
    of a polygon (order of symmetry = number of vertices). The value of
    the model at a distance is termed the relative position density (RPD)
    at that distance. Includes the effect of localisation precision and
    unresolvable substructure at the polygon vertices.

    In this model, the inter-vertex distances are allowed to have independent
    amplitude in the RPD.

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
    # Prepare amplitudes of the different inter-vertex distances
    # for easy use.
    vertices_contributions = [vertamp1, vertamp2, vertamp3, vertamp4]

    # Calculate the inter-vertex distances
    vertices = models.generate_polygon_points(sym_order.number, diameter)

    filter_distance = (2 * diameter)
    relative_positions = getdistances(vertices, filter_distance)
    xy_separations = np.sqrt(relative_positions[:, 0] ** 2
                             + relative_positions[:, 1] ** 2)
    # Select unduplicated inter-vertex distances
    # (round down from # vertices divided by 2).
    xy_separations = xy_separations[0:int(np.floor(sym_order / 2))]

    # Include the contributions from the inter-vertex distance in the RPD.
    rpd = separation_values * 0.
    for i, distance in enumerate(xy_separations):
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


def get_1sd_error_on_model_before_and_after_background_onset(
        x_values,
        params_optimised,
        params_covar,
        bg_onset):
    """Use stdev_of_models in modelling_general (uses numdifftools) to acquire
    stdev of the models before and after the background onset distance, and
    concatenate to provide output.

    Args:
        x_values (numpy array):
            Distances at which the models are evaluated.
        params_optimised (numpy array):
            The optimised parameter values for
            rot_sym_with_replocs_and_substructure_isotropic_bg_with_onset
            in rotationalsymettrymodelling, following a fit to
            experimental data.
        params_covar (numpy array):
            The covariance matrix for the parameters that have been optimised.
        vector_input_model (function):
            The name of a function that will be passed to
            numdifftools.Gradient. This function must have one arguments, which
            is a vector. The first element of the vector is also the vector of
            x_values.

    Returns:
        stdev_complete (numpy array):
            Numpy array of 1 standard deviation of uncertainty on the model
            evaluated at each x_value.

    """
    x_values_before_onset = x_values[x_values <= bg_onset]
    x_values_after_onset = x_values[x_values > bg_onset]

    # Use fitted parameters from
    # rot_sym_with_replocs_and_substructure_isotropic_bg_with_onset
    # in rotationalsymettrymodelling, except for background (final two)
    # parameters
    stdev_before_onset = stdev_of_model(x_values_before_onset,
                                        params_optimised[0:-2],
                                        params_covar[0:-2, 0:-2],
                                        model_replocs_substruct_no_bg_vectorargs
                                        )
    print('Calculated stdev before background onset.')

    # Use fitted parameters from
    # rot_sym_with_replocs_and_substructure_isotropic_bg_with_onset
    stdev_after_onset = stdev_of_model(x_values_after_onset,
                                       params_optimised,
                                       params_covar,
                                       model_linear_bg_after_onset_vectorargs
                                       )
    print('Calculated stdev after background onset.')

    stdev_complete = np.append(stdev_before_onset, stdev_after_onset)

    return stdev_complete


def nup_xy_fig_plot(
        relative_position_array,
        model_with_info,
        fitlength=200.):
    """Create a plot showing optimised model and uncertainties from fitting
    hard-onset Nup107 model.

    Args:
        relative_position_array (numpy array):
            Numpy array of relative positions vectors. First two columns
            are X and Y components of relative positions.
        model:
            A function. The parametric RPD model.
        fitlength (scalar):
            The extent across the XY plane within which
            relative positions are included in the histogram, fitting
            and plotting.
    Returns:
        stdev (numpy array):
            The uncertainties on the model at each separation used in the fit.
    """
    plt.figure()
    axes = plt.subplot(111)
    xy_histogram = models.make_xy_histogram_nm(relative_position_array,
                                               fitlength=fitlength,
                                               axes=axes)[0]

    (params_optimised,
     params_covar,
     params_1sd_error) = models.fit_model_to_experiment(xy_histogram,
                                                        model_with_info.model_rpd,
                                                        model_with_info.initial_params,
                                                        model_with_info.param_bounds,
                                                        fitlength=fitlength)
    x_values = np.arange(fitlength) + 0.5

    axes.plot(x_values, model_with_info.model_rpd(x_values,
                                                  *params_optimised
                                                  ),
              lw=0.5, color='xkcd:red'
              )

    # Find background onset distance from the optimised parameters
    bg_onset = params_optimised[-1]

    # Get uncertainties below and above background onset and concatenate
    stdev = get_1sd_error_on_model_before_and_after_background_onset(
                x_values,
                params_optimised,
                params_covar,
                bg_onset
                )
    print('Calculated stdev for whole model.')

    axes.fill_between(x_values,
                      (model_with_info.model_rpd(x_values,
                                                 *params_optimised
                                                 )
                       - stdev * 1.96
                       ),
                      (model_with_info.model_rpd(x_values,
                                                 *params_optimised
                                                 )
                       + stdev * 1.96
                       ),
                      color='xkcd:red', alpha=0.25
                      )

    return bg_onset, stdev


def nup_xy_plot_model_components(
        relpos,
        model_with_info,
        fitlength=200):
    """Creates a plot
    Args:
        relpos:
        model:
        fitlength:
    Returns:
        Nothing
    """
    plt.figure()
    axes = plt.subplot(111)
    xy_histogram = models.make_xy_histogram_nm(relpos,
                                               fitlength=fitlength,
                                               axes=axes)[0]

    (params_optimised,
     params_covar,
     params_1sd_error) = models.fit_model_to_experiment(xy_histogram,
                                                        model_with_info.model_rpd,
                                                        model_with_info.initial_params,
                                                        model_with_info.param_bounds,
                                                        fitlength=fitlength)
    dia,\
    vertssd, vertsamp,\
    replocssd, replocsamp,\
    substructsd, substructamp,\
    bggrad, bgonset\
    = params_optimised

    x_values = np.arange(fitlength) + 0.5
    # Plot fitted model curve
    axes.plot(x_values, model_with_info.model_rpd(x_values, *params_optimised))
    # Plot localisation precision component
    axes.plot(x_values,
              replocsamp
              * models.pairwise_correlation_2d(x_values,
                                               0.,
                                               np.sqrt(2) * replocssd
                                               )
              )
    # Plot substructure component
    axes.plot(x_values,
              substructamp
              * models.pairwise_correlation_2d(x_values,
                                               0.,
                                               np.sqrt(2) * substructsd
                                               )
              )
    # Plot symmetry related peaks
    verts = models.generate_polygon_points(sym_order.number, dia)
    filterdist = (2 * dia)

    relpos = getdistances(verts, filterdist)
    dists = np.sqrt(relpos[:, 0] ** 2 + relpos[:, 1] ** 2)
    dists = dists[0: 4]  # Select unduplicated distances between the vertices.
    contributions = np.array([2, 2, 2, 1])
    for peak, dist in enumerate(dists):
        axes.plot(x_values,
                  vertsamp * contributions[peak]
                  * models.pairwise_correlation_2d(x_values, dist, vertssd))

    # Plot background
    background = x_values * bggrad - bggrad * bgonset
    background[background < 0] = 0
    axes.plot(x_values, background)
    axes.set_xlim([0, 150])
    axes.set_xlabel('XY-separation (nm)')
    axes.set_ylabel('Counts (scaled)')


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
    lower_bound_dict = {'diameter': 0,
                        'vertices_sd': 0,
                        'vertices_amp': 0,
                        'vertex_amp_1': 0,
                        'vertex_amp_2': 0,
                        'vertex_amp_3': 0,
                        'vertex_amp_4': 0,
                        'loc_prec_sd': 0,
                        'loc_prec_amp': 0,
                        'substruct_sd': 0,
                        'substruct_amp': 0,
                        'bg_slope': 0,
                        'bg_onset': 0,
                        'bg_variation': 0,
                        'bg_dia': 0,
                        'bg_amp': 0
                        }

    upper_bound_dict = {'diameter': 200.,
                        'vertices_sd': 50.,
                        'vertices_amp': 100.,
                        'vertex_amp_1': 500.,
                        'vertex_amp_2': 500.,
                        'vertex_amp_3': 500.,
                        'vertex_amp_4': 500.,
                        'loc_prec_sd': 50.,
                        'loc_prec_amp': 50.,
                        'substruct_sd': 50.,
                        'substruct_amp': 50.,
                        'bg_slope': 1.,
                        'bg_onset': 150.,
                        'bg_variation': 1.,
                        'bg_dia': 1000.,
                        'bg_amp': 100.
                        }

    initial_params_dict = {'diameter': 100.,
                           'vertices_sd': 10.,
                           'vertices_amp': 10.,
                           'vertex_amp_1': 100.,
                           'vertex_amp_2': 100.,
                           'vertex_amp_3': 100.,
                           'vertex_amp_4': 100.,
                           'loc_prec_sd': 3.,
                           'loc_prec_amp': 10.,
                           'substruct_sd': 10.,
                           'substruct_amp': 10.,
                           'bg_slope': 0.1,
                           'bg_onset': 50.,
                           'bg_variation': 0.1,
                           'bg_dia': 300.,
                           'bg_amp': 10.
                           }

    return lower_bound_dict, upper_bound_dict, initial_params_dict


def set_up_model_replocs_substruct_iso_bg_with_onset_with_fit_settings():
    """Set up the RPD model with fitting settings,
    for a rotationally symmetric model with spread due to repeated
    localisations, spread to unresolvable substructure, and a background
    that increases linearly after an onset distance.
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
    model_replocs_substruct_iso_bg_with_onset_with_fit_settings = (
        ModelWithFitSettings(
            model_rpd=rot_sym_replocs_substructure_isotropic_bg_with_onset
            )
        )

    # Add fitting parameters to ModelWithFitSettings object
    (lower_bound_dict,
     upper_bound_dict,
     initial_params_dict) = create_default_fitting_params_dicts()

    # Can optionally modify these dictionaries here:

    initial_params = [initial_params_dict['diameter'],
                      initial_params_dict['vertices_sd'],
                      initial_params_dict['vertices_amp'],
                      initial_params_dict['loc_prec_sd'],
                      initial_params_dict['loc_prec_amp'],
                      initial_params_dict['substruct_sd'],
                      initial_params_dict['substruct_amp'],
                      initial_params_dict['bg_slope'],
                      initial_params_dict['bg_onset']
                      ]

    lower_bounds = [lower_bound_dict['diameter'],
                    lower_bound_dict['vertices_sd'],
                    lower_bound_dict['vertices_amp'],
                    lower_bound_dict['loc_prec_sd'],
                    lower_bound_dict['loc_prec_amp'],
                    lower_bound_dict['substruct_sd'],
                    lower_bound_dict['substruct_amp'],
                    lower_bound_dict['bg_slope'],
                    lower_bound_dict['bg_onset']
                    ]

    upper_bounds = [upper_bound_dict['diameter'],
                    upper_bound_dict['vertices_sd'],
                    upper_bound_dict['vertices_amp'],
                    upper_bound_dict['loc_prec_sd'],
                    upper_bound_dict['loc_prec_amp'],
                    upper_bound_dict['substruct_sd'],
                    upper_bound_dict['substruct_amp'],
                    upper_bound_dict['bg_slope'],
                    upper_bound_dict['bg_onset']
                    ]

    bounds = (lower_bounds, upper_bounds)

    model_replocs_substruct_iso_bg_with_onset_with_fit_settings.initial_params = (
        initial_params
        )
    model_replocs_substruct_iso_bg_with_onset_with_fit_settings.param_bounds = (
        bounds
        )
    model_replocs_substruct_iso_bg_with_onset_with_fit_settings.vector_input_model = (
        rot_sym_with_replocs_and_substructure_isotropic_bg_with_onset_vectorargs
        )

    return model_replocs_substruct_iso_bg_with_onset_with_fit_settings


def set_up_model_replocs_substruct_no_bg_with_fit_settings():
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
    # Generate ModelWithFitSettings object, conatining a model_rpd
    model_replocs_substruct_no_bg_with_fit_settings = (
        ModelWithFitSettings(
            model_rpd=model_replocs_substruct_no_bg
            )
        )

    # Add fitting parameters to ModelWithFitSettings object
    (lower_bound_dict,
     upper_bound_dict,
     initial_params_dict) = create_default_fitting_params_dicts()

    # Can optionally modify these dictionaries here:

#    initial_params = [initial_params_dict['diameter'],
#                      initial_params_dict['vertices_sd'],
#                      initial_params_dict['vertices_amp'],
#                      initial_params_dict['loc_prec_sd'],
#                      initial_params_dict['loc_prec_amp'],
#                      initial_params_dict['substruct_sd'],
#                      initial_params_dict['substruct_amp'],
#                      ]

    initial_params = [300.0, # diameter
                      35.0, # vertssd
                      100., # vertsamp
                      10.0, # replocssd
                      10.0, # replocsamp
                      10., # substructsd
                      10. # substructamp
                      ]

    lower_bounds = [lower_bound_dict['diameter'],
                    lower_bound_dict['vertices_sd'],
                    lower_bound_dict['vertices_amp'],
                    lower_bound_dict['loc_prec_sd'],
                    lower_bound_dict['loc_prec_amp'],
                    lower_bound_dict['substruct_sd'],
                    lower_bound_dict['substruct_amp'],
                    ]

    upper_bounds = [upper_bound_dict['diameter'],
                    upper_bound_dict['vertices_sd'],
                    upper_bound_dict['vertices_amp'],
                    upper_bound_dict['loc_prec_sd'],
                    upper_bound_dict['loc_prec_amp'],
                    upper_bound_dict['substruct_sd'],
                    upper_bound_dict['substruct_amp'],
                    ]

    upper_bounds = [400.0, # diameter
                    1000.0, # vertssd
                    500., #vertsamp
                    50.0, # replocssd
                    500.0, # replocsamp
                    50., # substructsd
                    500. # substructamp
                    ]

    bounds = (lower_bounds, upper_bounds)

    model_replocs_substruct_no_bg_with_fit_settings.initial_params = (
        initial_params
        )
    model_replocs_substruct_no_bg_with_fit_settings.param_bounds = (
        bounds
        )
    model_replocs_substruct_no_bg_with_fit_settings.vector_input_model = (
        model_replocs_substruct_no_bg_vectorargs
        )

    return model_replocs_substruct_no_bg_with_fit_settings



# Set up callable symmetry order object, with default value
sym_order = Number(8)


def main():
    """
    Reads input data of point density localisations that has been processed by the
    relative_positions.py script which calculates relative positions as
    vectors. These are compared to models of point density localisations with
    various levels of 2D rotational symmetry.

    Outputs are writen to a file in a directory with the name of the inputfile
    and a time stamp above the directory of the input file. There is an html
    report in this directory that provides information of how well each model
    fits the experimental data.
    If no input argments are provided inputs can be given from the command
    line as it executes.

    Args:
        input_file (FILE): File of localisations which is a .csv file.
        filter_dist (int): The filter distance.
        verbose (Boolean): Increases the output to screen during execution.

    Returns:
        Nothing
    """

    # handle the input arguments (flags)
    prog = 'rot_2d_symm_fit'
    prog_short_name = 'r2dsf'
    description = ('Fits rotational symmetry models to relative positions '
                   'among localisation microscopy data. The models are '
                   'generated from synthetic localisation data.')

    info = {'prog': prog,
            'prog_short_name': prog_short_name,
            'description': description}

    start = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    info['start'] = start

    parser = argparse.ArgumentParser(prog, description)

    parser.add_argument('-i', '--input_file',
                        dest='input_file',
                        type=argparse.FileType('r'),
                        help='File of localisations which is a .csv (or .txt '
                             'with comma delimiters) or .npy and containing N '
                             'localisations in N rows.',
                        metavar="FILE")

    parser.add_argument('-f', '--filter_distance',
                        dest='filter_dist',
                        type=int,
                        # default=150, PROBLEM LINE
                        default=200,
                        help="Filter distance.")

    parser.add_argument('-s', '--short_names',
                        help="Uses shortened names for the results files and "
                        "directories. While this makes the results less easy to"
                        " navigate it can be particularly useful on Windows"
                        " systems that do not allow long names and paths. "
                        "The input file name is used for the results filename"
                        " and this shortened names has the first and last 5"
                        " characters with -s- is the middle.",
                        action="store_true")
    
    parser.add_argument('-v', '--verbose',
                        help="Increase output verbosity",
                        action="store_true")

    args = parser.parse_args()

    print("\nargs: ", args)

    if args.verbose:
        print("Verbosity is turned on.\n")

    info['filter_dist'] = args.filter_dist
    info['verbose'] = args.verbose    
    info['short_names'] = args.short_names
    


    if args.input_file is None:
        get_inputs(info)
    else:
        info['in_file_and_path'] = args.input_file.name

    info['host'], info['ip_address'], info['operating_system'] = utils.find_hostname_and_ip()

    read_start = timeit.default_timer()
    xyz_values = utils.secondary_read_data_in(info)
    # print("data read!!\n")
    read_end = timeit.default_timer()
    reading_time = (read_end-read_start)/60

    utils.secondary_filename_and_path_setup(info)
        

    if info['verbose']:
        print("\nTime to read the input file was: "+str(round(reading_time, 3))+\
              " minutes.")
        print('This contains '+str(info['values'])+' xyz locations with '
              +str(info['columns'])+' columns.\n')

    if info['short_names'] is True:
        try:
            os.makedirs(info['short_results_dir'])
        except OSError:
            print("Unexpected error:", sys.exc_info()[0])
            sys.exit("Could not create directory for the results.")    
    else:
        try:
            os.makedirs(info['results_dir'])
        except OSError:
            print("Unexpected error:", sys.exc_info()[0])
            sys.exit("Could not create directory for the results.")

    # Get histogram data ready to fit to model
    #xy_histogram = models.make_xy_histogram_nm(xyz_values, fitlength=fitlength,
    #fig_toggle=False)[0]
    xydists = np.sqrt(xyz_values[:, 0] ** 2 + xyz_values[:, 1] ** 2)
    fitlength = info['filter_dist']
    bin_vals = np.arange(fitlength + 1)

    xy_histogram, bin_values = np.histogram(xydists,
                                            weights=np.repeat(float(fitlength) / len(xydists),
                                                              len(xydists)),
                                            bins=bin_vals)


    # Define symmetries over which to perform and evaluate fit
    symmetries = range(5, 12)

    # Prepare to record fit metric (AICc)
    aiccs = np.zeros(len(symmetries))


    # Define model, including parameter guesses and bounds to pass to
    # scipy.optimize.curve_fit
    model_with_info = (
        set_up_model_replocs_substruct_iso_bg_with_onset_with_fit_settings()
        )
    info['model_name'] = model_with_info.model_rpd.__name__


    info['p0'] = model_with_info.initial_params
    info['optimisation_bounds'] = model_with_info.param_bounds


    curve_values = []
    diameter_values = []
    table_param_values = []


    for i, sym in enumerate(symmetries):
        sym_order.number = sym

        (params_optimised,
         params_covar,
         params_1sd_error) = models.fit_model_to_experiment(xy_histogram,
                                                            model_with_info.model_rpd,
                                                            model_with_info.initial_params,
                                                            model_with_info.param_bounds,
                                                            fitlength=fitlength)
        aicc = stats.aic_from_least_sqr_fit(xy_histogram,
                                            model_with_info.model_rpd,
                                            params_optimised,
                                            fitlength=fitlength)[1]
        aiccs[i] = aicc
        x_values = np.arange(fitlength) + 0.5
        diameter_values.append(params_optimised[0])

        curve_values.append(model_with_info.model_rpd(x_values, *params_optimised))


        if info['verbose']:
            print('\nSymmetry:', sym)
            print("\nParameter    Optimised Value")

        param_string_values = []
        uncertainty_string_values = []

        for count, param_optimised in enumerate(params_optimised):
            uncertainty = params_1sd_error[count]

            value = param_optimised

            value_str, uncertainty_str = utils.plus_and_minus(value, uncertainty)

            param_string_values.append(value_str)
            uncertainty_string_values.append(uncertainty_str)

            if info['verbose']:
                print('  %d            %s +/- %s ' % (count+1, value_str, uncertainty_str))


        params = np.column_stack((params_optimised,
                                  params_1sd_error,
                                  param_string_values,
                                  uncertainty_string_values))


        table_param_values.append(params)

    plotting.plot_histogram_with_curves(bin_values,
                                        xy_histogram,
                                        symmetries,
                                        x_values,
                                        curve_values,
                                        info)



    for i, sym in enumerate(symmetries):
        sym_order.number = sym
        plotting.plot_rot_2d_geometry(sym, diameter_values[i], info)


    weights = stats.akaike_weights(aiccs)

    if info['verbose']:
        print('\nSymmetry\tAICc\t\tAkaike weight')

        for index, symmetry in enumerate(symmetries):
            print('{0:2d}\t\t{1:.2f} \t{2:.2}'.format(symmetry,
                                                      aiccs[index],
                                                      weights[index]))
                                                     

    reports.write_rot_2d_html_report(info, symmetries, aiccs, weights, table_param_values)








if __name__ == "__main__":
    #Tk().withdraw()
    main()
    #print('\nHit Enter to exit')
    #input()
    