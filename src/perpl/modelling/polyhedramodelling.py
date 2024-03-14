"""
polyhedramodelling.py

Library of non-statistical functions needed to create the polyhedral models.

Created on Thu Jun 20 16:32:11 2019

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
import modelling_general as model


def tri_prism_vertices(a, b):
    """Model triangular prism with triangular sides a and connecting sides b.
    End on.
    Args:
        a: Integer
        b: Integer
    """
    vv = np.zeros((6, 3))
    # vv[0] is [0, 0, 0]
    vv[1] = [a, 0, 0]
    vv[2] = [a / 2, a * np.sqrt(3) / 2, 0]
    vv[3] = [0, 0, b]
    vv[4] = vv[1] + vv[3]
    vv[5] = vv[2] + vv[3]
    return vv


def cuboid_vertices(a, b, c):
    """Model cuboid with sides a, b and c.
    Args:
        a: Integer
        b: Integer
        c: Integer
    """
    vv = np.zeros((8, 3))
    # vv[0] is [0, 0, 0]
    vv[1] = [a, 0, 0]
    vv[2] = [0, b, 0]
    vv[3] = [a, b, 0]
    vv[4] = [0, 0, c]
    vv[5] = vv[1] + vv[4]
    vv[6] = vv[2] + vv[4]
    vv[7] = vv[3] + vv[4]
    return vv


def tri_pyramid_vertices(a, b):
    """Model triangular pyramid with base sides a and height b."""
    vv = np.zeros((4, 3))
    # vv[0] is [0,0, 0]
    vv[1] = [a, 0, 0]
    vv[2] = [a / 2, a * np.sqrt(3) / 2, 0]
    vv[3] = [a / 2, a * np.sqrt(3) / 4, b]
    return vv


def get_1d_relpos_no_filter(xyz):
    """Store all relative positions in a numpy array. No need for a filter
    distance for searching.

    Args:
        xyz: numpy array of localisations with shape (N, 3),
        where N is the number of vertices.

    Returns:
        Numpy (N) array of Euclidean distances between vertices.
    """
    relpos = xyz - xyz[0]
    for i, loc in enumerate(xyz[1:len(xyz)]):
        relpos = np.append(relpos, xyz - loc, axis=0)

    # Remove [0., 0.] relative positions (self-referencing)
    relpos = relpos[np.any(relpos != 0., axis=1)]

    return(relpos)


def tri_prism_rpd(r, a, b, locamp, locprec, structamp, spread):
    """r are distances over which the model needs to be evaluated,
    e.g. 0.5, 1.5, 2.5, 3.5, ... nm
    """
    # Set up distances between tringular prism vertices
    verts = tri_prism_vertices(a, b)
    relpos = get_1d_relpos_no_filter(verts)
    dists = np.sqrt(relpos[:, 0] ** 2 + relpos[:, 1] ** 2 + relpos[:, 2] ** 2)
    # Initialise rpd array
    rpd = r * 0.
    # Fill with pair correlations between 3D Gaussian spreads at vertices
    for d in dists:
        rpd = rpd + structamp * model.pairwise_correlation_3d(r, d, spread)
    # Include single molecule localisation precision
    # This is approximated to isotropic
    rpd = rpd + locamp * model.pairwise_correlation_3d(r, 0., np.sqrt(2) * locprec)

    return rpd


def tri_prism_on_grid_rpd(r, a, b, locamp, locprec, structamp, spread,
                          gridspace, gridamp, gridspread):
    """r are distances over which the model needs to be evaluated,
    e.g. 0.5, 1.5, 2.5, 3.5, ... nm
    a: triangular side length
    b: connecting side length
    locamp: Amplitude of sinlge molecule localisation precision component
    locprec: Average single molecule localisation precision
    structamp: Amplitude of components reflecting the structural features
        of the complex.
    spread: Spread owing to unresolvable complexity or inhomogeneity between
        complexes.
    gridspace: The spacing of a square grid the complexes are found on.
    gridamp: Amplitude of components reflecting the nieghbouring complexes at
        nearby grid points.
    gridspread: Spread owing to different orientation at different grid points.
    """
    # Set up distances between tringular prism vertices
    verts = tri_prism_vertices(a, b)
    relpos = get_1d_relpos_no_filter(verts)
    dists = np.sqrt(relpos[:, 0] ** 2 + relpos[:, 1] ** 2 + relpos[:, 2] ** 2)
    # Initialise rpd array
    rpd = r * 0.
    # Fill with pair correlations between 3D Gaussian spreads at vertices
    for d in dists:
        rpd = rpd + structamp * model.pairwise_correlation_3d(r, d, spread)
    # Include single molecule localisation precision
    # This is approximated to isotropic
    rpd = rpd + locamp * model.pairwise_correlation_3d(r, 0., np.sqrt(2) * locprec)

    # Include neighbouring complexes on square grid:
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace, gridspread)
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace * np.sqrt(2),
                                                        gridspread)

    return rpd


def tri_prism_on_grid_substructure_rpd(r, a, b,
                                       locamp, locprec,
                                       structamp, spread,
                                       substructamp, substructspread,
                                       gridspace, gridamp, gridspread):
    """r are distances over which the model needs to be evaluated,
    e.g. 0.5, 1.5, 2.5, 3.5, ... nm
    a: triangular side length
    b: connecting side length
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
    """
    # Set up distances between tringular prism vertices
    verts = tri_prism_vertices(a, b)
    relpos = get_1d_relpos_no_filter(verts)
    dists = np.sqrt(relpos[:, 0] ** 2 + relpos[:, 1] ** 2 + relpos[:, 2] ** 2)
    # Initialise rpd array
    rpd = r * 0.
    # Fill with pair correlations between 3D Gaussian spreads at vertices
    for d in dists:
        rpd = rpd + structamp * model.pairwise_correlation_3d(r, d, spread)
    # Include single molecule localisation precision
    # This is approximated to isotropic
    rpd = rpd + locamp * model.pairwise_correlation_3d(r, 0., np.sqrt(2) * locprec)
    # Include substructure spread
    rpd = rpd + substructamp * model.pairwise_correlation_3d(r, 0.,
                                                             np.sqrt(2) * substructspread)

    # Include neighbouring complexes on square grid:
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace, gridspread)
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace * np.sqrt(2),
                                                        gridspread)

    return rpd


def tri_prism_on_grid_2disobg_substructure_rpd(r, a, b,
                                               locamp, locprec,
                                               structamp, spread,
                                               substructamp, substructspread,
                                               gridspace, gridamp, gridspread,
                                               bgslope):
    """r are distances over which the model needs to be evaluated,
    e.g. 0.5, 1.5, 2.5, 3.5, ... nm
    a: triangular side length
    b: connecting side length
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
    # Set up distances between tringular prism vertices
    verts = tri_prism_vertices(a, b)
    relpos = get_1d_relpos_no_filter(verts)
    dists = np.sqrt(relpos[:, 0] ** 2 + relpos[:, 1] ** 2 + relpos[:, 2] ** 2)
    # Initialise rpd array
    rpd = r * 0.
    # Fill with pair correlations between 3D Gaussian spreads at vertices
    for d in dists:
        rpd = rpd + structamp * model.pairwise_correlation_3d(r, d, spread)
    # Include single molecule localisation precision
    # This is approximated to isotropic
    rpd = rpd + locamp * model.pairwise_correlation_3d(r, 0., np.sqrt(2) * locprec)
    # Include substructure spread
    rpd = rpd + substructamp * model.pairwise_correlation_3d(r, 0.,
                                                             np.sqrt(2) * substructspread)

    # Include neighbouring complexes on square grid:
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace, gridspread)
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace * np.sqrt(2),
                                                        gridspread)
    # Include background
    rpd = rpd + r * bgslope

    return rpd


def tri_prism_on_grid_2disobg_substructure_rpd_vectorargs(
        input_vector):
    """Function to calculate the values given by
    tri_prism_on_grid_2disobg_substructure_rpd, but using a
    vector input for the parameters, so that the numdifftools package can be
    used to calculate partial derivatives for correct error propagation in the
    model.

    Args:
        input_vector (list or numpy array):
            A concatenation of:
                1. A distance at which density values of the model will be
                obtained (numpy array)

                2. The parameters used by
                tri_prism_on_grid_2disobg_substructure_rpd.
    Returns:
        rpd (numpy array):
            The relative position density given by the model at the input
            distances (called separation_values_1d).
    """
    (separation_values_1d,
     a, b,
     locamp, locprec,
     structamp, spread,
     substructamp, substructspread,
     gridspace, gridamp, gridspread,
     bgslope) = input_vector

    rpd = tri_prism_on_grid_2disobg_substructure_rpd(
            separation_values_1d,
            a, b,
            locamp, locprec,
            structamp, spread,
            substructamp, substructspread,
            gridspace, gridamp, gridspread,
            bgslope
            )

    return rpd


def tri_prism_on_grid_1_length(r, a, locamp, locprec, structamp, spread,
                                   gridspace, gridamp, gridspread):
    """r are distances over which the model needs to be evaluated,
    e.g. 0.5, 1.5, 2.5, 3.5, ... nm
    a: triangular side length
    b: connecting side length
    locamp: Amplitude of sinlge molecule localisation precision component
    locprec: Average single molecule localisation precision
    structamp: Amplitude of components reflecting the structural features
        of the complex.
    spread: Spread owing to unresolvable complexity or inhomogeneity between
        complexes.
    gridspace: The spacing of a square grid the complexes are found on.
    gridamp: Amplitude of components reflecting the nieghbouring complexes at
        nearby grid points.
    gridspread: Spread owing to different orientation at different grid points.
    """
    # Set up distances between tringular prism vertices
    verts = tri_prism_vertices(a, a)
    relpos = get_1d_relpos_no_filter(verts)
    dists = np.sqrt(relpos[:, 0] ** 2 + relpos[:, 1] ** 2 + relpos[:, 2] ** 2)
    # Initialise rpd array
    rpd = r * 0.
    # Fill with pair correlations between 3D Gaussian spreads at vertices
    for d in dists:
        rpd = rpd + structamp * model.pairwise_correlation_3d(r, d, spread)
    # Include single molecule localisation precision
    # This is approximated to isotropic
    rpd = rpd + locamp * model.pairwise_correlation_3d(r, 0., np.sqrt(2) * locprec)

    # Include neighbouring complexes on square grid:
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace, gridspread)
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace * np.sqrt(2),
                                                        gridspread)

    return rpd


def tri_prism_on_grid_1_length_2disobg(r,
                                       a,
                                       locamp, locprec,
                                       structamp, spread,
                                       gridspace, gridamp, gridspread,
                                       bgslope):
    """r are distances over which the model needs to be evaluated,
    e.g. 0.5, 1.5, 2.5, 3.5, ... nm
    a: triangular side length
    b: connecting side length
    locamp: Amplitude of sinlge molecule localisation precision component
    locprec: Average single molecule localisation precision
    structamp: Amplitude of components reflecting the structural features
        of the complex.
    spread: Spread owing to unresolvable complexity or inhomogeneity between
        complexes.
    gridspace: The spacing of a square grid the complexes are found on.
    gridamp: Amplitude of components reflecting the nieghbouring complexes at
        nearby grid points.
    gridspread: Spread owing to different orientation at different grid points.
    bgslope: Approximate background to 2D (relatively flat); this is the
        linear slope.
    """
    # Set up distances between tringular prism vertices
    verts = tri_prism_vertices(a, a)
    relpos = get_1d_relpos_no_filter(verts)
    dists = np.sqrt(relpos[:, 0] ** 2 + relpos[:, 1] ** 2 + relpos[:, 2] ** 2)
    # Initialise rpd array
    rpd = r * 0.
    # Fill with pair correlations between 3D Gaussian spreads at vertices
    for d in dists:
        rpd = rpd + structamp * model.pairwise_correlation_3d(r, d, spread)
    # Include single molecule localisation precision
    # This is approximated to isotropic
    rpd = rpd + locamp * model.pairwise_correlation_3d(r, 0., np.sqrt(2) * locprec)

    # Include neighbouring complexes on square grid:
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace, gridspread)
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace * np.sqrt(2),
                                                        gridspread)

    # Include background
    rpd = rpd + r * bgslope

    return rpd


def tri_prism_on_grid_1_length_substructure_rpd(r, a, locamp, locprec,
                                                structamp, spread,
                                                substructamp, substructspread,
                                                gridspace, gridamp, gridspread):
    """r are distances over which the model needs to be evaluated,
    e.g. 0.5, 1.5, 2.5, 3.5, ... nm
    a: triangular side length
    b: connecting side length
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
    """
    # Set up distances between tringular prism vertices
    verts = tri_prism_vertices(a, a)
    relpos = get_1d_relpos_no_filter(verts)
    dists = np.sqrt(relpos[:, 0] ** 2 + relpos[:, 1] ** 2 + relpos[:, 2] ** 2)
    # Initialise rpd array
    rpd = r * 0.
    # Fill with pair correlations between 3D Gaussian spreads at vertices
    for d in dists:
        rpd = rpd + structamp * model.pairwise_correlation_3d(r, d, spread)
    # Include single molecule localisation precision
    # This is approximated to isotropic
    rpd = rpd + locamp * model.pairwise_correlation_3d(r, 0., np.sqrt(2) * locprec)
    # Include substructure spread
    rpd = rpd + substructamp * model.pairwise_correlation_3d(r, 0.,
                                                             np.sqrt(2) * substructspread)
    # Include neighbouring complexes on square grid:
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace, gridspread)
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace * np.sqrt(2),
                                                        gridspread)

    return rpd


def tri_prism_on_grid_1_length_2disobg_substruct_rpd(r,
                                                     a,
                                                     locamp, locprec,
                                                     structamp, spread,
                                                     substructamp, substructspread,
                                                     gridspace, gridamp, gridspread,
                                                     bgslope):
    """r are distances over which the model needs to be evaluated,
    e.g. 0.5, 1.5, 2.5, 3.5, ... nm
    a: side length
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
    # Set up distances between tringular prism vertices
    verts = tri_prism_vertices(a, a)
    relpos = get_1d_relpos_no_filter(verts)
    dists = np.sqrt(relpos[:, 0] ** 2 + relpos[:, 1] ** 2 + relpos[:, 2] ** 2)
    # Initialise rpd array
    rpd = r * 0.
    # Fill with pair correlations between 3D Gaussian spreads at vertices
    for d in dists:
        rpd = rpd + structamp * model.pairwise_correlation_3d(r, d, spread)
    # Include single molecule localisation precision
    # This is approximated to isotropic
    rpd = rpd + locamp * model.pairwise_correlation_3d(r, 0., np.sqrt(2) * locprec)
    # Include substructure spread
    rpd = rpd + substructamp * model.pairwise_correlation_3d(r, 0.,
                                                             np.sqrt(2) * substructspread)

    # Include neighbouring complexes on square grid:
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace, gridspread)
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace * np.sqrt(2),
                                                        gridspread)

    # Include background
    rpd = rpd + r * bgslope

    return rpd


def tri_prism_on_grid_1_length_2disobg_substruct_rpd_vectorargs(input_vector):
    """Function to calculate the values given by
    tri_prism_on_grid_1_length_2disobg_substruct_rpd, but using a
    vector input for the parameters, so that the numdifftools package can be
    used to calculate partial derivatives for correct error propagation in the
    model.

    Args:
        input_vector (list or numpy array):
            A concatenation of:
                1. A distance at which density values of the model will be
                obtained (numpy array)

                2. The parameters used by
                tri_prism_on_grid_2disobg_substructure_rpd.
    Returns:
        rpd (numpy array):
            The relative position density given by the model at the input
            distances (called separation_values_1d).
    """
    (separation_values_1d,
     a,
     locamp, locprec,
     structamp, spread,
     substructamp, substructspread,
     gridspace, gridamp, gridspread,
     bgslope) = input_vector

    rpd = tri_prism_on_grid_1_length_2disobg_substruct_rpd(
            separation_values_1d,
            a,
            locamp, locprec,
            structamp, spread,
            substructamp, substructspread,
            gridspace, gridamp, gridspread,
            bgslope
            )

    return rpd


def tri_prism_on_grid_1_length_3disobg_substruct_rpd(r,
                                                     a,
                                                     locamp, locprec,
                                                     structamp, spread,
                                                     substructamp, substructspread,
                                                     gridspace, gridamp, gridspread,
                                                     bgscale):
    """r are distances over which the model needs to be evaluated,
    e.g. 0.5, 1.5, 2.5, 3.5, ... nm
    a: side length
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
    bgscale: 3D isotropic background.
    """
    # Set up distances between tringular prism vertices
    verts = tri_prism_vertices(a, a)
    relpos = get_1d_relpos_no_filter(verts)
    dists = np.sqrt(relpos[:, 0] ** 2 + relpos[:, 1] ** 2 + relpos[:, 2] ** 2)
    # Initialise rpd array
    rpd = r * 0.
    # Fill with pair correlations between 3D Gaussian spreads at vertices
    for d in dists:
        rpd = rpd + structamp * model.pairwise_correlation_3d(r, d, spread)
    # Include single molecule localisation precision
    # This is approximated to isotropic
    rpd = rpd + locamp * model.pairwise_correlation_3d(r, 0., np.sqrt(2) * locprec)
    # Include substructure spread
    rpd = rpd + substructamp * model.pairwise_correlation_3d(r, 0.,
                                                             np.sqrt(2) * substructspread)

    # Include neighbouring complexes on square grid:
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace, gridspread)
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace * np.sqrt(2),
                                                        gridspread)

    # Include background
    rpd = rpd + r * r * bgscale

    return rpd


def tri_prism_1_length_3disobg_substruct_rpd(r,
                                             a,
                                             locamp, locprec,
                                             structamp, spread,
                                             substructamp, substructspread,
                                             bgscale):
    """r are distances over which the model needs to be evaluated,
    e.g. 0.5, 1.5, 2.5, 3.5, ... nm
    a: side length
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
    bgscale: 3D isotropic background.
    """
    # Set up distances between tringular prism vertices
    verts = tri_prism_vertices(a, a)
    relpos = get_1d_relpos_no_filter(verts)
    dists = np.sqrt(relpos[:, 0] ** 2 + relpos[:, 1] ** 2 + relpos[:, 2] ** 2)
    # Initialise rpd array
    rpd = r * 0.
    # Fill with pair correlations between 3D Gaussian spreads at vertices
    for d in dists:
        rpd = rpd + structamp * model.pairwise_correlation_3d(r, d, spread)
    # Include single molecule localisation precision
    # This is approximated to isotropic
    rpd = rpd + locamp * model.pairwise_correlation_3d(r, 0., np.sqrt(2) * locprec)
    # Include substructure spread
    rpd = rpd + substructamp * model.pairwise_correlation_3d(r, 0.,
                                                             np.sqrt(2) * substructspread)

    # Include background
    rpd = rpd + r * r * bgscale

    return rpd


def tri_prism_on_grid_1_length_1_prec_rpd(r, a, locamp, locprec, structamp,
                                          gridspace, gridamp, gridspread):
    """r are distances over which the model needs to be evaluated,
    e.g. 0.5, 1.5, 2.5, 3.5, ... nm
    a: triangular side length
    b: connecting side length
    locamp: Amplitude of sinlge molecule localisation precision component
    locprec: Average single molecule localisation precision
    structamp: Amplitude of components reflecting the structural features
        of the complex.
    spread: Spread owing to unresolvable complexity or inhomogeneity between
        complexes.
    gridspace: The spacing of a square grid the complexes are found on.
    gridamp: Amplitude of components reflecting the nieghbouring complexes at
        nearby grid points.
    gridspread: Spread owing to different orientation at different grid points.
    """
    # Set up distances between tringular prism vertices
    verts = tri_prism_vertices(a, a)
    relpos = get_1d_relpos_no_filter(verts)
    dists = np.sqrt(relpos[:, 0] ** 2 + relpos[:, 1] ** 2 + relpos[:, 2] ** 2)
    # Initialise rpd array
    rpd = r * 0.
    # Fill with pair correlations between 3D Gaussian spreads at vertices
    for d in dists:
        rpd = rpd + structamp * model.pairwise_correlation_3d(r, d, np.sqrt(2) * locprec)
    # Include single molecule localisation precision
    # This is approximated to isotropic
    rpd = rpd + locamp * model.pairwise_correlation_3d(r, 0., np.sqrt(2) * locprec)

    # Include neighbouring complexes on square grid:
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace, gridspread)
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace * np.sqrt(2),
                                                        gridspread)

    return rpd


def tri_pyramid_on_grid_rpd(r, a, b, locamp, locprec, structamp, spread,
                            gridspace, gridamp, gridspread):
    """r are distances over which the model needs to be evaluated,
    e.g. 0.5, 1.5, 2.5, 3.5, ... nm
    a: triangular side length
    b: connecting side length
    locamp: Amplitude of sinlge molecule localisation precision component
    locprec: Average single molecule localisation precision
    structamp: Amplitude of components reflecting the structural features
        of the complex.
    spread: Spread owing to unresolvable complexity or inhomogeneity between
        complexes.
    gridspace: The spacing of a square grid the complexes are found on.
    gridamp: Amplitude of components reflecting the nieghbouring complexes at
        nearby grid points.
    gridspread: Spread owing to different orientation at different grid points.
    """
    # Set up distances between tringular prism vertices
    verts = tri_pyramid_vertices(a, b)
    relpos = get_1d_relpos_no_filter(verts)
    dists = np.sqrt(relpos[:, 0] ** 2 + relpos[:, 1] ** 2 + relpos[:, 2] ** 2)
    # Initialise rpd array
    rpd = r * 0.
    # Fill with pair correlations between 3D Gaussian spreads at vertices
    for d in dists:
        rpd = rpd + structamp * model.pairwise_correlation_3d(r, d, spread)
    # Include single molecule localisation precision
    # This is approximated to isotropic
    rpd = rpd + locamp * model.pairwise_correlation_3d(r, 0., np.sqrt(2) * locprec)

    # Include neighbouring complexes on square grid:
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace, gridspread)
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace * np.sqrt(2),
                                                        gridspread)

    return rpd


def tri_pyramid_on_grid_2disobg_substructure_rpd(r,
                                                 a, b,
                                                 locamp, locprec,
                                                 structamp, spread,
                                                 substructamp, substructspread,
                                                 gridspace, gridamp, gridspread,
                                                 bgslope):
    """r are distances over which the model needs to be evaluated,
    e.g. 0.5, 1.5, 2.5, 3.5, ... nm
    a: triangular side length
    b: connecting side length
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
    # Set up distances between tringular prism vertices
    verts = tri_pyramid_vertices(a, b)
    relpos = get_1d_relpos_no_filter(verts)
    dists = np.sqrt(relpos[:, 0] ** 2 + relpos[:, 1] ** 2 + relpos[:, 2] ** 2)
    # Initialise rpd array
    rpd = r * 0.
    # Fill with pair correlations between 3D Gaussian spreads at vertices
    for d in dists:
        rpd = rpd + structamp * model.pairwise_correlation_3d(r, d, spread)
    # Include single molecule localisation precision
    # This is approximated to isotropic
    rpd = rpd + locamp * model.pairwise_correlation_3d(r, 0., np.sqrt(2) * locprec)

    # Include substructure spread
    rpd = rpd + substructamp * model.pairwise_correlation_3d(r, 0.,
                                                             np.sqrt(2) * substructspread)

    # Include neighbouring complexes on square grid:
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace, gridspread)
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace * np.sqrt(2),
                                                        gridspread)

    # Include background
    rpd = rpd + r * bgslope

    return rpd


def cuboid_on_grid_rpd(r, a, b, c, locamp, locprec, structamp, spread,
                       gridspace, gridamp, gridspread):
    """r are distances over which the model needs to be evaluated,
    e.g. 0.5, 1.5, 2.5, 3.5, ... nm
    a, b, c: cuboid side lengths
    locamp: Amplitude of sinlge molecule localisation precision component
    locprec: Average single molecule localisation precision
    structamp: Amplitude of components reflecting the structural features
        of the complex.
    spread: Spread owing to unresolvable complexity or inhomogeneity between
        complexes.
    gridspace: The spacing of a square grid the complexes are found on.
    gridamp: Amplitude of components reflecting the nieghbouring complexes at
        nearby grid points.
    gridspread: Spread owing to different orientation at different grid points.
    """
    # Set up distances between tringular prism vertices
    verts = cuboid_vertices(a, b, c)
    relpos = get_1d_relpos_no_filter(verts)
    dists = np.sqrt(relpos[:, 0] ** 2 + relpos[:, 1] ** 2 + relpos[:, 2] ** 2)
    # Initialise rpd array
    rpd = r * 0.
    # Fill with pair correlations between 3D Gaussian spreads at vertices
    for d in dists:
        rpd = rpd + structamp * model.pairwise_correlation_3d(r, d, spread)
    # Include single molecule localisation precision
    # This is approximated to isotropic
    rpd = rpd + locamp * model.pairwise_correlation_3d(r, 0., np.sqrt(2) * locprec)

    # Include neighbouring complexes on square grid:
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace, gridspread)
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace * np.sqrt(2),
                                                        gridspread)

    return rpd


def cuboid_on_grid_substructure_rpd(r,
                                    a, b, c,
                                    locamp, locprec,
                                    structamp, spread,
                                    substructamp, substructspread,
                                    gridspace, gridamp, gridspread):
    """r are distances over which the model needs to be evaluated,
    e.g. 0.5, 1.5, 2.5, 3.5, ... nm
    a, b, c: cuboid side lengths
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
    """
    # Set up distances between tringular prism vertices
    verts = cuboid_vertices(a, b, c)
    relpos = get_1d_relpos_no_filter(verts)
    dists = np.sqrt(relpos[:, 0] ** 2 + relpos[:, 1] ** 2 + relpos[:, 2] ** 2)
    # Initialise rpd array
    rpd = r * 0.
    # Fill with pair correlations between 3D Gaussian spreads at vertices
    for d in dists:
        rpd = rpd + structamp * model.pairwise_correlation_3d(r, d, spread)
    # Include single molecule localisation precision
    # This is approximated to isotropic
    rpd = rpd + locamp * model.pairwise_correlation_3d(r, 0., np.sqrt(2) * locprec)
    # Include substructure spread
    rpd = rpd + substructamp * model.pairwise_correlation_3d(r, 0.,
                                                             np.sqrt(2) * substructspread)

    # Include neighbouring complexes on square grid:
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace, gridspread)
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace * np.sqrt(2),
                                                        gridspread)

    return rpd


def cuboid_on_grid_2disobg_substructure_rpd(r,
                                            a, b, c,
                                            locamp, locprec,
                                            structamp, spread,
                                            substructamp, substructspread,
                                            gridspace, gridamp, gridspread,
                                            bgslope):
    """r are distances over which the model needs to be evaluated,
    e.g. 0.5, 1.5, 2.5, 3.5, ... nm
    a, b, c: cuboid side lengths
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
    # Set up distances between tringular prism vertices
    verts = cuboid_vertices(a, b, c)
    relpos = get_1d_relpos_no_filter(verts)
    dists = np.sqrt(relpos[:, 0] ** 2 + relpos[:, 1] ** 2 + relpos[:, 2] ** 2)
    # Initialise rpd array
    rpd = r * 0.
    # Fill with pair correlations between 3D Gaussian spreads at vertices
    for d in dists:
        rpd = rpd + structamp * model.pairwise_correlation_3d(r, d, spread)
    # Include single molecule localisation precision
    # This is approximated to isotropic
    rpd = rpd + locamp * model.pairwise_correlation_3d(r, 0., np.sqrt(2) * locprec)
    # Include substructure spread
    rpd = rpd + substructamp * model.pairwise_correlation_3d(r, 0.,
                                                             np.sqrt(2) * substructspread)

    # Include neighbouring complexes on square grid:
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace, gridspread)
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace * np.sqrt(2),
                                                        gridspread)
    # Include background
    rpd = rpd + r * bgslope

    return rpd


def cuboid_on_grid_square_base_rpd(r, a, b, locamp, locprec, structamp, spread,
                                   gridspace, gridamp, gridspread):
    """r are distances over which the model needs to be evaluated,
    e.g. 0.5, 1.5, 2.5, 3.5, ... nm
    a, b, c: cuboid side lengths
    locamp: Amplitude of sinlge molecule localisation precision component
    locprec: Average single molecule localisation precision
    structamp: Amplitude of components reflecting the structural features
        of the complex.
    spread: Spread owing to unresolvable complexity or inhomogeneity between
        complexes.
    gridspace: The spacing of a square grid the complexes are found on.
    gridamp: Amplitude of components reflecting the nieghbouring complexes at
        nearby grid points.
    gridspread: Spread owing to different orientation at different grid points.
    """
    # Set up distances between tringular prism vertices
    verts = cuboid_vertices(a, a, b)
    relpos = get_1d_relpos_no_filter(verts)
    dists = np.sqrt(relpos[:, 0] ** 2 + relpos[:, 1] ** 2 + relpos[:, 2] ** 2)
    # Initialise rpd array
    rpd = r * 0.
    # Fill with pair correlations between 3D Gaussian spreads at vertices
    for d in dists:
        rpd = rpd + structamp * model.pairwise_correlation_3d(r, d, spread)
    # Include single molecule localisation precision
    # This is approximated to isotropic
    rpd = rpd + locamp * model.pairwise_correlation_3d(r, 0., np.sqrt(2) * locprec)

    # Include neighbouring complexes on square grid:
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace, gridspread)
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace * np.sqrt(2),
                                                        gridspread)

    return rpd


def cube_on_grid_rpd(r, a, locamp, locprec, structamp, spread,
                     gridspace, gridamp, gridspread):
    """r are distances over which the model needs to be evaluated,
    e.g. 0.5, 1.5, 2.5, 3.5, ... nm
    a, b, c: cuboid side lengths
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
    """
    # Set up distances between tringular prism vertices
    verts = cuboid_vertices(a, a, a)
    relpos = get_1d_relpos_no_filter(verts)
    dists = np.sqrt(relpos[:, 0] ** 2 + relpos[:, 1] ** 2 + relpos[:, 2] ** 2)
    # Initialise rpd array
    rpd = r * 0.
    # Fill with pair correlations between 3D Gaussian spreads at vertices
    for d in dists:
        rpd = rpd + structamp * model.pairwise_correlation_3d(r, d, spread)
    # Include single molecule localisation precision
    # This is approximated to isotropic
    rpd = rpd + locamp * model.pairwise_correlation_3d(r, 0., np.sqrt(2) * locprec)

    # Include neighbouring complexes on square grid:
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace, gridspread)
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace * np.sqrt(2),
                                                        gridspread)

    return rpd


def cube_on_grid_substructure_rpd(r, a, locamp, locprec, structamp, spread,
                                  substructamp, substructspread,
                                  gridspace, gridamp, gridspread):
    """r are distances over which the model needs to be evaluated,
    e.g. 0.5, 1.5, 2.5, 3.5, ... nm
    a, b, c: cuboid side lengths
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
    """
    # Set up distances between tringular prism vertices
    verts = cuboid_vertices(a, a, a)
    relpos = get_1d_relpos_no_filter(verts)
    dists = np.sqrt(relpos[:, 0] ** 2 + relpos[:, 1] ** 2 + relpos[:, 2] ** 2)
    # Initialise rpd array
    rpd = r * 0.
    # Fill with pair correlations between 3D Gaussian spreads at vertices
    for d in dists:
        rpd = rpd + structamp * model.pairwise_correlation_3d(r, d, spread)
    # Include single molecule localisation precision
    # This is approximated to isotropic
    rpd = rpd + locamp * model.pairwise_correlation_3d(r, 0., np.sqrt(2) * locprec)
    # Include substructure spread                                                    np.sqrt(2) * substructspread)
    rpd = rpd + substructamp * model.pairwise_correlation_3d(r, 0.,
                                                             np.sqrt(2) * substructspread)
    # Include neighbouring complexes on square grid:
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace, gridspread)
    rpd = rpd + gridamp \
            * model.pairwise_correlation_2d(r, gridspace * np.sqrt(2), gridspread)

    return rpd


def cube_on_grid_2disobg_substructure_rpd(r, a, locamp, locprec, structamp, spread,
                                          substructamp, substructspread,
                                          gridspace, gridamp, gridspread,
                                          bgslope):
    """r are distances over which the model needs to be evaluated,
    e.g. 0.5, 1.5, 2.5, 3.5, ... nm
    a, b, c: cuboid side lengths
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
    # Set up distances between tringular prism vertices
    verts = cuboid_vertices(a, a, a)
    relpos = get_1d_relpos_no_filter(verts)
    dists = np.sqrt(relpos[:, 0] ** 2 + relpos[:, 1] ** 2 + relpos[:, 2] ** 2)
    # Initialise rpd array
    rpd = r * 0.
    # Fill with pair correlations between 3D Gaussian spreads at vertices
    for d in dists:
        rpd = rpd + structamp * model.pairwise_correlation_3d(r, d, spread)
    # Include single molecule localisation precision
    # This is approximated to isotropic
    rpd = rpd + locamp * model.pairwise_correlation_3d(r, 0., np.sqrt(2) * locprec)
    # Include substructure spread
    rpd = rpd + substructamp \
            * model.pairwise_correlation_3d(r, 0., np.sqrt(2) * substructspread)

    # Include neighbouring complexes on square grid:
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace, gridspread)
    rpd = rpd + gridamp * model.pairwise_correlation_2d(r, gridspace * np.sqrt(2),
                                                        gridspread)
    # Include background
    rpd = rpd + r * bgslope

    return rpd
