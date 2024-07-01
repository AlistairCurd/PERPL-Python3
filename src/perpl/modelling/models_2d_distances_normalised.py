"""
Created on Tue Jun 9 2020

Functions for creating models for distances in 2D space, normalised for
increasing search radius.

Alistair Curd
University of Leeds
9 June 2020

---
Copyright 2020 Peckham Lab

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
import perpl.modelling.modelling_general as model

"""Various models for the relative position distribution (hence
*returns* **rpd**).

Args are from:
    x_values (float or numpy array):
        Distances at which the model will be evaluated.
    dist_int (float):
        Characteristic distance between two localisations in a 2D distribution.
    broadening (float):
        Sigma in the function used for distances between Gaussian
        distributions.
    amp or amp_int (float):
        Amplitude of the peak representing the characteristic distance
    locprec (float):
        Localisation precision/broadening for repeated localisations of the
        same or unresolvable molecule(s).
    ampreplocs (float):
        Amplitude for the repeated localisations peak.
    bgoffset (float):
        Magnitude of a uniform backgroun level of distances.
"""

def onepeakplusreps_normalised_flat_bg(x_values, dist_1, broadening,
                            amp,
                            locprec, ampreplocs,
                            bgoffset):
    """A model for the normalised distance distribution for 2D localisations with
    a characteristic distance between them.
    
    See docstring below imports.
    """
    # Background
    rpd = 0. * x_values + bgoffset
    # Characteristic distance peak
    peak_1 = amp * model.pairwise_correlation_2d(x_values, dist_1, broadening)
    peak_1 = peak_1 / x_values # Normalisation
    rpd = rpd + peak_1
    # Repeated localisations of the same/unresolvable molecule(s)
    reps = ampreplocs * model.pairwise_correlation_2d(x_values, 0., np.sqrt(2) * locprec)
    rpd = rpd + reps

    return rpd


def onepeakplusreps_normalised_flat_bg_vectorinput(vector_input):
    """Version of onepeakplusreps_normalised_flat_bg to take vector input
    so that numdifftools can calculate the Jacobian, and we can get confidence
    intervals.
    """
    (x_values, rep, broadening,
     amp,
     locprec, ampreplocs,
     bgoffset) = vector_input
    rpd = onepeakplusreps_normalised_flat_bg(x_values, rep, broadening,
                                  amp,
                                  locprec, ampreplocs,
                                  bgoffset)
    return rpd


def twopeaks_normalised_flat_bg(x_values,
                                dist_1, dist_2, broadening,
                                amp_1, amp_2,
                                bgoffset):
    """A model for the normalised distance distribution for 2D localisations with
    a characteristic distance between them.
    
    See docstring below imports.
    """
    # Background
    rpd = 0. * x_values + bgoffset
    # Characteristic distance peaks
    peak_1 = (amp_1 * model.pairwise_correlation_2d(x_values, dist_1, broadening)
              / x_values # Normalisation
              )
    peak_2 = (amp_2 * model.pairwise_correlation_2d(x_values, dist_2, broadening)
              / x_values # Normalisation
              )
    rpd = rpd + peak_1 + peak_2

    return rpd


def twopeaks_normalised_flat_bg_vectorinput(vector_input):
    (x_values,
     dist_1, dist_2, broadening,
     amp_1, amp_2,
     bgoffset) = vector_input
    rpd = twopeaks_normalised_flat_bg(x_values,
                                      dist_1, dist_2, broadening,
                                      amp_1, amp_2,
                                      bgoffset)
    return rpd
