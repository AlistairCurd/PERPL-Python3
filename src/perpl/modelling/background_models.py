"""
background_densities.py

Created on Wed Sep 18 09:55:09 2019

Functions for modelling relative positions among background localisations.

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

import numpy as np


def exponential_decay_1d_pair_corr(separation_values,
                                   amplitude,
                                   scale_param):
    """1D pair-correlation function for localisations where the probability of
    detection decreases exponentially along the relevant direction, starting
    from any location (the location parameter of the exponential p.d.f. does
    not affect the pair-correlation function).

    (Derived in Alistair's lab book, 18/09/2019)

    Args:
        separation_values (numpy array):
            The separations between localisations, measured along one
            direction, at which the values of the pair-correlation function
            will be generated.
        amplitude (float):
            Amplitude of the returned function.
        scale_param (float):
            Scale parameter (1 / decay rate) of the exponential detection
            p.d.f.. Also the decay rate of the returned function.

    Returns:
        rpd (numpy array):
            The 1D pair-correlation function, evaluated at separation_values.
    """
    rpd = amplitude * np.exp(-separation_values / scale_param)
    return rpd


def zero_to_constant_gradient(separation_values,
                              gradient,
                              crossover_distance,
                              variation_amplitude):
    """Function which will tend to zero for low separation values and to a
    constant gradient at high separation values, relative to a crossover
    distance.

    This can be used as an approximate differentiable background function when
    modelling XY-separations between Nup107 localisations, where we expect few
    localisations within the ring structures.

    (See notes in Alistair's lab book 19 Sep 2019.)

    Args:
        separation_values (numpy array):
            The separations between localisations at which the values of the
            pair-correlation function will be generated. In the case of the
            nuclear pore data, for instance, these are distances measured
            across the XY plane.

        gradient (float):
            The asymptotic gradient for
            separation_values >> crossover_distance.

        crossover_distance (float):
            Characteristic distance for the function.

            If separation_value << crossover_distance,
                result --> 0;

            If separation_value >> crossover_distance,
                result --> separation_value * gradient

        variation_amplitude (float):
            Determines the size of the variation as the function leaves the
            asymptotes.

    Returns:
        rpd (numpy array):
            The values of the background function, evaluated at
            separation_values.
    """
    x_values = separation_values
    amplitude = variation_amplitude
    crossover = crossover_distance

    # It is possible the powers of x below should be greater,
    # and could possibly be fitted,
    # but should be one greater in the numerator than in the denominator.
    rpd = gradient * ((amplitude * x_values ** 3)
                      /
                      (crossover + amplitude * x_values ** 2)
                      )

    return rpd


def pair_correlation_disk(separation_values, radius):
    """Pair-correlation function for points with uniform density
    within a disk.

    See journal and lab book.

    Args:
        separation_values (float):
            Separations between two points with the disk.
        radius (float):
            Radius of the disk.

    Returns:
        rpd (float):
            Relative position distribution, evaluated at the
            separation_values between points on the disk.
            Amplitude is scaled by 10e-7.
            Provides values for separation_values upto the diameter of the
            disk.
    """
    rpd = (2 * np.pi
           * separation_values
           * ((np.arccos(separation_values / (2 * radius))
               -
               np.arctan(separation_values
                         /
                         np.sqrt(4 * radius ** 2 - separation_values ** 2)
                         )
               + np.pi / 2
               )
              * radius ** 2
              - separation_values * np.sqrt(4 * radius ** 2
                                            - separation_values ** 2
                                            ) / 2
              ) * 10 ** -7
           )
    return rpd


def internalbg(separation_values, diameter, amp):
    """Background RPD term modeling isotripc localisations across a disk.
    See pair_correlation_disk().

    Args:
        separation_values (numpy array):
            Distances at which density values of the background component
            will be obtained.
        diameter (float):
            Diameter of the disk containing the background.
        amp (float)
            Amplitude of the background components.

    Returns:
        rpd (numpy array):
            The relative position density given by the background model
            at the separation_values between localisations.
    """
    # Generate the background distribution, for separation_values upto
    # the diameter of the disk.
    disk_background = (amp
                       * pair_correlation_disk(np.arange(np.round(diameter)),
                                               radius=diameter/2.
                                               )
                       )

    # Pad out to cover the rest of the required distance range with zeros,
    # for use with other model components.
    rpd = np.pad(disk_background,
                 (0, len(separation_values) - len(disk_background)), 'constant'
                 )
    return rpd
