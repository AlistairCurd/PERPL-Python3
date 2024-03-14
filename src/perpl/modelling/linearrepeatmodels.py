"""
linearrepeatmodels.py

Library of functions for regularly repeating linear patterns in models of
cellular structures.

Alistair Curd
University of Leeds
August 2019

Software Engineering practices applied

Joanna Leng (an EPSRC funded Research Software Engineering Fellow (EP/R025819/1)
University of Leeds
August 2019

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
from modelling_general import pairwise_correlation_1d


def noslope(x_values, mean):
    """Calculates the rpd (relative position density) which is a meansures
    the variation in a set of data.
    Args:
        x_values: A numpy array
        mean: A single float or int value
    Return:
        rpd: A numpy array
    """
    rpd = mean + 0. * x_values
    return rpd


def linear_fit(x_values, slope, offset):
    rpd = offset + slope * x_values
    return rpd


def linear_fit_vector_args(vector_input):
    (x_values,
     bgslope, bgoffset) = vector_input
    y = linear_fit(x_values,
                   bgslope, bgoffset
                   )
    return y


def justreplocs(x_values,
                locprec, ampreplocs,
                bgslope, bgoffset
                ):
    rpd = bgoffset + bgslope * x_values  # Linear background

    rpd = rpd + ampreplocs * model.pairwise_correlation_1d(x_values, 0., np.sqrt(2) * locprec)

    return rpd


def justreplocs_vectorinput(vector_input):
    (x_values,
     locprec, ampreplocs,
     bgslope, bgoffset) = vector_input
    y = justreplocs(x_values,
                   locprec, ampreplocs,
                   bgslope, bgoffset
                   )
    return y


def onepeakplusreps(x_values, rep, broadening,
                    a,
                    locprec, ampreplocs,
                    bgslope, bgoffset
                    ):
    rpd = bgoffset + bgslope * x_values  # Linear background
    amps = [a]
    for i, amp in enumerate(amps):
        rpd = rpd + amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)

    reps = ampreplocs * model.pairwise_correlation_1d(x_values, 0., np.sqrt(2) * locprec)
    rpd = rpd + reps

    return rpd


def onepeakplusreps_vectorinput(vector_input):
    (x_values, rep, broadening,
     a,
     locprec, ampreplocs,
     bgslope, bgoffset) = vector_input
    rpd = onepeakplusreps(x_values, rep, broadening,
                          a,
                          locprec, ampreplocs,
                          bgslope, bgoffset)
    return rpd


def onepeakplusreps_no_bg(x_values, rep, broadening,
                          a,
                          locprec, ampreplocs):
    rpd = 0. * x_values
    amps = [a]
    for i, amp in enumerate(amps):
        rpd = rpd + amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)

    reps = ampreplocs * model.pairwise_correlation_1d(x_values, 0., np.sqrt(2) * locprec)
    rpd = rpd + reps

    return rpd


def onepeakplusreps_no_bg_vectorinput(vector_input):
    (x_values, rep, broadening,
     a,
     locprec, ampreplocs) = vector_input
    rpd = onepeakplusreps_no_bg(x_values, rep, broadening,
                                a,
                                locprec, ampreplocs)
    return rpd


def onepeakplusreps_flat_bg(x_values, rep, broadening,
                          a,
                          locprec, ampreplocs,
                          bgoffset):
    rpd = 0. * x_values + bgoffset
    amps = [a]
    for i, amp in enumerate(amps):
        rpd = rpd + amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)

    reps = ampreplocs * model.pairwise_correlation_1d(x_values, 0., np.sqrt(2) * locprec)
    rpd = rpd + reps

    return rpd


def onepeakplusreps_flat_bg_vectorinput(vector_input):
    (x_values, rep, broadening,
     a,
     locprec, ampreplocs,
     bgoffset) = vector_input
    rpd = onepeakplusreps_flat_bg(x_values, rep, broadening,
                                a,
                                locprec, ampreplocs,
                                bgoffset)
    return rpd


def onepeaknoreps(x_values, rep, broadening,
                  a,
                  bgslope, bgoffset
                  ):
    rpd = bgoffset + bgslope * x_values  # Linear background
    amps = [a]
    for i, amp in enumerate(amps):
        rpd = rpd + amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)

    return rpd


def onepeaknoreps_vectorinput(vector_input):
    (x_values, rep, broadening,
     a,
     bgslope, bgoffset) = vector_input
    rpd = onepeaknoreps(x_values, rep, broadening,
                      a,
                      bgslope, bgoffset
                      )
    return rpd


def linrepplusreps(x_values, rep, broadening,
                   a, b, c, d,  # e, f,
                   locprec, ampreplocs,
                   bgslope, bgoffset
                   ):
    rpd = np.zeros(len(x_values))
    amps = [a, b, c, d]  # , e, f]
    for i, amp in enumerate(amps):
        if (i + 1) * rep < broadening * 10:
            rpd = rpd + amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)
        else:
            rpd = rpd + amp * model.gauss1d(x_values, (i + 1) * rep, broadening)

    background = bgoffset + bgslope * x_values
    rpd = rpd + background
    reps = ampreplocs * x_values / (2 * locprec ** 2) * np.exp(
                                               -(x_values ** 2) / (4 * locprec ** 2))
    rpd = rpd + reps

    return rpd


def linrepplusreps3(x_values, rep, broadening,
                    a, b, c,
                    locprec, ampreplocs,
                    bgslope, bgoffset
                    ):
    rpd = bgoffset + bgslope * x_values  # Linear background
    amps = [a, b, c]
    for i, amp in enumerate(amps):
        rpd = rpd + amp * model.pairwise_correlation_1d(x_values,
                                                        (i + 1) * rep,
                                                        broadening)
    # Repeated localisations
    rpd = rpd + ampreplocs * model.pairwise_correlation_1d(x_values,
                                                           0.,
                                                           np.sqrt(2) * locprec)

    return rpd


def offsetlinrepplusreps3flatbg(x_values, rep, broadening,
                                a, b, c,
                                peakoffset,
                                locprec, ampreplocs,
                                bgoffset
                                ):
    rpd = bgoffset + 0. * x_values  # Linear background
    amp = [a, b, c]
    for i in range(3):
        rpd = (rpd
               + amp[i] * model.pairwise_correlation_1d(x_values,
                                           peakoffset + (i + 1) * rep,
                                           broadening)
               )

    rpd = rpd + ampreplocs * model.pairwise_correlation_1d(x_values, 0., np.sqrt(2) * locprec)

    return rpd

def linrepplusreps3fixedpeaks(x_values, rep, broadening,
                   amp,
                   locprec, ampreplocs,
                   bgslope, bgoffset
                   ):
    rpd = bgoffset + bgslope * x_values # Linear background
    for i in range(3):
        rpd = rpd + amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)    
    
    reps = ampreplocs * x_values / (2 * locprec ** 2) * np.exp(
                                               -(x_values ** 2) / (4 * locprec ** 2))
    rpd = rpd + reps
    #reps2 = ampreplocs2 * x_values / (2 * locprec2 ** 2) * np.exp(
    #                                           -(x_values ** 2) / (4 * locprec2 ** 2))
    #rpd = rpd + reps2
    return rpd

def linrepplusreps3fixedpeaksflatbg(x_values, rep, broadening,
                   amp,
                   locprec, ampreplocs,
                   bgoffset
                   ):
    rpd = bgoffset + 0. * x_values # Linear background
    for i in range(3):
        rpd = rpd + amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)    
    
    rpd = rpd + ampreplocs * model.pairwise_correlation_1d(x_values, 0., np.sqrt(2) * locprec) # rep locs
    return rpd

def linrepplusreps3fixedpeakratio(x_values, rep, broadening,
                   amp,
                   locprec, ampreplocs,
                   bgslope, bgoffset
                   ):
    rpd = bgoffset + bgslope * x_values # Linear background
    for i in range(3):
        rpd = rpd + (1. - i / 3.) * amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)    
    
    rpd = rpd + ampreplocs * model.pairwise_correlation_1d(x_values, 0., np.sqrt(2) * locprec) # rep locs

    return rpd


def linrepplusreps3fixedpeakratiovectorinput(vectorin):
    (x_values, rep, broadening,
     amp,
     locprec, ampreplocs,
     bgslope, bgoffset) = vectorin
    y = linrepplusreps3fixedpeakratio(x_values, rep, broadening,
                                      amp,
                                      locprec, ampreplocs,
                                      bgslope, bgoffset
                                      )
    return y


def linrepplusreps3fixedpeakamp(x_values, rep, broadening,
                   amp,
                   locprec, ampreplocs,
                   bgslope, bgoffset
                   ):
    rpd = bgoffset + bgslope * x_values # Linear background
    for i in range(3):
        rpd = rpd +  amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)    
    
    reps = ampreplocs * x_values / (2 * locprec ** 2) * np.exp(
                                               -(x_values ** 2) / (4 * locprec ** 2))
    rpd = rpd + reps
    #reps2 = ampreplocs2 * x_values / (2 * locprec2 ** 2) * np.exp(
    #                                           -(x_values ** 2) / (4 * locprec2 ** 2))
    #rpd = rpd + reps2
    return rpd

def offsetlinrepplusreps3fixedpeakampflatbg(x_values, rep, broadening,
                   amp,
                   peakoffset,
                   locprec, ampreplocs,
                   bgoffset
                   ):
    rpd = bgoffset + 0. * x_values # Linear background
    for i in range(3):
        rpd = rpd +  amp * model.pairwise_correlation_1d(
                                    x_values, peakoffset + (i + 1) * rep, broadening)    
    
    rpd = rpd + ampreplocs * model.pairwise_correlation_1d(x_values, 0., np.sqrt(2) * locprec)

    return rpd

def offsetlinrep3fixedpeakamp(x_values, rep, broadening,
                   amp,
                   peakoffset,
                   bgoffset
                   ):
    """Offset peak plus 3 more at repeat distance.
    """
    rpd = bgoffset + 0. * x_values # Linear background
    for i in range(4):
        rpd = rpd +  amp * model.pairwise_correlation_1d(
                                    x_values, peakoffset + i * rep, broadening)    
    
    #reps = ampreplocs * x_values / (2 * locprec ** 2) * np.exp(
    #                                           -(x_values ** 2) / (4 * locprec ** 2))
    #rpd = rpd + reps
    #reps2 = ampreplocs2 * x_values / (2 * locprec2 ** 2) * np.exp(
    #                                           -(x_values ** 2) / (4 * locprec2 ** 2))
    #rpd = rpd + reps2
    return rpd

def offsetlinrep3fixedpeakampnobg(x_values, rep, broadening,
                   amp,
                   peakoffset
                   ):
    """Offset peak plus 3 more at repeat distance.
    """
    rpd = 0. * x_values # Linear background
    for i in range(4):
        rpd = rpd +  amp * model.pairwise_correlation_1d(
                                    x_values, peakoffset + i * rep, broadening)    
    
    #reps = ampreplocs * x_values / (2 * locprec ** 2) * np.exp(
    #                                           -(x_values ** 2) / (4 * locprec ** 2))
    #rpd = rpd + reps
    #reps2 = ampreplocs2 * x_values / (2 * locprec2 ** 2) * np.exp(
    #                                           -(x_values ** 2) / (4 * locprec2 ** 2))
    #rpd = rpd + reps2
    return rpd

def linrepbroadsplusreps3fixedpeakratio(x_values, rep,
                                  b1, b2, b3,
                   amp,
                   locprec, ampreplocs,
                   bgslope, bgoffset
                   ):
    rpd = bgoffset + bgslope * x_values # Linear background
    b = [b1, b2, b3]
    for i in range(3):
        rpd = rpd + (1. - i / 3.) * amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, b[i])    
    
    reps = ampreplocs * x_values / (2 * locprec ** 2) * np.exp(
                                               -(x_values ** 2) / (4 * locprec ** 2))
    rpd = rpd + reps
    #reps2 = ampreplocs2 * x_values / (2 * locprec2 ** 2) * np.exp(
    #                                           -(x_values ** 2) / (4 * locprec2 ** 2))
    #rpd = rpd + reps2
    return rpd

def linrep19plusreps3fixedpeakratio(x_values, broadening,
                   amp,
                   locprec, ampreplocs,
                   bgslope, bgoffset
                   ):
    rpd = bgoffset + bgslope * x_values # Linear background
    rep = 19.2
    for i in range(3):
        rpd = rpd + (1. - i / 3.) * amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)    
    
    reps = ampreplocs * x_values / (2 * locprec ** 2) * np.exp(
                                               -(x_values ** 2) / (4 * locprec ** 2))
    rpd = rpd + reps
    #reps2 = ampreplocs2 * x_values / (2 * locprec2 ** 2) * np.exp(
    #                                           -(x_values ** 2) / (4 * locprec2 ** 2))
    #rpd = rpd + reps2
    return rpd

def linrepnoreps3fixedpeakratio(x_values, rep, broadening,
                   amp,
                   bgslope, bgoffset
                   ):
    rpd = bgoffset + bgslope * x_values # Linear background
    for i in range(3):
        rpd = rpd + (1. - i / 3.) * amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)    

    #reps2 = ampreplocs2 * x_values / (2 * locprec2 ** 2) * np.exp(
    #                                           -(x_values ** 2) / (4 * locprec2 ** 2))
    #rpd = rpd + reps2
    return rpd

def linrep19noreps3fixedpeakratio(x_values, broadening,
                   amp,
                   bgslope, bgoffset
                   ):
    rpd = bgoffset + bgslope * x_values # Linear background
    rep = 19.2
    for i in range(3):
        rpd = rpd + (1. - i / 3.) * amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)    

    #reps2 = ampreplocs2 * x_values / (2 * locprec2 ** 2) * np.exp(
    #                                           -(x_values ** 2) / (4 * locprec2 ** 2))
    #rpd = rpd + reps2
    return rpd

def linrepnoreps3fixedpeaks(x_values, rep, broadening,
                   amp,
                   bgslope, bgoffset
                   ):
    rpd = bgoffset + bgslope * x_values # Linear background
    for i in range(3):
        rpd = rpd + amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)    
 
    #reps2 = ampreplocs2 * x_values / (2 * locprec2 ** 2) * np.exp(
    #                                           -(x_values ** 2) / (4 * locprec2 ** 2))
    #rpd = rpd + reps2
    return rpd

def linrepplusreps4(x_values, rep, broadening,
                   a, b, c, d,
                   locprec, ampreplocs,
                   bgslope, bgoffset
                   ):
    rpd = bgoffset + bgslope * x_values # Linear background
    amps = [a, b, c, d]
    for i, amp in enumerate(amps):
        rpd = rpd + amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)    
    rpd = rpd + ampreplocs * model.pairwise_correlation_1d(x_values, 0., np.sqrt(2) * locprec)

    return rpd

def linrepplusreps4fixedpeakratio(x_values, rep, broadening,
                   amp,
                   locprec, ampreplocs,
                   bgslope, bgoffset
                   ):
    rpd = bgoffset + bgslope * x_values # Linear background
    for i in range(4):
        rpd = rpd + (1. - i / 4.) * amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)
    
    rpd = rpd + ampreplocs * model.pairwise_correlation_1d(x_values, 0., np.sqrt(2) * locprec)

    return rpd


def linrepplusreps4fixedpeakratiovectorinput(vectorin):
    (x_values, rep, broadening,
     amp,
     locprec, ampreplocs,
     bgslope, bgoffset) = vectorin
    y = linrepplusreps4fixedpeakratio(x_values, rep, broadening,
                                      amp,
                                      locprec, ampreplocs,
                                      bgslope, bgoffset
                                      )
    return y


def linrepplusreps4fixedpeakrationobg(x_values, rep, broadening,
                   amp,
                   locprec, ampreplocs,
                   ):
    rpd = 0. * x_values # Linear background
    for i in range(4):
        rpd = rpd + (1. - i / 4.) * amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)
    
    rpd = rpd + ampreplocs * model.pairwise_correlation_1d(x_values, 0., np.sqrt(2) * locprec)

    return rpd

def linrepbroadsplusreps4fixedpeakratio(x_values, rep,
                    b1, b2, b3, b4,
                   amp,
                   locprec, ampreplocs,
                   bgslope, bgoffset
                   ):
    rpd = bgoffset + bgslope * x_values # Linear background
    b = [b1, b2, b3, b4]
    for i in range(4):
        rpd = rpd + (1. - i / 4.) * amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, b[i])
    
    reps = ampreplocs * x_values / (2 * locprec ** 2) * np.exp(
                                               -(x_values ** 2) / (4 * locprec ** 2))
    rpd = rpd + reps

    #reps2 = ampreplocs2 * x_values / (2 * locprec2 ** 2) * np.exp(
    #                                           -(x_values ** 2) / (4 * locprec2 ** 2))
    #rpd = rpd + reps2
    return rpd

def linrep19plusreps4fixedpeakratio(x_values, broadening,
                   amp,
                   locprec, ampreplocs,
                   bgslope, bgoffset
                   ):
    rpd = bgoffset + bgslope * x_values # Linear background
    rep = 19.2
    for i in range(4):
        rpd = rpd + (1. - i / 4.) * amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)
    
    reps = ampreplocs * x_values / (2 * locprec ** 2) * np.exp(
                                               -(x_values ** 2) / (4 * locprec ** 2))
    rpd = rpd + reps

    #reps2 = ampreplocs2 * x_values / (2 * locprec2 ** 2) * np.exp(
    #                                           -(x_values ** 2) / (4 * locprec2 ** 2))
    #rpd = rpd + reps2
    return rpd


def linrepnoreps4_bg_linear(x_values, rep, broadening,
                  a, b, c, d,
                  bgslope, bgoffset
                  ):
    rpd = bgoffset + bgslope * x_values  # Background
    amps = [a, b, c, d]
    for i, amp in enumerate(amps):
        rpd = rpd + amp * model.pairwise_correlation_1d(x_values,
                                                        (i + 1) * rep,
                                                        broadening)
    return rpd


def linrepnoreps4_bg_non_negative(x_values, rep, broadening,
                                  a, b, c, d,
                                  bgslope, bgoffset
                                  ):
    background = bgoffset + bgslope * x_values  # Background
    background[background < 0.] = 0. # Cannot be negative
    amps = [a, b, c, d] # Amplitudes of four peaks
    for i, amp in enumerate(amps):
        rpd = background + amp * model.pairwise_correlation_1d(x_values,
                                                        (i + 1) * rep,
                                                        broadening)
    return rpd


def linrepnoreps4_bg_zero(x_values, rep, broadening,
                  a, b, c, d
                  ):
    rpd = 0
    amps = [a, b, c, d]
    for i, amp in enumerate(amps):
        rpd = rpd + amp * model.pairwise_correlation_1d(x_values,
                                                        (i + 1) * rep,
                                                        broadening)
    return rpd


def linrepnoreps4_bg_linear_vectorinput(vector_input):
    (x_values,
     rep, broadening,
     a, b, c, d,
     bgslope, bgoffset) = vector_input

    y = linrepnoreps4_bg_linear(x_values,
                      rep, broadening,
                      a, b, c, d,
                      bgslope, bgoffset)
    return y


def linrepnoreps4_bg_zero_vectorinput(vector_input):
    (x_values,
     rep, broadening,
     a, b, c, d) = vector_input

    y = linrepnoreps4_bg_zero(x_values,
                      rep, broadening,
                      a, b, c, d
                      )
    return y


def linrepnoreps4fixedpeaks(x_values, rep, broadening,
                   amp,
                   bgslope, bgoffset
                   ):
    rpd = bgoffset + bgslope * x_values # Linear background
    for i in range(4):
        rpd = rpd + amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)    

    #reps2 = ampreplocs2 * x_values / (2 * locprec2 ** 2) * np.exp(
    #                                           -(x_values ** 2) / (4 * locprec2 ** 2))
    #rpd = rpd + reps2
    return rpd

def linrepnoreps4fixedpeakratio(x_values, rep, broadening,
                   amp,
                   bgslope, bgoffset
                   ):
    rpd = bgoffset + bgslope * x_values # Linear background
    for i in range(4):
        rpd = rpd + (1. - i / 4.) * amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)    

    #reps2 = ampreplocs2 * x_values / (2 * locprec2 ** 2) * np.exp(
    #                                           -(x_values ** 2) / (4 * locprec2 ** 2))
    #rpd = rpd + reps2
    return rpd


def linrep19noreps4fixedpeakratio(x_values, broadening,
                   amp,
                   bgslope, bgoffset
                   ):
    rpd = bgoffset + bgslope * x_values # Linear background
    rep = 19.2
    for i in range(4):
        rpd = rpd + (1. - i / 4.) * amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)    

    #reps2 = ampreplocs2 * x_values / (2 * locprec2 ** 2) * np.exp(
    #                                           -(x_values ** 2) / (4 * locprec2 ** 2))
    #rpd = rpd + reps2
    return rpd


def lin_repeat_after_offset_4(x_values, rep, broadening,
                              first_peak_offset,
                              a, b, c, d,
                              bgslope, bgoffset
                              ):
    rpd = bgoffset + bgslope * x_values  # Linear background
    amps = [a, b, c, d]
    for i, amp in enumerate(amps):
        rpd = rpd + amp * pairwise_correlation_1d(x_values,
                                                  first_peak_offset
                                                  + (i + 1) * rep,
                                                  broadening)
    return rpd


def lin_repeat_after_offset_4_vectorargs(vector_input):
    (x_values, rep, broadening,
     first_peak_offset,
     a, b, c, d,
     bgslope, bgoffset) = vector_input
    y = lin_repeat_after_offset_4(x_values, rep, broadening,
                                  first_peak_offset,
                                  a, b, c, d,
                                  bgslope, bgoffset
                                  )
    return y


def linrepplusreps5(x_values, rep, broadening,
                   a, b, c, d, e,
                   locprec, ampreplocs,
                   bgslope, bgoffset
                   ):
    rpd = bgoffset + bgslope * x_values # Linear background
    amps = [a, b, c, d, e]
    for i, amp in enumerate(amps):
        rpd = rpd + amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)    
    rpd = rpd + ampreplocs * model.pairwise_correlation_1d(x_values, 0., np.sqrt(2) * locprec) # rep locs

    return rpd


def linrepplusreps5_bg_flat(x_values, rep, broadening,
                   a, b, c, d, e,
                   locprec, ampreplocs,
                   bgoffset
                   ):
    rpd = x_values * 0. + bgoffset # Background
    amps = [a, b, c, d, e]
    for i, amp in enumerate(amps):
        rpd = rpd + amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)    
    rpd = rpd + ampreplocs * model.pairwise_correlation_1d(x_values, 0., np.sqrt(2) * locprec) # rep locs

    return rpd


def linrepplusreps5_bg_flat_vectorinput(vector_input):
    (x_values, rep, broadening,
     a, b, c, d, e,
     locprec, ampreplocs,
     bgoffset) = vector_input
    rpd = linrepplusreps5_bg_flat(x_values, rep, broadening,
                                  a, b, c, d, e,
                                  locprec, ampreplocs,
                                  bgoffset)

    return rpd


def linrepnoreps5_bg_linear(x_values, rep, broadening,
                  a, b, c, d, e,
                  bgslope, bgoffset
                  ):
    rpd = bgoffset + bgslope * x_values  # Background
    amps = [a, b, c, d, e]
    for i, amp in enumerate(amps):
        rpd = rpd + amp * model.pairwise_correlation_1d(x_values,
                                                        (i + 1) * rep,
                                                        broadening)
    return rpd


def linrepnoreps5_bg_flat(x_values, rep, broadening,
                  a, b, c, d, e,
                  bgoffset
                  ):
    rpd = x_values * 0. + bgoffset  # Background
    amps = [a, b, c, d, e]
    for i, amp in enumerate(amps):
        rpd = rpd + amp * model.pairwise_correlation_1d(x_values,
                                                        (i + 1) * rep,
                                                        broadening)
    return rpd


def linrepnoreps5_bg_non_negative(x_values, rep, broadening,
                                  a, b, c, d, e,
                                  bgslope, bgoffset
                                  ):
    background = bgoffset + bgslope * x_values  # Background
    background[background < 0.] = 0. # Cannot be negative
    
    # Include five peaks on linear repeat
    rpd = np.zeros(len(x_values))
    amps = [a, b, c, d, e] # Amplitudes of five peaks
    for i, amp in enumerate(amps):
        rpd = rpd + amp * model.pairwise_correlation_1d(x_values,
                                                        (i + 1) * rep,
                                                        broadening)
    
    # Add background
    rpd = rpd + background

    return rpd


def linrepnoreps5_bg_zero(x_values, rep, broadening,
                  a, b, c, d, e
                  ):
    rpd = 0
    amps = [a, b, c, d, e]
    for i, amp in enumerate(amps):
        rpd = rpd + amp * model.pairwise_correlation_1d(x_values,
                                                        (i + 1) * rep,
                                                        broadening)
    return rpd


def linrepnoreps5_bg_linear_vectorinput(vector_input):
    (x_values,
     rep, broadening,
     a, b, c, d, e,
     bgslope, bgoffset) = vector_input

    y = linrepnoreps5_bg_linear(x_values,
                      rep, broadening,
                      a, b, c, d, e,
                      bgslope, bgoffset)
    return y


def linrepnoreps5_bg_flat_vectorinput(vector_input):
    (x_values,
     rep, broadening,
     a, b, c, d, e,
     bgoffset) = vector_input

    y = linrepnoreps5_bg_flat(x_values,
                      rep, broadening,
                      a, b, c, d, e,
                      bgoffset)
    return y


def linrepnoreps5_bg_zero_vectorinput(vector_input):
    (x_values,
     rep, broadening,
     a, b, c, d, e) = vector_input

    y = linrepnoreps5_bg_zero(x_values,
                      rep, broadening,
                      a, b, c, d, e
                      )
    return y


def randplusreps5(x_values, broadening,
                   xa, xb, xc, xd, xe,
                   a, b, c, d, e,
                   locprec, ampreplocs,
                   bgslope, bgoffset
                   ):
    rpd = bgoffset + bgslope * x_values # Linear background
    xs = [xa, xb, xc, xd, xe]
    amps = [a, b, c, d, e]
    for i, amp in enumerate(amps):
        rpd = rpd + amp * model.pairwise_correlation_1d(x_values, xs[i], broadening)    
    reps = ampreplocs * x_values / (2 * locprec ** 2) * np.exp(
                                               -(x_values ** 2) / (4 * locprec ** 2))
    rpd = rpd + reps
    #reps2 = ampreplocs2 * x_values / (2 * locprec2 ** 2) * np.exp(
    #                                           -(x_values ** 2) / (4 * locprec2 ** 2))
    #rpd = rpd + reps2
    return rpd

def linrepplusreps5fixedpeak(x_values, rep, broadening,
                   amp,
                   locprec, ampreplocs,
                   bgslope, bgoffset
                   ):
    rpd = bgoffset + bgslope * x_values # Linear background
    for i in range(5):
        rpd = rpd + amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)    
    reps = ampreplocs * x_values / (2 * locprec ** 2) * np.exp(
                                               -(x_values ** 2) / (4 * locprec ** 2))
    rpd = rpd + reps
    #reps2 = ampreplocs2 * x_values / (2 * locprec2 ** 2) * np.exp(
    #                                           -(x_values ** 2) / (4 * locprec2 ** 2))
    #rpd = rpd + reps2
    return rpd

def linrepplusreps5fixedpeakratio(x_values, rep, broadening,
                   amp,
                   locprec, ampreplocs,
                   bgslope, bgoffset
                   ):
    rpd = bgoffset + bgslope * x_values # Linear background
    for i in range(5):
        rpd = rpd + (1. - i / 5.) * amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)    
    
    rpd = rpd + ampreplocs * model.pairwise_correlation_1d(x_values, 0., np.sqrt(2) * locprec) # rep locs

    return rpd


def linrepplusreps5fixedpeakratiovectorinput(vectorin):
    (x_values, rep, broadening,
     amp,
     locprec, ampreplocs,
     bgslope, bgoffset) = vectorin
    y = linrepplusreps5fixedpeakratio(x_values, rep, broadening,
                                      amp,
                                      locprec, ampreplocs,
                                      bgslope, bgoffset
                                      )
    return y


def linrepnoreps5_fixedpeakratio(x_values, rep, broadening,
                   amp,
                   bgslope, bgoffset
                   ):
    rpd = bgoffset + bgslope * x_values # Linear background
    for i in range(5):
        rpd = rpd + (1. - i / 5.) * amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)    

    return rpd


def linrepnoreps5_fixedpeakratio_vectorinput(vectorin):
    (x_values, rep, broadening,
     amp,
     bgslope, bgoffset) = vectorin
    y = linrepnoreps5_fixedpeakratio(x_values, rep, broadening,
                                      amp,
                                      bgslope, bgoffset
                                      )
    return y


def lin_repeat_after_offset_5(x_values, rep, broadening,
                              first_peak_offset,
                              amp0, a, b, c, d, e,
                              bgslope, bgoffset
                              ):
    rpd = bgoffset + bgslope * x_values  # Linear background
    amps = [amp0, a, b, c, d, e]
    for i, amp in enumerate(amps):
        rpd = rpd + amp * pairwise_correlation_1d(x_values,
                                                  first_peak_offset
                                                  + i * rep,
                                                  broadening)
    return rpd


def lin_repeat_after_offset_5_vectorargs(vector_input):
    (x_values, rep, broadening,
     first_peak_offset,
     amp0, a, b, c, d, e,
     bgslope, bgoffset) = vector_input
    y = lin_repeat_after_offset_5(x_values, rep, broadening,
                                  first_peak_offset,
                                  amp0, a, b, c, d, e,
                                  bgslope, bgoffset
                                  )
    return y


def lin_repeat_after_offset_5_no_bg(x_values, rep, broadening,
                              first_peak_offset,
                              amp0, a, b, c, d, e
                              ):
    rpd = 0. * x_values
    amps = [amp0, a, b, c, d, e]
    for i, amp in enumerate(amps):
        rpd = rpd + amp * pairwise_correlation_1d(x_values,
                                                  first_peak_offset
                                                  + (i) * rep,
                                                  broadening)
    return rpd


def lin_repeat_after_offset_5_no_bg_vectorargs(vector_input):
    (x_values, rep, broadening,
     first_peak_offset,
     amp0, a, b, c, d, e) = vector_input
    y = lin_repeat_after_offset_5_no_bg(x_values, rep, broadening,
                                  first_peak_offset,
                                  amp0, a, b, c, d, e
                                  )
    return y


def lin_repeat_after_offset_5_flat_bg_replocs(x_values, rep, broadening,
                                              first_peak_offset,
                                              amp0, a, b, c, d, e,
                                              locprec, ampreplocs,
                                              bgoffset):
    rpd = 0. * x_values + bgoffset # Background
    
    amps = [amp0, a, b, c, d, e]
    for i, amp in enumerate(amps):
        rpd = rpd + amp * pairwise_correlation_1d(x_values,
                                                  first_peak_offset
                                                  + (i) * rep,
                                                  broadening)
    
    rpd = rpd + ampreplocs * model.pairwise_correlation_1d(x_values, 0., np.sqrt(2) * locprec) # rep locs
    
    return rpd


def lin_repeat_after_offset_5_flat_bg_replocs_vectorargs(vector_input):
    (x_values, rep, broadening,
     first_peak_offset,
     amp0, a, b, c, d, e,
     locprec, ampreplocs,
     bgoffset) = vector_input
    y = lin_repeat_after_offset_5_flat_bg_replocs(x_values, rep, broadening,
                                  first_peak_offset,
                                  amp0, a, b, c, d, e,
                                  locprec, ampreplocs,
                                  bgoffset)
    return y


def linrepnoreps5fixedpeak(x_values, rep, broadening,
                   amp,
                   bgslope, bgoffset
                   ):
    rpd = bgoffset + bgslope * x_values # Linear background
    for i in range(5):
        rpd = rpd + amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)    
    
    #reps2 = ampreplocs2 * x_values / (2 * locprec2 ** 2) * np.exp(
    #                                           -(x_values ** 2) / (4 * locprec2 ** 2))
    #rpd = rpd + reps2
    return rpd

def linrepnoreps5fixedpeakratios(x_values, rep, broadening,
                   amp,
                   bgslope, bgoffset
                   ):
    rpd = bgoffset + bgslope * x_values # Linear background
    for i in range(5):
        rpd = rpd + (1. - i / 5.) * amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)    
    
    #reps2 = ampreplocs2 * x_values / (2 * locprec2 ** 2) * np.exp(
    #                                           -(x_values ** 2) / (4 * locprec2 ** 2))
    #rpd = rpd + reps2
    return rpd

def nobg5fixedpeakratios(x_values, rep, broadening,
                   amp
                   ):
    rpd = x_values - x_values # Linear background
    for i in range(5):
        rpd = rpd + (1. - i / 5.) * amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)    
    
    #reps2 = ampreplocs2 * x_values / (2 * locprec2 ** 2) * np.exp(
    #                                           -(x_values ** 2) / (4 * locprec2 ** 2))
    #rpd = rpd + reps2
    return rpd

def nobg5fixedpeakratiosplusreps(x_values, rep, broadening,
                   amp,
                   locprec, ampreplocs
                   ):
    rpd = x_values - x_values # Linear background
    for i in range(5):
        rpd = rpd + (1. - i / 5.) * amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)    

    reps = ampreplocs * x_values / (2 * locprec ** 2) * np.exp(
                                               -(x_values ** 2) / (4 * locprec ** 2))
    rpd = rpd + reps    
    #reps2 = ampreplocs2 * x_values / (2 * locprec2 ** 2) * np.exp(
    #                                           -(x_values ** 2) / (4 * locprec2 ** 2))
    #rpd = rpd + reps2
    return rpd

def linrepplusreps6(x_values, rep, broadening,
                   a, b, c, d, e, f,
                   locprec, ampreplocs,
                   bgslope, bgoffset
                   ):
    rpd = bgoffset + bgslope * x_values # Linear background
    amps = [a, b, c, d, e, f]
    for i, amp in enumerate(amps):
        rpd = rpd + amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)    
    
    reps = ampreplocs * x_values / (2 * locprec ** 2) * np.exp(
                                               -(x_values ** 2) / (4 * locprec ** 2))
    rpd = rpd + reps
    #reps2 = ampreplocs2 * x_values / (2 * locprec2 ** 2) * np.exp(
    #                                           -(x_values ** 2) / (4 * locprec2 ** 2))
    #rpd = rpd + reps2
    return rpd


def linrepnoreps6(x_values, rep, broadening,
                  a, b, c, d, e, f,
                  bgslope, bgoffset
                  ):
    rpd = np.zeros(len(x_values))
    amps = [a, b, c, d, e, f]
    for i, amp in enumerate(amps):
        if (i + 1) * rep < broadening * 10:
            rpd = rpd + amp * model.pairwise_correlation_1d(x_values, (i + 1) * rep, broadening)
        else:
            rpd = rpd + amp * model.gauss1d(x_values, (i + 1) * rep, broadening)

    background = bgoffset + bgslope * x_values
    rpd = rpd + background

    return rpd


def randplusreps6(x_values, broadening,
                   xa, xb, xc, xd, xe, xf,
                   a, b, c, d, e, f,
                   locprec, ampreplocs,
                   bgslope, bgoffset
                   ):
    rpd = bgoffset + bgslope * x_values # Linear background
    xs = [xa, xb, xc, xd, xe, xf]
    amps = [a, b, c, d, e, f]
    for i, amp in enumerate(amps):
        rpd = rpd + amp * model.pairwise_correlation_1d(x_values, xs[i], broadening)    
    reps = ampreplocs * x_values / (2 * locprec ** 2) * np.exp(
                                               -(x_values ** 2) / (4 * locprec ** 2))
    rpd = rpd + reps
    #reps2 = ampreplocs2 * x_values / (2 * locprec2 ** 2) * np.exp(
    #                                           -(x_values ** 2) / (4 * locprec2 ** 2))
    #rpd = rpd + reps2
    return rpd