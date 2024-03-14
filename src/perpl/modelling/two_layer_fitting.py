"""
two_layer_fitting.py

Functions for fitting relative positions of localisations to a two layer model.

Created on Mon Sep 16 13:45:00 2019

Alistair Curd
University of Leeds
16 September 2019

Software Engineering practices applied

Joanna Leng (an EPSRC funded Research Software Engineering Fellow (EP/R025819/1)
University of Leeds
January 2019

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
import platform
import datetime
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from perpl.modelling.background_models import exponential_decay_1d_pair_corr as expo_bg
from perpl.modelling.modelling_general import pairwise_correlation_1d
from perpl.modelling.modelling_general import stdev_of_model
from perpl.io.utils import find_hostname_and_ip


def get_input(info):
    """Load relative positions.

    Returns:
        Relative position data (2D or 3D) loaded from .csv file.
    """
    Tk().withdraw()
    print('Please select input file containing relative positions to assess '
          'for two-layer structure (.csv or .txt with comma delimiters).')
    print('The file should contain an array '
          'with one relative position per row.\n')
    infile = askopenfilename()
    print("The file you selected is: ", infile, "\n")
    if not os.path.exists(infile):
        sys.exit("ERROR; The input file does not exist.")

    path, in_file_no_path = os.path.split(infile)

    index_of_dot = in_file_no_path.index(".")
    filename_without_extension = in_file_no_path[:index_of_dot]

    results_dir = (path + r"/" + info['prog'] + r"_"
                   + filename_without_extension + r"_"+info['start'])

    if infile[-4:] == '.npy':
        try:
            xyz_values = np.load(infile)
        except (EOFError, IOError, OSError) as exception:
            print("\n\nCould not read file: ", infile)
            print("\n\n", type(exception))
            sys.exit("Could not read the input file "+infile+".\n")
    elif infile[-4:] == '.csv' or infile[-4:] == '.txt':
        try:
            xyz_values = np.loadtxt(infile, delimiter=',', skiprows=1)
            # For files in .txt format, there may be no header, so one data
            # point may be missing
        except (EOFError, IOError, OSError) as exception:
            print("\n\nCould not read file: ", infile)
            print("\n\n", type(exception))
            sys.exit("Could not read the input file "+infile+".\n")
    else:
        xyz_values = 'Ouch'
        print('Sorry, wrong format!\n')
        sys.exit("The input file "+infile+" has the wrong format.\n")

    info['results_dir'] = results_dir
    info['in_file_and_path'] = infile
    info['in_file_no_extension'] = filename_without_extension
    info['in_file_no_path'] = in_file_no_path

    # Set longest distance used between localisations when producing
    # distance histograms and fitting
    try:
        fitlength = int(input('What maximum distance would you like '
                              'to set (nm)? '))
    except ValueError:
        print('This must be an integer.\n')
        sys.exit("The filter distance must be an integer.\n")

    info['fitlength'] = fitlength

    print('Do you want updates printed to the screen as the analysis '
          'progresses?')

    silent = True
    answer = input('yes/no \n').lower()
    if answer.startswith('y'):
        silent = False

    info['silent'] = silent

    #relpos = np.loadtxt(infile, delimiter=',')
    #print('The file you selected is:')
    #print(infile, ',', sep='')
    #print('which contains', relpos.shape[0], 'relative positions.')

    return xyz_values


def log_file_header(log_file, info):
    """Writes key info to the top of the log file. We continually write to the
    log file and the script is executed so we have something if it crashes.

    Args:
       log_file (file handler):
           Allows you to write to the already open log file.
       info (dict):
           A python dictionary containing a collection of useful parameters
           such as the filenames and paths.

    Returns:
       Nothing is returned.

    """
    log_file.write(info['prog']+r': '+info['description']+"\n\n")

    log_file.write(r"This program ran at " + info['start']
                   + r" on the host system " + info['host'] + ".")

    log_file.write("\n System's IP address is: "+info['ip_address'])
    log_file.write("\n\n")
    log_file.flush()

    log_file.write('Versions of python and key libraries used:\n\n')
    if 'conda' in sys.version:
        log_file.write('Python '+platform.version()+' and Anaconda, Inc.')
    else:
        log_file.write('Python '+platform.version())
#    log_file.write('\n\nPython {0} and {1}'.format((platform.version).split('|')[0],\
#                   (sys.version).split('|')[1]))

    log_file.write('\nnumpy version is: '+np.__version__)
    #log_file.write('\nMatplotlib version is: '+plt.__version__)
    #log_file.write('\nScipy version is: '+scipy.__version__)

    log_file.write('\n\nInput file: '+info['in_file_and_path']+"\n")

    #log_file.write('This files contains '+str(info['values'])+' locs with '\
    #               +str(info['columns'])+' columns.\n')

    log_file.flush()

    return


def two_layer_model_constant_bg(distance_values_1d,
                                layer_separation,
                                amplitude_within_layer,
                                amplitude_between_layers,
                                broadening,
                                background_offset):
    """Generate the values of a relative position density (RPD) for a
    two-layer, 1D distribution. Comprised of two pair-correlation functions
    for Gaussian distributions (within and between the two layers), plus a
    constant background density value (assuming isotropic localisations).

    Args:
        distance_values_1d (numpy array):
            The separations between localisations, measured along one
            direction, at which the values of the RPD will be generated.
        layer_separation (float):
            The separation between the two layers of the model structure.
        amplitude_within_layer (float):
            The amplitude of the peak of the distribution describing
            within-layer separations.
        broadening (float):
            Reflects the spread of each layer, assumed to be the same for both.
        background_offset:
            Constant background density for an isotropic sample.
    """

    within_layer_peak = (
        amplitude_within_layer
        * pairwise_correlation_1d(distance_values_1d,
                                  0,
                                  broadening)
    )

    between_layers_peak = (
        amplitude_between_layers
        * pairwise_correlation_1d(distance_values_1d,
                                  layer_separation,
                                  broadening)
    )

    model_rpd = within_layer_peak + between_layers_peak + background_offset

    return model_rpd


def two_layer_model_exp_decay_bg(distance_values_1d,
                                 layer_separation,
                                 amplitude_within_layer,
                                 amplitude_between_layers,
                                 broadening,
                                 bg_amplitude,
                                 bg_scale_param):
    """Generate the values of a relative position density (RPD) for a
    two-layer, 1D distribution. Comprised of two pair-correlation functions
    for Gaussian distributions (within and between the two layers), plus an
    exponentially decaying background density function (e.g. as arising from
    localisations in Z in the evanescent field in a TIRF experiment).

    Args:
        distance_values_1d (numpy array):
            The separations between localisations, measured along one
            direction, at which the values of the RPD will be generated.
        layer_separation (float):
            The separation between the two layers of the model structure.
        amplitude_within_layer (float):
            The amplitude of the peak of the distribution describing
            within-layer separations.
        broadening (float):
            Reflects the spread of each layer, assumed to be the same for both.
        bg_amplitude:
            Amplitude of the exponentially decaying background density.
        bg_scale_param:
            Scale parameter for the exponentially decaying background term.
    """

    within_layer_peak = (
        amplitude_within_layer
        * pairwise_correlation_1d(distance_values_1d,
                                  0,
                                  broadening)
    )

    between_layers_peak = (
        amplitude_between_layers
        * pairwise_correlation_1d(distance_values_1d,
                                  layer_separation,
                                  broadening)
    )

    background = expo_bg(distance_values_1d, bg_amplitude, bg_scale_param)

    model_rpd = within_layer_peak + between_layers_peak + background

    return model_rpd


def two_layer_model_exp_decay_bg_vectorargs(input_vector):
    """Function to calculate the values given by two_layer_model_exp_decay_bg,
    but using a vector input for the parameters, so that the numdifftools
    package can be used to calculate partial derivatives for correct error
    propagation in the model.

    Args:
        input_vector (list or numpy array):
            A concatenation of:
                1. Distances at which density values of the model will be
                obtained (numpy array)

                2. The parameters used by
                two_layer_model_exp_decay_bg (list or numpy vector).
    Returns:
        rpd (numpy array):
            The relative position density given by the model at the input
            distances (called separation_values_1d).
    """
    # Get the variables back out of the vector for the non-vector-input
    # function.
    (separation_values_1d,
     layer_separation,
     amplitude_within_layer,
     amplitude_between_layers,
     broadening,
     bg_amplitude,
     bg_scale_param) = input_vector

    # Evaluate
    rpd = two_layer_model_exp_decay_bg(separation_values_1d,
                                       layer_separation,
                                       amplitude_within_layer,
                                       amplitude_between_layers,
                                       broadening,
                                       bg_amplitude,
                                       bg_scale_param)

    return rpd


def fit_two_layer_model(experimental_data,
                        model=two_layer_model_constant_bg,
                        fitlength=200.):
    """Use scipy.optimize.curve_fit to do non-linear least-squares fitting
    of a model two-layer relative position distribution to an experimental
    distribution, e.g. histogram or kernel density estimation.
    Args:
        experimental_data:
            Experimental pairwise distance distribution along one
            direction, evaluated at n + 0.5 nm, where n is an integer
            (histogram bin centres for distance histograms).
        model:
            Parametric model distribution.
            Defaults to two_layer_model().
        fitlength:
            Maximum distance included in the fit.

    Returns:
        params_optimised:
            Optimised parameters.
        params_covar:
            Covariance matrix between parameters.
        params_1sd_err:
            Error (1 SD) on parameters.
    """
    # Independent variable: distances at which the experimental distance
    # histogram or density has been obtained and the model generated.
    distance_values = np.arange(fitlength) + 0.5

    # Find estimates and covariances of model parameters
    params_optimised, params_covar = curve_fit(
        model, distance_values, experimental_data,
        p0=[60.,  # layer_separation
            10.,  # amplitude_within_layer
            10.,  # amplitude_between_layers
            20.,  # broadening
            # 0.01  # background_offset (constant background)
            10.,  # bg amplitude (exponential decay background)
            100  # bg scale parameter (exponential decay background)
            ],
        bounds=(0.,
                [120.,  # layer_separation
                 100.,  # amplitude_within_layer
                 100.,  # amplitude_between_layers
                 100.,  # broadening
                 # 0.1  # background_offset (constant background)
                 100.,  # bg amplitude (exponential decay background)
                 fitlength  # bg scale parameter (exponential decay background)
                 ])
        )

    # Calculate uncertainty (1 SD)
    params_1sd_err = np.sqrt(np.diag(params_covar))
    # print('Fitted parameters:')
    # Print out parameter estimates and errors:
    # print(np.column_stack((params_optimised, params_1sd_err)))

    return params_optimised, params_covar, params_1sd_err


def main():
    """GET DATA AND RUN ANALYSIS."""
    info = {'prog': 'two_layer_fitting',
            'description': 'Fits a 1D two-layer model to \
            relative positions among localisation microscopy data.'}

    start = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    info['start'] = start

    xyz_values = get_input(info)

    fitlength = info['fitlength']

    info['values'] = xyz_values.shape[0]

    info['columns'] = xyz_values.shape[1]

    info['total_values'] = xyz_values.shape[0]
    info['total_columns'] = xyz_values.shape[1]

    print('This contains '+str(info['values'])+' relative positions with '
          + str(info['columns'])+' columns.\n')

    try:
        os.makedirs(info['results_dir'])
    except OSError:
        print("Unexpected error:", sys.exc_info()[0])
        sys.exit("Could not create directory for the results.")

    log_file_name = info['results_dir']+r"//log.txt"

    try:
        log_file = open(log_file_name, "w")
    except (EOFError, IOError, OSError):
        print("Unexpected error:", sys.exc_info()[0])
        sys.exit("Could not create and open the log file.")

    info['host'], info['ip_address'] = find_hostname_and_ip()

    log_file_header(log_file, info)

    # Get distances and distance histogram, and plot.
    dimension_to_analyse = (
        int(input('Which column of the relative positions data '
                  '(which dimension) would you like to fit '
                  'to a two-layer model (start counting at 0)? '))
    )
    distances = xyz_values[:, dimension_to_analyse]
    distance_histogram_values, bin_values = np.histogram(
            distances,
            weights=np.repeat(float(fitlength) / len(distances),
                              len(distances)),
            bins=np.arange(fitlength + 1)
            )

    fig_histogram = plt.figure(num=None, figsize=(10, 8), dpi=100,
                               facecolor='w', edgecolor='k')
    axes = fig_histogram.add_subplot(111)
    center = (bin_values[:-1] + bin_values[1:]) / 2
    width = 1.0
    axes.bar(center, distance_histogram_values,
             align='center', width=width, alpha=0.5, color='lightgrey')

    # Fit two-layer model and plot
    model = two_layer_model_exp_decay_bg
    info['model_name'] = model.__name__

    params_optimised, params_covar, params_1sd_error = (
            fit_two_layer_model(distance_histogram_values,
                                model=model,
                                fitlength=fitlength)
    )

    x_values = center
    fitted_curve = model(x_values, *params_optimised)
    axes.plot(x_values, fitted_curve)

    # Find 95% confidence interval at each x-value and plot
    plt.figure()
    axes = plt.subplot(111)
    bin_centres = (bin_values[:-1] + bin_values[1:]) / 2
    width = 1.0
    axes.bar(bin_centres,
            distance_histogram_values,
            align='center', width=width, alpha=0.5,                       color='lightgrey'
            )
    axes.set_xlim([0, fitlength])
    axes.set_xlabel(r'$\Delta$Z (nm)')
    axes.set_ylabel('Counts (scaled: mean = 1)')

    # Plot model
    axes.plot(bin_centres,
            model(bin_centres, *params_optimised),
            color='xkcd:red', lw=0.75
            )
    vector_input_model = two_layer_model_exp_decay_bg_vectorargs
    stdev = stdev_of_model(x_values,
                           params_optimised,
                           params_covar,
                           vector_input_model)
    axes.fill_between(x_values,
                      model(x_values, *params_optimised) - stdev * 1.96,
                      model(x_values, *params_optimised) + stdev * 1.96,
                      alpha=0.25
                      )

    filename = info['results_dir']+r'/'+r'Histogram_with_Fitted_Curve.png'
    fig_histogram.savefig(filename, bbox_inches='tight')
    fig_histogram.show()

    # Table of optimised parameters and uncertainties
    params_table = np.column_stack((params_optimised, params_1sd_error))
    print(params_table)

    return params_table


if __name__ == '__main__':
    Tk().withdraw()
    main()
    print('\nHit Enter to exit')
    input()

"""For figure, used lightblue for histogram, and xkcd:red for curve and
error plot.
"""
