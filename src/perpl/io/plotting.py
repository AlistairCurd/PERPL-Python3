"""
plotting.py

Library to creates plots of the modelled data.

Created on Mon Apr 29 14:52:57 2019

Alistair Curd
University of Leeds
30 July 2018

Software Engineering practices applied

Joanna Leng
(an EPSRC funded Research Software Engineering Fellow (EP/R025819/1))
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

import sys
from sys import platform as _platform
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage.filters import gaussian_filter
from tifffile import TiffWriter
import perpl.modelling_general as models

# information on backends is here
# https://matplotlib.org/3.1.1/tutorials/introductory/usage.html#backends
# if _platform == "linux" or _platform == "linux2":

# # linux
#   matplotlib.use('TkAgg')

# elif _platform == "darwin":
# # MAC OS X
#   matplotlib.use('MacOSX')

# elif _platform == "win32":
# # Windows
#   matplotlib.use('TkAgg')

# elif _platform == "win64":
# # Windows 64-bit
#    matplotlib.use('TkAgg')

if _platform == "darwin":
    # MAC OS X
    matplotlib.use('MacOSX')


def draw_2d_scatter_plots(xyzcolour_values, dims, info, zoom):  # xyz_values
    """Farms out the scatter plots that need to be plotted to the
    draw_2d_scatter_plot function.

    Args:
       xyz_values (numpy array): A numpy array of the localizations.
       dims (int): The dimensions of the data ie 2D or 3D.
       info (dict): A python dictionary containing a collection of useful parameters.
       zoom (int): A zoom of the central region can be created to this zoom factor.

    Returns:
       Nothing is returned.
    """
    if zoom == 0:
        title = "Scatter plot of localisations in XY"
        fig_name = r'scatter_plot_xy_localisations.png'
        filename = info['results_dir']+r'/'+fig_name
        if info['short_names'] is True:
            filename = info['short_results_dir']+r'/'+fig_name
        draw_2col_2d_scatter_plot(xyzcolour_values,
            title, filename, 'X (nm)', 'Y (nm)', info)

        if dims == 3:
            title = "Scatter plot of localisations in XZ"
            fig_name = r'scatter_plot_xz_localisations.png'
            filename = info['results_dir']+r'/'+fig_name
            if info['short_names'] is True:
                filename = info['short_results_dir']+r'/'+fig_name
            draw_2col_2d_scatter_plot(xyzcolour_values,
                title, filename, 'X (nm)', 'Z (nm)', info, axes=(0, 2))

            title = "Scatter plot of localisations in YZ"
            fig_name = r'scatter_plot_yz_localisations.png'
            filename = info['results_dir']+r'/'+fig_name
            if info['short_names'] is True:
                filename = info['short_results_dir']+r'/'+fig_name
            draw_2col_2d_scatter_plot(xyzcolour_values,
                title, filename, 'X (nm)', 'Z (nm)', info, axes=(1, 2))

    # Zoomed-in xy-plot
    if zoom > 0:  # FIX ZOOM!!!!!
        # Find total data ranges, zoom area and data subset
        x_range = np.max(xyzcolour_values[:, 0]) - np.min(xyzcolour_values[:, 0])
        x_centre = np.min(xyzcolour_values[:, 0] + x_range / 2)
        y_range = np.max(xyzcolour_values[:, 1]) - np.min(xyzcolour_values[:, 1])
        y_centre = np.min(xyzcolour_values[:, 1] + y_range / 2)

        zoomed_xmin = x_centre - x_range / zoom / 2
        zoomed_xmax = x_centre + x_range / zoom / 2
        zoomed_ymin = y_centre - y_range / zoom / 2
        zoomed_ymax = y_centre + y_range / zoom / 2

        # Filter in x and y
        zoomed_xyzc = xyzcolour_values[xyzcolour_values[:, 0] > zoomed_xmin]
        zoomed_xyzc = zoomed_xyzc[zoomed_xyzc[:, 0] < zoomed_xmax]
        zoomed_xyzc = zoomed_xyzc[zoomed_xyzc[:, 1] > zoomed_ymin]
        zoomed_xyzc = zoomed_xyzc[zoomed_xyzc[:, 1] < zoomed_ymax]

        title = "Scatter plot of localisations in XY: zoom x{:d} on central region".format(zoom)
        fig_name = r'scatter_plot_xy_localisations_x'+str(zoom)+'.png'
        filename = info['results_dir']+r'/'+fig_name
        if info['short_names'] is True:
            filename = info['short_results_dir']+r'/'+fig_name
        draw_2col_2d_scatter_plot(zoomed_xyzc, title, filename, 'X (nm)', 'Y (nm)', info)

        #if dims == 3:
            #title = "Scatter Plot of XZ Locations: Center with Zoom of x{:d}".format(zoom)
            #fig_name = r'scatter_plot_xz_locations_x'+str(zoom)+'.png'
            #filename = info['results_dir']+r'/'+fig_name
            #draw_2d_scatter_plot(x_values_masked, z_values_masked,
             #                    title, filename, 'x (nm)', 'z (nm)')

            #title = "Scatter Plot of YZ Locations: Center with Zoom of x{:d}".format(zoom)
            #fig_name = r'scatter_plot_yz_locations_x'+str(zoom)+'.png'
            #filename = info['results_dir']+r'/'+fig_name
            #draw_2d_scatter_plot(y_values_masked, z_values_masked,
            #                     title, filename, 'y (nm)', 'z (nm)')


def draw_2col_2d_scatter_plot(xyzcolour_values,
    title, filename, x_label, y_label, info, axes=(0, 1)
    ):
    """Creates a scatter plot and saves it to a .png file.

    Args:
        xyzcolour_values (numpy array): A numpy array of the xy(and z)-coordinates of the
            localizations (first 2 or 3 columns) and their colour channels (final column).
        title (str): the text string that has the title for the plot.
        filename (str): The output file has all the text of the input
            file plus some more information so it is easy to recognise which
            it has been derived from.
        info (dict): A python dictionary containing a collection of useful parameters
            such as the filenames and paths.
        axes (tuple): Axis indices for the coordinates to be plotted.

    Returns:
       Nothing is returned.
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200, facecolor='w', edgecolor='k')
    if info['colours_analysed'] is None:
        ax.scatter(xyzcolour_values[:, axes[0]], xyzcolour_values[:, axes[1]], s=1)
    else:
        scatterplot = ax.scatter(xyzcolour_values[:, axes[0]], xyzcolour_values[:, axes[1]],
            c=xyzcolour_values[:, -1], s=1
            )
        legend = ax.legend(*scatterplot.legend_elements(),
            loc='upper right', title='Channel'
            )
        ax.add_artist(legend)
    ax.set_title(title)
    ax.set_xlabel(x_label, {'fontsize':'14'})
    ax.set_ylabel(y_label, {'fontsize':'14'})
    # options on scaling are here
    # https://matplotlib.org/devdocs/api/_as_gen/matplotlib.axes.Axes.set_aspect.html
    ax.axis('scaled')
    fig.savefig(filename, bbox_inches='tight')


def find_thresholds(arr, zoom):
    """Calculates the thresholds for the zoom into the scatter plots and applies
    them to mask of the input array.

    Args:
       arr (numpy array): A numpy array of one of the co-ordinates for the locationation.
       zoom (int): A zoom of the central region can be created to this zoom factor.

    Returns:
       arr (numpy.ma.core.MaskedArray): A numpy mask array of one of the
            thresholded co-ordinates for the locationation.
    """
    arr_min = np.min(arr)
    arr_max = np.max(arr)

    data_range = arr_max - arr_min
    zoom_bin = data_range / zoom

    if zoom%2 == 1:
        min_threshold = arr_min + (zoom_bin * int(zoom/2))
        max_threshold = arr_max - (zoom_bin * int(zoom/2))
    else:
        min_threshold = arr_min + (zoom_bin * (int(zoom/2)-0.5))
        max_threshold = arr_max - (zoom_bin * (int(zoom/2)-0.5))

    arr = np.ma.array(arr)
    arr_masked = \
        np.ma.masked_where(\
                (np.logical_xor((min_threshold < arr), (arr < max_threshold))), \
                arr, np.nan)

    return arr_masked


def plot_histograms(d_values, dims, filter_distance, info, binsize=1):
    """Farms out the histograms that need to be plotted to the plot_histogram
    function.

    Args:
       d_values (numpy array): A numpy array of the localizations and distance
            between localisations.
       dims (int): The dimensions of the data ie 2D or 3D.
       filterdist (int): The distance within which relative positions were calculated.
       info (dict): A python dictionary containing a collection of useful parameters
            such as the filenames and paths.

    Returns:
        Nothing is returned.

    """


    plot_histogram(np.absolute(d_values[:, 3]), "xy", filter_distance, info, binsize=binsize)
    #print("xy d_values[:, 3][0]: ", d_values[:, 3][0])

    plot_histogram(np.absolute(d_values[:, 3]),
        "xy", filter_distance, info, standardise='2d', binsize=binsize
        )

    plot_histogram(np.absolute(d_values[:, 0]), "x", filter_distance, info, binsize=binsize)
    #print("x d_values[:, 0][0]: ", d_values[:, 0][0])
    plot_histogram(np.absolute(d_values[:, 1]), "y", filter_distance, info, binsize=binsize)


    if dims == 3:
        plot_histogram(np.absolute(d_values[:, 4]), "xz", filter_distance, info, binsize=binsize)
        plot_histogram(np.absolute(d_values[:, 5]), "yz", filter_distance, info, binsize=binsize)
        plot_histogram(np.absolute(d_values[:, 6]), "xyz", filter_distance, info, binsize=binsize)

        plot_histogram(np.absolute(d_values[:, 2]), "z", filter_distance, info, binsize=binsize)


def plot_new_histogram(data_values):
    """Creates a histogram with a smooth curve fitted to it.

    Args:
       data_values (numpy array): A numpy array of the distance between localisations.

    Returns: No return value

    """

    # Unbinned axial relative positions

    fitlength = 100.

    # In this case my relative positions
    #ax_points = relpos.axial[(relpos.axial < fitlength) & (relpos.transverse < 10)]
    ax_points = data_values[(data_values[:0] < fitlength) & (data_values[:, 6] < 10)]

    # processing resulted in X distances (relpos.axial) and YZ distances (relpos.tranverse).

    # Sort and remove duplicates
    ax_points = np.sort(ax_points)
    ax_points = ax_points[::2]

    # Plot histogram
    axes = plt.hist(ax_points, bins=np.arange(fitlength + 1), color='xkcd:lightblue')[0]

    # Smooth and plot histogram
    # 4.4 here is np.sqrt(2) * localisation precision estimate
    line_smooth = gaussian_filter1d(axes, 4.4)

    # (which the user could input)
    plt.plot(np.arange(fitlength) + 0.5, line_smooth, color='xkcd:red')


def plot_histogram(data_values, data_description, filterdist, info, binsize=1, standardise=None):
    """Creates a histogram and saves it into the .png file.

    Args:
        data_values (numpy array):
            A numpy array of the distance between localisations.
        data_description (str):
            A description of the distance eg xy or xz that
            is added to the name of the .png file.
        filterdist (int):
            The distance within which relative positions were calculated.
        info (dict):
            A python dictionary containing a collection of useful parameters
            such as the filenames and paths.
        standardise (string): None, '2d' or '3d'
            Standardisation of distance distributions:
                '2d': divides histogram bins by distance
                '3d': divides histogram bins by distance ^ 2

    Returns:
       histogram_values (numpy array): A numpy array of the normalised probability
           density values for each bin.

    """

    data_max = data_values.max()

    if data_max < 5:
        start = 0.0
        end = math.ceil(data_max*10)/10
        # edges = math.ceil(data_max*10) + 1
        # bins = np.linspace(start, end, edges)
        # Include end as final edge if end is integer * binsize
        if end % binsize == 0:
            bins = np.arange(start, end + binsize, binsize)
        # Else stop before final edge
        else:
            bins = np.arange(start, end, binsize)
    else:
        start = 0.0
        end = filterdist
        # edges = filterdist + 1
        # bins = np.linspace(start, end, edges)
        # Include end as final edge if end is integer * binsize
        if end % binsize == 0:
            bins = np.arange(start, end + binsize, binsize)
        # Else stop before final edge
        else:
            bins = np.arange(start, end, binsize)

    # Set up axes object for plotting
    fig_hist = plt.figure(num=None,
                          figsize=(10, 8),
                          dpi=100,
                          facecolor='w',
                          edgecolor='k')
    axes = fig_hist.add_subplot(111)

    # Set up filename
    fig_name_base = r'histogram_'+data_description.replace(" ", "_")+r'_separation_in_nm'
    # Include potential standardisation of histogram
    if standardise is not None:
        fig_name_base = fig_name_base+ '_' +standardise+ '_standardised'
    fig_name = fig_name_base + '.png'
    filename = info['results_dir']+r'/'+fig_name
    if info['short_names'] is True:
        filename = info['short_results_dir']+r'/'+fig_name

    # Title and axis labels
    title = r"Distance histogram of "+data_description.upper()+r" separations"
    if standardise is not None:
        title = title + ', ' + standardise + ' standardised'

    plt.title(title)

    plt.xlabel(data_description.upper()+r' separation (nm)')
    plt.ylabel('Counts')
    if standardise == '2d':
        plt.ylabel('Counts / distance (nm)')
    if standardise == '3d':
        plt.ylabel('Counts / distance ^ 2 (nm ^ 2)')

    # Get histogram count values and bin positions
    bin_heights, bin_edges = np.histogram(data_values[data_values < filterdist], bins)
    bin_centres = (bin_edges[0:len(bin_edges) - 1] + bin_edges[1:len(bin_edges)]) / 2

    # Standardise bin counts if desired
    if standardise == '2d':
        bin_heights = bin_heights / bin_centres[0:len(bin_centres)]
    if standardise == '3d':
        bin_heights = bin_heights / bin_centres[0:len(bin_centres)] ** 2

    # PREVIOUS VERSION
    # axes.hist(data_values, bins, color='darkblue', edgecolor='k', linewidth=1,\
    #           alpha=0.5)

    # Plot data and save figure
    width = binsize
    axes.bar(bin_centres,
             bin_heights,
             align='center',
             width=width,
             alpha=0.5,
             color='lightgrey')

    fig_hist.savefig(filename, bbox_inches='tight')
    #fig_hist.show()

    # Save histogram data
    histo_name_base = r'histogram_'+data_description.replace(" ", "_")
    # Add standardisation info if relevant
    if standardise is not None:
        histo_name_base = histo_name_base + '_' + standardise + '_standardised'
    histo_name = histo_name_base + r'.csv'
    filename1 = info['results_dir']+r'/'+histo_name
    if info['short_names'] is True:
        filename1 = info['short_results_dir']+r'/'+histo_name

    data_values = pd.concat([pd.DataFrame(bin_heights),
                             pd.DataFrame(bin_edges)], axis=1)

    head = "normalised probability density,bin edges"

    try:
        np.savetxt(filename1, data_values, delimiter=',', header=head, comments='')
    except (EOFError, IOError, OSError):
        print("Unexpected error:", sys.exc_info()[0])
        sys.exit("Could not create and open the output data file.")

    return bin_heights


def estimate_rpd_churchman_1d(input_distances,
                              calculation_points,
                              combined_precision):
    """Estimates a smooth 1D RPD (relative position distribution) from a set
    of distances, using Churchman's distribution for distances between
    localisations in two clusters as a smoothing kernel.

    Args:
        input_distances (numpy array):
            The distances between localisations in the input data.
        calculation_points (numpy array):
            The distances at which the RPD will be estimated.
        combined_precision (float):
            The width (sigma) of the smoothing function. In later version,
            this may be an array with one values per input distance.

    Returns:
        estimated_rpd (numpy array):
            The 1D RPD estimated at the calculation points.
    """
    estimated_rpd = np.zeros(len(calculation_points))
    for input_distance in input_distances:
        estimated_rpd = estimated_rpd + models.pairwise_correlation_1d(
            calculation_points, input_distance, combined_precision
        )
    return estimated_rpd


def estimate_rpd_churchman_2d(input_distances,
                              calculation_points,
                              combined_precision):
    """Estimates a smooth distance distribuion from a 2D RPD
    (relative position distribution) from a set of distances, using
    Churchman's distribution for distances between localisations in
    two clusters as a smoothing kernel.

    Args:
        input_distances (numpy array):
            The distances between localisations in the input data.
        calculation_points (numpy array):
            The distances at which the RPD will be estimated.
        combined_precision (float):
            The width (sigma) of the smoothing function. In later versions,
            this may be an array with one values per input distance.

    Returns:
        estimated_rpd (numpy array):
            The distance distribution for the 2D RPD estimated at the
            calculation points.
    """
    estimated_rpd = np.zeros(len(calculation_points))
    for input_distance in input_distances:
        estimated_rpd = estimated_rpd + models.pairwise_correlation_2d(
            calculation_points, input_distance, combined_precision
        )
    return estimated_rpd


def create_histogram_3d(rel_pos_xyz, filterdist, smoothing=None):
    """Generates a 3D histogram of relative positions among localisations,
    in bins of 1 nm^3.

    Args:
        rel_pos_xyz (numpy array):
            At least three columns, where the first three columns are
            X, Y, Z coordinates of relative positions (nm).
        filterdist (int):
            The maximum distance (nm) in X, Y and Z upto which the histogram
            should be generated.
        smoothing (float):
            Kernel size (SD, in nm) for isotropic 3D Gaussian smoothing
            of the histogram.

    Returns:
        histogram (numpy array):
            3D histogram bin values.
            Dimensions X, Y, Z will be histogrammed along dimensions 1, 2, 3.
            Includes central reference point sharing the maximum bin value
            in the histogram.
        bin_edge_vector (numpy array):
            1D vector of bin_edges, which are equal for all dimensions.
    """
    # Set Gaussian smoothing kernel, if any
    if smoothing is None:
        print('Would you like to apply 3D smoothing to the histogram (y/n)?')
        smoothing_choice = input('').lower()
        print('')
        if smoothing_choice.startswith('y'):
            print('Please provide the SD (in nm)')
            smoothing = float(input(' for the Gaussian smoothing kernel: '))
        else:
            print('Ok, no smoothing.')

    # Make 3D histogram
    bin_edge_vector = np.array(range(-filterdist, filterdist + 2)) - 0.5
    histogram = np.histogramdd(rel_pos_xyz, bins=(bin_edge_vector,
                                                  bin_edge_vector,
                                                  bin_edge_vector
                                                  )
                               )[0] # No need for returned variable [1] (edges)

    # Apply smoothing, if any
    if smoothing is not None:
        histogram = gaussian_filter(histogram, sigma=smoothing)

    # Add central reference point
    histogram[filterdist, filterdist, filterdist] = np.max(histogram)

    return histogram, bin_edge_vector


def save_tiff_histogram_3d(rel_pos_xyz, filterdist, smoothing=None):
    """Generates a 3D histogram of relative positions among localisations,
    in bins of 1 mm^3. Then saves as a 32-bit multipage tiff file.

    Args:
        rel_pos_xyz (numpy array):
            At least three columns, where the first three columns are
            X, Y, Z coordinates of relative positions (nm).
        filterdist (int):
            The maximum distance (nm) in X, Y and Z upto which the histogram
            should be generated.
        smoothing (float):
            Kernel size (SD, in nm) for isotropic 3D Gaussian smoothing
            of the histogram.

    Returns:
        Nothing

    Saves:
        temp.tif in the working directory, containing the 3D histogram.
    """
    histogram = create_histogram_3d(rel_pos_xyz,
                                    filterdist,
                                    smoothing)[0] # Output [1] not needed
    histogram = histogram.astype(np.float32)
    # Re-order as ZYX.
    histogram = np.transpose(histogram, (2, 1, 0))
    # Save
    with TiffWriter('temp.tif', byteorder='<') as tif:
        tif.save(histogram)


def plot_histogram_with_curves(bin_values, xy_histogram, symmetries, x_values, curve_values, info):
    ''' Plots a histogram of seprations with curves that show how well various models
    fit this histogram.
    Args:
        bin_values: An array of the histogram bin boundaries.
        xy_histogram: An array of floats giving the separations.
        symmetries: Array of the range of rotational symmerties the model tests.
        x_values: An array of the points on the x axesthat the curve are on the plot.
        curve_values: A 2d array of the curves for each symmertry.
        info (dict): A python dictionary containing a collection of useful
                     parameters such as the filenames and paths.

    Returns:
        nothing
    '''

    fig_histogram = plt.figure(num=None,
                               figsize=(10, 8),
                               dpi=100,
                               facecolor='w',
                               edgecolor='k')

    axes = fig_histogram.add_subplot(111)
    center = (bin_values[:-1] + bin_values[1:]) / 2
    width = 1.0
    axes.bar(center,
             xy_histogram,
             align='center',
             width=width,
             alpha=0.5,
             color='lightgrey')

    for i, symm in enumerate(symmetries):
        line_label = str(symm)+"-fold symmetry"
        axes.plot(x_values,
                  curve_values[i],
                  label=line_label)

    plt.title('5-fold to 11-fold fits',
              fontsize=14,
              color='black')
    plt.xlabel('XY separation (nm)', fontsize=14, color='black')
    plt.ylabel('Counts (scaled)', fontsize=14, color='black')

    plt.legend()
    filename = info['results_dir']+r'/'+r'Histogram_with_Fitted_Curves.png'
    if info['short_names'] is True:
        filename = info['short_results_dir']+r'/'+r'Histogram_with_Fitted_Curves.png'
    fig_histogram.savefig(filename, bbox_inches='tight')
    #fig_histogram.show()


def plot_rot_2d_geometry(sym, diameter, info):
    ''' Plots the rotational geometry used in a model.
    Args:
        sym: Integer that represents the rotational symetry of the model.
        diamter: Integer that represents the diamter of the model
        info (dict): A python dictionary containing a collection of useful
                     parameters such as the filenames and paths.
    Returns:
        nothing

    '''

    #small_font = 6
    #medium_font = 8
    #plt.rc('axes', titlesize=medium_font)     # fontsize of the axes title
    #plt.rc('axes', labelsize=medium_font)    # fontsize of the x and y labels
    #plt.rc('xtick', labelsize=small_font)    # fontsize of the tick labels
    #plt.rc('ytick', labelsize=small_font)
    #plt.rc('legend', fontsize=small_font)    # legend fontsize
    #plt.rc('figure', titlesize=medium_font)  # fontsize of the figure title

    fig = plt.figure(num=None,
                     figsize=(1.5, 1.5),
                     dpi=200,
                     facecolor='w',
                     edgecolor='k')

    title = "Plot of geometry with\n"+str(sym)+"-fold rotational symmetry"

    fig_name = "GeometryPlotRotationqalSymmetry"+str(sym)+r"Fold.png"

    filename = info['results_dir']+r'/'+fig_name
    if info['short_names'] is True:
        filename = info['short_results_dir']+r'/'+fig_name

    verts = models.generate_polygon_points(sym, diameter)

    x_values = verts[:, 0]
    y_values = verts[:, 1]

    area = np.pi*20

    plt.scatter(x_values, y_values, s=area, c='blue')
    for point in range(0, sym, 1):
        if point < sym-1:
            x_point = (x_values[point], x_values[point+1])
            y_point = (y_values[point], y_values[point+1])
        if point == sym-1:
            x_point = (x_values[point], x_values[0])
            y_point = (y_values[point], y_values[0])
        plt.plot(x_point, y_point, 'k-')
    plt.title(title)
    plt.xlabel('x (nm)')
    plt.ylabel('y (nm)')

    fig.savefig(filename, bbox_inches='tight')
    #fig.show()
