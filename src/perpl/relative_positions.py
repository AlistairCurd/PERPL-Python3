"""
relative_positions.py

Functions for finding relative positions between points in 3D space and
plotting as a relative positions density function.

Will take localisations and save the relative positions between them,
within a filter distance applied in 3D.

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


import os
import sys
import argparse
import datetime
import timeit
import time
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import numpy as np
from scipy import spatial

from perpl.io import utils
from perpl.io import plotting, reports


def get_inputs(info):
    """Creates a file browser to read the filename of the input data. Then asks
    for takes other inputs as text from the command line. Puts these inputs
    parameters into the dictionary that is easy to pass to many functions.
    These are:
        in_file_and_path (string):
            The input filename and path.
        dims (int):
            The dimensions of the data ie 2D or 3D.
        filterdist (int):
            The distance within which relative positions were calculated.
        zoom (int):
            Magnification factor for the centre of the data in a scatter plot.
        verbose (Boolean):
            If True prints outputs to screen as program executes.
    Args:
        info (dict):
            A python dictionary containing a collection of useful parameters
            such as the filenames and paths.
            This dictionary is modified during the function.

    Returns:
        Nothing
    """

    #root = Tk()
    Tk().withdraw()

    # Find input file.
    print('\n\nPlease select a localisations file. This should be a .csv '
          '(or .txt with comma delimiters) or .npy, containing one '
          'localisation per row.'
          '\nThe first two or three columns, assumed to be X and Y '
          '(and Z, if 3D is chosen later), will be used for analysis.\n')

    in_file = askopenfilename()

    #root.destroy()

    print("The file you selected is: ", in_file, "\n")

    info['in_file_and_path'] = in_file

    # Get spatial dimensionality.
    print('How many spatial dimensions shall we use (2 or 3)?')

    try:
        data_dims = int(input('These should be the first 2 or 3 columns of '
                              'your input file: '))
    except ValueError:
        sys.exit('\nThe number of dimensions must be an integer.\n')

    if data_dims not in (2, 3):
        sys.exit('\nThe number of dimensions must be 2 or 3.\n')

    print("\n"+str(data_dims)+" dimensions were selected.\n")

    info['dims'] = data_dims

    ### Include colour information if applicable.
    print('Do localisations also have colour-channel information (yes/no)?')
    colour_answer = input(
        'The colour channel information should be the last column of your '
        'input file: ').lower()
    # Stop if not answered well enough.
    if colour_answer.startswith('y') or colour_answer.startswith('n'):
        pass
    else:
        sys.exit('\nYou must answer yes or no.\n')

    # Set colour channel info to 'None' or not None.
    if colour_answer.startswith('n'):
        info['colours_analysed'] = None
    if colour_answer.startswith('y'):
        info['colours_analysed'] = 'Some'

    # print(info['colours_analysed']) # Debug

    # Set filter distance to use for relative position calculations.
    print('\nWe will identify neighbouring localisations within a set distance '
          '(filter distance). The smaller the distance the quicker the '
          'calculation.')

    try:
        filterdist = int(input('What is the filter distance between '
                               'localisations that you want to apply (nm)? '))
    except ValueError:
        print('This must be an integer.\n')
        sys.exit("The filter distance must be an integer.\n")

    print("\n"+str(filterdist)+" nm filter distance was selected.")

    info['filter_dist'] = filterdist

    # Set histogram bin values for distance histograms.
    print('\nWhat bin size (integer) do you want for the distance histograms (nm)?')
    print('...Choose 1 for model curve fitting functions...')
    info['bin_size'] = int(input(
        '...Choose a factor of the filter distance to histogram '
        'all the way upto the filter distance: ')
        )

    print("\n"+str(info['bin_size'])+" nm bin size was selected.\n")

    #print('Scatter plots of the raw data are plotted. A zoom scatter plot of '
    #      'the centre is also plotted.')
    #print('If you need a zoom but not of the centre please use an interactive '
    #      'visualization system.\n')

    #try:
    #    zoom = int(input('What zoom magnification do you want (any non '
    #                     'integer answer sets the value to 10)? '))
    #except ValueError:
    #    print('\nYou did not select a suitable value so this is set to 10.\n')
    #    zoom = 10

    #if zoom < 1:
    #    zoom = 10

    #info['zoom'] = zoom

    print('Do you want updates printed to the screen as the analysis progresses?')

    silent = False
    answer = input('yes/no \n').lower()
    if answer.startswith('y'):
        silent = True

    info['verbose'] = silent


def read_data_in(info):
    """Reads data from the input file thats filename is provided as an argument
       to this program (relative_positions.py) or from the command line while
       this program executes. Also extracts unful substrings from the input
       filename that will be used to outpur results files and puts them in the
       info dictionary. These are:
          results_dir (str): All output files are saved in a directory at the same
                             level in the directory structure as the input data and with the
                             name that consists of the input file and a date stamp.
          in_file_no_path (str): The input file name with no path.
          filename_without_extension (str): Input file name wihtout the path and
                            file extension. It is used to create a unique name of the output
                            data file and directory.

    Args:
        info (dict): A python dictionary containing a collection of useful parameters
            such as the filenames and paths.
            This dictionary is modified to contain information about the data read
            during the function.
    Returns:
               xyz_values (numpy array): A numpy array of the x, y (and z) localisations.
    """

    in_file = info['in_file_and_path']

    if not os.path.exists(in_file):
        sys.exit("ERROR; The input file does not exist.")


    if in_file[-4:] == '.npy':
        try:
            xyzcolour_values = np.load(in_file)
        except (EOFError, IOError, OSError) as exception:
            print("\n\nCould not read file: ", in_file)
            print("\n\n", type(exception))
            sys.exit("Could not read the input file "+in_file+".\n")
    elif in_file[-4:] == '.csv' or in_file[-4:] == '.txt':
        try:
            skip = 0
            line = open(in_file).readline()
            for cell in line.split(','):
                try:
                    float(cell)
                except ValueError:
                    #print("Not a float")
                    skip = 1
            xyzcolour_values = np.loadtxt(in_file,
                                          delimiter=',',
                                          skiprows=skip,
                                          # Remove to allow colours at end
                                          # of many columns
                                          # usecols=range(info['dims'])
                                          )
        except (EOFError, IOError, OSError) as exception:
            print("\n\nCould not read file: ", in_file)
            print("\n\n", type(exception))
            sys.exit("Could not read the input file "+in_file+".\n")
    else:
        xyzcolour_values = 'Ouch'
        print('Sorry, wrong format!\n')
        sys.exit("The input file "+in_file+" has the wrong format.\n")


    info['values'] = xyzcolour_values.shape[0]
    info['columns'] = xyzcolour_values.shape[1]
    info['total_values'] = xyzcolour_values.shape[0]
    info['total_columns'] = xyzcolour_values.shape[1]
    # Get the unique channel numbers in use
    if info['colours_analysed'] is not None:
        info['unique_colour_values'] = np.unique(xyzcolour_values[:,-1])
        if len(info['unique_colour_values']) > 100:
            sys.exit(
                '\nThere are ' +repr(len(info['unique_colour_values']))+
                ' unique values (channel values) in the final column of you data.\n'
                '\nExiting because these are unlikely to be the correct channel values.')
        # Print channel options if not specified in shell command arguments.
        if info['start_channel'] is None:
            print('\nThe colour channels present are: '
                  +repr(info['unique_colour_values'].tolist()))

    return xyzcolour_values


def choose_channels(info):
    """Choose 'from' and 'to' colour channels for between colour relative
    positions.

    Args:
        info (dict): A python dictionary containing a collection of useful parameters
            such as the filenames and paths.
            This dictionary is modified to contain information about the data read
            during the function.

    Returns:
        start_channel (float):
            The value of the colour channel for 'from' locs.
        end_channel (float):
            The value of the colour channel for 'to' locs.
            'None' if analysis is of a single specific channel.
    """
    # User input for the number of channels.
    if len(info['unique_colour_values']) == 1:
        start_channel = info['unique_colour_values'][0]
        end_channel = None
        print('Using the only colour channel: ' +repr(start_channel))
        return start_channel, end_channel

    # If > 1 channel - won't get this far if only 1.
    try:
        print('\nHow many colour channels would you like to use in the analysis?')
        colour_number = int(input('You can currently choose 1 or 2: ' ))
    except ValueError:
        sys.exit('\nThe number of colour channels to use must be an integer.\n')
    if colour_number == 0 or colour_number > 2:
        sys.exit('\nSorry, only 1 or 2.\n')
    else:
        print('\n'+str(colour_number)+' colour channels were selected.\n')
        info['colours_analysed'] = colour_number

    # Get the colour channel values.
    if info['colours_analysed'] == 1:
        start_channel = float(input('Which colour channel do you want to analyse? '))
        end_channel = None

    if info['colours_analysed'] == 2:
        start_channel = float(input('Which colour channel do you want to measure FROM? '))
        end_channel = float(input('Which colour channel do you want to measure TO? '))

    # Check the input values match channel values in the data
    # For 'from' channel
    valid_colour = False
    for colour in info['unique_colour_values']:
        if start_channel == colour:
            valid_colour = True
    if valid_colour is False:
        print(repr(start_channel)+ ' is not one of your colour channel values.')
        retry = input('Do you want to select a different colour channel value (yes/no)?')
        if retry.lower()[0] == 'y':
            # info['start_channel'] == 'Incorrect input' # Debug option
            start_channel, end_channel = choose_channels(info)
        else:
            sys.exit('Exiting.')

    # For 'to' channel
    if end_channel is not None:
        valid_colour = False
        for colour in info['unique_colour_values']:
            if end_channel == colour:
                valid_colour = True
        if valid_colour is False:
            print(repr(end_channel)+ ' is not one of your colour channel values.')
            retry = input('Do you want to select a different colour channel value (yes/no)?')
            if retry.lower()[0] == 'y':
                # info['end_channel'] == 'Incorrect input' # Debug option
                start_channel, end_channel = choose_channels(info)
            else:
                sys.exit('Exiting.')

    return start_channel, end_channel


def getdistances_two_colours(
    xyz_values_start, filterdist, xyz_values_end,
    verbose=False
    ):
    """Store all vectors (relative positions) between points within a chosen
    distance of each other in 3D from a list of points in one numpy array
    to another.
    Also works for 2D.

    Args:
        xyz_values_start (numpy array):
            Numpy array of localisations with one row per localisation and
            spatial coordinates in 2 or 3 columns,
            to calculate relative positions 'from'.
        filterdist (float):
            Distance (in all three dimensions) between points within
            which relative positions are calculated. This can be chosen by
            user input as the function runs, or by specifying when calling
            the function from a script.
        xyz_values_end (float):
            A second, optional set of points to calculate relative positions
            'to', from the xyz_values_start values.
            Defaults to None.
        verbose (Boolean):
            Choice whether to print updates to screen. Defaults to False.

    Returns:
        d (numpy array):
            A numpy array of vectors of neighbours within the filter distance
            for every localisation.
    """

    start_time = time.time()  # Start timing it.

    # If two set of points are in use:
    # REMOVE THIS AND PUT IN MAIN AS WILL SLOW DOWN FITTING LATER
    if xyz_values_end is not None:
        # Check they have the same dimensionality. Exit if not.
        if xyz_values_start.shape[1] != xyz_values_end.shape[1]:
            sys.exit(
                '\nYour start and end sets of posistions must have '
                'the same dimensionality.\n'
                'They currently have ' +repr(xyz_values_start.shape[1])+
                ' and ' +repr(xyz_values_end.shape[1])+ 'dimensions, '
                'respectively.\n')
        # Add 3rd column to 2D localisations
        if xyz_values_end.shape[1] == 2:
            xyz_values_end = np.column_stack((xyz_values_end, np.zeros(xyz_values_end.shape[0])))

    # Set up distance filter
    xyz_filter_values = np.array([filterdist, filterdist, filterdist])

    if verbose:
        print('\nFinding vectors to nearby localisations:\n')

    # Add 3rd column to 2D 'from' localisations
    if xyz_values_start.shape[1] == 2:
        xyz_values_start = np.column_stack((xyz_values_start, np.zeros(xyz_values_start.shape[0])))

    # Initialise d (array of relative positions)
    separation_values = []

    # Find relative positions of near neighbours in 'from' to the first point
    # in the 'from' set. Keep going through the list of points until at least
    # one near neighbour is found.
    xyz_index = 0

    # Find all the separtions in x, y (and z) from the first 'from'
    # localisation to the 'to' localisations.
    while len(separation_values) == 0:
        if xyz_index > len(xyz_values_start) -1:
            break

        one_xyz_value = xyz_values_start[xyz_index]  # Reference 'from' loc

        # A filter to find 'to' localisation coordinates within filterdist of
        # 'from' loc. Gives np.array of True / False
        boolean_test_results = np.logical_and(
            xyz_values_end > one_xyz_value - xyz_filter_values,
            xyz_values_end < one_xyz_value + xyz_filter_values
            )

        # Find indices of 'to' locs within filterdist of 'from'
        # loc in all three dimensions (all True) and select these.
        test_results_indices = np.all(boolean_test_results, axis=1)
        subxyz_end = xyz_values_end[test_results_indices]

        sys.stdout.flush()

        # Populate separation_values with any near neighbours.
        # len(subxyz) == 1 would mean the reference 'from' loc was
        # duplicated in the 'to' set and was the only locs within
        # filterdist of itself.
        # Remove [0,0,0], these are duplicates and can overwhelm the result
        # when there is not a second 'to' dataset.
        # Very unlikely that [0,0,0] occurs at all when there are different
        # from and two datasets.
        # Store the vectors from the reference 'from' loc to the filtered subset
        # of all 'from' locs.
        if len(subxyz_end) != 1:
            separation_values = subxyz_end - one_xyz_value
            selectnonzeros = np.any(separation_values != 0, axis=1)
            separation_values = np.compress(selectnonzeros, separation_values, axis=0)

        # Increment reference 'from' localisation
        # in search of next 'from' localisation with near neighbours.
        xyz_index = xyz_index + 1

    # Continue appending to the array separation_values
    # with vectors to the near-enough neighbours of other localisations.
    for xyz_index in range(xyz_index, len(xyz_values_start)):
        one_xyz_value = xyz_values_start[xyz_index]
        boolean_test_results = np.logical_and(
            xyz_values_end > one_xyz_value - xyz_filter_values,
            xyz_values_end < one_xyz_value + xyz_filter_values)
        test_results_indices = np.all(boolean_test_results, axis=1)
        subxyz_end = xyz_values_end[test_results_indices]
        if len(subxyz_end) != 1:
            subd = subxyz_end - one_xyz_value  #  Array of vectors to near-enough neighbours
            selectnonzeros = np.any(subd != 0, axis=1)
            subd = np.compress(selectnonzeros, subd, axis=0)
            separation_values = np.append(separation_values, subd, axis=0)

        # Progress message
        if verbose and xyz_index % 5000 == 0:
            print('Found neighbours for localisation', xyz_index, 'of',
                repr(len(xyz_values_start)) +'.')
            print('%i seconds so far.' % (time.time() - start_time))

    if verbose:
        print('Found %i vectors between all localisations' % len(separation_values))
        print('in %i seconds.' % (time.time() - start_time))

    return separation_values


def get_distances(kdtree, filterdist, verbose=False):
    """Calculates relative positions from positions in a scipy KDTree object,
    up to a maximum distance.
    
    The relative positions do not contain duplicates (for calculating both ways) between a pair.
    
    Args:
        kdtree (scipy KDTree):
            kdtree from input localisations.
        filterdist (float):
            Maximum distance up to which to calculate relative positions.
        verbose (bool):
            Choose whether to print updates to screen.
    
    Returns:
        loc_pairs (set):
            Pairs of localisations (labelled by row number in input array) within filterdist.
        separation_values (numpy array):
            Relative positions of the pairs of localisations.
    """
    start_time = time.time()

    loc_pairs = kdtree.query_pairs(r=filterdist, output_type='ndarray')
    separation_values = kdtree.data[loc_pairs[:, 0]] - kdtree.data[loc_pairs[:, 1]]

    if verbose:
        print('Found %i vectors between all localisations' % len(separation_values))
        print('in %i seconds.' % (time.time() - start_time))

    return loc_pairs, separation_values


def getdistances_two_colours_kdtree(
    xyz_values_start, filterdist, kdtree_end,
    verbose=False
    ):
    """Store all vectors (relative positions) between points within a chosen
    distance of each other in 3D from a list of points in one numpy array
    to another.
    Also works for 2D.

    Args:
        xyz_values_start (numpy array):
            Numpy array of localisations with one row per localisation and
            spatial coordinates in 2 or 3 columns,
            to calculate relative positions 'from'.
        filterdist (float):
            Distance (in all three dimensions) between points within
            which relative positions are calculated. This can be chosen by
            user input as the function runs, or by specifying when calling
            the function from a script.
        kdtree_end (scipy KDTree object):
            The kdtree from points to calculate relative positions
            'to', from the xyz_values_start values.
            Defaults to None.
        verbose (Boolean):
            Choice whether to print updates to screen. Defaults to False.

    Returns:
        relposns_list (list of numpy arrays):
            A list with each element being a numpy array of vectors to neighbours
            from a start point to an end point within the filter distance.
    """

    start_time = time.time()  # Start timing it.

    end_points_within_distance = kdtree_end.query_ball_point(
        xyz_values_start, filterdist)
    relposns_list = []
    for i, start_point in xyz_values_start:
        relposns_list.append(
            end_points_within_distance[i] - start_point)

    if verbose:
        print('Found vectors between %i start and %i end localisations'
              % (len(xyz_values_start), len(kdtree_end.data)))
        print('in %i seconds.' % (time.time() - start_time))

    return relposns_list


def get_vectors(d_values, dims):
    """Calculates the distances in 2D and 3D for relative position vectors.
    This function saves both 2D and 3D data.
    NOTE: THIS FUNCTION HAS AN UNHELPFUL FILE NAME AT THE MOMENT.

    Args:
        d_values: numpy array of localisations with distances between the
                  localisations.
        dim: The dimensions of the data ie 2D or 3D.

    Returns:
        d_values (numpy array): Array of vector components.
    """
    #for i in range(0, 10):
    #    print(d_values[i])
    x_square_values = np.square(d_values[:, 0])
    y_square_values = np.square(d_values[:, 1])

    if dims == 3:
        z_square_values = np.square(d_values[:, 2])

    # Calculate distances across planes and 3D space
    xy_distance_values = np.sqrt(x_square_values + y_square_values)

    if dims == 3:
        xz_distance_values = np.sqrt(x_square_values + z_square_values)
        yz_distance_values = np.sqrt(y_square_values + z_square_values)
        xyz_distance_values = np.sqrt(x_square_values + y_square_values + z_square_values)

    # Include these distances in the output table
    d_values = np.column_stack((d_values, xy_distance_values))

    if dims == 3:
        d_values = np.column_stack((d_values, xz_distance_values))
        d_values = np.column_stack((d_values, yz_distance_values))
        d_values = np.column_stack((d_values, xyz_distance_values))

    # I DON'T THINK THIS SORTING IS NEEDED
    # if dims == 2:
        # d_values.view('f8,f8,f8,f8').sort(order=['f3'], axis=0)

    # if dims == 3:
        # d_values.view('f8,f8,f8,f8,f8,f8,f8').sort(order=['f6'], axis=0)

    return d_values


def save_relative_positions(d_values, filterdist, dims, info):
    """Saves the relative positions that have been found in a csv file. This
    function saves both 2D and 3D data.

    Args:
        d_values: numpy array of localisations with distances between the
                  localisations.
        filterdist: distance (in all three dimensions) between points within
            which relative positions are calculated. This can be chosen by user
            input as the function runs, or by specifying when calling the
            function from a script.
        dim: The dimensions of the data ie 2D or 3D.
        info (dict): A python dictionary containing a collection of useful parameters
            such as the filenames and paths.

    Returns:
        outfilename: The path and filename of the output data file. This is
           recorded in the log file.
    """
    out_file_name = info['results_dir']+r'//'+ info['in_file_no_extension'] + \
        '_PERPL-relpos_%.1ffilter.csv' % filterdist

    if info['short_names']:
        out_file_name = info['short_results_dir']+r'//'+ \
            info['short_filename_without_extension'] + \
            '_PERPL-relpos_%.1ffilter.csv' % filterdist

    head = None
    if dims == 2:
        head = "xx_separation,yy_separation, ,xy_separation"
    elif dims == 3:
        head = ("xx_separation,yy_separation,zz_separation,xy_separation,"
                "xz_separation,yz_separation,xyz_separation")


    try:
        np.savetxt(out_file_name, d_values, delimiter=',', header=head, comments='')
    except (EOFError, IOError, OSError):
        print("Unexpected error:", sys.exc_info()[0])
        sys.exit("Could not create and open the output data file.")

    return out_file_name








def main():
    """Reads input data of point density locations and calculates relative
        poasitions as vectors. Outputs are writen to a file in a directory
        with the name of the inputfile and a time stamp above the directory of
        the input file. The files contains the x, y (and z) localisations.
        If no input argments are provided inputs can be given from the command
        line as it executes.

    Args:
        input_file (FILE): File of localisations which is a .csv (or .txt with
                           comma delimiters) or .npy and containing N
                           localisations in N rows.
        dims (int): The dimensions of the data. This can be 2 or 3.
        filter_dist (int): The filter distance.
        zoom (int): A magnified scatter plot of the centre of the principal
                    view is produced at this level of zoom. Default is 10.
        verbose (Boolean): Increases the output to screen during execution.

    Returns:
        Nothing
    """

    # Handle any input arguments (flags) and set up info dictionary.
    prog = 'relative_positions'
    prog_short_name = 'rp'
    description = 'Calculating the relative positions of points as vectors.'

    info = {'prog':prog,
            'prog_short_name':prog_short_name,
            'description':description}

    info['start'] = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


    parser = argparse.ArgumentParser(prog, description)

    parser.add_argument('-i', '--input_file',
                        dest='input_file',
                        type=argparse.FileType('r'),
                        help='File of localisations which is a .csv (or .txt '
                             'with comma delimiters) or .npy and containing N '
                             'localisations in N rows.',
                        metavar="FILE")

    parser.add_argument('-d', '--dims',
                        dest='dims',
                        type=int,
                        default=3,
                        help="Dimensions of the data. It can be 2 or 3.")

    parser.add_argument('-c', '--colours',
                        dest='colours',
                        type=int,
                        default=None,
                        help="Number of colour channels. It can be 1 or 2.")

    parser.add_argument('--from',
                        dest='start_channel',
                        type=int,
                        default=None,
                        help="Colour channel to measure FROM. "
                            'Use to specify channel for 1-colour data, '
                            'as well as localisations to measure FROM in 2-colour data.')

    parser.add_argument('--to',
                        dest='end_channel',
                        type=int,
                        default=None,
                        help="Colour channel to measure TO. "
                            'Use to specify channel for localisations to measure TO '
                            'in 2-colour data.')

    parser.add_argument('-f', '--filter_distance',
                        dest='filter_dist',
                        type=int,
                        default=150,
                        help="Filter distance.")

    parser.add_argument('-b', '--bin_size',
                        dest='bin_size',
                        type=int,
                        default=1,
                        help="Bin size in distance histograms (nm).")

    parser.add_argument('-z', '--zoom',
                        dest='zoom',
                        type=int,
                        default=10,
                        help='Magnification applied to the scatter plot of the '
                             'principal view of the data.')

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


    if args.verbose:
        print("Verbosity is turned on.\n")

    if args.dims < 2 or args.dims > 3:
        sys.exit("ERROR; The data can only have 2 or 3 dimensions.")

    info['dims'] = args.dims
    info['bin_size'] = args.bin_size
    info['colours_analysed'] = args.colours
    info['start_channel'] = args.start_channel
    info['end_channel'] = args.end_channel
    info['filter_dist'] = args.filter_dist


    info['zoom'] = args.zoom
    info['verbose'] = args.verbose
    info['short_names'] = args.short_names

    if args.input_file is None:
        #print("Get the data from the command line as the program executes.")
        get_inputs(info)
        # print('Colours: ' + repr(info['colours_analysed'])) # Debug
    else:
        info['in_file_and_path'] = args.input_file.name

    info['host'], info['ip_address'], info['operating_system'] = utils.find_hostname_and_ip()

    # GET THE INPUT LOCALISATIONS with possible colour channels
    read_start = timeit.default_timer()
    xyzcolour_values = read_data_in(info)
    read_end = timeit.default_timer()
    reading_time = (read_end-read_start)/60

    if info['verbose']:
        print('\nInput file:')
        print(info['in_file_and_path'])
        print("\nTime to read the input file was: "+str(round(reading_time, 3))+\
              " minutes.\n")
        print('This file contains '+str(info['values'])+' localisations with '
              +str(info['columns'])+' columns per localisation.')

    # For colour channel information, choose channel(s) to analyse,
    # if not given as arguments in the shell command
    if info['colours_analysed'] is not None and info['start_channel'] is None:
        info['start_channel'], info['end_channel'] = choose_channels(info)

    utils.primary_filename_and_path_setup(info)

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

    # GET RELATIVE POSITIONS!
    d_values = []
    # For single channel
    if info['colours_analysed'] is None:
        xyz_values_start = xyzcolour_values[:, 0:info['dims']]
        kdtree = spatial.KDTree(xyz_values_start)
        d_values = get_distances(kdtree, info['filter_dist'], verbose=info['verbose'])[1]

    if info['colours_analysed'] == 1:
        xyz_values_start = \
            xyzcolour_values[:, 0:info['dims']][xyzcolour_values[:, -1] == info['start_channel']]
        kdtree = spatial.KDTree(xyz_values_start)
        d_values = get_distances(kdtree, info['filter_dist'], verbose=info['verbose'])[1]

    # For two channels
    if info['colours_analysed'] == 2:
        xyz_values_start = \
            xyzcolour_values[:, 0:info['dims']][xyzcolour_values[:, -1] == info['start_channel']]
        xyz_values_end = \
            xyzcolour_values[:, 0:info['dims']][xyzcolour_values[:, -1] == info['end_channel']]
        kdtree_end = spatial.KDTree(xyz_values_end)
        d_values = getdistances_two_colours_kdtree(
            xyz_values_start, info['filter_dist'], kdtree_end, verbose=info['verbose'])
        d_values = np.vstack(d_values)
        # d_values = getdistances_two_colours(
        #     xyz_values_start, info['filter_dist'], xyz_values_end, verbose=info['verbose']
        #     )


    # Draw scatterplot and zoomed region
    plotting.draw_2d_scatter_plots(xyzcolour_values, info['dims'], info, 0)
    plotting.draw_2d_scatter_plots(xyzcolour_values, info['dims'], info, info['zoom'])

    # Get distances in 2D and 3D for relative positions.
    # Note, get_vectors() is an unhelpful name as it takes the vectors we already have
    # and calculates distances.
    if len(d_values) > 0:
        # CURRENTLY NEED TO ADD ZEROS COLUMN TO 2D KDTREE VERSION:
        if d_values.shape[1] == 2:
            d_values = np.column_stack((d_values, np.zeros(d_values.shape[0])))
        d_values = get_vectors(d_values, info['dims'])
    else:
        print("No data found so we are exiting.")
        sys.exit("No data found so we are exiting.")

    # Summarise
    if info['verbose']:
        print(
            '\nWhen symmetric duplicates are removed (done by default for a '
            'single colour channel, '
            'never for two-colour data), there are %i vectors within the '
            'filter distance in all dimensions for all localisations.'
             % len(d_values))

    # Plot vector component results
    plotting.plot_histograms(
        d_values, info['dims'], info['filter_dist'], info, binsize=info['bin_size']
        )

    filter_end = timeit.default_timer()
    filter_time = (filter_end-read_end)/60

    if info['verbose']:
        print("\nTime to filter the data was: "+ str(round(filter_time, 3)) +\
              " minutes.")


    # Save relative positions and vector components.
    xyz_filename = save_relative_positions(d_values, info['filter_dist'], info['dims'], info)

    save_data_end = timeit.default_timer()
    filtering_time = (save_data_end-filter_end)/60
    if info['verbose']:
        print("\nTime to write the data was: "+str(round(filtering_time, 3))+" minutes.")

    # Create html report.
    reports.write_rel_pos_html_report(info)

    # Direct user to the location of the output.
    if info['verbose']:
        print('\nRelative positions are saved in the file:\n' + xyz_filename)












if __name__ == "__main__":
    #Tk().withdraw()
    main()
    #print('\nHit Enter to exit')
    #input()
    