"""
utils.py

Created on Tue Sep 10 13:34:20 2019

Authored by Joanna Leng who works at the University of Leeds who is funded by
EPSRC as a Research Software Engineering Fellow (EP/R025819/1).

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
import socket
from sys import platform as _platform
import numpy as np


def find_hostname_and_ip():
    """Finds the hostname and IP address to go in the log file.

    Args:
       No arguments

    Returns:
       host (str): Name of the host machine executing the script.
       ip_address (str): IP adress of the machine that runs the script.
       operating_system (str): Operating system of the machine that runs the script.

    """
    host = 'undetermined'
    ip_address = 'undetermined'
    operating_system = 'undetermined'

    try:
        host = socket.gethostbyaddr(socket.gethostname())[0]
    except socket.herror:
        host = "undetermined"

    my_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        # doesn't even have to be reachable
        my_socket.connect(('10.255.255.255', 1))
        ip_address = my_socket.getsockname()[0]
    except socket.error:
        ip_address = '127.0.0.1'
    finally:
        my_socket.close()

    if _platform in ("linux", "linux2"):
        # linux
        operating_system = 'Linux'
    elif _platform == "darwin":
        # MAC OS X
        operating_system = 'Mac OSX'
    elif _platform == "win32":
        # Windows
        operating_system = 'Windows'
    elif _platform == "win64":
    # Windows 64-bit
        operating_system = 'Windows'

    return host, ip_address, operating_system


def find_exponent(number_string):
    '''Returns the number of significant digits in a number. This takes into account
       strings formatted in 1.23e+3 format and even strings such as 123.450'''
    # change all the 'E' to 'e'
    number_string = number_string.lower()
    if 'e' in number_string:
        # return the length of the numbers before the 'e'
        exponent_string = number_string.split('e', 1)[1]
        return exponent_string # to compenstate for the decimal point
    else:
        # put it in e format and return the result of that
        ### NOTE: the 8 below is picked as on my system floats convert to the
        ### mantissa and exponent format when the exponent reaches 5 and the
        ### 8 allows some change for different systems
        number = ('%.*e' %(8, float(number_string))).split('e')
    #pass it back to the beginning to be parsed
    return find_exponent('e'.join(number))
    #return "error, number not recognised as a float"


def plus_and_minus(value, uncertainty):
    ''' Takes a value and its uncertainty and provides string values for each
    which can be used in the format 'value +/- uncertainty'

    Args:
        value: A float the value that has an error value.
        uncertainty: A float the uncetainty of the value which is generally
                     smaller than the value.
    Returns:
        value_str: A string with the value given to the number of relevant places
                   as determined by the uncertainty.
        uncertainty_str: A string with the uncertainty given to 2 decimal places.
    '''


    uncertainty_str = "%.2g" % (uncertainty)
    uncertainty_expo = int(find_exponent(uncertainty_str))
    value_str = "%.2g" % (value)
    value_expo = int(find_exponent(value_str))
    if value_expo >= uncertainty_expo:
        whole_number = 1
        fraction = 1
        if uncertainty_expo > 0:
            whole_number = uncertainty_expo
        if uncertainty_expo < 0:
            fraction = abs(uncertainty_expo)+1

        uncertainty_str = "%.2f" % (uncertainty)
        value_str = '{0:{1}.{2}f}'.format(value, whole_number, fraction)

        uncertainty_str = '{0:.2f}'.format(uncertainty)
    if value_expo < uncertainty_expo:
        whole_number = 1
        fraction = 1
        if value_expo > 0:
            whole_number = value_expo
        if value_expo < 0:
            fraction = abs(value_expo)+1

        value_str = "%.2f" % (value)
        uncertainty_str = '{0:{1}.{2}f}'.format(uncertainty, whole_number, fraction)

    return value_str, uncertainty_str



def primary_filename_and_path_setup(info):
    ''' Takes information from the python dictionary, info, on the location of the
    input data file to create the data strings nescessary to create the output
    path and file names for images of plots, the html report and output data
    files. The filename and path is different for relative_positions.py as it
    reads experimental data as a primary source to models which read the output
    of relative_positions.py. This function is for relative_positions.py.

    Args:
    info (dict):
            A python dictionary containing a collection of useful
            parameters such as the filenames and paths. New values written
            to the dictionary do not need to be explicitly returned by
            the function as they can be seen in info in other functions.
    '''

    path, in_file_no_path = os.path.split(info['in_file_and_path'])

    index_of_dot = in_file_no_path.index(".")
    filename_without_extension = in_file_no_path[:index_of_dot]

    # Put parameters in directory name
    parameter_str = "filter_"+str(info['filter_dist'])+"_"+str(info['dims'])+"D_"

    results_dir = (path+r"/PERPL_"+info['prog']+r"/"+filename_without_extension
                   +r"/"+parameter_str)

    # Include colour channel information, if used
    if info['colours_analysed'] == 1:
        results_dir = results_dir + 'col' +repr(info['start_channel'])+ '_'
    if info['colours_analysed'] == 2:
        results_dir = (results_dir
                       + 'cols' +repr(info['start_channel'])+ 'to' +repr(info['end_channel'])+ '_'
                       )

    # Include number of nearest neighbours, if used
    if info['nns'] > 0:
        results_dir = results_dir + f'{info["nns"]}nn_'

    # Include histogram bin-size
    results_dir = results_dir + 'bin' +repr(info['bin_size'])+ '_'

    # Include start time
    results_dir = results_dir + info['start']

    # Set up short directory name to save space
    short_filename_without_extension = \
        filename_without_extension[:5]+r"-s-"+filename_without_extension[-5:]

    # Include some parameters
    short_parameter_str = "f_"+str(info['filter_dist'])+"_"+str(info['dims'])+"D_"

    short_results_dir = (
        path+r"/PERPL_"
        +info['prog_short_name']
        +r"/"+short_filename_without_extension
        +r"/"+short_parameter_str
        )
    if info['colours_analysed'] == 1:
        short_results_dir = short_results_dir + 'c' +repr(info['start_channel'])+ '_'
    if info['colours_analysed'] == 2:
        short_results_dir = (short_results_dir
                             + 'c' +repr(info['start_channel'])+ '-' +repr(info['end_channel'])
                             + '_'
                             )
    short_results_dir = short_results_dir + 'b' + repr(info['bin_size'])+ '_'
    short_results_dir = short_results_dir + info['start']

    info['results_dir'] = results_dir
    info['in_file_no_extension'] = filename_without_extension
    info['in_file_no_path'] = in_file_no_path
    info['short_results_dir'] = short_results_dir
    info['short_filename_without_extension'] = short_filename_without_extension


def secondary_filename_and_path_setup(info):
    '''  Takes information from the python dictionary, info, on the location of the
    input data file to create the data strings nescessary to create the output
    path and file names for images of plots, the html report and output data
    files. The filename and path is different for relative_positions.py as it
    reads experimental data as a primary source to models which read the output
    of relative_positions.py. This function is for model scripts.

    Args:
    info (dict):
            A python dictionary containing a collection of useful
            parameters such as the filenames and paths. New values written
            to the dictionary do not need to be explicitly returned by
            the function as they can be seen in info in other functions.
    '''

    path, in_file_no_path = os.path.split(info['in_file_and_path'])

    index_of_dot = in_file_no_path.index(".")
    filename_without_extension = in_file_no_path[:index_of_dot]

    parameter_str = "filter_"+str(info['filter_dist'])+"_"


    results_dir = (path+r"/"+info['prog']+r"/"+r"/"+parameter_str+info['start'])


    short_filename_without_extension = \
        filename_without_extension[:5]+r"-s-"+filename_without_extension[-5:]
    short_parameter_str = "f_"+str(info['filter_dist'])+"_"

    short_results_dir = (path+r"/"+info['prog_short_name']+r"/"+short_filename_without_extension
                   +r"/"+short_parameter_str+info['start'])

    info['results_dir'] = results_dir
    info['in_file_no_extension'] = filename_without_extension
    info['in_file_no_path'] = in_file_no_path
    info['short_results_dir'] = short_results_dir
    info['short_filename_without_extension'] = short_filename_without_extension


def secondary_read_data_in(info):
    """Reads data from the input file thats filename is provided as an argument
       to this program or from the command line while this program executes.
       This reader is for a model and so only reads data that is ouput from
       relative_positions.py.
       Also extracts unful substrings from the input filename that will be used
       to outpur results files and puts them in the
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
    Returns:
               xyz_values (numpy array): A numpy array of the x, y (and z) localisations.
    """

    in_file = info['in_file_and_path']

    if not os.path.exists(in_file):
        sys.exit("ERROR; The input file does not exist.")

    if in_file[-4:] == '.csv':
        try:
            line = open(in_file).readline()
        except (EOFError, IOError, OSError) as exception:
            print("\n\nCould not open file: ", in_file)
            print("\n\n", type(exception))
            sys.exit("Could not open the input file "+in_file+".\n")
        if (line.__contains__("xx_separation,yy_separation, ,xy_separation") or
                line.__contains__("xx_separation,yy_separation,zz_separation,"
                                  "xy_separation,xz_separation,yz_separation,"
                                  "xyz_separation")):
            skip = 1
            try:
                xyz_values = np.loadtxt(in_file, delimiter=',', skiprows=skip)
            except (EOFError, IOError, OSError) as exception:
                print("\n\nCould not read file: ", in_file)
                print("\n\n", type(exception))
                sys.exit("Could not read the input file "+in_file+".\n")
        else:
            xyz_values = 'Ouch'
            print('Sorry, wrong format! This program needs a file output from '
                  'relative_positions.py\n')
            sys.exit("The input file "+in_file+" has the wrong format. It needs "
                     "a file output form relative_positions\n")
    else:
        xyz_values = 'Ouch'
        print('Sorry, wrong format! This program needs a file output from '
              'relative_positions.py\n')
        sys.exit("The input file "+in_file+" has the wrong format. It needs "
                 "a file output form relative_positions\n")


    info['values'] = xyz_values.shape[0]
    info['columns'] = xyz_values.shape[1]
    info['total_values'] = xyz_values.shape[0]
    info['total_columns'] = xyz_values.shape[1]


    return xyz_values
