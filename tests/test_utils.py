"""
test_utils.py

Created on Wed Aug 21 16:33:05 2019

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

import unittest
import utils as ut



class TestFindExponent(unittest.TestCase):
    """
    Test the find_exponent function from the utils library. This is used by the
    function plus_and_minus that formats a value and its uncertainty as 2 strings.
    Key to getting the formatting correct in all cases is knowing the relative
    sizes of numbers and from that calulating the precission and relevant number
    of significant figures needed.
    """


    def test_positve_exponent_float(self):
        """
        Test the exponent for a large number in float format. The exponent will
        be a positive number.
        """
        print("Start TestFindExponent test_positive_float", flush=True)

        value = 20886.96

        result = ut.find_exponent(str(value))

        self.assertEqual(int(result), 4)

    def test_negative_exponent_float(self):
        """
        Test the exponent for a fraction in float format. The exponent will
        be a negative number.
        """
        print("Start TestFindExponent test_negative_exponent_float", flush=True)

        value = 0.06

        result = ut.find_exponent(str(value))

        self.assertEqual(int(result), -2)

    def test_zero_exponent_float(self):
        """
        Test the exponent for a number in float format which has the exponent
        of zero.
        """
        print("Start TestFindExponent test_zero_exponent_float", flush=True)

        value = 1.06

        result = ut.find_exponent(str(value))

        self.assertEqual(int(result), 0)

    def test_positive_exponent_complex(self):
        """
        Test the exponent for a large number in complex format. The exponent will
        be a positive number.
        """
        print("Start TestFindExponent test_positve_exponent_complex", flush=True)

        value = 2.088696E+4

        result = ut.find_exponent(str(value))

        self.assertEqual(int(result), 4)

    def test_negative_exponent_complex(self):
        """
        Test the exponent for a fraction in complex format. The exponent will
        be a negative number.
        """
        print("Start TestFindExponent test_negative_exponent_complex", flush=True)

        value = 6.0E-2

        result = ut.find_exponent(str(value))

        self.assertEqual(int(result), -2)

    def test_find_exponent_size_zero_complex(self):
        """
        Test the exponent for a number in complex format which has the exponent
        of zero.
        """
        print("Start TestFindExponent test_zero_exponent_complex", flush=True)

        value = 1.06E0

        result = ut.find_exponent(str(value))

        self.assertEqual(int(result), 0)


class TestPlusAndMinus(unittest.TestCase):
    """
    Test the plus_and_minus function from the utils library that formats a value
    and its uncertainty as 2 strings and uses the get_exponent function
    Key to getting the formatting correct in all cases is knowing the relative
    sizes of numbers and from that calulating the precission and relevant number
    of significant figures needed.
    """

    def test_larger_pos_expo_value_smaller_neg_expo_uncertainty_floats(self):
        """
        Tests that when there is a large float (bigger than 10 and so with a
        positive value for the exponent) and a smaller uncertianty (a fraction
        smaller than 1.0 with a negative value for the exponent) that the
        correct strings for the value and the uncertainty are produced.

        The string formatting functions enforce particular significant figure
        formats that slightly break the rules of what we want here. We need the
        number of significant figures in the value to alter those of the uncertainty
        and vice versa. Python number formatting is not conditional in this way
        so we just check that the returned strings are in rather than equal to
        each other.
        """
        print(("Start TestPlusAndMinus test_larger_pos_expo_value_smaller_neg_"
               "expo_uncertainty_floats"), flush=True)

        value = 86.96

        uncertainty = 0.011

        result1, result2 = ut.plus_and_minus(value, uncertainty)

        #print(result1)
        #print(value)

        #print(result2)
        #print(uncertainty)

        self.assertIn(str(result1).strip(), "86.960".strip())
        self.assertIn(str(result2), "0.01")

    def test_larger_pos_expo_value_smaller_pos_expo_uncertainty_floats(self):
        """
        Tests that when there is a large float (bigger than 10 ie with a positive
        value for the exponent) and an uncertainty smaller than the value but
        also with a positive exponent that the correct strings for the value
        and the uncertainty are produced.

        The string formatting functions enforce particular significant figure
        formats that slightly break the rules of what we want here. We need the
        number of significant figures in the value to alter those of the uncertainty
        and vice versa. Python number formatting is not conditional in this way
        so we just check that the returned strings are in rather than equal to
        each other.
        """
        print(("Start TestPlusAndMinus test_larger_pos_expo_value_smaller_pos_"
               "expo_uncertainty_floats"), flush=True)

        value = 86.96

        uncertainty = 10.0

        result1, result2 = ut.plus_and_minus(value, uncertainty)

        #print(result1)
        #print(value)

        #print(result2)
        #print(uncertainty)

        self.assertIn(str(result1).strip(), "87.00".strip())
        self.assertIn(str(result2), "10.00")

    def test_smaller_pos_expo_value_larger_pos_expo_uncertainty_floats(self):
        """
        Tests that when there is a large float (bigger than 10 ie with a positive
        exponent) and an error larger than the value that the correct strings
        for the value and the uncertainty are produced.

        The string formatting functions enforce particular significant figure
        formats that slightly break the rules of what we want here. We need the
        number of significant figures in the value to alter those of the uncertainty
        and vice versa. Python number formatting is not conditional in this way
        so we just check that the returned strings are in rather than equal to
        each other.
        """
        print(("Start TestPlusAndMinus test_smaller_pos_expo_value_larger_pos_"
               "expo_uncertainty_floats"), flush=True)

        value = 86.96

        uncertainty = 300.01

        result1, result2 = ut.plus_and_minus(value, uncertainty)

        #print(result1)
        #print(value)

        #print(result2)
        #print(uncertainty)

        self.assertIn(str(result1).strip(), "86.96".strip())
        self.assertIn(str(result2), "300.0")

    def test_larger_neg_expo_value_smaller_neg_expo_uncertainty_floats(self):
        """
        Tests that when there is a fraction float value (smaller than 1.0 so with
        a negative exponent) and a smaller uncertainty (a fraction smaller with a
        negative exponent of grater magnitude) that the correct strings for the value
        and the uncertainty are produced.

        The string formatting functions enforce particular significant figure
        formats that slightly break the rules of what we want here. We need the
        number of significant figures in the value to alter those of the uncertainty
        and vice versa. Python number formatting is not conditional in this way
        so we just check that the returned strings are in rather than equal to
        each other.
        """
        print(("Start TestPlusAndMinus test_larger_neg_expo_value_smaller_neg_"
               "expo_uncertainty_floats"), flush=True)

        value = 0.8696

        uncertainty = 0.011

        result1, result2 = ut.plus_and_minus(value, uncertainty)

        #print(result1)
        #print(value)

        #print(result2)
        #print(uncertainty)

        self.assertIn(str(result1).strip(), "0.870".strip())
        self.assertIn(str(result2), "0.011")

    def test_smaller_neg_expo_value_larger_neg_expo_uncertainty_floats(self):
        """
        Tests that when there is a fraction float value (smaller than 1.0 so with
        a negative exponent) and a larger uncertainty (a fraction with a
        negative exponent of grater magnitude) that the correct strings for the value
        and the uncertainty are produced.

        The string formatting functions enforce particular significant figure
        formats that slightly break the rules of what we want here. We need the
        number of significant figures in the value to alter those of the uncertainty
        and vice versa. Python number formatting is not conditional in this way
        so we just check that the returned strings are in rather than equal to
        each other.
        """
        print(("Start TestPlusAndMinus test_smaller_neg_expo_value_larger_neg_"
               "expo_uncertainty_floats"), flush=True)

        value = 0.08696

        uncertainty = 0.101

        result1, result2 = ut.plus_and_minus(value, uncertainty)

        #print(result1)
        #print(value)

        #print(result2)
        #print(uncertainty)

        self.assertIn(str(result1).strip(), "0.09".strip())
        self.assertIn(str(result2), "0.101")

    def test_smaller_neg_expo_value_larger_neg_expo_uncertainty_floats1(self):
        """
        Tests that when there is a fraction float value (smaller than 1.0 so with
        a negative exponent) and a larger uncertainty (a fraction larger with a
        negative exponent of grater magnitude) that the correct strings for the value
        and the uncertainty are produced.

        The string formatting functions enforce particular significant figure
        formats that slightly break the rules of what we want here. We need the
        number of significant figures in the value to alter those of the uncertainty
        and vice versa. Python number formatting is not conditional in this way
        so we just check that the returned strings are in rather than equal to
        each other.
        """
        print(("Start TestPlusAndMinus test_smaller_neg_expo_value_larger_neg_"
               "expo_uncertainty_floats1"), flush=True)

        value = 0.08696

        uncertainty = 0.330001

        result1, result2 = ut.plus_and_minus(value, uncertainty)

        #print(result1)
        #print(value)

        #print(result2)
        #print(uncertainty)

        self.assertIn(str(result1).strip(), "0.09".strip())
        self.assertIn(str(result2), "0.330")


    def test_smaller_neg_expo_value_larger_pos_expo_uncertainty_floats(self):
        """
        Tests that when there is a fraction float value (smaller than 1.0 so with
        a negative exponent) and a larger nonfraction uncertainty (greater than
        1.0 larger so with a positive exponent) that the correct strings for the
        value and the uncertainty are produced.

        The string formatting functions enforce particular significant figure
        formats that slightly break the rules of what we want here. We need the
        number of significant figures in the value to alter those of the uncertainty
        and vice versa. Python number formatting is not conditional in this way
        so we just check that the returned strings are in rather than equal to
        each other.
        """
        print(("Start TestPlusAndMinus test_smaller_neg_expo_value_larger_pos_"
               "expo_uncertainty_floats"), flush=True)

        value = 0.08696

        uncertainty = 33.0001

        result1, result2 = ut.plus_and_minus(value, uncertainty)

        #print(result1)
        #print(value)

        #print(result2)
        #print(uncertainty)

        self.assertIn(str(result1).strip(), "0.09".strip())
        self.assertIn(str(result2), "33.000")


    def test_larger_pos_expo_value_smaller_zero_expo_uncertainty_floats(self):
        """
        Tests that when there is a large float (bigger than 10 and so with a
        positive value for the exponent) and a smaller uncertianty (a smaller
        number with a zero value for the exponent) that the
        correct strings for the value and the uncertainty are produced.

        The string formatting functions enforce particular significant figure
        formats that slightly break the rules of what we want here. We need the
        number of significant figures in the value to alter those of the uncertainty
        and vice versa. Python number formatting is not conditional in this way
        so we just check that the returned strings are in rather than equal to
        each other.
        """
        print(("Start TestPlusAndMinus test_larger_pos_expo_value_smaller_zero_"
               "expo_uncertainty_floats"), flush=True)

        value = 86.96

        uncertainty = 1.1

        result1, result2 = ut.plus_and_minus(value, uncertainty)

        #print(result1)
        #print(value)

        #print(result2)
        #print(uncertainty)

        self.assertIn(str(result1).strip(), "87.0".strip())
        self.assertIn(str(result2), "1.10")

    def test_larger_zero_expo_value_smaller_pos_expo_uncertainty_floats(self):
        """
        Tests that when there is a float with a zero value for the exponent and
        an uncertainty smaller than the value but also with a zero exponent that
        the correct strings for the value and the uncertainty are produced.

        The string formatting functions enforce particular significant figure
        formats that slightly break the rules of what we want here. We need the
        number of significant figures in the value to alter those of the uncertainty
        and vice versa. Python number formatting is not conditional in this way
        so we just check that the returned strings are in rather than equal to
        each other.
        """
        print(("Start TestPlusAndMinus test_larger_zero_expo_value_smaller_pos_"
               "expo_uncertainty_floats"), flush=True)

        value = 8.696

        uncertainty = 1.1

        result1, result2 = ut.plus_and_minus(value, uncertainty)

        #print(result1)
        #print(value)

        #print(result2)
        #print(uncertainty)

        self.assertIn(str(result1).strip(), "8.70".strip())
        self.assertIn(str(result2), "1.10")



    def test_larger_pos_expo_value_smaller_neg_expo_uncertainty_complex(self):
        """
        Tests that when there is a large complex (bigger than 10 and so with a
        positive value for the exponent) and a smaller uncertianty (a fraction
        smaller than 1.0 with a negative value for the exponent) that the
        correct strings for the value and the uncertainty are produced.

        The string formatting functions enforce particular significant figure
        formats that slightly break the rules of what we want here. We need the
        number of significant figures in the value to alter those of the uncertainty
        and vice versa. Python number formatting is not conditional in this way
        so we just check that the returned strings are in rather than equal to
        each other.
        """
        print(("Start TestPlusAndMinus test_larger_pos_expo_value_smaller_neg_"
               "expo_uncertainty_complex"), flush=True)

        value = 8.696E+1

        uncertainty = 1.1E-2

        result1, result2 = ut.plus_and_minus(value, uncertainty)

        #print(result1)
        #print(value)

        #print(result2)
        #print(uncertainty)

        self.assertIn(str(result1).strip(), "86.960".strip())
        self.assertIn(str(result2), "0.01")

    def test_larger_pos_expo_value_smaller_pos_expo_uncertainty_complex(self):
        """
        Tests that when there is a large complex (bigger than 10 ie with a positive
        value for the exponent) and an uncertainty smaller than the value but
        also with a positive exponent that the correct strings for the value
        and the uncertainty are produced.

        The string formatting functions enforce particular significant figure
        formats that slightly break the rules of what we want here. We need the
        number of significant figures in the value to alter those of the uncertainty
        and vice versa. Python number formatting is not conditional in this way
        so we just check that the returned strings are in rather than equal to
        each other.
        """
        print(("Start TestPlusAndMinus test_larger_pos_expo_value_smaller_pos_"
               "expo_uncertainty_complex"), flush=True)

        value = 8.696E+1

        uncertainty = 1.00E+1

        result1, result2 = ut.plus_and_minus(value, uncertainty)

        #print(result1)
        #print(value)

        #print(result2)
        #print(uncertainty)

        self.assertIn(str(result1).strip(), "87.00".strip())
        self.assertIn(str(result2), "10.00")

    def test_smaller_pos_expo_value_larger_pos_expo_uncertainty_complex(self):
        """
        Tests that when there is a large float (bigger than 10 ie with a positive
        exponent) and an error larger than the value that the correct strings
        for the value and the uncertainty are produced.

        The string formatting functions enforce particular significant figure
        formats that slightly break the rules of what we want here. We need the
        number of significant figures in the value to alter those of the uncertainty
        and vice versa. Python number formatting is not conditional in this way
        so we just check that the returned strings are in rather than equal to
        each other.
        """
        print(("Start TestPlusAndMinus test_smaller_pos_expo_value_larger_pos_"
               "expo_uncertainty_complex"), flush=True)

        value = 8.696E+1

        uncertainty = 3.0001E+2

        result1, result2 = ut.plus_and_minus(value, uncertainty)

        #print(result1)
        #print(value)

        #print(result2)
        #print(uncertainty)

        self.assertIn(str(result1).strip(), "86.96".strip())
        self.assertIn(str(result2), "300.0")

    def test_larger_neg_expo_value_smaller_neg_expo_uncertainty_complex(self):
        """
        Tests that when there is a fraction float value (smaller than 1.0 so with
        a negative exponent) and a smaller uncertainty (a fraction smaller with a
        negative exponent of grater magnitude) that the correct strings for the value
        and the uncertainty are produced.

        The string formatting functions enforce particular significant figure
        formats that slightly break the rules of what we want here. We need the
        number of significant figures in the value to alter those of the uncertainty
        and vice versa. Python number formatting is not conditional in this way
        so we just check that the returned strings are in rather than equal to
        each other.
        """
        print(("Start TestPlusAndMinus test_larger_neg_expo_value_smaller_neg_"
               "expo_uncertainty_complex"), flush=True)

        value = 8.696E-1

        uncertainty = 1.1E-2

        result1, result2 = ut.plus_and_minus(value, uncertainty)

        #print(result1)
        #print(value)

        #print(result2)
        #print(uncertainty)

        self.assertIn(str(result1).strip(), "0.870".strip())
        self.assertIn(str(result2), "0.011")

    def test_smaller_neg_expo_value_larger_neg_expo_uncertainty_complex(self):
        """
        Tests that when there is a fraction float value (smaller than 1.0 so with
        a negative exponent) and a larger uncertainty (a fraction with a
        negative exponent of grater magnitude) that the correct strings for the value
        and the uncertainty are produced.

        The string formatting functions enforce particular significant figure
        formats that slightly break the rules of what we want here. We need the
        number of significant figures in the value to alter those of the uncertainty
        and vice versa. Python number formatting is not conditional in this way
        so we just check that the returned strings are in rather than equal to
        each other.
        """
        print(("Start TestPlusAndMinus test_smaller_neg_expo_value_larger_neg_"
               "expo_uncertainty_complex"), flush=True)

        value = 8.696E-3

        uncertainty = 1.01E-1

        result1, result2 = ut.plus_and_minus(value, uncertainty)

        #print(result1)
        #print(value)

        #print(result2)
        #print(uncertainty)

        self.assertIn(str(result1).strip(), "0.01".strip())
        self.assertIn(str(result2), "0.1010")

    def test_smaller_neg_expo_value_larger_neg_expo_uncertainty_complex1(self):
        """
        Tests that when there is a fraction float value (smaller than 1.0 so with
        a negative exponent) and a larger uncertainty (a fraction larger with a
        negative exponent of grater magnitude) that the correct strings for the value
        and the uncertainty are produced.

        The string formatting functions enforce particular significant figure
        formats that slightly break the rules of what we want here. We need the
        number of significant figures in the value to alter those of the uncertainty
        and vice versa. Python number formatting is not conditional in this way
        so we just check that the returned strings are in rather than equal to
        each other.
        """
        print(("Start TestPlusAndMinus test_smaller_neg_expo_value_larger_neg_"
               "expo_uncertainty_complex1"), flush=True)

        value = 8.696E-2

        uncertainty = 3.30001E-1

        result1, result2 = ut.plus_and_minus(value, uncertainty)

        #print(result1)
        #print(value)

        #print(result2)
        #print(uncertainty)

        self.assertIn(str(result1).strip(), "0.09".strip())
        self.assertIn(str(result2), "0.330")


    def test_smaller_neg_expo_value_larger_pos_expo_uncertainty_complex(self):
        """
        Tests that when there is a fraction float value (smaller than 1.0 so with
        a negative exponent) and a larger nonfraction uncertainty (greater than
        1.0 larger so with a positive exponent) that the correct strings for the
        value and the uncertainty are produced.

        The string formatting functions enforce particular significant figure
        formats that slightly break the rules of what we want here. We need the
        number of significant figures in the value to alter those of the uncertainty
        and vice versa. Python number formatting is not conditional in this way
        so we just check that the returned strings are in rather than equal to
        each other.
        """
        print(("Start TestPlusAndMinus test_smaller_neg_expo_value_larger_pos_"
               "expo_uncertainty_complex"), flush=True)

        value = 8.696E-2

        uncertainty = 3.30001E+1

        result1, result2 = ut.plus_and_minus(value, uncertainty)

        #print(result1)
        #print(value)

        #print(result2)
        #print(uncertainty)

        self.assertIn(str(result1).strip(), "0.09".strip())
        self.assertIn(str(result2), "33.000")


    def test_larger_pos_expo_value_smaller_zero_expo_uncertainty_complex(self):
        """
        Tests that when there is a large float (bigger than 10 and so with a
        positive value for the exponent) and a smaller uncertianty (a smaller
        number with a zero value for the exponent) that the
        correct strings for the value and the uncertainty are produced.

        The string formatting functions enforce particular significant figure
        formats that slightly break the rules of what we want here. We need the
        number of significant figures in the value to alter those of the uncertainty
        and vice versa. Python number formatting is not conditional in this way
        so we just check that the returned strings are in rather than equal to
        each other.
        """
        print(("Start TestPlusAndMinus test_larger_pos_expo_value_smaller_zero_"
               "expo_uncertainty_complex"), flush=True)

        value = 8.696E+1

        uncertainty = 1.1E-0

        result1, result2 = ut.plus_and_minus(value, uncertainty)

        #print(result1)
        #print(value)

        #print(result2)
        #print(uncertainty)

        self.assertIn(str(result1).strip(), "87.0".strip())
        self.assertIn(str(result2), "1.10")

    def test_larger_zero_expo_value_smaller_pos_expo_uncertainty_complex(self):
        """
        Tests that when there is a float with a zero value for the exponent and
        an uncertainty smaller than the value but also with a zero exponent that
        the correct strings for the value and the uncertainty are produced.

        The string formatting functions enforce particular significant figure
        formats that slightly break the rules of what we want here. We need the
        number of significant figures in the value to alter those of the uncertainty
        and vice versa. Python number formatting is not conditional in this way
        so we just check that the returned strings are in rather than equal to
        each other.
        """
        print(("Start TestPlusAndMinus test_larger_zero_expo_value_smaller_pos_"
               "expo_uncertainty_complex"), flush=True)

        value = 8.696E+0

        uncertainty = 1.1E-0

        result1, result2 = ut.plus_and_minus(value, uncertainty)

        #print(result1)
        #print(value)

        #print(result2)
        #print(uncertainty)

        self.assertIn(str(result1).strip(), "8.70".strip())
        self.assertIn(str(result2), "1.10")


class TestPrimaryFilenameAndPathSetup(unittest.TestCase):
    """
    Test the primary_filename_and_path_setup function from the utils library.
    This is used to store data, images and a html report for the
    relative_positions script that analyses experimental data.
    """

    def test_linux_filenames(self):
        """
        Test the input expected from a Linux system.
        """
        print("Start TestPrimaryFilenameAndPathSetup test_linux_filename", flush=True)

        prog = 'relative_positions'
        prog_short_name = 'rp'
        description = 'Calculating the relative positions of points as vectors.'
        info = {'prog':prog, 'prog_short_name':prog_short_name, 'description':description}
        info['start'] = '2019-11-07_15-49-55'
        info['dims'] = 2
        info['filter_dist'] = 200
        info['zoom'] = 10
        info['verbose'] = True
        info['in_file_and_path'] = ("/localhome/joanna/PERPL_data/Nup107_SNAP_3D"
                                    "_GRROUPED_10nmZprec.csv")

        ut.primary_filename_and_path_setup(info)

        result1 = info['results_dir']
        result2 = info['in_file_no_extension']
        result3 = info['in_file_no_path']
        result4 = info['short_results_dir']
        result5 = info['short_filename_without_extension']

        self.assertIn(result1, '/localhome/joanna/PERPL_data/PERPL_relative_'
                      'positions/Nup107_SNAP_3D_GRROUPED_10nmZprec/'
                      'filter_200_2D_2019-11-07_15-49-55')
        self.assertIn(result2, 'Nup107_SNAP_3D_GRROUPED_10nmZprec')
        self.assertIn(result3, 'Nup107_SNAP_3D_GRROUPED_10nmZprec.csv')
        self.assertIn(result4, '/localhome/joanna/PERPL_data/PERPL_rp/Nup10-s-Zprec/'
                      'f_200_2D_2019-11-07_15-49-55')
        self.assertIn(result5, 'Nup10-s-Zprec')
        
        

class TestSecondaryFilenameAndPathSetup(unittest.TestCase):
    """
    Test the primary_filename_and_path_setup function from the utils library.
    This is used to store data, images and a html report for the
    rot_2d_symm script that analyses experimental data.
    """


    def test_linux_filenames(self):
        """
        Test the input expected from a Linux system.
        """
        print("Start TestSecondaryFilenameAndPathSetup test_linux_filename", flush=True)

        prog = 'relative_positions'
        prog_short_name = 'rp'
        description = 'Calculating the relative positions of points as vectors.'
        info = {'prog':prog, 'prog_short_name':prog_short_name, 'description':description}
        info['start'] = '2019-11-07_16-49-55'
        info['dims'] = 2
        info['filter_dist'] = 200
        info['zoom'] = 10
        info['verbose'] = True
        info['in_file_and_path'] = ('/localhome/joanna/PERPL_data/PERPL_relative'
                                    '_positions/Nup107_SNAP_3D_GRROUPED_10nmZprec/'
                                    'filter_200_2019-11-07_15-49-55'
                                    '/Nup107_SNAP_3D_GRROUPED_10nmZprec_PERPL'
                                    '-relpos_200.0filter.csv')

        ut.secondary_filename_and_path_setup(info)

        result1 = info['results_dir']
        result2 = info['in_file_no_extension']
        result3 = info['in_file_no_path']
        result4 = info['short_results_dir']
        result5 = info['short_filename_without_extension']

        print("result1: ", result1)
        print("result2: ", result2)
        print("result3: ", result3)
        print("result4: ", result4)
        print("result5: ", result5)        

        self.assertIn(result1, '/localhome/joanna/PERPL_data/PERPL_relative_'
                      'positions/Nup107_SNAP_3D_GRROUPED_10nmZprec/filter_200'
                      '_2019-11-07_15-49-55/relative_positions//filter_200_'
                      '2019-11-07_16-49-55')
        self.assertIn(result2, 'Nup107_SNAP_3D_GRROUPED_10nmZprec_PERPL-relpos_200')
        self.assertIn(result3, 'Nup107_SNAP_3D_GRROUPED_10nmZprec_PERPL-relpos_200.0filter.csv')
        self.assertIn(result4, '/localhome/joanna/PERPL_data/PERPL_relative'
                      '_positions/Nup107_SNAP_3D_GRROUPED_10nmZprec/'
                      'filter_200_2019-11-07_15-49-55'
                      '/rp/Nup10-s-s_200/f_200_2019-11-07_16-49-55')
        self.assertIn(result5, 'Nup10-s-s_200')



if __name__ == '__main__':
    unittest.main()
    