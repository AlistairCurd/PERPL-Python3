"""
test_linearrepeatmodels.py

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
import numpy as np
import linearrepeatmodels as lin



class TestNoslope(unittest.TestCase):
    """
    Test the noslope function from the linearrepeatmodels library
    """

    def test_noslope_floats(self):
        """
        Test that the input of two floats returns the correct rpd value
        """
        print("Start test_noslope_floats")
        x_float = 1.0
        mean_float = 1.0
        rpd_float = 1.0
        result = lin.noslope(x_float, mean_float)
        self.assertEqual(result, rpd_float)

    def test_noslope_int(self):
        """
        Test that the input of two floats returns the correct rpd value
        """
        print("Start test_noslope_int")
        x_int = 1
        mean_int = 1
        rpd_int = 1
        result = lin.noslope(x_int, mean_int)
        self.assertEqual(result, rpd_int)

    def test_noslope_mean_int(self):
        """
        Test that the input of two floats returns the correct rpd value
        """
        print("Start test_noslope_mean_int")
        x_float = 5.3
        mean_int = 100
        result = lin.noslope(x_float, mean_int)
        self.assertEqual(result, mean_int)


    def test_noslope_mean_float(self):
        """
        Test that the input of two floats returns the correct rpd value
        """
        print("Start test_noslope_mean_float")
        x_int = 5
        mean_float = 24.3
        result = lin.noslope(x_int, mean_float)
        self.assertEqual(result, mean_float)

    def test_noslope_mean_float_array(self):
        """
        Test that the input of two floats returns the correct rpd value
        """
        print("Start test_noslope_mean_float_array")
        x_int_array = np.array([5, 2, 4])
        mean_float = 24.3
        rpd = np.array([24.3, 24.3, 24.3])
        result = lin.noslope(x_int_array, mean_float)
        self.assertEqual(result.tolist(), rpd.tolist())


class TestLinearFit(unittest.TestCase):
    """
    Test the linear_fit function from the linearrepeatmodels library
    """

    def test_linear_fit(self):
        """
        Test that the linear fit of a float input values returns the correct
        rpd value
        """
        print("Start test_linear_fit")
        x_float = 3.2
        slope_float = 6.4
        offset_float = 2.3

        result = lin.linear_fit(x_float, slope_float, offset_float)
        self.assertEqual(result, 22.780000000000005)

    def test_linear_fit_zero_slope_float(self):
        """
        Test that the linear fit of a float input values returns the correct
        rpd value
        """
        print("Start test_linear_fit_zero_slope_float")
        x_float = 3.2
        slope_float_zero = 0.0
        offset_float = 2.3

        result = lin.linear_fit(x_float, slope_float_zero, offset_float)
        self.assertEqual(result, offset_float)

    def test_linear_fit_zero_slope_int(self):
        """
        Test that the linear fit of a float input values returns the correct
        rpd value
        """
        print("Start test_linear_fit_zero_slope_int")
        x_float = 3.2
        slope_int_zero = 0
        offset_float = 2.3

        result = lin.linear_fit(x_float, slope_int_zero, offset_float)
        self.assertEqual(result, offset_float)



if __name__ == '__main__':
    unittest.main()
