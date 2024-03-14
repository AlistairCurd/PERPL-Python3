"""
test_relative_positions.py

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
#import tempfile
import unittest
import numpy as np
import perpl.relative_positions as rp



class TestGetdistances(unittest.TestCase):
    """
    Test the getdistances function from the relative_positions library
    """

    def test_2d_array_of_2(self):
        """
        Tests the getdistances function with an array representing 2
        localisations in 2d space (x and y coordinates) with values that are
        relative close ie with a separation of less than 10nm.
        """
        print("Start TestGetdistances test_2d_array_of_2", flush=True)
        xyz_values = np.array([[20886.96, 30248.96],
                               [20891.08, 30246.26]])
        filterdist = 150
        separation_values = np.array([[4.12, -2.7, 0.]])

        result = rp.getdistances(xyz_values, filterdist)
        try:
            np.testing.assert_array_almost_equal(separation_values, result)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_2d_array_of_4_with_2_close_pairs(self):
        """
        Tests the getdistances function with an array representing 4
        localisations in 2d space (x and y coordinates) with 2 pairs of values
        that are relative close ie with each pair having a separation of less
        than 10nm but each pair being further than the 150 filterdistance.
        """
        print("Start TestGetdistances test_2d_array_of_4_with_2_close_pairs", flush=True)
        xyz_values = np.array([[21740.77, 26400.65],
                               [21734.15, 26386.16],
                               [20886.96, 30248.96],
                               [20891.08, 30346.26]])
        filterdist = 150
        separation_values = np.array([[4.12, 97.3, 0.],
                                      [6.62, 14.49, 0.]])

        result = rp.getdistances(xyz_values, filterdist)
        try:
            np.testing.assert_array_almost_equal(separation_values, result)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_2d_array_of_4_with_1_close_pair_y_altered(self):
        """
        Tests the getdistances function with an array representing 4
        localisations in 2d space (x and y coordinates) with 2 candidate pairs
        of localisations only 1 pair has a separation of less than 10nm. The
        other pair of localisation are further than the 150 filterdistance.
        """
        print(("Start TestGetdistances test_2d_array_of_4_with_1_close_pair_y_"
               "altered"), flush=True)
        xyz_values = np.array([[21740.77, 26400.65],
                               [21734.15, 26386.16],
                               [20886.96, 30248.96],
                               [20891.08, 30446.26]])
        filterdist = 150
        separation_values = np.array([[6.62, 14.49, 0.]])

        result = rp.getdistances(xyz_values, filterdist)
        try:
            np.testing.assert_array_almost_equal(separation_values, result)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_2d_array_of_4_1_close_pair_filterdist_altered(self):
        """
        Tests the getdistances function with an array representing 4
        localisations in 2d space (x and y coordinates) with 2 candidate pairs
        of localisations only 1 pair has a separation of less than 10nm. The
        other pair of localisation are further than the 150 filterdistance but
        this time a filterdistance of 200 is used.
        """
        print(("Start TestGetdistances test_2d_array_of_4_1_close_pair_"
               "filterdist_altered"), flush=True)
        xyz_values = np.array([[21740.77, 26400.65],
                               [21734.15, 26386.16],
                               [20886.96, 30248.96],
                               [20891.08, 30446.26]])
        filterdist = 200
        separation_values = np.array([[4.12, 197.3, 0.],
                                      [6.62, 14.49, 0.],])

        result = rp.getdistances(xyz_values, filterdist)
        try:
            np.testing.assert_array_almost_equal(separation_values, result)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_2d_array_of_4_1_close_pair_x_altered(self):
        """
        Tests the getdistances function with an array representing 4
        localisations in 2d space (x and y coordinates) with 2 candidate pairs
        of localisations only 1 pair has a separation of less than 10nm. The
        other pair of localisation are further than the 150 filterdistance.
        """
        print("Start TestGetdistances test_2d_array_of_4_1_close_pair_x_altered", flush=True)
        xyz_values = np.array([[21740.77, 26400.65],
                               [21734.15, 26386.16],
                               [20886.96, 30248.96],
                               [20791.08, 30446.26]])
        filterdist = 150
        separation_values = np.array([[6.62, 14.49, 0.]])

        result = rp.getdistances(xyz_values, filterdist)
        try:
            np.testing.assert_array_almost_equal(separation_values, result)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)


    def test_2d_array_of_5_with_1_close_pair(self):
        """
        Tests the getdistances function with an array representing 4
        localisations in 2d space (x and y coordinates) with only 1 pair
        of localisations only 1 pair. The other localisations are further
        than the 150 filterdistance.
        """
        print("Start TestGetdistances test_2d_array_of_5_with_1_close_pair", flush=True)
        xyz_values = np.array([[20886.96, 30248.96],
                               [35950.99, 41226.77],
                               [22019.19, 36913.71],
                               [21740.77, 26400.65],
                               [20891.08, 30246.26]])
        filterdist = 150
        separation_values = np.array([[4.12, -2.7, 0.]])

        result = rp.getdistances(xyz_values, filterdist)
        try:
            np.testing.assert_array_almost_equal(separation_values, result)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)


    def test_2d_array_of_6_with_2_pairs_2_same_place(self):
        """
        Tests the getdistances function with an array representing 6
        localisations in 2d space (x and y coordinates) with 2 localisations
        in the same place. The other localisations are further than the
        150 filterdistance.
        """
        print("Start TestGetdistances test_2d_array_of_6_with_2_pairs_2_same_place", flush=True)
        xyz_values = np.array([[20886.96, 30248.96],
                               [35950.99, 41226.77],
                               [22019.19, 36913.71],
                               [21740.77, 26400.65],
                               [20891.08, 30246.26],
                               [20891.08, 30246.26]])
        filterdist = 150
        separation_values = np.array([[4.12, -2.7, 0.],
                                      [4.12, -2.7, 0.]])

        result = rp.getdistances(xyz_values, filterdist)
        try:
            np.testing.assert_array_almost_equal(separation_values, result)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)


    def test_getdistances_2d_array_of_10(self):
        """
        Test that the input of 10 lines of a numpy array of a 2D sample returns
        the correct values in a numpy array
        """
        print("Start TestGetDistances test_getdistances_2d_array_of_10", flush=True)

        xyz_values = np.array([[20886.96, 30248.96],
                               [35950.99, 41226.77],
                               [22019.19, 36913.71],
                               [21740.77, 26400.65],
                               [20891.08, 30246.26],
                               [21769.1, 32955.15],
                               [23850.47, 28568.08],
                               [32170.57, 39171.85],
                               [30441.26, 45623.22],
                               [21734.15, 26386.16]])

        filterdist = 150
        separation_values = np.array([[4.12, -2.7, 0.],
                                      [6.62, 14.49, 0.]])

        result = rp.getdistances(xyz_values, filterdist)

        try:
            np.testing.assert_array_almost_equal(separation_values, result)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)


class TestGetVectors(unittest.TestCase):
    """
    Test the get_vectors function from the relative_positions library
    """


    def test_2d_array_of_1(self):
        """
        Tests the get_vectors function with an array with values for 1
        2d localisation with x and y coordinates. The vector is calculated
        as the sqareroot(x**2+y**2).
        """
        print("Start TestGetVectors test_2d_array_of_1", flush=True)

        d_values = np.array([[20886.96, 30248.96, 0.]])

        dims = 2

        v_values = np.array([[20886.96, 30248.96, 0., 36759.55221603]])

        result = rp.get_vectors(d_values, dims)

        try:
            np.testing.assert_array_almost_equal(v_values, result)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)


    def test_2d_array_of_1_negative(self):
        """
        Tests the get_vectors function with an array with values for 1
        2d localisation with negative x and y coordinates. The vector is calculated
        as the sqareroot(x**2+y**2).
        """
        print("Start TestGetVectors test_2d_array_of_1_negative", flush=True)

        d_values = np.array([[-20886.96, -30248.96, 0.]])

        dims = 2

        v_values = np.array([[-20886.96, -30248.96, 0., 36759.55221603]])

        result = rp.get_vectors(d_values, dims)

        try:
            np.testing.assert_array_almost_equal(v_values, result)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)


    def test_2d_array_of_2(self):
        """
        Tests the get_vectors function with an array with values for 2
        2d localisation with x and y coordinates. The vector is calculated
        as the sqareroot(x**2+y**2).
        """
        print("Start TestGetVectors test_2d_array_of_2", flush=True)

        d_values = np.array([[20886.96, 30248.96, 0.],
                             [20886.97, 30248.95, 0.]])

        dims = 2

        v_values = np.array([[20886.97, 30248.95, 0., 36759.549669],
                             [20886.96, 30248.96, 0., 36759.552216]])

        result = rp.get_vectors(d_values, dims)

        try:
            np.testing.assert_array_almost_equal(v_values, result)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_2d_array_of_2_with_negatives(self):
        """
        Tests the get_vectors function with an array with values for 1
        2d localisation with some negative x and y coordinates. The vector is
        calculated as the sqareroot(x**2+y**2).
        """
        print("Start TestGetVectors test_2d_array_of_2_with_negatives", flush=True)

        d_values = np.array([[-20886.96, 30248.96, 0.],
                             [20886.97, -30248.95, 0.]])

        dims = 2

        v_values = np.array([[20886.97, -30248.95, 0., 36759.549669],
                             [-20886.96, 30248.96, 0., 36759.552216]])

        result = rp.get_vectors(d_values, dims)

        try:
            np.testing.assert_array_almost_equal(v_values, result)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

    def test_2d_array_of_10(self):
        """
        Tests the get_vectors function with an array with values for 10
        2d localisation with x and y coordinates. The vector is
        calculated as the sqareroot(x**2+y**2).
        """
        print("Start TestGetVectors test_2d_array_of_10", flush=True)

        d_values = np.array([[20886.96, 30248.96, 0.],
                             [35950.99, 41226.77, 0.],
                             [22019.19, 36913.71, 0.],
                             [21740.77, 26400.65, 0.],
                             [20891.08, 30246.26, 0.],
                             [21769.1, 32955.15, 0.],
                             [23850.47, 28568.08, 0.],
                             [32170.57, 39171.85, 0.],
                             [30441.26, 45623.22, 0.],
                             [21734.15, 26386.16, 0.]])

        dims = 2

        v_values = np.array([[21734.15, 26386.16, 0., 34184.83166213],
                             [21740.77, 26400.65, 0., 34200.22515445],
                             [20886.96, 30248.96, 0., 36759.55221603],
                             [20891.08, 30246.26, 0., 36759.67175525],
                             [23850.47, 28568.08, 0., 37215.32095935],
                             [21769.1, 32955.15, 0., 39496.02038601],
                             [22019.19, 36913.71, 0., 42982.16739789],
                             [32170.57, 39171.85, 0., 50689.04621856],
                             [35950.99, 41226.77, 0., 54700.27647657],
                             [30441.26, 45623.22, 0., 54846.59071953]])

        result = rp.get_vectors(d_values, dims)

        try:
            np.testing.assert_array_almost_equal(v_values, result)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)


    def test_3d_array_of_10_with_negative_floats_and_complex(self):
        """
        Tests the get_vectors function with an array with values for 10
        2d localisation with x and y coordinates. The vector is
        calculated as the sqrt(x**2+y**2+z**2).
        """
        print(("Start TestGetVectors test_3d_array_of_10_with_negative_floats_"
               "and_complex"), flush=True)

        d_values = np.array([[0.,         2.41,    -193.],
                             [0.,         2.64,      40.],
                             [0.,         3.36,       9.],
                             [1.000e-02, -3.942e+01, -1.820e+02],
                             [1.00e-02,  -7.42e+00,  -1.32e+02],
                             [1.00e-02,   1.42e+00,  -1.64e+02],
                             [1.00e-02,   3.81e+00,  -1.38e+02],
                             [2.00e-02,   1.43e+00,  -2.10e+01],
                             [2.0e-02,   -2.3e+00,    9.1e+01],
                             [0.02,      -0.66,     -12.]])

        dims = 3

        v_values = np.array([[0.00000000e+00, 3.36000000e+00, 9.00000000e+00, 3.36000000e+00,
                              9.00000000e+00, 9.60674763e+00, 9.60674763e+00],
                             [2.00000000e-02, -6.60000000e-01, -1.20000000e+01, 6.60302961e-01,
                              1.20000167e+01, 1.20181363e+01, 1.20181529e+01],
                             [2.00000000e-02, 1.43000000e+00, -2.10000000e+01, 1.43013985e+00,
                              2.10000095e+01, 2.10486318e+01, 2.10486413e+01],
                             [0.00000000e+00, 2.64000000e+00, 4.00000000e+01, 2.64000000e+00,
                              4.00000000e+01, 4.00870253e+01, 4.00870253e+01],
                             [2.00000000e-02, -2.30000000e+00, 9.10000000e+01, 2.30008695e+00,
                              9.10000022e+01, 9.10290613e+01, 9.10290635e+01],
                             [1.00000000e-02, -7.42000000e+00, -1.32000000e+02, 7.42000674e+00,
                              1.32000000e+02, 1.32208382e+02, 1.32208383e+02],
                             [1.00000000e-02, 3.81000000e+00, -1.38000000e+02, 3.81001312e+00,
                              1.38000000e+02, 1.38052585e+02, 1.38052585e+02],
                             [1.00000000e-02, 1.42000000e+00, -1.64000000e+02, 1.42003521e+00,
                              1.64000000e+02, 1.64006147e+02, 1.64006148e+02],
                             [1.00000000e-02, -3.94200000e+01, -1.82000000e+02, 3.94200013e+01,
                              1.82000000e+02, 1.86220129e+02, 1.86220129e+02],
                             [0.00000000e+00, 2.41000000e+00, -1.93000000e+02, 2.41000000e+00,
                              1.93000000e+02, 1.93015046e+02, 1.93015046e+02]])

        result = rp.get_vectors(d_values, dims)

        try:
            np.testing.assert_array_almost_equal(v_values, result)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)


if __name__ == '__main__':
    unittest.main()
