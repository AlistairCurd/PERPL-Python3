"""
test_plotting.py

Created on Fri Nov  8 11:32:53 2019

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
from unittest import mock

import matplotlib
import numpy as np
import perpl.io.plotting as plots



class TestDraw2dScatterPlot(unittest.TestCase):
    """
    Test the draw_2col_2d_scatter_plot function from the plotting library
    """
    @mock.patch.object(matplotlib.figure.Figure, "savefig")
    @mock.patch.object(matplotlib.axes.Axes, "set_ylabel")
    @mock.patch.object(matplotlib.axes.Axes, "set_xlabel")
    @mock.patch.object(matplotlib.axes.Axes, "set_title")
    @mock.patch.object(matplotlib.axes.Axes, "scatter")
    def test_mock_inputs(self, mock_scatter, mock_set_title,
                         mock_set_xlabel, mock_set_ylabel, mock_savefig):
        """
        Tests the getdistances function with an array representing 2
        localisations in 2d space (x and y coordinates) with values that are
        relative close ie with a separation of less than 10nm.
        """
        print("Start TestDraw2dScatterPlot test_mock_inputs", flush=True)

        x_values = np.arange(10)
        y_values = np.linspace(0, 200, num=10)
        title = "2D Scatter Plot for Tests"
        filename = "./2dScatterTest.png"
        x_label = "X Label"
        y_label = "Y Label"

        plots.draw_2col_2d_scatter_plot(np.column_stack((x_values, y_values)),
                                        title, filename, x_label, y_label,
                                        info={'colours_analysed': None}
                                        )

        x_values_dash = mock_scatter.call_args_list[0][0][0]
        y_values_dash = mock_scatter.call_args_list[0][0][1]

        try:
            np.testing.assert_array_almost_equal(x_values, x_values_dash)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

        try:
            np.testing.assert_array_almost_equal(y_values, y_values_dash)
            res = True
        except AssertionError as err:
            res = False
            print(err)
        self.assertTrue(res)

        assert x_values_dash.shape == y_values_dash.shape

        assert mock_set_title.call_args_list[0][0][0] == "2D Scatter Plot for Tests"
        assert mock_set_xlabel.call_args_list[0][0][0] == "X Label"
        assert mock_set_ylabel.call_args_list[0][0][0] == "Y Label"

        assert mock_savefig.called


if __name__ == '__main__':
    unittest.main()
    