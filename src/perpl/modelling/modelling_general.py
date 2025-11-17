"""
modelling_general.py

Library of non-statistical functions needed to create the models.

Created on Wed Dec 12 15:06:11 2018

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

# import time
import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in font_manager.findSystemFonts(fontpaths=None, fontext="ttf"):
    # to install arial on linux had to run these commands...
    # sudo apt install ttf-mscorefonts-installer
    # sudo fc-cache -f
    # rm -rf ~/.cache/matplotlib
    if "Arial" in f:
        plt.rcParams["font.family"] = ["Arial"]
from scipy.linalg import svd
from scipy.spatial.transform import Rotation
from scipy.special import i0
from scipy import stats
from scipy.optimize import curve_fit, least_squares


class PERPLModel:
    """Generic model to be fit

    Attributes:

        pairwise correlation: Pairwise correlation function for the data
            based on dimension
        n_params: Number of parameters in the data
    """

    def __init__(
        self,
        dimension=1,
        background=None,
        n_peaks=0,
        peak_type="variable",
        characteristic_distance="one",
        characteristic_distance_ratio=None,
        repeats=False,
        offset=False,
        normalise=False,
        params_initial=None,
        params_lower=None,
        params_upper=None,
        name=None,
    ):
        """Initialise

        Args:
            dimension: Dimension of the fit (1, 2 or 3)
            background: None, "flat", "linear" or "linear_flat" background
            n_peaks: Number of peaks (0, 1, 2, ...)
            peak_type: "variable" or "fixed_ratio" peaks
            characteristic_distance: How to model the characteristic distance.
                one: One characteristic distance, that is repeated at each
                    peak.
                multiple_fixed : As many characteristic distances as peaks at
                    user defined distances
                multiple_ratio: As many characteristic distances as peaks, at
                    distances calculated by scaling up one characteristic distance
                    by different ratios
            characteristic_distance_ratio: Ratio of characteristic distances
            repeats: TRUE or FALSE to have repeated localisations
            offset: TRUE or FALSE to include offset
            normalise: TRUE or FALSE to normalise
            params_initial: Dictionary of initial param guesses
            params_lower: Dictionary of lower bounds on params
            params_upper: Dictionary of upper bounds on params
            name: Name of the model
        """

        if dimension not in [1, 2, 3]:
            raise ValueError("Dimension should be 1,2 or 3")
        if background not in [None, "flat", "linear", "linear_flat"]:
            raise ValueError("Background should be None, flat, linear, linear_flat")
        if not type(n_peaks) == int:
            raise ValueError("n peaks should be an integer")
        if peak_type not in ["variable", "fixed_ratio"]:
            raise ValueError("Peak type should be variable or fixed_ratio")
        if characteristic_distance not in ["one", "multiple_fixed", "multiple_ratio"]:
            raise ValueError(
                "Characteristic distance should be one, multiple_fixed or multiple_ratio"
            )
        if characteristic_distance == "multiple_ratio":
            if len(characteristic_distance_ratio) < n_peaks:
                raise ValueError(
                    "Should be a characteristic distance ratio for each peak"
                )
            if characteristic_distance_ratio[0] != 1.0:
                raise ValueError("First peak should have ratio 1.0")
        if not type(repeats) == bool:
            raise ValueError("Repeats should be True or False")
        if not type(offset) == bool:
            raise ValueError("Offset should be True or False")
        if not type(normalise) == bool:
            raise ValueError("Normalise should be True or False")

        # number of params
        n_params = 0
        self.initial_params = {}
        self.params_lower = {}
        self.params_upper = {}

        # Model name
        self.name = name

        # background
        self.background = background
        if background == "flat":
            self.initial_params["bg_offset"] = params_initial["bg_offset"]
            self.params_lower["bg_offset"] = params_lower["bg_offset"]
            self.params_upper["bg_offset"] = params_upper["bg_offset"]
            n_params += 1
        elif background == "linear" or background == "linear_flat":
            self.initial_params["bg_offset"] = params_initial["bg_offset"]
            self.params_lower["bg_offset"] = params_lower["bg_offset"]
            self.params_upper["bg_offset"] = params_upper["bg_offset"]
            self.initial_params["bg_slope"] = params_initial["bg_slope"]
            self.params_lower["bg_slope"] = params_lower["bg_slope"]
            self.params_upper["bg_slope"] = params_upper["bg_slope"]
            n_params += 2

        # dimension of data
        self.dimension = dimension
        if dimension == 1:
            self.pairwise_correlation = pairwise_correlation_1d
        elif dimension == 2:
            self.pairwise_correlation = pairwise_correlation_2d
        elif dimension == 3:
            self.pairwise_correlation = pairwise_correlation_3d

        # peaks in the data
        self.n_peaks = n_peaks
        self.peak_type = peak_type
        if n_peaks > 0:
            if peak_type == "fixed_ratio":
                self.initial_params[f"amp_peak_1"] = params_initial["amp_peak_1"]
                self.params_lower[f"amp_peak_1"] = params_lower["amp_peak_1"]
                self.params_upper[f"amp_peak_1"] = params_upper["amp_peak_1"]
                n_params += 1
            elif peak_type == "variable":
                for i in range(n_peaks):
                    self.initial_params[f"amp_peak_{i+1}"] = params_initial[
                        f"amp_peak_{i+1}"
                    ]
                    self.params_lower[f"amp_peak_{i+1}"] = params_lower[
                        f"amp_peak_{i+1}"
                    ]
                    self.params_upper[f"amp_peak_{i+1}"] = params_upper[
                        f"amp_peak_{i+1}"
                    ]
                n_params += n_peaks

            # characteristic distances
            self.characteristic_distance_type = characteristic_distance
            self.characteristic_distances_ratio = characteristic_distance_ratio

            self.initial_params["characteristic_distance_1"] = params_initial[
                "characteristic_distance_1"
            ]
            self.params_lower["characteristic_distance_1"] = params_lower[
                "characteristic_distance_1"
            ]
            self.params_upper["characteristic_distance_1"] = params_upper[
                "characteristic_distance_1"
            ]

            self.initial_params["characteristic_distance_broadening"] = params_initial[
                "characteristic_distance_broadening"
            ]
            self.params_lower["characteristic_distance_broadening"] = params_lower[
                "characteristic_distance_broadening"
            ]
            self.params_upper["characteristic_distance_broadening"] = params_upper[
                "characteristic_distance_broadening"
            ]

            n_params += 2

            if characteristic_distance == "multiple_fixed":
                for i in range(n_peaks):
                    if i == 0:
                        continue
                    self.initial_params[f"characteristic_distance_{i+1}"] = (
                        params_initial[f"characteristic_distance_{i+1}"]
                    )
                    self.params_lower[f"characteristic_distance_{i+1}"] = params_lower[
                        f"characteristic_distance_{i+1}"
                    ]
                    self.params_upper[f"characteristic_distance_{i+1}"] = params_upper[
                        f"characteristic_distance_{i+1}"
                    ]

                n_params += n_peaks - 1

        # repeats
        if repeats:
            self.initial_params["loc_prec_amp"] = params_initial["loc_prec_amp"]
            self.params_lower["loc_prec_amp"] = params_lower["loc_prec_amp"]
            self.params_upper["loc_prec_amp"] = params_upper["loc_prec_amp"]

            self.initial_params["loc_prec_sd"] = params_initial["loc_prec_sd"]
            self.params_lower["loc_prec_sd"] = params_lower["loc_prec_sd"]
            self.params_upper["loc_prec_sd"] = params_upper["loc_prec_sd"]

            n_params += 2

        # normalise
        self.normalise = normalise

        # offset
        if offset:
            raise NotImplementedError("Offset not implemented yet")

        self.param_bounds = (
            np.array(list(self.params_lower.values())),
            np.array(list(self.params_upper.values())),
        )
        self.n_params = n_params
        self.param_names = list(self.initial_params.keys())

    def model_rpd(
        self,
        x_values,
        characteristic_distances=[],
        characteristic_distance_broadening=0.0,
        loc_prec_sd=0.0,
        loc_prec_amp=None,
        bg_slope=0.0,
        bg_offset=0.0,
        amps=[],
    ):

        rpd = 0.0 * x_values

        # background
        background = bg_offset + bg_slope * x_values
        if self.background == "linear_flat":
            if type(x_values) == np.float64:
                if x_values >= -bg_offset / bg_slope:
                    background = 0.0
            else:
                background[x_values >= -bg_offset / bg_slope] = 0.0

        rpd += background

        # peaks
        if self.n_peaks == 0:
            characteristic_distance_terms = None
        else:
            characteristic_distance_terms = []
        for peak_idx in range(self.n_peaks):
            if self.peak_type == "fixed_ratio":
                amp = amps[0] * (1 - peak_idx / self.n_peaks)
            elif self.peak_type == "variable":
                amp = amps[peak_idx]

            if self.characteristic_distance_type == "one":
                characteristic_distance = characteristic_distances[0] * (peak_idx + 1)

            elif self.characteristic_distance_type == "multiple_fixed":
                characteristic_distance = characteristic_distances[peak_idx]

            elif self.characteristic_distance_type == "multiple_ratio":
                characteristic_distance = (
                    characteristic_distances[0]
                    * self.characteristic_distances_ratio[peak_idx]
                )

            characteristic_distance_term = amp * self.pairwise_correlation(
                x_values,
                characteristic_distance,
                characteristic_distance_broadening,
            )
            rpd += characteristic_distance_term
            characteristic_distance_terms.append(characteristic_distance_term)

        # repeats
        if loc_prec_amp is not None:
            rep_locs = loc_prec_amp * self.pairwise_correlation(
                x_values, 0.0, np.sqrt(2) * (loc_prec_sd)
            )
            rpd += rep_locs
        else:
            rep_locs = None

        # normalise
        if self.normalise:
            # if dimension = 1 --> divide by 1s
            # if dimension = 2 --> divide by r (perimeter of circle)
            # if dimension = 3 --> divide by r^2 (surface area of sphere)
            rpd /= x_values ** (self.dimension - 1)
            background /= x_values ** (self.dimension - 1)
            if rep_locs is not None:
                rep_locs /= x_values ** (self.dimension - 1)
            if characteristic_distance_terms is not None:
                characteristic_distance_terms = [
                    i / (x_values ** (self.dimension - 1))
                    for i in characteristic_distance_terms
                ]

        # offset

        return rpd, background, characteristic_distance_terms, rep_locs

    def model_rpd_wrapper(self, x, params):

        # dict(zip(self.initial_params.keys(), params)
        kwargs = dict(zip(self.param_names, params))

        if self.n_peaks != 0:
            amps = [kwargs["amp_peak_1"]]
            kwargs.pop("amp_peak_1")
            if self.peak_type == "variable":
                for i in range(self.n_peaks):
                    if i == 0:
                        continue
                    amps.append(kwargs[f"amp_peak_{i+1}"])
                    kwargs.pop(f"amp_peak_{i+1}")

            characteristic_distances = [kwargs["characteristic_distance_1"]]
            kwargs.pop("characteristic_distance_1")
            if self.characteristic_distance_type == "multiple_fixed":
                for i in range(self.n_peaks):
                    if i == 0:
                        continue
                    characteristic_distances.append(
                        kwargs[f"characteristic_distance_{i+1}"]
                    )
                    kwargs.pop(f"characteristic_distance_{i+1}")

        else:
            amps = []
            characteristic_distances = []

        return self.model_rpd(
            x, **kwargs, amps=amps, characteristic_distances=characteristic_distances
        )

    def model_rpd_wrapper_vector(self, vector_input):

        return self.model_rpd_wrapper(vector_input[0], vector_input[1:])[0]

    def error_fn(self, params, x, y):

        output = self.model_rpd_wrapper(x, params)[0]

        return output - y

    def fit_to_experiment(self, x, y):
        """Use scipy.optimize.curve_fit to do non-linear least-squares fitting
        of model relative position distribution to an experimental distribution,
        e.g. histogram or kernel density estimation.

        Args:
             x:
                Distance values at which the experimental distribution of
                distances and the model are evaluated i.e. calculation
                points
            y:
                Experimental pairwise distance distribution."""

        # Find estimates and covariances of model parameters
        if self.normalise:
            # can't put y /= x**(self.dimnension-1)
            # as y may be integers so get error
            y = y / x ** (self.dimension - 1)
        try:
            res = least_squares(
                self.error_fn,
                np.array(list(self.initial_params.values())),
                bounds=self.param_bounds,
                args=(x, y),
                # not sure whether to include this or not...
                x_scale=abs((self.param_bounds[0] + self.param_bounds[1]) / 2),
            )

            # Param optimal and covariance
            popt = res.x

            # Do Moore-Penrose inverse discarding zero singular values [from scipy.linalg]
            _, s, VT = svd(res.jac, full_matrices=False)
            threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
            s = s[s > threshold]
            VT = VT[: s.size]
            pcov = np.dot(VT.T / s**2, VT)

            ysize = len(res.fun)
            cost = 2 * res.cost  # res.cost is half sum of squares!

            if ysize > self.n_params:
                s_sq = cost / (ysize - self.n_params)
                pcov = pcov * s_sq
            else:
                print(f"Covariance calculation failed for model: {self.name}")
                self.params_optimised = None
                self.params_covar = None
                self.params_err = None
                self.sum_of_squares_error = 1e10
                self.aic = 1e10
                self.aic_corrected = 1e10
                self.bgbelowzero = None
                return

        except RuntimeError:
            print(f"Exceeded runtime to fit for model: {self.name}")
            # assign values
            self.params_optimised = None
            self.params_covar = None
            self.params_err = None
            self.sum_of_squares_error = 1e10
            self.aic = 1e10
            self.aic_corrected = 1e10
            self.bgbelowzero = None
            return

        # plt.plot(x, model(x, *popt))
        perr = np.sqrt(np.diag(pcov))
        # params = np.column_stack((popt, perr))
        # print(params)
        k = float(self.n_params + 1)  # No. free parameters,
        # including var. of residuals
        # for least squares fit.

        ssr = np.sum((self.model_rpd_wrapper(x, popt)[0] - y) ** 2)
        aic = len(x) * np.log(ssr / len(x)) + 2 * k
        if len(x) - k - 1 == 0:
            aiccorr = 1e6
        else:
            aiccorr = aic + 2 * k * (k + 1) / (len(x) - k - 1)

        # check bg
        bg = self.model_rpd_wrapper(x, popt)[1]
        if (bg < 0).any():
            self.bgbelowzero = True
        else:
            self.bgbelowzero = False

        # check if params at edge of bound
        if (np.round(popt, 2) == self.param_bounds[0]).any() or (
            np.round(popt, 2) == self.param_bounds[1]
        ).any():
            self.popt_at_bound = True
        else:
            self.popt_at_bound = False

        # check large uncertainties
        if (perr > abs(popt)).any():
            self.large_uncertainty = True
        else:
            self.large_uncertainty = False

        # assign values
        self.params_optimised = popt
        self.params_covar = pcov
        self.params_err = perr
        self.sum_of_squares_error = ssr
        self.aic = aic
        self.aic_corrected = aiccorr

    def calculate_stdev(self, x):
        """Use automatic differentiation to acquire derivatives, and multiply
        with covariance matrix to acquire variance, then sd of a model relative
        position distribution. When numdifftools.Gradient runs, output to screen
        from the function it is differentiating is suppressed.

        Args:
            x (numpy array):
                Distances at which the model is evaluated.
            params_optimised (numpy array):
                The optimised parameter values of the model, following a fit to
                experimental data.
            params_covar (numpy array):
                The covariance matrix for the parameters that have been optimised.
            vector_input_model (function):
                The name of a function that will be passed to
                numdifftools.Gradient. This function must have one arguments, which
                is a vector. The first element of the vector is also the vector of
                x_values.

        Returns:
            stdev (numpy array):
                Numpy array of 1 standard deviation of uncertaintly on the model
                evaluated at each x_value.

        """

        # We will calculate 1 sd at each x-value
        stdev = np.zeros(len(x))

        for i, x_value in enumerate(x):

            # Pass arguments as required for differentiation
            vector_input = np.concatenate(([x_value], self.params_optimised))
            grads = nd.Gradient(self.model_rpd_wrapper_vector)(vector_input)

            # From gradients with respect to each parameter and the covariance
            # matrix, calculate the total variance.
            jacobian = grads[1:]
            variance = np.dot(jacobian, np.dot(self.params_covar, jacobian))
            stdev[i] = np.sqrt(variance)

        return stdev

    def plot_distance_hist_and_fit(
        self,
        distances,
        bin_edges,
        bin_centres,
        fitlength,
        plot_95ci=True,
        color="xkcd:red",
    ):
        """Plot the distance histogram and overlay the fit. And save the plots

        Args:
            distances: Distances being plotted as histogram
            bin_edges: Edges of histogram bins
            bin_centres: Centres of histogram bins
            fitlength: Distance up to which to fit
            plot_95ci: Whether to plot the 95% CI
            color: Colour of the fit line
        """

        if self.params_optimised is None:
            return None

        fig = plt.figure()
        axes = plt.subplot(111)

        hist_counts, _ = np.histogram(distances, bins=bin_edges)
        if self.normalise:
            # /= gives different type error
            hist_counts = hist_counts / bin_centres ** (self.dimension - 1)

        axes.stairs(hist_counts, bin_edges, color="grey", alpha=0.5, fill=True)

        axes.plot(
            bin_centres,
            self.model_rpd_wrapper(bin_centres, self.params_optimised)[0],
            color=color,
            lw=0.75,
        )
        axes.set_xlim([0, fitlength])
        axes.set_ylim(bottom=0)
        axes.set_ylabel("Counts")
        axes.set_xlabel("Distance between localisations")

        # Get 1 SD uncertainty on model result from uncertainty on parameters
        # and plot 95% CI.
        if plot_95ci is True:

            stdev = self.calculate_stdev(
                bin_centres,
            )

            axes.fill_between(
                bin_centres,
                self.model_rpd_wrapper(bin_centres, self.params_optimised)[0]
                - stdev * 1.96,
                self.model_rpd_wrapper(bin_centres, self.params_optimised)[0]
                + stdev * 1.96,
                facecolor=color,
                alpha=0.25,
            )

        return fig

    def plot_distance_kde_and_fit(
        self,
        calc_points,
        rpd,
        fitlength,
        plot_95ci=True,
    ):
        """Plot the KDE of the RPD and overlay the fit. And save the plots

        Args:
            calc_points: Points where RPD was calculated
            rpd: Value of RPD at each calculation point
            fitlength: Distance up to which to fit
            plot_95ci: Whether to plot the 95% CI
        """

        if self.params_optimised is None:
            return None

        calc_points = calc_points[calc_points <= fitlength]

        fig = plt.figure()
        axes = plt.subplot(111)

        if self.normalise:
            rpd /= calc_points ** (self.dimension - 1)

        axes.scatter(
            calc_points, rpd, s=10, marker="x", color="C0", label="Experimental data"
        )
        axes.plot(
            calc_points,
            self.model_rpd_wrapper(calc_points, self.params_optimised)[0],
            color="C1",
            lw=0.75,
            label="Model",
        )
        axes.set_xlim([0, fitlength])
        bottom, top = axes.get_ylim()
        if bottom < 0:
            axes.set_ylim(bottom=0)
        axes.set_ylabel("Counts")
        axes.set_xlabel("Distance between localisations")

        # Get 1 SD uncertainty on model result from uncertainty on parameters
        # and plot 95% CI.
        if plot_95ci is True:

            stdev = self.calculate_stdev(
                calc_points,
            )

            axes.fill_between(
                calc_points,
                self.model_rpd_wrapper(calc_points, self.params_optimised)[0]
                - stdev * 1.96,
                self.model_rpd_wrapper(calc_points, self.params_optimised)[0]
                + stdev * 1.96,
                facecolor="C2",
                alpha=0.25,
            )

        axes.legend()

        return fig

    def plot_model_components(
        self,
        fitlength,
    ):
        """Plot the components of this Z-disk model.
        Args:
            fitlength (integer): Distance up to to which the model is plotted.
        """

        if self.params_optimised is None:
            return None

        x = np.arange(0, fitlength + 1, 1)

        fig = plt.figure()
        axes = plt.subplot(111)

        rpd, background, characteristic_distance_terms, rep_locs = (
            self.model_rpd_wrapper(x, self.params_optimised)
        )

        # Plot background term
        if not self.background is None:
            axes.plot(x, background, label="Background", color="C0")
            if (background < 0).any():
                axes.set_title("BG. goes below zero")
                assert self.bgbelowzero

        # Plot characteristic distance term
        if characteristic_distance_terms is not None:
            for i, characteristic_distance_term in enumerate(
                characteristic_distance_terms
            ):
                if self.characteristic_distance_type == "one":
                    label = f"Repeat distance {i+1}"
                else:
                    label = f"Characteristic distance {i+1}"
                axes.plot(
                    x,
                    characteristic_distance_term,
                    label=label,
                    color=f"C{i+4}",
                )

        # Plot repeat term for localisations of the same molecule
        if rep_locs is not None:
            axes.plot(x, rep_locs, label="Repeated localisations", color="C2")

        # Plot full model
        axes.plot(x, rpd, label="Full model", color="C3")

        axes.set_xlim([0, fitlength])
        axes.set_ylim(0, None)
        axes.set_ylabel("Counts")
        axes.set_xlabel("Distance between localisations")
        axes.legend()

        return fig


class ModelWithFitSettings:
    """Class containing a model relative position distribution (model_rpd)
    that will be fitted, together with other input required
    by scipy.optimize.curve_fit.

    Attributes
    ----------
    model_rpd (function):
        The parameterised model rpd that can be fitted.

    initial_params (array_like):
        Initial guesses for the model parameters to pass to
        scipy.optimize.curve_fit as p0.  If p0 is passed as None,
        then the initial values used by scipy.optimize.curve_fit will all be 1.

    param_bounds (2-tuple of array_like):
        Lower and upper bounds on parameters to pass to
        scipy.optimize.curve_fit as bounds.
        ...curve_fit defaults to no bounds if not present.
        Each element of the tuple must be either an array with the length
        equal to the number of parameters, or a scalar (in which case the
        bound is taken to be the same for all parameters.)
        Use np.inf with an appropriate sign to disable bounds on all or some
        parameters.
    """

    def __init__(
        self, model_rpd, initial_params=None, param_bounds=None, vector_input_model=None
    ):
        self.model_rpd = model_rpd
        self.initial_params = initial_params
        self.param_bounds = param_bounds
        self.vector_input_model = vector_input_model


def pairwise_correlation_3d(r, rmean, sigma):
    """
    Apparent density of separations(r) for two repeatedly localised fluorophores in 3D
    with true separation rmean.
    sigma = sum in quadrature of sigma for each fluorophore,
    so sigma ** 2 = 2 * loc.prec ** 2. for repeated locs of the same molecule.
    From Churchman, Biophys J 90, 668-671 (2006).
    Need Bessel function, imported as i0(x)
    Args:
        r:
        r_mean:
        sigma:
    Return:
        p:
    """
    if rmean == 0:
        # Using approximation of sinh from
        # http://mathworld.wolfram.com/SeriesExpansion.html
        p = (
            np.sqrt(2 / np.pi)
            * (r**2 / sigma**3)
            * (np.exp(-(rmean**2 + r**2) / (2 * sigma**2)))
        )
    # if rmean < (sigma * 5.):
    elif (np.max(r) * rmean / sigma**2) < 700.0:
        p = (
            np.sqrt(2 / np.pi)
            * (r / (sigma * rmean))
            * (
                np.exp(-(rmean**2 + r**2) / (2 * sigma**2))
                * np.sinh(r * rmean / sigma**2)
            )
        )
    else:  # Approximate overly large sinh()
        p = (
            1.0
            / 2.0
            * np.sqrt(2 / np.pi)
            * (r / (sigma * rmean))
            * np.exp(-((r - rmean) ** 2) / (2 * sigma**2))
        )
    return p


def pairwise_correlation_2d(r, rmean, sigma):
    """
    The main bulding block of the 2D model. Although it does not model the
    background.
    Apparent density of separations(r) for two repeatedly localised fluorophores in 2D
    with true separation rmean.
    sigma = sum in quadrature of sigma for each fluorophore,
    so sigma ** 2 = 2 * loc.prec ** 2. for repeated locs of the same molecule.
    From Churchman, Biophys J 90, 668-671 (2006).
    Need Bessel function, imported as i0(x)

    Args:
        r: Numpy array of distances at which you want to evaluate a correlation
           function.
        rmean: Single float value that is the mean distance between the 2
               Gaussian densities.
        sigma is the spread of the Gaussian function.
    Returns:
        p: Numpy array of the probability density of the correclation function
           at the values of r.
    """
    # If r is a single value, rather than an array:
    if np.isscalar(r):
        p = 0  # Might help to diagnose if there are problems
        if (rmean * r / sigma**2) < 700:
            p = (r / sigma**2) * (
                np.exp(-(rmean**2 + r**2) / (2 * sigma**2)) * i0(r * rmean / sigma**2)
            )
        else:  # Approximate overly large i0()
            p = (
                1
                / (np.sqrt(2 * np.pi) * sigma)
                * np.sqrt(r / rmean)
                * np.exp(-((r - rmean) ** 2) / (2 * sigma**2))
            )

    # If z is an array of distances at which to evaluate the function:
    else:
        if (np.max(r) * rmean / sigma**2) < 700.0:
            p = (
                r
                / sigma**2
                * (
                    np.exp(-(rmean**2 + r**2) / (2 * sigma**2))
                    * i0(r * rmean / sigma**2)
                )
            )
        else:  # Approximate for overly large i0()
            p = (
                1
                / (np.sqrt(2 * np.pi) * sigma)
                * np.sqrt(r / rmean)
                * np.exp(-((r - rmean) ** 2) / (2 * sigma**2))
            )
    return p


def pair_corr_2d_standardised(r, rmean, sigma):
    """
    The main bulding block of the 2D model. Although it does not model the
    background.
    Apparent density of separations(r) for two repeatedly localised fluorophores in 2D
    with true separation rmean, standardised by dividing the separations.
    sigma = sum in quadrature of sigma for each fluorophore,
    so sigma ** 2 = 2 * loc.prec ** 2. for repeated locs of the same molecule.
    From Churchman, Biophys J 90, 668-671 (2006).
    Need Bessel function, imported as i0(x)

    Args:
        r: Numpy array of distances at which you want to evaluate a correlation
           function.
        rmean: Single float value that is the mean distance between the 2
               Gaussian densities.
        sigma is the spread of the Gaussian function.
    Returns:
        p: Numpy array of the probability density of the correclation function
           at the values of r.
    """
    # If r is a single value, rather than an array:
    if np.isscalar(r):
        p = 0  # Might help to diagnose if there are problems
        if (rmean * r / sigma**2) < 700:
            p = (
                1
                / r
                * (r / sigma**2)
                * (
                    np.exp(-(rmean**2 + r**2) / (2 * sigma**2))
                    * i0(r * rmean / sigma**2)
                )
            )
        else:  # Approximate overly large i0()
            p = (
                1
                / r
                / (np.sqrt(2 * np.pi) * sigma)
                * np.sqrt(r / rmean)
                * np.exp(-((r - rmean) ** 2) / (2 * sigma**2))
            )

    # If z is an array of distances at which to evaluate the function:
    else:
        if (np.max(r) * rmean / sigma**2) < 700.0:
            p = (
                1
                / r
                * (
                    r
                    / sigma**2
                    * (
                        np.exp(-(rmean**2 + r**2) / (2 * sigma**2))
                        * i0(r * rmean / sigma**2)
                    )
                )
            )
        else:  # Approximate for overly large i0()
            p = (
                1
                / r
                * (
                    1
                    / (np.sqrt(2 * np.pi) * sigma)
                    * np.sqrt(r / rmean)
                    * np.exp(-((r - rmean) ** 2) / (2 * sigma**2))
                )
            )
    return p


def pairwise_correlation_1d(z, zmean, sigma):
    """
    Apparent density of separations(z) for two repeatedly localised fluorophores
    with true separation rmean.
    Sigma = sum in quadrature of sigma for each fluorophore,
    so sigma ** 2 = 2 * loc.prec ** 2. for repeated locs of the same molecule.
    From Churchman, Biophys J 90, 668-671 (2006).
    Need Bessel function, imported as j0(x)
    """
    # Improve approximation to depend on z as well as zmean wrt sigma

    # If z is a single value, rather than an array:
    if np.isscalar(z):
        p = 0  # Might help to diagnose if there are problems
        if (zmean * z / sigma**2) < 500:
            p = (
                np.sqrt(2 / np.pi)
                * 1.0
                / sigma
                * (
                    np.exp(-(zmean**2 + z**2) / (2 * sigma**2))
                    * np.cosh(zmean * z / sigma**2)
                )
            )
        else:
            p = (
                (1.0 / 2.0)
                * np.sqrt(2 / np.pi)
                * 1.0
                / sigma
                * (np.exp(-((z - zmean) ** 2) / (2 * sigma**2)))
            )

    # If z is an array of distances at which to evaluate the function:
    else:
        p = z * 0.0
        for i, test_z in enumerate(z):
            if (zmean * test_z / sigma**2) < 500:
                p[i] = (
                    np.sqrt(2 / np.pi)
                    * 1.0
                    / sigma
                    * np.exp(-(zmean**2 + test_z**2) / (2 * sigma**2))
                    * np.cosh(zmean * test_z / sigma**2)
                )
            else:
                p[i] = (
                    (1.0 / 2.0)
                    * np.sqrt(2 / np.pi)
                    * 1.0
                    / sigma
                    * (np.exp(-((test_z - zmean) ** 2) / (2 * sigma**2)))
                )

    # if zmean < (sigma * 10.):
    #    p = np.sqrt(2 / np.pi) * 1 / sigma * (np.exp(-(zmean ** 2 + z ** 2) \
    #                / (2 * sigma ** 2)) * np.cosh(zmean * z / sigma ** 2))
    # else:  # Approximate overly large np.cosh()
    #    """ Old version - mistakenly kept z/zmean factor from 2D case
    #    in multiplier. Obviously, this is 1 at z=zmean. """
    # p = 1. / 2. * np.sqrt(2 / np.pi) * 1 / sigma * np.sqrt(z / zmean) * (
    #        np.exp(-((z - zmean) ** 2) / (2 * sigma ** 2))
    #        )
    #    """ New version - removed z/zmean in multiplier. """
    #    p = (1. / 2.) * np.sqrt(2 / np.pi) * 1 / sigma \
    #        * (np.exp(-((z - zmean) ** 2) / (2 * sigma ** 2)))
    return p


def gauss1d(x_values, xmean, sigma):
    """Apparent density of separations(z) for two repeatedly localised fluorophores
    with true separation rmean, when zmean >> sigma.
    Sigma = sum in quadrature of sigma for each fluorophore.
    """
    p = (
        np.sqrt(2 / np.pi)
        * 1
        / sigma
        * (np.exp(-((x_values - xmean) ** 2) / (2 * sigma**2)))
    )
    return p


def generate_polygon_points(n, d):
    """Generate the coordinates of points on an polygon.

    Args:
        n: Number of vertices.
        d: Diameter of circle on which the vertices are found.

    Returns:
        v: Numpy [x, y] coordinates of vertices. One [x, y] row
            per vertex.
    """
    v = np.zeros((n, 2))
    for a in range(n):
        v[a, 0] = d / 2 * np.cos(a * 2 * np.pi / n)
        v[a, 1] = d / 2 * np.sin(a * 2 * np.pi / n)

    return v


def linear_fit(x_values, slope, offset):
    """Generate (return) y values for a linear function at given values of
    x, slope and offset.
    """
    rpd = offset + slope * x_values
    return rpd


def rotate_3d_vectors(vectors, axis, angle):
    """Rotate 3D vectors (or coordinates) by an angle about an axis. Normalise
    the axis vector so that it does not need to be already normalised when
    passed to this function.

    Parameters
    ----------
    vectors : numpy array, shape (n, 3), where columns are XYZ
        3D vectors to be rotated.
    axis : numpy array, shape (3,)
        Vector with direction of the rotation axis.
    angle : float
        Angle of rotation about the axis (degrees).

    Returns
    -------
    rotated_vectors : numpy array, shape (n, 3)
        Input vectors rotated according to the input parameters.
    """
    # Normalise rotation axis vector, in case it is not normalised
    length = np.linalg.norm(axis)
    axis = axis / length

    # Initialise scipy Rotation from rotation vector
    angle_radians = np.radians(angle)
    rotation_vector = angle_radians * axis
    rotation = Rotation.from_rotvec(rotation_vector)

    # Calculate output vectors
    rotated_vectors = rotation.apply(vectors)

    return rotated_vectors


def make_x_histogram_nm(dists, fitlength=400.0, axes=None):
    """Generate a histogram of distances in one direction (e.g. X) between
    localisations, with 1-nm bins. Bin values averaged so that the mean = 1
    (this helps with using scipy.optimise.curve_fit).

    Args:
        dists:     numpy array of distances between localisations (1D).
        fitlength:  Maximum distance included in the histogram.
    Returns:
        xyhist:     Histogram bin values with mean = 1.
        fitlength:  Maximum distance included in the histogram.
    """
    if axes is None:
        plt.figure()
        axes = plt.subplot(111)
    distancehist = plt.hist(
        dists,
        weights=np.repeat(float(fitlength) / len(dists), len(dists)),
        bins=np.arange(float(fitlength) + 1),
        color="lightblue",
    )[0]
    return distancehist, fitlength


def make_xy_histogram_nm(
    relpos, fitlength=400.0, axes=None, fig_toggle=True
):  # plot_toggle):
    """Generate a histogram of Euclidean distance in XY between localisations,
    with 1-nm bins. Bin values averaged so that the mean = 1
    (this helps with using scipy.optimise.curve_fit).

    Args:
        relpos:     numpy array of 2D or 3D relative positions
                        (N relative positions in N rows).
        fitlength:  Maximum distance included in the histogram.
    Returns:
        xyhist:     Histogram bin values with mean = 1.
        fitlength:  Maximum distance included in the histogram.
    """
    xydists = np.sqrt(relpos[:, 0] ** 2 + relpos[:, 1] ** 2)

    if fig_toggle is True:
        if axes is None:
            plt.figure()
            axes = plt.subplot(111)
        xyhist = plt.hist(
            xydists,
            weights=np.repeat(float(fitlength) / len(xydists), len(xydists)),
            bins=np.arange(float(fitlength) + 1),
            color="lightblue",
            alpha=0.5,
        )[0]
        return xyhist, fitlength
    if fig_toggle is False:
        xyhist = np.histogram(
            xydists,
            weights=np.repeat(float(fitlength) / len(xydists), len(xydists)),
            bins=np.arange(float(fitlength) + 1),
        )[0]
        return xyhist, fitlength


def make_xyz_histogram_nm(relpos, fitlength=400.0, axes=None):
    """Generate a histogram of 3D Euclidean distances between localisations,
    with 1-nm bins. Bin values averaged so that the mean = 1
    (this helps with using scipy.optimise.curve_fit).

    Args:
        relpos:     numpy array of 3D relative positions
                        (N relative positions in N rows).
        fitlength:  Maximum distance included in the histogram.
    Returns:
        xyzhist:     Histogram bin values with mean = 1.
        fitlength:  Maximum distance included in the histogram.
    """
    xyzdists = np.sqrt(relpos[:, 0] ** 2 + relpos[:, 1] ** 2 + relpos[:, 2] ** 2)
    if axes is None:
        plt.figure()
        axes = plt.subplot(111)
    xyzhist = plt.hist(
        xyzdists,
        weights=np.repeat(float(fitlength) / len(xyzdists), len(xyzdists)),
        bins=np.arange(float(fitlength) + 1),
        color="lightblue",
        alpha=0.5,
    )[0]
    return xyzhist, fitlength


def kde_1nm(distances, locprec, fitlength):
    """Evaluate KDE of distances between localisations at steps of
    1 nm, starting at 0.5 nm. Use localisation precision * np.sqrt(2) as the
    Gaussian kernel.

    Args:
        distances (numpy-like array):
            Distances between localisations (e.g. axial separation
                                                    in the Z-disk).
        locprec (float):
            Average localisation precision of the localisations involved.
        fitlength (float):
            The distance upto which the KDE will be evaluated.
    Returns:
        x (numpy array):
            The distances at which the KDE was evaluated.
        kde (numpy array):
            The KDE of the distance distribution, evaluated at x.
    """
    smoothing = np.sqrt(2) * locprec
    kernel_scalar = smoothing / np.std(distances)
    kernel = stats.gaussian_kde(distances, bw_method=kernel_scalar)
    x = np.arange(np.round(smoothing * 3), fitlength + 1, 1.0)

    # Multiply to match histogram, as KDE gets normalised as a pdf.
    kde = kernel(x) * len(distances)

    return x, kde


def fit_model_to_experiment(
    expt, model, param_guesses, param_bounds, fitlength=400.0, **kwargs
):
    """Use scipy.optimize.curve_fit to do non-linear least-squares fitting
    of model relative position distribution to an experimental distribution,
    e.g. histogram or kernel density estimation.

    Args:
        expt:
            Experimental pairwise distance distribution, evaluated
            at n + 0.5 nm, when n is an integer (histogram bin values).
        model:
            Parametric model distribution.
        param_guesses:
            Starting guesses for parameter values in scipy's curve_fit.
        param_bounds:
            Bounds on the allowed parameter values during optimisation
            in scipy's curve_fit.
        fitlength:
            Maximum distance included in the fit.

    Returns:
        params_optimised:
            Optimised parameters.
        params_covar:
            Covariance matrix between parameters.
        params_1sd_error:
            Error (1 SD) on parameters.
    """
    # Find estimates and covariances of model parameters
    (params_optimised, params_covar) = curve_fit(
        model,
        np.arange(fitlength) + 0.5,
        expt,
        p0=param_guesses,
        bounds=param_bounds,
        **kwargs,
    )

    # Calculate uncertainty (1 SD)
    params_1sd_err = np.sqrt(np.diag(params_covar))

    # Print out parameter estimates and errors.
    # print('Fitted parameters:')
    # print(np.column_stack((params_optimised, params_1sd_err)))

    return params_optimised, params_covar, params_1sd_err


def stdev_of_model(x_values, params_optimised, params_covar, vector_input_model):
    """Use automatic differentiation to acquire derivatives, and multiply
    with covariance matrix to acquire variance, then sd of a model relative
    position distribution. When numdifftools.Gradient runs, output to screen
    from the function it is differentiating is suppressed.

    Args:
        x_values (numpy array):
            Distances at which the model is evaluated.
        params_optimised (numpy array):
            The optimised parameter values of the model, following a fit to
            experimental data.
        params_covar (numpy array):
            The covariance matrix for the parameters that have been optimised.
        vector_input_model (function):
            The name of a function that will be passed to
            numdifftools.Gradient. This function must have one arguments, which
            is a vector. The first element of the vector is also the vector of
            x_values.

    Returns:
        stdev (numpy array):
            Numpy array of 1 standard deviation of uncertaintly on the model
            evaluated at each x_value.

    """
    # print('Starting calculation of uncertainty on the model estimates.')
    # start_time = time.time()

    # We will calculate 1 sd at each x-value
    stdev = np.zeros(len(x_values))

    for i, x_value in enumerate(x_values):

        # Pass arguments as required for differentiation
        vector_input = np.concatenate(([x_value], params_optimised))
        grads = nd.Gradient(vector_input_model)(vector_input)

        # From gradients with respect to each parameter and the covariance
        # matrix, calculate the total variance.
        jacobian = grads[1:]
        variance = np.dot(jacobian, np.dot(params_covar, jacobian))
        stdev[i] = np.sqrt(variance)

        # Update
        # if (i + 1) % 20 == 0:
        #     time_elapsed = time.time() - start_time
        #     print('Done ' + repr(i + 1) + ' calculations of '
        #           + repr(len(x_values)) + ' in '
        #           + repr(time_elapsed) + '.')

    return stdev


def correlation_matrix_from_covariance(pcov):
    """Calculate the correlation matrix for parameters from their
    covariance matrix (pcov).
    """
    perr = np.sqrt(np.diag(pcov))
    inverr = 1 / perr
    inverr = np.diag(inverr)
    corrmatrix = np.matmul(inverr, np.matmul(pcov, inverr))
    return corrmatrix
