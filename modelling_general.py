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
from scipy.spatial.transform import Rotation
from scipy import i0
from scipy import stats
from scipy.optimize import curve_fit


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
    def __init__(self, model_rpd,
                 initial_params=None,
                 param_bounds=None,
                 vector_input_model=None):
        self.model_rpd = model_rpd
        self.initial_params = initial_params
        self.param_bounds = param_bounds
        self.vector_input_model = vector_input_model


def pairwise_correlation_3d(r, rmean, sigma):
    """
    Apparent density of separations(r) for two repeatedly localised fluorophores in 2D
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
        p = np.sqrt(2 / np.pi) \
            * (r ** 2 / sigma ** 3) \
            * (np.exp(-(rmean ** 2 + r ** 2)
               / (2 * sigma ** 2))
               )
    # if rmean < (sigma * 5.):
    elif (np.max(r) * rmean / sigma ** 2) < 700.:
        p = np.sqrt(2 / np.pi) * (r / (sigma * rmean)) \
                * (np.exp(-(rmean ** 2 + r ** 2) / (2 * sigma ** 2)) \
                * np.sinh(r * rmean / sigma ** 2))
    else:  # Approximate overly large sinh()
        p = 1. / 2. * np.sqrt(2 / np.pi) * (r / (sigma * rmean)) \
            * np.exp(-((r - rmean) ** 2) / (2 * sigma ** 2))
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
        p = 0 # Might help to diagnose if there are problems
        if (rmean * r / sigma ** 2) < 700:   
            p = (r / sigma ** 2) * (np.exp(-(rmean ** 2 + r ** 2) \
                / (2 * sigma ** 2)) * i0(r * rmean / sigma ** 2))
        else:  # Approximate overly large i0()
            p = 1 / (np.sqrt(2 * np.pi) * sigma) * np.sqrt(r / rmean) \
                * np.exp(-((r - rmean) ** 2) / (2 * sigma ** 2))

    # If z is an array of distances at which to evaluate the function:    
    else:
        if (np.max(r) * rmean / sigma ** 2) < 700.:
            p = (r / sigma ** 2
                    * (np.exp(-(rmean ** 2 + r ** 2)
                                / (2 * sigma ** 2)
                                )
                        * i0(r * rmean / sigma ** 2)
                        )
                    )
        else:  # Approximate for overly large i0()
            p = (1 / (np.sqrt(2 * np.pi) * sigma)
                    * np.sqrt(r / rmean)
                    * np.exp(-((r - rmean) ** 2) / (2 * sigma ** 2))
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
        p = 0 # Might help to diagnose if there are problems
        if (rmean * r / sigma ** 2) < 700:   
            p = 1 / r * (r / sigma ** 2) * (np.exp(-(rmean ** 2 + r ** 2) \
                / (2 * sigma ** 2)) * i0(r * rmean / sigma ** 2))
        else:  # Approximate overly large i0()
            p = 1 / r / (np.sqrt(2 * np.pi) * sigma) * np.sqrt(r / rmean) \
                * np.exp(-((r - rmean) ** 2) / (2 * sigma ** 2))

    # If z is an array of distances at which to evaluate the function:    
    else:
        if (np.max(r) * rmean / sigma ** 2) < 700.:
            p = 1 / r * (r / sigma ** 2
                    * (np.exp(-(rmean ** 2 + r ** 2)
                                / (2 * sigma ** 2)
                                )
                        * i0(r * rmean / sigma ** 2)
                        )
                    )
        else:  # Approximate for overly large i0()
            p = 1 / r * (1 / (np.sqrt(2 * np.pi) * sigma)
                    * np.sqrt(r / rmean)
                    * np.exp(-((r - rmean) ** 2) / (2 * sigma ** 2))
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
        p = 0 # Might help to diagnose if there are problems
        if (zmean * z / sigma ** 2) < 500:
            p = (np.sqrt(2 / np.pi) * 1. / sigma
                 * (np.exp(-(zmean ** 2 + z ** 2)
                 / (2 * sigma ** 2)) * np.cosh(zmean * z / sigma ** 2))
            )
        else:
            p = ((1. / 2.) * np.sqrt(2 / np.pi) * 1. / sigma
                 * (np.exp(-((z - zmean) ** 2) / (2 * sigma ** 2)))
            )

    # If z is an array of distances at which to evaluate the function:
    else:
        p = z * 0.
        for i, test_z in enumerate(z):
            if (zmean * test_z / sigma ** 2) < 500:
                p[i] = (np.sqrt(2 / np.pi) * 1. / sigma
                        * np.exp(-(zmean ** 2 + test_z ** 2)
                                 / (2 * sigma ** 2)
                                )
                        * np.cosh(zmean * test_z / sigma ** 2)
                )
            else:
                p[i] = ((1. / 2.) * np.sqrt(2 / np.pi) * 1. / sigma
                        * (np.exp(-((test_z - zmean) ** 2) / (2 * sigma ** 2)))
                )

    #if zmean < (sigma * 10.):
    #    p = np.sqrt(2 / np.pi) * 1 / sigma * (np.exp(-(zmean ** 2 + z ** 2) \
    #                / (2 * sigma ** 2)) * np.cosh(zmean * z / sigma ** 2))
    #else:  # Approximate overly large np.cosh()
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
    p = np.sqrt(2 / np.pi) * 1 / sigma \
        * (np.exp(-((x_values - xmean) ** 2) / (2 * sigma ** 2)))
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


def make_x_histogram_nm(dists, fitlength=400., axes=None):
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
    distancehist = plt.hist(dists,
                            weights=np.repeat(float(fitlength) / len(dists),
                                              len(dists)),
                            bins=np.arange(float(fitlength) + 1),
                            color='lightblue')[0]
    return distancehist, fitlength


def make_xy_histogram_nm(relpos, fitlength=400., axes=None, fig_toggle=True):  # plot_toggle):
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
        xyhist = plt.hist(xydists,
                          weights=np.repeat(float(fitlength) / len(xydists),
                                            len(xydists)),
                          bins=np.arange(float(fitlength) + 1),
                          color='lightblue',
                          alpha=0.5)[0]
        return xyhist, fitlength
    if fig_toggle is False:
        xyhist = np.histogram(
                    xydists,
                    weights=np.repeat(float(fitlength) / len(xydists),
                                      len(xydists)
                                      ),
                    bins=np.arange(float(fitlength) + 1)
                    )[0]
        return xyhist, fitlength


def make_xyz_histogram_nm(relpos, fitlength=400., axes=None):
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
    xyzdists = np.sqrt(relpos[:, 0] ** 2
                       + relpos[:, 1] ** 2
                       + relpos[:, 2] ** 2)
    if axes is None:
        plt.figure()
        axes = plt.subplot(111)
    xyzhist = plt.hist(xyzdists,
                       weights=np.repeat(float(fitlength) / len(xyzdists),
                                         len(xyzdists)
                                         ),
                       bins=np.arange(float(fitlength) + 1),
                       color='lightblue',
                       alpha=0.5)[0]
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
    x = np.arange(np.round(smoothing * 3), fitlength + 1, 1.)

    # Multiply to match histogram, as KDE gets normalised as a pdf.
    kde = kernel(x) * len(distances)

    return x, kde


def fit_model_to_experiment(expt,
                            model,
                            param_guesses,
                            param_bounds,
                            fitlength=400., **kwargs):
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
    (params_optimised,
     params_covar) = curve_fit(model,
                               np.arange(fitlength) + 0.5,
                               expt,
                               p0=param_guesses,
                               bounds=param_bounds,
                               **kwargs
                               )

    # Calculate uncertainty (1 SD)
    params_1sd_err = np.sqrt(np.diag(params_covar))

    # Print out parameter estimates and errors.
    # print('Fitted parameters:')
    # print(np.column_stack((params_optimised, params_1sd_err)))

    return params_optimised, params_covar, params_1sd_err


def stdev_of_model(x_values,
                   params_optimised,
                   params_covar,
                   vector_input_model):
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
    #print('Starting calculation of uncertainty on the model estimates.')
    #start_time = time.time()

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
