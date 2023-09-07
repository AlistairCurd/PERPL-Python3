"""
zdisk_plots.py

Functions for plotting figures for manuscript on PERPL
technique.

Alistair Curd
University of Leeds
30 March 2020

---
Copyright 2020 Peckham Lab

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at
http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import linearrepeatmodels as linmods
import zdisk_modelling
from modelling_general import kde_1nm
from modelling_general import pairwise_correlation_1d
from modelling_general import stdev_of_model
from plotting import estimate_rpd_churchman_1d
from zdisk_modelling import read_relpos_from_pickles
from zdisk_modelling import getaxialseparations_no_smoothing
from zdisk_modelling import remove_duplicates
from zdisk_modelling import set_up_model_4_variable_peaks_with_fit_settings
from zdisk_modelling import set_up_model_5_variable_peaks_with_fit_settings
from zdisk_modelling import fitmodel_to_hist


def plot_distance_hist(distances, fitlength, color='gray'):
    """Plot histogram of experimental distances, with 1 nm bins.

    Args:
        distances (numpy array-like):
            Set of distances (nm) to plot.
        fitlength (float):
            The distance upto which the histogram will be calculated.
        color (string):
            Colour for the matplotlib histogram.
    Returns:
        hist_values (numpy array):
            Histogram bin values.
        bin_edges (numpy array):
            Histogram bin edge positions. 
    """
    # Histogram figure with 1-nm bins
    histfig = plt.figure()
    histaxes = histfig.add_subplot(111)
    hist_values, bin_edges = histaxes.hist(distances,
                                           bins=np.arange(fitlength + 1),
                                           color=color, alpha=0.5
                                           )[0:2] # 2 not required
    histaxes.set_xlim([0, fitlength])
    # histaxes.set_ylim([0, 82])
    histaxes.set_ylim(bottom=0)
    histaxes.set_title('Histogram')
    histaxes.set_xlabel(r'$\Delta$X (nm) ($\Delta$YZ < 10 nm)')
    histaxes.set_ylabel('Counts')

    return hist_values, bin_edges


def plot_distance_kde(distances, locprec, fitlength, color='gray'):
    """Plot KDE (using localisation precision) of experimental distances.

    Args:
        distances (numpy array-like):
            Set of distances (nm) to plot.
        locprec:
            Average localisation precision of the localisations involved.
        fitlength (float):
            The distance upto which the KDE will be evaluated.
        color (string):
            Colour for the shaded KDE.
    Returns:
        kde_x_values (numpy array):
            Distances at which the kde is evaluated.
        kde (numpy array):
            The density estimated at kde_x_points.
    """
    # Set kde kernel size
    kernel_size = np.sqrt(2) * locprec

    kdefig = plt.figure()
    kdeaxes = kdefig.add_subplot(122)
    kde_x_values, kde = kde_1nm(distances, locprec, fitlength)
    kdeaxes.fill_between(kde_x_values, 0, kde, color=color, alpha=0.5, lw=0)
    # Start x-axis from 3 * kernel size, to avoid kde problems around zero.
    kdeaxes.set_xlim([3 * kernel_size, fitlength])
    # kdeaxes.set_ylim([0, 82])
    kdeaxes.set_ylim(bottom=0)
    kdeaxes.set_xlabel(r'$\Delta$X (nm) ($\Delta$YZ < 10 nm)')
    kdeaxes.set_title('KDE\n(Gaussian kernel, '
                      +r'$\sigma$ = {:.1f} nm)'.format(kernel_size))
    kdeaxes.set_ylabel('Counts')

    return kde_x_values, kde


def plot_distance_hist_and_fit(
                axpoints,
                fitlength,
                params_optimised,
                params_covar,
                model_with_info,
                plot_95ci=False,
                color='xkcd:red'):
    fig = plt.figure()

    axes = plt.subplot(111)

    bin_edges = np.arange(fitlength + 1)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    axes.hist(axpoints,
              bins=bin_edges,
              color='grey', alpha=0.5)[0]

    # x, kde = kde_1nm(axpoints)
    # ax.plot(x, kde)

    axes.plot(bin_centres,
              model_with_info.model_rpd(bin_centres, *params_optimised),
              color=color,
              lw=0.75
              )
    axes.set_xlim([0, fitlength])
    # axes.set_xlabel(r'$\Delta$X (nm) ($\Delta$YZ < 10 nm)')
    # axes.set_ylim([0, 82])
    axes.set_ylim(bottom=0)
    axes.set_ylabel('Counts')
    axes.set_title('Model: ' +model_with_info.model_rpd.__name__)

    # Get 1 SD uncertainty on model result from uncertainty on parameters
    # and plot 95% CI.
    if plot_95ci is True:
        stdev = stdev_of_model(bin_centres,
                               params_optimised,
                               params_covar,
                               model_with_info.vector_input_model
                               )

        axes.fill_between(bin_centres,
                          model_with_info.model_rpd(bin_centres,
                                                    *params_optimised)
                          - stdev * 1.96,
                          model_with_info.model_rpd(bin_centres,
                                                    *params_optimised)
                          + stdev * 1.96,
                          facecolor=color,
                          alpha=0.25
                          )

    return fig, axes


def plot_fitted_model(
    axpoints,
    fitlength,
    params_optimised,
    params_covar,
    model_with_info,
    plot_95ci=False,
    color='xkcd:red'
    ):
    """Plot model fitted to experimental data, with or without
    confidence intervals.
    """
    axpoints_fitted = axpoints[axpoints <= fitlength]
    plt.plot(axpoints_fitted,
              model_with_info.model_rpd(axpoints_fitted, *params_optimised),
              color=color,
              lw=0.75
              )

    # Get 1 SD uncertainty on model result from uncertainty on parameters
    # at n + 0.5 nm, and plot 95% CI.
    if plot_95ci is True:
        # Distances at which model is fitted
        estimate_points = np.arange(fitlength) + 1.
        # Distance at which background estimate gets to zero
        bg_offset = params_optimised[-1]
        bg_slope = params_optimised[-2]
        x_when_bg_is_zero = -bg_offset / bg_slope
        print('x at zero bg is ' + repr(x_when_bg_is_zero) + '.')
        
        # Calculate CIs for bi-partite model separately, if necessary
        # e.g. background is linear for one section and zero for another
        if type(model_with_info.vector_input_model) is list:
            # Confidence intervals (stdev) before background get to zero
            estimate_points_bg_linear = estimate_points[estimate_points
                                                        < x_when_bg_is_zero
                                                        ]
            stdev_bg_linear = stdev_of_model(estimate_points_bg_linear,
                                            params_optimised,
                                            params_covar,
                                            model_with_info.vector_input_model[0]
                                            )
            # Confidence intervals (stdev) when background is zero
            estimate_points_bg_zero = estimate_points[estimate_points
                                                    >= x_when_bg_is_zero
                                                    ]
            # print('Number of points with bg zero is ' + repr(len(estimate_points_bg_zero)))
            stdev_bg_zero = stdev_of_model(estimate_points_bg_zero,
                                        params_optimised[0:-2],
                                        params_covar[0:-2, 0:-2],
                                        model_with_info.vector_input_model[1]
                                        )
            # Combine confidence intervals
            stdev = np.append(stdev_bg_linear, stdev_bg_zero)
        
        else: # For one continuous model:
            stdev = stdev_of_model(estimate_points,
                                   params_optimised,
                                   params_covar,
                                   model_with_info.vector_input_model
                                   )

        # Plot confidence intervals (95%)
        plt.fill_between(estimate_points,
                         model_with_info.model_rpd(estimate_points,
                                                   *params_optimised)
                         - stdev * 1.96,
                         model_with_info.model_rpd(estimate_points,
                                                   *params_optimised)
                         + stdev * 1.96,
                         facecolor=color,
                         alpha=0.25
                         )


def plot_model_components_4peaks_fixed_peak_ratio(
    fitlength,
    repeat_distance,
    repeat_broadening,
    repeat_amplitude,
    loc_precision,
    loc_precision_amplitude,
    bg_slope,
    bg_offset
    ):
    """Plot the components of this Z-disk model.
    Args
    ----
    fitlength (integer):
        Cell-axial distance upto to which the model is plotted.
    repeat_distance (float):
        Axial repeat distance through the Z-disk.
    repeat_broadening (float):
        Broadening on the cell-axial repeat term.
    repeat_amplitude (float):
        Amplitude of the first peak of the 4. Amplitudes decrease
        with ratios 4:3:2:1.
    loc_precision (float):
        Broadening on the peak representing repeated localisations
        of the same molecule.
    loc_precision_amplitude (float):
        Amplitude of the peak representing repeated localisations
        of the same molecule.
    bg_slope (float):  
        Slope of the background term (isotropic within the thickness
        of the Z-disk)
    bg_offset (float):
        Value of the background term at distance = 0.
    """
    distance_values = np.arange(0, fitlength + 1, 1)

    plt.figure()
    axes = plt.subplot(111)
    axes.set_xlim([0, fitlength])
    axes.set_xlabel(r'$\Delta$X (nm) ($\Delta$YZ < 10 nm)')
    axes.set_ylim([0, 82])
    axes.set_ylabel('Counts')
    axes.set_title('Model: 5-layer Z-disk (4 peaks, fixed ratios)')
 
    # Plot background term
    axes.plot(distance_values,
              bg_offset + bg_slope * distance_values
              )

    # Plot linear repeat term
    repeat_component = np.zeros(len(distance_values))
    for i in range(4):
        repeat_component = (
            repeat_component
            + (1. - i / 4.) * repeat_amplitude
                            * pairwise_correlation_1d(
                                distance_values,
                                (i + 1) * repeat_distance,
                                repeat_broadening
                                )
            )    
    axes.plot(distance_values, repeat_component)

    # Plot term for localisations of the same molecule
    axes.plot(loc_precision_amplitude
              * pairwise_correlation_1d(distance_values,
                                        0.,
                                        np.sqrt(2) * loc_precision
                                        )
              )

    # Plot full model
    axes.plot(distance_values,
              linmods.linrepplusreps4fixedpeakratio(
                distance_values,
                repeat_distance,
                repeat_broadening,
                repeat_amplitude,
                loc_precision,
                loc_precision_amplitude,
                bg_slope,
                bg_offset                
                ),
              color='xkcd:red'
        )


def plot_model_components_5peaks_variable(
    distance_values,
    repeat_distance,
    repeat_broadening,
    peak1_amplitude,
    peak2_amplitude,
    peak3_amplitude,
    peak4_amplitude,
    peak5_amplitude,
    bg_slope,
    bg_offset
    ):
    """Plot the components of this Z-disk model.
    Args
    ----
    fitlength (integer):
        Cell-axial distance upto to which the model is plotted.
    repeat_distance (float):
        Axial repeat distance through the Z-disk.
    repeat_broadening (float):
        Broadening on the cell-axial repeat term.
    peak1_amplitude (float):
        Amplitude of the first peak due to the linear repeating pattern
        of localisations.
    peak2_amplitude to peak5_amplitude (float):
        As peak1_amplitude, but for peaks 2 to 5.
    bg_slope (float):  
        Slope of the background term (isotropic within the thickness
        of the Z-disk)
    bg_offset (float):
        Value of the background term at distance = 0.
    """
    plt.figure()
    axes = plt.subplot(111)
    axes.set_xlabel(r'$\Delta$X (nm) ($\Delta$YZ < 10 nm)')
    axes.set_ylabel('Counts')
    axes.set_title('Model: 6-layer Z-disk (5 peaks, independent amplitudes)')
 
    # Plot background term
    background = bg_offset + bg_slope * distance_values  # Background
    background[background < 0.] = 0. # Cannot be negative    
    axes.plot(distance_values, background)

    # Plot linear repeat terms
    amplitudes = [peak1_amplitude,
                  peak2_amplitude,
                  peak3_amplitude,
                  peak4_amplitude,
                  peak5_amplitude
                  ]
    for i in range(5):
        axes.plot(distance_values,
                  amplitudes[i]
                  * pairwise_correlation_1d(distance_values,
                                            (i + 1) * repeat_distance,
                                            repeat_broadening
                                            )
                  )

    # Plot full model
    axes.plot(distance_values,
              linmods.linrepnoreps5_bg_non_negative(distance_values,
                                                    repeat_distance,
                                                    repeat_broadening,
                                                    peak1_amplitude,
                                                    peak2_amplitude,
                                                    peak3_amplitude,
                                                    peak4_amplitude,
                                                    peak5_amplitude,
                                                    bg_slope,
                                                    bg_offset                
                                                    ),
              color='xkcd:red'
              )
    axes.set_xlim(distance_values.min(), distance_values.max())
    axes.set_ylim(bottom=0)


def actn_mEos_x_plot():
    # Get 2D (axial, transverse) relative positions
    relpos = pd.read_pickle('../perpl_test_data/ACTN2-mEos2_PERPL-relpos_200.0filter_6FOVs_aligned_len1229656.pkl')

    # Get subset of axial relative positions
    relpos.axial = abs(relpos.axial)
    fitlength = 100.
    # loc_precision = 3.4
    # separation_precision = np.sqrt(2) * loc_precision
    separation_precision = 6
    # The smoothing is used like this so that points beyond those not
    # included would only have 1.1% effect on the values of the kernal
    # density estimate (KDE) at fitlength.
    axpoints = getaxialseparations_no_smoothing(
                relpos=relpos,
                max_distance=fitlength + 3 * separation_precision,
                transverse_limit=10.
                )

    # Remove duplicates
    axpoints = remove_duplicates(axpoints)

    # Find kernel density estimate using localisation precision * sqrt(2)
    # as the Gaussian kernel.
    # Ues mean localisation precision estimate for mEos2:ACTN2,
    # after filtering to < 5 nm.
    # kde_x_values, kde = kde_1nm(axpoints,
    #                             locprec=loc_precision,
    #                             fitlength=fitlength)

    # Histogram of axial separation, in 1-nm bins
    bin_vals = np.arange(fitlength + 1)
    ax_histogram, bin_values = np.histogram(axpoints,
                                            bins=bin_vals)

    # Centre and width values for histogram bars
    ax_bar_points = (bin_values[:-1] + bin_values[1:]) / 2
    ax_fit_points = bin_vals[0:-1]
    bar_width = 1.

    # Get Churchman-smoothed distribution of cell-axial distances
    smoothed_1d_rpd = estimate_rpd_churchman_1d(axpoints,
                                                ax_fit_points,
                                                separation_precision)

    # Plot histogram and kde for axial separations
    plt.figure(figsize=[3*2.54, 2*2.54])
    axes = plt.subplot(111)
    axes.bar(ax_bar_points,
             ax_histogram,
             align='center',
             width=bar_width,
             color='lightblue', alpha=0.5,
             edgecolor='none')
    axes.fill_between(ax_fit_points,
                      0, smoothed_1d_rpd,
                      lw=0.5, color='blue', alpha=0.5)
    # axes.plot(kde_x_values, kde,
    #          lw=0.5, color='blue',
    #          )



    # Set up models and fit:
    # model_with_info = set_up_model_5_variable_peaks_with_fit_settings()
    # model_with_info = zdisk_modelling.set_up_model_5_variable_peaks_with_fit_settings()
    # model_with_info = zdisk_modelling.set_up_model_5_variable_peaks_bg_flat_with_fit_settings()
    model_with_info = zdisk_modelling.set_up_model_5_variable_peaks_with_replocs_bg_flat_with_fit_settings()
    # model_with_info = zdisk_modelling.set_up_model_5_peaks_fixed_ratio_with_fit_settings()
    # model_with_info = zdisk_modelling.set_up_model_5_peaks_fixed_ratio_no_replocs_with_fit_settings()
    # model_with_info = zdisk_modelling.set_up_model_linear_fit_plusreplocs_with_fit_settings()
    # model_with_info = zdisk_modelling.set_up_model_onepeak_with_fit_settings()
    # model_with_info = zdisk_modelling.set_up_model_onepeak_plus_replocs_with_fit_settings()
    # model_with_info = zdisk_modelling.set_up_model_onepeak_plus_replocs_flat_bg_with_fit_settings()

    (params_optimised,
     params_covar,
     params_1sd_error) = fitmodel_to_hist(ax_fit_points,
                                          smoothed_1d_rpd,
                                          model_with_info.model_rpd,
                                          model_with_info.initial_params,
                                          model_with_info.param_bounds,
                                          )
    del(params_1sd_error)

    axes.plot(ax_fit_points,
              model_with_info.model_rpd(ax_fit_points, *params_optimised),
              color='xkcd:red', lw=0.5)
    axes.set_xlim([0, 100])

    # Get 1 SD uncertainty on model result from uncertainty on parameters.
    stdev = stdev_of_model(ax_fit_points,
                           params_optimised,
                           params_covar,
                           model_with_info.vector_input_model
                           )

    # Plot 95% confidence interval on model
    axes.fill_between(ax_fit_points,
                      model_with_info.model_rpd(ax_fit_points,
                                                *params_optimised)
                      - stdev * 1.96,
                      model_with_info.model_rpd(ax_fit_points,
                                                *params_optimised)
                      + stdev * 1.96,
                      color='xkcd:red', alpha=0.25
                      )
    
    plt.savefig('mEos_Xfit.pdf')


def lasp_mEos_plot():
    """Fit axial KDE of relative positions of myopalladin:mEos localisations.
    """
    # Get 2D (axial, transverse) relative positions
    relpos = pd.read_pickle('../data-perpl/mEos3-LASP2_PERPL-relpos_200.0filter_5FOVs_aligned.pkl')
        
    # 'S:/Peckham/Bioimaging/Alistair/PALM-STORM'
    # '/AlexasNearestNeighbours/LASP2'
    # '/nns-aligned-5fovs.pkl')

    # Get subset of axial relative positions
    relpos.axial = abs(relpos.axial)
    fitlength = 100.
    loc_precision = 3.4
    separation_precision = np.sqrt(2) * loc_precision
    # The smoothing is used like this so that points beyond those not
    # included would only have 0.03% effect on the values of the kernal
    # density estimate (KDE) at fitlength.
    axpoints = getaxialseparations_no_smoothing(
                relpos=relpos,
                max_distance=fitlength + 3 * separation_precision,
                transverse_limit=10.
                )

    # Remove duplicates
    axpoints = remove_duplicates(axpoints)

    # Find kernel density estimate using localisation precision * sqrt(2)
    # as the Gaussian kernel.
    # Ues mean localisation precision estimate for mEos2:ACTN2,
    # after filtering to < 5 nm.
    # kde_x_values, kde = kde_1nm(axpoints,
    #                            locprec=loc_precision,
    #                            fitlength=fitlength)

    # Histogram of axial separation, in 1-nm bins
    bin_vals = np.arange(fitlength + 1)
    ax_histogram, bin_values = np.histogram(axpoints,
                                            bins=bin_vals)

    # Centre and width values for histogram bars
    ax_plot_points = (bin_values[:-1] + bin_values[1:]) / 2
    bar_width = 1.
    
    # Get Churchman-smoothed distribution of cell-axial distances
    smoothed_1d_rpd = estimate_rpd_churchman_1d(axpoints,
                                                ax_plot_points,
                                                separation_precision)

    # Set up models and fit:
    # for model_with_info in [# set_up_model_5_variable_peaks_with_fit_settings(),
    #                        set_up_model_4_variable_peaks_with_fit_settings()
    #                        ]:
    # model_with_info = set_up_model_5_variable_peaks_with_fit_settings()
    # model_with_info = set_up_model_5_variable_peaks_after_offset_with_fit_settings()
    # model_with_info = set_up_model_linear_fit_with_fit_settings()

    # Plot histogram and kde for axial separations
    plt.figure(figsize=[3/2.54, 2/2.54])
    axes = plt.subplot(111)
    axes.bar(ax_plot_points,
             ax_histogram,
             align='center',
             width=bar_width,
             color='lightblue', alpha=0.5)
    axes.fill_between(ax_plot_points,
                      0, smoothed_1d_rpd,
                      lw=0, color='blue', alpha=0.5)
    # axes.plot(kde_x_values, kde, lw=0.5, color='blue')

    axes.set_xlim([0, 100])

    # print('\n'+ model_with_info.model_rpd.__name__)
    # print('________________________________')

    #(params_optimised,
    # params_covar,
    # params_1sd_error) = fitmodel_to_hist(kde_x_values,
    #                                      kde,
    #                                      model_with_info.model_rpd,
    #                                      model_with_info.initial_params,
    #                                      model_with_info.param_bounds,
    #                                      )
    #del(params_1sd_error)

    #axes.plot(kde_x_values,
    #          model_with_info.model_rpd(kde_x_values, *params_optimised),
    #          color='xkcd:red', lw=0.5)

    # Get 1 SD uncertainty on model result from uncertainty on parameters.
    #stdev = stdev_of_model(kde_x_values,
    #                       params_optimised,
    #                       params_covar,
    #                       model_with_info.vector_input_model
    #                       )

    # Plot 95% confidence interval on model
    #axes.fill_between(kde_x_values,
    #                  model_with_info.model_rpd(kde_x_values,
    #                                            *params_optimised)
    #                                            - stdev * 1.96,
    #                  model_with_info.model_rpd(kde_x_values,
    #                                            *params_optimised)
    #                                            + stdev * 1.96,
    #                  color='xkcd:red', alpha=0.25
    #                  )

    #axes.set_title(model_with_info.model_rpd.__name__)
    #axes.set_xlabel('Axial separation (nm)')
    #axes.set_ylabel('Counts')

    plt.savefig('LASP2_Xfit.pdf')


def mypn_mEos_plot():
    """Fit axial KDE of relative positions of myopalladin:mEos localisations.
    """
    # Get 2D (axial, transverse) relative positions
    relpos = pd.read_pickle('../data-perpl/MYPN_NNS_aligned_5_FOVs_len1445698.pkl')
    
    # ('S:/Peckham/Bioimaging/Alistair/PALM-STORM'
    #                        '/AlexasNearestNeighbours/Myopalladin'
    #                        '/MYPN_NNS_aligned_5_FOVs_len1445698.pkl')

    # Get subset of axial relative positions
    relpos.axial = abs(relpos.axial)
    fitlength = 100.
    loc_precision = 3.4
    separation_precision = np.sqrt(2) * loc_precision
    # The smoothing is used like this so that points beyond those not
    # included would only have 0.03% effect on the values of the kernal
    # density estimate (KDE) at fitlength.
    axpoints = getaxialseparations_no_smoothing(
                relpos=relpos,
                max_distance=fitlength + 3 * separation_precision,
                transverse_limit=10.
                )

    # Remove duplicates
    axpoints = remove_duplicates(axpoints)

    # Find kernel density estimate using localisation precision * sqrt(2)
    # as the Gaussian kernel.
    # Ues mean localisation precision estimate for mEos2:ACTN2,
    # after filtering to < 5 nm.
    # kde_x_values, kde = kde_1nm(axpoints,
    #                            locprec=loc_precision,
    #                            fitlength=fitlength)

    # Histogram of axial separation, in 1-nm bins
    bin_vals = np.arange(fitlength + 1)
    ax_histogram, bin_values = np.histogram(axpoints,
                                            bins=bin_vals)

    # Centre and width values for histogram bars
    ax_plot_points = (bin_values[:-1] + bin_values[1:]) / 2
    bar_width = 1.
    
    # Get Churchman-smoothed distribution of cell-axial distances
    smoothed_1d_rpd = estimate_rpd_churchman_1d(axpoints,
                                                ax_plot_points,
                                                separation_precision)

    # Plot histogram and kde for axial separations
    plt.figure(figsize=[3/2.54, 2/2.54])
    axes = plt.subplot(111)
    axes.bar(ax_plot_points,
             ax_histogram,
             align='center',
             width=bar_width,
             color='lightblue', alpha=0.5)
    # axes.plot(kde_x_values, kde, lw=0.5, color='blue')

    axes.fill_between(ax_plot_points,
                      0, smoothed_1d_rpd,
                      lw=0, color='blue', alpha=0.5)
    # axes.plot(kde_x_values, kde, lw=0.5, color='blue')

    axes.set_xlim([0, 100])
    
    plt.savefig('MYPN_Xfit.pdf')

    # Set up models and fit:
    # model_with_info = set_up_model_5_variable_peaks_with_fit_settings()
    # model_with_info = set_up_model_linear_fit_with_fit_settings()

    # (params_optimised,
    # params_covar,
    # params_1sd_error) = fitmodel_to_hist(kde_x_values,
    #                                      kde,
    #                                      model_with_info.model_rpd,
    #                                      model_with_info.initial_params,
    #                                      model_with_info.param_bounds,
    #                                      )
    # del(params_1sd_error)

    # axes.plot(kde_x_values,
    #          model_with_info.model_rpd(kde_x_values, *params_optimised),
    #          color='xkcd:red', lw=0.5)

    # Get 1 SD uncertainty on model result from uncertainty on parameters.
    # stdev = stdev_of_model(kde_x_values,
    #                       params_optimised,
    #                       params_covar,
    #                       model_with_info.vector_input_model
    #                       )

    # Plot 95% confidence interval on model
    #axes.fill_between(kde_x_values,
    #                  model_with_info.model_rpd(kde_x_values,
    #                                            *params_optimised)
    #                  - stdev * 1.96,
    #                  model_with_info.model_rpd(kde_x_values,
    #                                            *params_optimised)
    #                  + stdev * 1.96,
    #                  color='xkcd:red', alpha=0.25
    #                  )
