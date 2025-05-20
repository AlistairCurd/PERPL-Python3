# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Make ACTN2 Affimer PERPL analysis plots

# # IMPORTANT
# # Disable autosave for Jupytext version control with a paired .py script
# # But manually saving the notebook frequently is still good

# %autosave 0

# ## Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perpl.modelling import modelling_general, zdisk_modelling, zdisk_plots
from perpl.io import plotting

# ## Set average estimated localisation precision for Affimer and PALM data.
# This is the mean after filtering for localisation precision < 5 nm.

af647_precision = 3.1 # Mean value after filtering for precision < 5 nm

# ## Set standard maximum distance over which to plot distances and fit models.

fitlength = 100.

# ## Choose Affimer relative position data and combine files if necessary
# ### Insert paths to the Affimer relative position data here:

actn_affimer_relpos_path = r"../../perpl_test_data/ACTN2-Affimer_PERPL-relpos_200.0filter_len2440488.pkl"

# ### Select desired datasets to combine here:
# These contain all data attributes, including the relative positions in both directions for each pair of localisations. We will filter e.g. for cell-axial data later.

path_list = [actn_affimer_relpos_path]
print(path_list)
relpos = zdisk_modelling.read_relpos_from_pickles(path_list)

# ### Data attributes and number of data points:

relpos.iloc[0, :] # This shows the first relative position.

len(relpos) # This shows how many relative positions.

# ## Get the axial (X) distances, without duplicates
# The YZ-distance limit for pairs of localisations to include can be set here.

# +
# This is the YZ-distance limit for X-distances to include:
transverse_limit = 10.

axial_distances = zdisk_modelling.getaxialseparations_no_smoothing(
    relpos,
    max_distance=relpos.axial.max(),
    transverse_limit=transverse_limit
    )
axial_distances = zdisk_modelling.remove_duplicates(axial_distances)
# -

# ## Get the 1-nm bin histogram data
# Up to distance = fitlength

hist_values, bin_edges = zdisk_plots.plot_distance_hist(
    axial_distances,
    fitlength
    )
bin_centres = (bin_edges[0:(len(bin_edges) - 1)]
               + bin_edges[1:]
               ) / 2

# ## Get the KDE data
# Estimate every 1 nm, with kernel size based on localisation precision estimate.

kde_x_values, kde = zdisk_plots.plot_distance_kde(
    axial_distances,
    af647_precision,
    100.
    )

# ## Calculate the axial RPD with smoothing for Churchman 1D function

calculation_points = np.arange(fitlength + 1.)
axial_rpd = plotting.estimate_rpd_churchman_1d(
    input_distances=axial_distances,
    calculation_points=calculation_points,
    combined_precision=(np.sqrt(2) * af647_precision)
)
plt.plot(calculation_points, axial_rpd)

# ## Choose axial model:

axial_model_with_info = zdisk_modelling.set_up_model_4_peaks_fixed_ratio_with_fit_settings()


# ## Fit model to histogram bin values, at bin centres

(params_optimised,
 params_covar,
 params_1sd_error) = zdisk_modelling.fitmodel_to_hist(
    bin_centres,
    hist_values,
    axial_model_with_info.model_rpd,
    axial_model_with_info.initial_params,
    axial_model_with_info.param_bounds,
    )
print('')
print('Initial parameter guesses:')
print(axial_model_with_info.initial_params)
print('')
print('Parameter bounds:')
print(axial_model_with_info.param_bounds)

# ## Plot fitted model over histogram data

fig, axes = zdisk_plots.plot_distance_hist_and_fit(
    axial_distances,
    fitlength,
    params_optimised,
    params_covar,
    axial_model_with_info
)

# ## Plot fitted model over histogram data, with confidence intervals on the model
# ### NOTE: IT TAKES A WHILE TO CALCULATE THE CONFIDENCE INTERVALS
# ### Skip this if you don't need it right now.

zdisk_plots.plot_distance_hist_and_fit(
    axial_distances,
    fitlength,
    params_optimised,
    params_covar,
    axial_model_with_info,
    plot_95ci=True
)

# ## Akaike weights for the models
# Typed in AICc values for the different models here, to obtain relative likelihood, summing to one:

from perpl.statistics.modelstats import akaike_weights
weights = akaike_weights([
    364.35,
    364.51,
    366.38,
    370.28,
    374.67
])
print(weights)

# ## Plot model components for best model (4 peaks with fixed peak ratio)

zdisk_plots.plot_model_components_4peaks_fixed_peak_ratio(
    fitlength,
    *params_optimised)

# # Transverse distances

# ## Get the transverse (YZ) distances, without duplicates
# The X-distance limit for pairs of localisations to include can be set here.

# +
# This is the YZ-distance limit for X-distances to include:
axial_limit = 10.
print(relpos.shape)

trans_distances = zdisk_modelling.get_transverse_separations(
    relpos,
    max_distance=relpos.transverse.max(),
    axial_limit=axial_limit
    )
trans_distances = zdisk_modelling.remove_duplicates(trans_distances)
# -

# ## Choose analysis lengthscale for transverse distance

fitlength = 50.

hist_1nm_bins = plt.hist(trans_distances, bins=np.arange(fitlength + 1.))

# ## Estimate RPD using Churchman's function

fitlength = 50.
calculation_points = np.arange(fitlength + 1.)
combined_precision= np.sqrt(2) * af647_precision
transverse_rpd = plotting.estimate_rpd_churchman_2d(
    input_distances=trans_distances[trans_distances < (fitlength + 3 * combined_precision)],
    calculation_points=calculation_points,
    combined_precision=combined_precision
)
plt.plot(calculation_points, transverse_rpd)

# ## Normalise for increasing search circle with increasing distance

normalised_transverse_rpd = transverse_rpd[calculation_points > 0.] / calculation_points[calculation_points > 0.]
norm_rpd_calculation_points = calculation_points[calculation_points > 0.]
plt.plot(norm_rpd_calculation_points, normalised_transverse_rpd)

# ### 1 nm-bin  histogram result for comparison

plt.plot(hist_1nm_bins[1][0:-1] + 0.5, hist_1nm_bins[0]/(hist_1nm_bins[1][0:-1] + 0.5))

# ### Optional save/load to save time

np.save('normalised_transverse_rpd_smoothed_Churchman-4p4', normalised_transverse_rpd)
# normalised_transverse_rpd = np.load('normalised_transverse_rpd_smoothed_Churchman-4p4.npy')

# ## Set up an RPD model and fit
# I've tried a few smoothing kernel widths here.

trans_model_with_info = zdisk_modelling.set_up_model_2d_onepeak_plus_replocs_flat_bg_with_fit_settings()

(params_optimised,
 params_covar,
 params_1sd_error) = zdisk_modelling.fitmodel_to_hist(
    norm_rpd_calculation_points[0:31],
    normalised_transverse_rpd[0:31],
    trans_model_with_info.model_rpd,
    trans_model_with_info.initial_params,
    trans_model_with_info.param_bounds
    )
print('')
print('Initial parameter guesses:')
print(trans_model_with_info.initial_params)
print('')
print('Parameter bounds:')
print(trans_model_with_info.param_bounds)

plt.plot(norm_rpd_calculation_points,
    normalised_transverse_rpd)
zdisk_plots.plot_fitted_model(
    norm_rpd_calculation_points,
    fitlength,
    params_optimised,
    params_covar,
    trans_model_with_info,
    plot_95ci=False
    )

fitlength = 50.
calculation_points = np.arange(fitlength + 1.)
combined_precision=8.
transverse_rpd_s8 = plotting.estimate_rpd_churchman_2d(
    input_distances=trans_distances[trans_distances < (fitlength + 3 * combined_precision)],
    calculation_points=calculation_points,
    combined_precision=combined_precision
)

normalised_transverse_rpd_s8 = transverse_rpd_s8[calculation_points > 0.] / calculation_points[calculation_points > 0.]
norm_rpd_calculation_points = calculation_points[calculation_points > 0.]
plt.plot(norm_rpd_calculation_points, normalised_transverse_rpd_s8)
np.save('..\data-perpl\ACTN2-Affimer\\normalised_transverse_rpd_smoothed_Churchman-8', normalised_transverse_rpd_s8)
# normalised_transverse_rpd = np.load('normalised_transverse_rpd_smoothed_Churchman-4p4.npy')

(params_optimised,
 params_covar,
 params_1sd_error) = zdisk_modelling.fitmodel_to_hist(
    norm_rpd_calculation_points[0:31],
    normalised_transverse_rpd_s8[0:31],
    trans_model_with_info.model_rpd,
    trans_model_with_info.initial_params,
    trans_model_with_info.param_bounds
    )
print('')
print('Initial parameter guesses:')
print(trans_model_with_info.initial_params)
print('')
print('Parameter bounds:')
print(trans_model_with_info.param_bounds)

plt.plot(norm_rpd_calculation_points,
    normalised_transverse_rpd_s8)
zdisk_plots.plot_fitted_model(
    norm_rpd_calculation_points,
    fitlength,
    params_optimised,
    params_covar,
    trans_model_with_info,
    plot_95ci=False
    )

fitlength = 50.
calculation_points = np.arange(fitlength + 1.)
combined_precision=5.
transverse_rpd_s5 = plotting.estimate_rpd_churchman_2d(
    input_distances=trans_distances[trans_distances < (fitlength + 3 * combined_precision)],
    calculation_points=calculation_points,
    combined_precision=combined_precision
)

normalised_transverse_rpd_s5 = transverse_rpd_s5[calculation_points > 0.] / calculation_points[calculation_points > 0.]
norm_rpd_calculation_points = calculation_points[calculation_points > 0.]
plt.plot(norm_rpd_calculation_points, normalised_transverse_rpd_s5)
np.save('normalised_transverse_rpd_smoothed_Churchman-5', normalised_transverse_rpd_s5)
# normalised_transverse_rpd = np.load('normalised_transverse_rpd_smoothed_Churchman-4p4.npy')

fitlength = 50.
calculation_points = np.arange(fitlength + 1.)
combined_precision=6.
transverse_rpd_s6 = plotting.estimate_rpd_churchman_2d(
    input_distances=trans_distances[trans_distances < (fitlength + 3 * combined_precision)],
    calculation_points=calculation_points,
    combined_precision=combined_precision
)

normalised_transverse_rpd_s6 = transverse_rpd_s6[calculation_points > 0.] / calculation_points[calculation_points > 0.]
norm_rpd_calculation_points = calculation_points[calculation_points > 0.]
plt.plot(norm_rpd_calculation_points, normalised_transverse_rpd_s6)
np.save('normalised_transverse_rpd_smoothed_Churchman-6', normalised_transverse_rpd_s6)
# normalised_transverse_rpd = np.load('normalised_transverse_rpd_smoothed_Churchman-4p4.npy')

(params_optimised,
 params_covar,
 params_1sd_error) = zdisk_modelling.fitmodel_to_hist(
    norm_rpd_calculation_points[0:31],
    normalised_transverse_rpd_s6[0:31],
    trans_model_with_info.model_rpd,
    trans_model_with_info.initial_params,
    trans_model_with_info.param_bounds
    )
print('')
print('Initial parameter guesses:')
print(trans_model_with_info.initial_params)
print('')
print('Parameter bounds:')
print(trans_model_with_info.param_bounds)

plt.plot(norm_rpd_calculation_points,
    normalised_transverse_rpd_s8)
zdisk_plots.plot_fitted_model(
    norm_rpd_calculation_points,
    fitlength,
    params_optimised,
    params_covar,
    trans_model_with_info,
    plot_95ci=False
    )

# ## Set up another model and fit
# Tried a few smoothing widths again
# ### This model fits much better

trans_model_with_info = zdisk_modelling.set_up_model_2d_twopeaks_flat_bg_with_fit_settings()

normalised_transverse_rpd_s6 = np.load('./normalised_transverse_rpd_smoothed_Churchman-6.npy')
calculation_points = np.arange(fitlength + 1.)
norm_rpd_calculation_points = calculation_points[calculation_points > 0.]

(params_optimised,
 params_covar,
 params_1sd_error) = zdisk_modelling.fitmodel_to_hist(
    norm_rpd_calculation_points[0:31],
    normalised_transverse_rpd_s6[0:31],
    trans_model_with_info.model_rpd,
    trans_model_with_info.initial_params,
    trans_model_with_info.param_bounds
    )
print('')
print('Initial parameter guesses:')
print(trans_model_with_info.initial_params)
print('')
print('Parameter bounds:')
print(trans_model_with_info.param_bounds)

plt.plot(norm_rpd_calculation_points[0:31],
    normalised_transverse_rpd_s6[0:31])
zdisk_plots.plot_fitted_model(
    norm_rpd_calculation_points[0:31],
    31.,
    params_optimised,
    params_covar,
    trans_model_with_info,
    plot_95ci=True
    )

(params_optimised,
 params_covar,
 params_1sd_error) = zdisk_modelling.fitmodel_to_hist(
    norm_rpd_calculation_points[0:31],
    normalised_transverse_rpd[0:31],
    trans_model_with_info.model_rpd,
    trans_model_with_info.initial_params,
    trans_model_with_info.param_bounds
    )
print('')
print('Initial parameter guesses:')
print(trans_model_with_info.initial_params)
print('')
print('Parameter bounds:')
print(trans_model_with_info.param_bounds)

plt.plot(norm_rpd_calculation_points[0:31],
    normalised_transverse_rpd[0:31])
zdisk_plots.plot_fitted_model(
    norm_rpd_calculation_points[0:31],
    31.,
    params_optimised,
    params_covar,
    trans_model_with_info,
    plot_95ci=True
    )

normalised_transverse_rpd_s8 = np.load('./normalised_transverse_rpd_smoothed_Churchman-8.npy')
calculation_points = np.arange(fitlength + 1.)
norm_rpd_calculation_points = calculation_points[calculation_points > 0.]

(params_optimised,
 params_covar,
 params_1sd_error) = zdisk_modelling.fitmodel_to_hist(
    norm_rpd_calculation_points[0:31],
    normalised_transverse_rpd_s8[0:31],
    trans_model_with_info.model_rpd,
    trans_model_with_info.initial_params,
    trans_model_with_info.param_bounds
    )
print('')
print('Initial parameter guesses:')
print(trans_model_with_info.initial_params)
print('')
print('Parameter bounds:')
print(trans_model_with_info.param_bounds)

plt.plot(norm_rpd_calculation_points[0:31],
    normalised_transverse_rpd_s8[0:31])
zdisk_plots.plot_fitted_model(
    norm_rpd_calculation_points[0:31],
    31.,
    params_optimised,
    params_covar,
    trans_model_with_info,
    plot_95ci=True
    )

(params_optimised,
 params_covar,
 params_1sd_error) = zdisk_modelling.fitmodel_to_hist(
    norm_rpd_calculation_points[0:31],
    normalised_transverse_rpd_s5[0:31],
    trans_model_with_info.model_rpd,
    trans_model_with_info.initial_params,
    trans_model_with_info.param_bounds
    )
print('')
print('Initial parameter guesses:')
print(trans_model_with_info.initial_params)
print('')
print('Parameter bounds:')
print(trans_model_with_info.param_bounds)

plt.plot(norm_rpd_calculation_points[0:31],
    normalised_transverse_rpd_s5[0:31])
zdisk_plots.plot_fitted_model(
    norm_rpd_calculation_points[0:31],
    31.,
    params_optimised,
    params_covar,
    trans_model_with_info,
    plot_95ci=True
    )


