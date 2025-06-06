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

# +
import numpy as np
import matplotlib.pyplot as plt

from perpl.modelling import modelling_general, zdisk_modelling, zdisk_plots
from perpl.io import plotting
# -

# ## Set average estimated localisation precision for Affimer and PALM data.
# This is the mean after filtering for localisation precision < 5 nm.

meos_precision = 3.4 # Mean value after filtering for precision < 5 nm

# ## Set standard maximum distance over which to plot distances and fit models.

fitlength = 100.

# ## Choose Affimer relative position data and combine files if necessary
# ### Insert paths to the Affimer relative position data here:

actn_meos_relpos_path = r'../../perpl_test_data/ACTN2-mEos2_PERPL-relpos_200.0filter_6FOVs_aligned_len1229656.pkl'


# ### Select desired datasets to combine here:
# These contain all data attributes, including the relative positions in both directions for each pair of localisations. We will filter e.g. for cell-axial data later.

path_list = [actn_meos_relpos_path]
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

# ## Set the X-distances at which to calculate the axial RPD

calculation_points = np.arange(fitlength + 1.)

# ## Calculate the axial RPD with smoothing for Churchman 1D function

axial_rpd = plotting.estimate_rpd_churchman_1d(
    input_distances=axial_distances,
    calculation_points=calculation_points,
    combined_precision=6
    # combined_precision=(np.sqrt(2) * meos_precision)
)
plt.plot(calculation_points, axial_rpd)

# ## Get the 1-nm bin histogram data
# Up to distance = fitlength

hist_values, bin_edges = zdisk_plots.plot_distance_hist(
    axial_distances,
    fitlength
    )
bin_centres = (bin_edges[0:(len(bin_edges) - 1)] + bin_edges[1:]) / 2

# ## Get the KDE data (estimate every 1 nm)

kde_x_values, kde = zdisk_plots.plot_distance_kde(
    axial_distances,
    meos_precision,
    fitlength
    )

# ## Choose axial model

# axial_model_with_info = zdisk_modelling.set_up_model_5_variable_peaks_with_fit_settings()
# axial_model_with_info = zdisk_modelling.set_up_model_5_variable_peaks_bg_flat_with_fit_settings()
axial_model_with_info = zdisk_modelling.set_up_model_5_variable_peaks_with_replocs_bg_flat_with_fit_settings()
# axial_model_with_info = zdisk_modelling.set_up_model_5_peaks_fixed_ratio_with_fit_settings()
# axial_model_with_info = zdisk_modelling.set_up_model_5_peaks_fixed_ratio_no_replocs_with_fit_settings()
# axial_model_with_info = zdisk_modelling.set_up_model_linear_fit_plusreplocs_with_fit_settings()
# axial_model_with_info = zdisk_modelling.set_up_model_onepeak_with_fit_settings()
# axial_model_with_info = zdisk_modelling.set_up_model_onepeak_plus_replocs_with_fit_settings()
# axial_model_with_info = zdisk_modelling.set_up_model_onepeak_plus_replocs_flat_bg_with_fit_settings()

# ## Fit model to Churchman-smoothed RPD

(params_optimised,
 params_covar,
 params_1sd_error) = zdisk_modelling.fitmodel_to_hist(
    calculation_points,
    axial_rpd,
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

plt.plot(calculation_points, axial_rpd)
zdisk_plots.plot_fitted_model(
    axial_distances,
    fitlength,
    params_optimised,
    params_covar,
    axial_model_with_info,
    plot_95ci=True
    )

# ## Plot histogram, Churchman KDE and fitted model with confidence intervals.

zdisk_plots.actn_mEos_x_plot()

# ## Fit model to histogram values
# This results in an error for some models because of the bin-to-bin noise of the histogram giving the fitting algorithm a challenge.

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

hist_values, bin_edges = zdisk_plots.plot_distance_hist(
    axial_distances,
    fitlength
    )
zdisk_plots.plot_fitted_model(
    axial_distances,
    fitlength,
    params_optimised,
    params_covar,
    axial_model_with_info,
    plot_95ci=True
    )



# ## Plot fitted model over KDE data

zdisk_plots.plot_distance_kde(
    axial_distances,
    meos_precision,
    fitlength
    )
zdisk_plots.plot_fitted_model(
    axial_distances,
    fitlength,
    params_optimised,
    params_covar,
    axial_model_with_info
    )

# ## Plot fitted model over histogram data, with confidence intervals on the model
# ### NOTE: IT TAKES A WHILE TO CALCULATE THE CONFIDENCE INTERVALS
# ### Skip this if you don't need it right now.

zdisk_plots.plot_distance_kde(
    axial_distances,
    meos_precision,
    fitlength
    )
zdisk_plots.plot_fitted_model(
    axial_distances,
    fitlength,
    params_optimised,
    params_covar,
    axial_model_with_info,
    plot_95ci=True
    )

# ## Plot model components for best model (5 peaks with independent amplitudes)
# Only for illustrating axial RPD model with 5 independent-amplitude peaks, plus linear background.

# +
#zdisk_plots.plot_model_components_5peaks_variable(
#    kde_x_values,
#    *params_optimised)
# -

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

fitlength = 50.
calculation_points = np.arange(fitlength) + 0.5
combined_precision = np.sqrt(2) * meos_precision
combined_precision = np.sqrt(2) * meos_precision
transverse_rpd = plotting.estimate_rpd_churchman_2d(
    input_distances=trans_distances[trans_distances < (fitlength + 3 * combined_precision)],
    calculation_points=calculation_points,
    combined_precision=combined_precision
)
plt.plot(calculation_points, transverse_rpd)

# ## Normalise for increasing search circle with increasing distance

# +
normalised_transverse_rpd = transverse_rpd[calculation_points > 0.] / calculation_points[calculation_points > 0.]
norm_rpd_calculation_points = calculation_points[calculation_points > 0.]

plt.plot(norm_rpd_calculation_points, normalised_transverse_rpd)
# -

yz_fig = plt.figure()
yz_axes = yz_fig.add_subplot(111)
yz_axes.fill_between(calculation_points, 0, normalised_transverse_rpd,
                    color='blue', alpha=0.25)
yz_axes.set_xlim(0, 49.5)
yz_axes.set_ylim(0,)

import numpy as np
24. * np.sqrt(2)

# ### Optional save/load to save time

# +
# np.save('normalised_transverse_rpd_smoothed_Churchman-4p8', normalised_transverse_rpd)
# normalised_transverse_rpd = np.load('normalised_transverse_rpd_smoothed_Churchman-4p8.npy')
# -

# ## Set up model RPD and fit
# Tried a few smoothing kernel widthes and fitted up to 50 nm.

trans_model_with_info = zdisk_modelling.set_up_model_2d_twopeaks_flat_bg_with_fit_settings()

(params_optimised,
 params_covar,
 params_1sd_error) = zdisk_modelling.fitmodel_to_hist(
    norm_rpd_calculation_points[0:50],
    normalised_transverse_rpd[0:50],
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

# ## Calculate 95% CIs

stdev = modelling_general.stdev_of_model(calculation_points,
                            params_optimised,
                            params_covar,
                                trans_model_with_info.vector_input_model)

plt.plot(norm_rpd_calculation_points[0:50],
    normalised_transverse_rpd[0:50])
zdisk_plots.plot_fitted_model(
    norm_rpd_calculation_points[0:50],
    51.,
    params_optimised,
    params_covar,
    trans_model_with_info,
    plot_95ci=True
    )

yz_fig = plt.figure()
yz_axes = yz_fig.add_subplot(111)
yz_axes.fill_between(calculation_points, 0, normalised_transverse_rpd,
                    color='blue', alpha=0.25)
yz_axes.set_xlim(0, 49.5)
yz_axes.set_ylim(0,)
yz_axes.plot(calculation_points,
             trans_model_with_info.model_rpd(calculation_points, *params_optimised),
            color='red')
yz_axes.fill_between(calculation_points,
                     trans_model_with_info.model_rpd(calculation_points, *params_optimised) - 1.96 * stdev,
                     trans_model_with_info.model_rpd(calculation_points, *params_optimised) + 1.96 * stdev,
                     color='red', alpha=0.25
                    )
yz_axes.set_ylabel('Standardaised counts')
yz_axes.set_xlabel('$\Delta$YZ (nm)')
yz_fig.savefig(r'..\..\perpl_test_data\ACTN2_mEos-YZ-dists-standardised-smoothed2xprec-fit-95ci.pdf',
              bbox_inches='tight')

fitlength = 50.
calculation_points = np.arange(fitlength + 1.)
combined_precision = 6.
transverse_rpd = plotting.estimate_rpd_churchman_2d(
    input_distances=trans_distances[trans_distances < (fitlength + 3 * combined_precision)],
    calculation_points=calculation_points,
    combined_precision=combined_precision
)
plt.plot(calculation_points, transverse_rpd)

normalised_transverse_rpd = transverse_rpd[calculation_points > 0.] / calculation_points[calculation_points > 0.]
norm_rpd_calculation_points = calculation_points[calculation_points > 0.]
plt.plot(norm_rpd_calculation_points, normalised_transverse_rpd)

# ### Optional save/load to save time

# +
# np.save('normalised_transverse_rpd_smoothed_Churchman-6', normalised_transverse_rpd)
# normalised_transverse_rpd = np.load('normalised_transverse_rpd_smoothed_Churchman-6.npy')
# -

calculation_points = np.arange(fitlength + 1.)
norm_rpd_calculation_points = calculation_points[calculation_points > 0.]
(params_optimised,
 params_covar,
 params_1sd_error) = zdisk_modelling.fitmodel_to_hist(
    norm_rpd_calculation_points[0:50],
    normalised_transverse_rpd[0:50],
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

plt.plot(norm_rpd_calculation_points[0:50],
    normalised_transverse_rpd[0:50])
zdisk_plots.plot_fitted_model(
    norm_rpd_calculation_points[0:50],
    51.,
    params_optimised,
    params_covar,
    trans_model_with_info,
    plot_95ci=True
    )

fitlength = 50.
calculation_points = np.arange(fitlength + 1.)
combined_precision = 8.
transverse_rpd = plotting.estimate_rpd_churchman_2d(
    input_distances=trans_distances[trans_distances < (fitlength + 3 * combined_precision)],
    calculation_points=calculation_points,
    combined_precision=combined_precision
)
plt.plot(calculation_points, transverse_rpd)

normalised_transverse_rpd = transverse_rpd[calculation_points > 0.] / calculation_points[calculation_points > 0.]
norm_rpd_calculation_points = calculation_points[calculation_points > 0.]
plt.plot(norm_rpd_calculation_points, normalised_transverse_rpd)

# ### Optional save/load to save time

# +
# np.save('normalised_transverse_rpd_smoothed_Churchman-8', normalised_transverse_rpd)
# normalised_transverse_rpd = np.load('normalised_transverse_rpd_smoothed_Churchman-6.npy')
# -

# calculation_points = np.arange(fitlength + 1.)
# norm_rpd_calculation_points = calculation_points[calculation_points > 0.]
(params_optimised,
 params_covar,
 params_1sd_error) = zdisk_modelling.fitmodel_to_hist(
    norm_rpd_calculation_points[0:50],
    normalised_transverse_rpd[0:50],
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

plt.plot(norm_rpd_calculation_points[0:50],
    normalised_transverse_rpd[0:50])
zdisk_plots.plot_fitted_model(
    norm_rpd_calculation_points[0:50],
    51.,
    params_optimised,
    params_covar,
    trans_model_with_info,
    plot_95ci=True
    )

fitlength = 50.
calculation_points = np.arange(fitlength + 1.)
combined_precision = 10.
transverse_rpd = plotting.estimate_rpd_churchman_2d(
    input_distances=trans_distances[trans_distances < (fitlength + 3 * combined_precision)],
    calculation_points=calculation_points,
    combined_precision=combined_precision
)
plt.plot(calculation_points, transverse_rpd)

normalised_transverse_rpd = transverse_rpd[calculation_points > 0.] / calculation_points[calculation_points > 0.]
norm_rpd_calculation_points = calculation_points[calculation_points > 0.]
plt.plot(norm_rpd_calculation_points, normalised_transverse_rpd)

# ### Optional save/load to save time

# +
# np.save('normalised_transverse_rpd_smoothed_Churchman-10', normalised_transverse_rpd)
# normalised_transverse_rpd = np.load('normalised_transverse_rpd_smoothed_Churchman-6.npy')
# -

# calculation_points = np.arange(fitlength + 1.)
# norm_rpd_calculation_points = calculation_points[calculation_points > 0.]
(params_optimised,
 params_covar,
 params_1sd_error) = zdisk_modelling.fitmodel_to_hist(
    norm_rpd_calculation_points[0:50],
    normalised_transverse_rpd[0:50],
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

plt.plot(norm_rpd_calculation_points[0:50],
    normalised_transverse_rpd[0:50])
zdisk_plots.plot_fitted_model(
    norm_rpd_calculation_points[0:50],
    51.,
    params_optimised,
    params_covar,
    trans_model_with_info,
    plot_95ci=True
    )


