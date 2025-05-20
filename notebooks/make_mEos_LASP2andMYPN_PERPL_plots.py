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

# # Make LASP2 mEos PERPL analysis plots

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

meos_lasp_relpos_path = r'../perpl_test_data/mEos3-LASP2_PERPL-relpos_200.0filter_5FOVs_aligned_len533140.pkl'

# ### Select desired datasets to combine here:
# These contain all data attributes, including the relative positions in both directions for each pair of localisations. We will filter e.g. for cell-axial data later.

path_list = [meos_lasp_relpos_path]
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
bin_centres = (bin_edges[0:(len(bin_edges) - 1)] + bin_edges[1:]) / 2

# ## Get the KDE data (estimate every 1 nm)

kde_x_values, kde = zdisk_plots.plot_distance_kde(
    axial_distances,
    meos_precision,
    fitlength
    )

# ## Set the X-distances at which to calculate the axial RPD

calculation_points = np.arange(fitlength + 1.)

# ## Calculate the axial RPD with smoothing for Churchman 1D function

axial_rpd = plotting.estimate_rpd_churchman_1d(
    input_distances=axial_distances,
    calculation_points=calculation_points,
    # combined_precision=6.
    combined_precision=(np.sqrt(2) * meos_precision)
)
plt.plot(calculation_points, axial_rpd)

# ## Choose axial model

axial_model_with_info = zdisk_modelling.set_up_model_5_variable_peaks_after_offset_flat_bg_with_fit_settings()

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

# ## Plot including histogram

zdisk_plots.lasp_mEos_plot()

# # MYPN

zdisk_plots.mypn_mEos_plot()


