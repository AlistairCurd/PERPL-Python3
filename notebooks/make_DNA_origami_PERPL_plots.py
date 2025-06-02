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

# # Make DNA-origami PERPL analysis plots

# # IMPORTANT
# # Disable autosave for Jupytext version control with a paired .py script
# # But manually saving the notebook frequently is still good

# %autosave 0

# ## Imports

# +
import time
import numpy as np

from perpl.modelling import modelling_general, dna_paint_data_fitting
from perpl.statistics import modelstats
from perpl.statistics.modelstats import akaike_weights
# -

# ## Set standard maximum distance over which to plot distances and fit models

fitlength = 250.

# ## Load relative position data
# ### Insert path to the relative position data here:

dna_origami_relpos_path = r'../../perpl_test_data/DNA-origami_DNA-PAINT_locs_xyz_PERPL-relpos_250.0filter.csv'


start_time = time.time()
relpos = np.loadtxt(dna_origami_relpos_path, delimiter=',', skiprows=1)
print('This took ' +repr(time.time() - start_time)+ ' s.')

# ### One data point and number of data points:

relpos[0] # This shows the first relative position.
# Distances are in X, Y, Z, XY, XZ, YZ, XYZ

len(relpos) # This shows how many relative positions.

# ## Get XYZ distances and plot 1-nm bin histogram
# Distances used up to fitlength.

xyz_distances = relpos[:, 6]
xyz_distances = xyz_distances[xyz_distances <= fitlength]
hist_values, bin_edges = dna_paint_data_fitting.plot_xyz_distance_histogram(
    xyz_distances,
    fitlength
    )

# ## Choose RPD model:

model_with_info = dna_paint_data_fitting.set_up_tri_prism_on_grid_1_length_2disobg_substruct_with_fit_info()

# ## Fit model to histogram bin values, at bin centres

(params_optimised,
 params_covar,
 params_1sd_error) = modelling_general.fit_model_to_experiment(
                        hist_values,
                        model_with_info.model_rpd,
                        model_with_info.initial_params,
                        model_with_info.param_bounds,
                        fitlength=fitlength
                        )
ssr, aicc = modelstats.aic_from_least_sqr_fit(
    hist_values,
    model_with_info.model_rpd,
    params_optimised,
    fitlength
    )
print('Parameter estimates and uncertainties (stdev):')
print(np.column_stack((params_optimised, params_1sd_error)))
print('\nInitial parameter guesses:')
print(model_with_info.initial_params)
print('\nParameter bounds:')
print(model_with_info.param_bounds)
print('\nSSR = ' +repr(ssr))
print('AICc = ' +repr(aicc))

# ## Plot fitted model over histogram data

fig, axes = dna_paint_data_fitting.plot_distance_hist_and_fit(
    xyz_distances,
    fitlength,
    params_optimised,
    params_covar,
    model_with_info
)

# ## Plot fitted model over histogram data, with confidence intervals on the model
# ### NOTE: IT TAKES A WHILE TO CALCULATE THE CONFIDENCE INTERVALS
# ### Skip this if you don't need it right now.

start_time = time.time()
fig, axes = dna_paint_data_fitting.plot_distance_hist_and_fit(
    xyz_distances,
    fitlength,
    params_optimised,
    params_covar,
    model_with_info,
    plot_95ci=True
)


# ## Akaike weights for the models
# Typed in AICc values for the different models here, to obtain relative likelihood, summing to one:

weights = akaike_weights([1.])
print(weights)

# ## Plot model components for best model: triangular prism with equal sides
# Isotropic 2D background, features repeating on square grid.
# Includes localisation precision (repeated localisations of the same molecule) and an unresolvable substructure term.

dna_paint_data_fitting.plot_model_components_tri_prism(
    fitlength,
    *params_optimised
    )

dna_paint_data_fitting.plot_three_models(relpos, fitlength=fitlength)


