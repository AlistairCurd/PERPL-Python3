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

# # Display 2D histograms of Nup107 FOV used in PERPL manuscript

# # IMPORTANT
# # Disable autosave for Jupytext version control with a paired .py script
# # But manually saving the notebook frequently is still good

# %autosave 0

# ## Import modules

import numpy as np
import matplotlib.pyplot as plt

# ## Load localisations and build histogram
#
# Choose paths to load from and save to.

input_path = r'..\..\perpl_test_data\Nup107_SNAP_3D_GRROUPED_10nmZprec.txt'
locs = np.loadtxt(input_path, delimiter=',')
print(locs.shape)
# locs.columns

# ### Change binsize as required for histogram

# +
binsize = 50 # Edit
edgex = np.arange(locs[:, 0].min() - 1.5 * binsize, locs[:, 0].max() + 2.5 * binsize, binsize)
edgey = np.arange(locs[:, 1].min() - 1.5 * binsize,
                  locs[:, 1].max() + 2.5 * binsize,
                  binsize)
hist_2d = np.histogram2d(locs[:, 1], locs[:, 0], bins=(edgey, edgex))[0]

plt.matshow(hist_2d, cmap='gray')

# +
## Option to save as binary
# -

hist_2d.tofile(input_path[0:-4] + '_xyhist_' + repr(binsize) + 'nmbins_64bit_w{}_h{}.raw'.format(hist_2d.shape[1], hist_2d.shape[0]))
print('2D histogram shape = ' + repr(hist_2d.shape) + ' (columns, rows).')

# ## Zoom in on centre of FOV
# ### Change binsize and zoom factor (centred on mean x and y values) as required

# +
binsize = 5 # Edit
zoomfactor = 10 # Edit
rangex = locs[:, 0].max() - locs[:, 0].min()
rangey = locs[:, 1].max() - locs[:, 1].min()

edgex = np.arange(locs[:, 0].mean() - 1./ (zoomfactor * 2.) * rangex,
                  locs[:, 0].mean() + 1./ (zoomfactor * 2.) * rangex,
                  binsize)
edgey = np.arange(locs[:, 1].mean() - 1./ (zoomfactor * 2.) * rangey,
                  locs[:, 1].mean() + 1./ (zoomfactor * 2.) * rangey,
                  binsize)
hist_2d = np.histogram2d(locs[:, 1], locs[:, 0], bins=(edgey, edgex))[0]

plt.matshow(hist_2d, cmap='gray')
# -

# ## Option to save as binary

hist_2d.tofile(input_path[0:-4] + '_xyhist_zoomfactor' + repr(zoomfactor) + '_' + repr(binsize) + 'nmbins_64bit_w{}_h{}.raw'.format(hist_2d.shape[1], hist_2d.shape[0]))
print('2D histogram shape = ' + repr(hist_2d.shape) + ' (columns, rows).')


