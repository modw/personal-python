# Python convenience modules

A collection of simple modules to facilitate repeated tasks in research projects.

## Modules


`S12.py` : Module to calculate S_1/2. Metric used to characterize lack of large-angle correlations in the CMB. Works for masked skies.

`constrained_realizations.py` : Generate constrained realizations of correlated spherical harmonics coefficients. Given auto and cross spectra calculate either random constrained realization, or individual correlated piece and random uncorrelated piece. Includes convenience functions to transform between real and complex basis.

`map_rotation.py` : Couple of functions that make use of `healpy`'s rotation functions to facilitate coordinate change of alm's and pixel maps.

`planck_colormap.py` : Contains function to add colormap used in `Planck`'s temperature map.

`wiener_filter.py` : Contains Class to perform pixel-space wiener filtering of CMB map given noisy data `d` and pixel and noise covariance matrices `S` and `N`. Works with masked skies. Very memory intensive for resolutions above `Nside=32` since the covariance matrix is of size `(12*Nside^2)x(12*Nside^2)`.
