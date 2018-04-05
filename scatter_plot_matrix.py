"""Script to create lower-triangula grid with methods to plot statistical plots such as
confidence curves and 2d histograms.

Instructions
-------
 - Create grid object with AxesGrid()
 - Add desired curves with methods add_hist, add_hist2d and add_confidence curves
 - If using multiple datasets, add data labels with method add_data_label; plot labels with
 add_data_legend

 Pending
 -------
 - Adding method to map any function of two variables to off diagonal
 - Adding method to map any function of one variable to diagonal
"""


import numpy as np
import matplotlib.pyplot as plt

from statsmodels.stats.weightstats import DescrStatsW
from scipy.interpolate import CubicSpline
import scipy.ndimage as ndimage


class AxesGrid:
    def __init__(self, n_params, labels=None, size=12, hspace=0.08, wspace=0.08, **kwargs):
        """Creates a lower-triangular grid on which cross-correlation-like plots can be drawn
        off diagonal, and histogram-like plots on the diagonal. Matplotlib axes can be accessed with
        .axes attribute. Labels can be either added in init or later via method add_labels.

        Parameters:
        ----------
        n_params : {int}
            Number of variables that are going to be plotted against each other.
        labels : {list of strings}, optional
            Plot labels for each variable (the default is None)
        size : {int}, optional
            Side size of generated figure (the default is 12)
        hspace : {float}, optional
            vertical space between each subplot (the default is 0.08)
        wspace : {float}, optional
            horizontal space between each subplot (the default is 0.08)

        """

        # create figure and axes object
        fig, axes = plt.subplots(n_params, n_params, figsize=(
            size, size), sharex='col', sharey='row', **kwargs)
        self.nax = n_params
        self.axes = axes
        # turn off top and right spines for every ax
        for ax in self.axes.flatten():
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        # add labels
        if labels is not None:
            self.add_labels(labels)
        # turn off y ticks, spine and labels of very fist ax
        self.axes[0, 0].set_ylabel("")
        self.axes[0, 0].set_yticks([])
        self.axes[0, 0].spines['left'].set_visible(False)
        # adjust spacing and delete upper diag of axes
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        self._del_upper_triangle()
        # initialize label ax
        self.label_ax = self.axes[-2, -1]._make_twin_axes(frameon=False)
        self.label_ax.set_axis_off()
        self.label_ax.set_yticks([])
        return

    # grid methods

    def add_labels(self, labels):
        """Add labels for each variable.

        Parameters:
        ----------
        labels : {list of strings}
            List of labels for each variable. Should be in same order as the
            columns in the data.
        """

        # loop over left and bottom axes to label them
        for i in range(self.nax):
            self.axes[i, 0].yaxis.set_label_text(labels[i])
            self.axes[-1, i].xaxis.set_label_text(labels[i])

    def _del_upper_triangle(self):
        """Deletes upper triangle of square grid of subplots.

        """

        for i in range(self.nax):
            for j in range(self.nax):
                if j > i:
                    plt.delaxes(self.axes[i, j])

    # plotting methods

    def add_hist(self, df,  bins=50, histtype='step', color='k', lw=2, weights=None, **kwargs):
        """Adds histogram to diagonal of grid.

        Parameters:
        ----------
        df : {Pandas DataFrame object}
            DataFrame with data to be plotted. Each column being a different variable.
        bins : {int}, optional
            Number of bins in histrogram (the default is 50)
        histtype : {str}, optional
            Type of histogram (the default is 'step')
        color : {str}, optional
            Color of histogram (the default is 'k')
        lw : {int}, optional
            Line width in case of 'step' histogram (the default is 2)
        weights : {array}, optional
            Array of weights of length len(df), to be used in case the
            data is weighted (the default is None)

        """

        for i in range(self.nax):
            diag_ax = self.axes[i, i]._make_twin_axes(
                sharex=self.axes[i, i], frameon=False)
            diag_ax.set_axis_off()
            diag_ax.set_yticks([])
            diag_ax.hist(df.iloc[:, i], bins=bins, weights=weights, histtype=histtype,
                         color=color, lw=lw, **kwargs)

    def add_confidence_curves(self, df, targets=[0.9545, 0.6827], bins=50,
                              fill=True, smoothing_beam=1, colors='k', cmap=None,
                              lw=2, weights=None, **kwargs):
        """Adds confidence curves to off diagonal of grid.

        Parameters:
        ----------
        df : {pandas DataFrame object}
            Object with data to be plotted. Each off column being a different variable.
        targets : {list}, optional
            Desired confidence levels. (the default is [0.9545, 0.6827], which signals standard
            one and two sigma curves.)
        bins : {int}, optional
            Number of bins to generate hist2d from which curves are calculated.
             (the default is 50)
        fill : {bool}, optional
            Whether or not the curves should be filled. (the default is True)
        smoothing_beam : {float}, optional
            Standard deviation for smoothing Gaussian kernel. Always make sure smoothing
            beam is narrow enough not to spread out the data. (the default is 1)
        colors : {str or list of strings}, optional
            Color or colors of confidence level curves if fill=False (the default is 'k', which is black)
        cmap : {str}, optional
            Colormap to be used to plot curves (the default is None)
        lw : {int}, optional
            Line width of curves if fill = False (the default is 2)
        weights : {array}, optional
            Array of weights of length len(df), to be used in case the
            data is weighted (the default is None)

        """

        for i in range(self.nax):
            for j in range(i):
                _plot_confidence_levels_2d(self.axes[i, j], df.iloc[:, j], df.iloc[:, i],
                                           targets=targets, bins=bins, weights=weights,
                                           fill=fill, smoothing_beam=smoothing_beam, colors=colors,
                                           cmap=cmap, linewidths=lw, **kwargs)

    def add_hist2d(self, df, bins=50, cmap='Greys', weights=None, **kwargs):
        """Adds hist2d plot to off dioganal of grid.

        Parameters:
        ----------
        df : {pandas DataFrame object}
            Object with data to be plotted. Each off column being a different variable.
        bins : {int}, optional
            Number of bins to generate hist2d (the default is 50)
        cmap : {str}, optional
            Colormap of hist2d (the default is 'Greys')
        weights : {array}, optional
            Array of weights of length len(df), to be used in case the
            data is weighted (the default is None)

        """

        for i in range(self.nax):
            for j in range(i):
                _plot_hist2d(self.axes[i, j], df.iloc[:, j], df.iloc[:, i],
                             bins=bins, weights=weights, cmap=cmap, **kwargs)

    def add_data_label(self, label, color, histtype, linestyle=None, linewidth=2):
        """Adds a label for a dataset using specified aesthetic. Should be followed by
        add_data_legend in order to display legends.

        Parameters:
        ----------
        label : {str}
            Label of dataset
        color : {str}
            Color used to be represent data set
        histtype : {str}
            To specify whether or not label should have a filled or unfilled square.
        linestyle : {str}, optional
            Line style in case of unfilled square. (the default is None)
        linewidth : {float}, optional
            Width of label line (the default is 2)

        """

        self.bars = self.label_ax.hist([], label=label, histtype=histtype,
                                       lw=linewidth, linestyle=linestyle,
                                       color=color)[2]

    def add_data_legend(self):
        """Should be used after specifying labels with add_data_label method.

        """

        self.label_ax.legend(loc='center')
        [b.remove() for b in self.bars]

# internal plotting functions


def _plot_hist2d(ax, *args, **kwargs):
    return ax.hist2d(*args, **kwargs)[3]


def _plot_confidence_levels_2d(ax, x, y, targets, bins=50, weights=None,
                               fill=False, smoothing_beam=0,
                               colors=None, cmap=None, label=None,
                               linewidths=2, **kwargs):
    """
    Plots contourf on input ax with contour selected by target levels. 

    Parameters
    ----------
    ax : Axes object
        Matplotlib Axes object onto which confidence curves will be plotted.
    x, y : array_like, shape (n, )
        Input data
    targets: list of floats normalized to one in decreasing order
        Target confidence levels
    bins: int
        Number of bins to perform histogram2d 
    weights: array, shape of x, y
        In case x and y are weighted
    fill: boolean
        Whether or not to fill curves.
    smoothing_beam: float
        Standard deviation for Gaussian kernel to be convolved with 2d histogram.
    """
    # calculating 2d histogram
    Z, xedges, yedges = np.histogram2d(x, y, bins, weights=weights)
    if smoothing_beam > 0:
        Z = ndimage.gaussian_filter(Z, sigma=smoothing_beam, order=0)
    dx = (xedges[1] - xedges[0])
    dy = (yedges[1] - yedges[0])
    xplot = xedges[1:] - dx/2
    yplot = yedges[1:] - dy/2
    # calculating areas for each height
    heights = np.linspace(0, Z.max()*0.9, 500)
    areas = [Z[Z > i].sum() for i in heights]
    areas /= Z.sum()
    # getting levels for each target
    clevels = []
    for t in targets:
        clevels.append(CubicSpline(heights, areas -
                                   t).roots(extrapolate=False)[0])
    clevels = np.array(clevels)
    # for better transition in plot
    Z = np.log(Z+1)
    clevels = np.log(clevels+1)
    if fill == True:
        return ax.contourf(xplot, yplot, Z.T, levels=np.append(clevels, Z.max()),
                           cmap=cmap, antialiased=True, **kwargs)
    else:
        return ax.contour(xplot, yplot, Z.T, levels=np.append(clevels, Z.max()),
                          colors=colors, linewidths=linewidths, cmap=cmap, **kwargs)


def _hist_stats(ax, arr, label, weights, fontsize=12):
    stats = DescrStatsW(arr, weights=weights)
    mean = np.round(stats.mean, 3)
    std = np.round(stats.std, 3)
    ax.set_title('{} = {} $\pm$ {}'.format(label, mean, std),
                 fontsize=fontsize)
