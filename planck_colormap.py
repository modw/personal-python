
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# importing colormap and registering it
cmap_rgb = np.loadtxt("./files/Planck_T_Colormap_RGB.dat")
ptmap = mcolors.ListedColormap(cmap_rgb, name='PlanckTMap')
plt.register_cmap(cmap=ptmap)


def colormap(s='PlanckTMap'):
    """
    Function to set colormap while keeping proper background and badpix colors.
    ----------
    Parameters
    ----------
    s: String
    Name of the desired colormap. Default is 'PlanckTMap'.
    For non-divergent colormaps the recommended is 'viridis'.
    """
    cm = plt.get_cmap(s)
    cm.set_under('w', alpha=0)
    cm.set_bad('gray')
    return cm
