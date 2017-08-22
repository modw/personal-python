
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os

### home dir

cwd = os.getcwd().split('/')
HOME = cwd[1:cwd.index('Dropbox')]
HOME = "/"+"/".join(HOME)

### importing colormap and registering it
cmap_rgb = np.loadtxt(HOME + "/Dropbox/Doutorado/Research/Wiener Filtering/Messenger method code/Planck_T_Colormap_RGB.dat")
ptmap = mcolors.ListedColormap(cmap_rgb,name='PlanckTMap')
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
    #global cm
    cm = plt.get_cmap(s)
    cm.set_under('w',alpha=0)
    cm.set_bad('gray')
    return cm
