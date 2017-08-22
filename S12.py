import numpy as np
from scipy import special
import healpy as hp
import os

badval = -1.6375e+30
#importing support files
DATAPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files')
Ill = np.load(os.path.join(DATAPATH, 'I_TT_060_small.npy'))
xsp, Csp = np.loadtxt(os.path.join(DATAPATH, 'spice.cor'),unpack=True, usecols=(1,2))

def get_s12(map1,lmax,mask=None,wl=None):
    """
    Calculate S12 for a given map and mask (given by wl)
    WARNING: maximum lmax supported is 399
    ----------
    Parameters
    ----------
    map1 : float, array-like
    An array containing the map.

    mask : int, array-like
    Mask used to calculate cut sky S12. 1s and 0s.

    wl: float, array-like
    Power spectrum of the mask. Using this as input will make the calculation
    more efficient when compared to using the mask as input.

    lmax : int, scalar
    Maximum multipole to which perform the sum to calculate S12
    """
    l = np.arange(lmax+1)

    if mask is None and wl is None:
        Cl = hp.anafast(map1,lmax=lmax)
        return np.dot(Cl[2:lmax],np.dot(Ill[2:lmax,2:lmax],Cl[2:lmax]))
    elif mask is not None:
        # masking map
        map1_masked = np.copy(map1)
        map1_masked[mask == 0] = badval
        wl = hp.anafast(mask, lmax=lmax)
    elif wl is not None:
        map1_masked = np.copy(map1)
        pass

    # power spectrum
    Cl_til = hp.anafast(map1_masked,lmax=lmax)

    # defining support functions to calculate S12
    def A(x):
        return np.sum(2*np.pi*(2*l+1)*wl*special.lpn(lmax,x)[0])**(-1)
    def C_cut(x):
        return 2*np.pi*A(x)*np.sum((2*l+1)*Cl_til*special.lpn(lmax,x)[0])
    C_cut = np.vectorize(C_cut)
    # calculating S12
    Cl_cutt = 4*np.pi*np.polynomial.legendre.legfit(xsp, C_cut(xsp), lmax, rcond=None, full=False, w=None)/(2*l+1)
    S12_cut = np.dot(Cl_cutt[2:lmax],np.dot(Ill[2:lmax,2:lmax],Cl_cutt[2:lmax]))

    return S12_cut
