import healpy as hp
import numpy as np
from sympy.physics.wigner import wigner_3j


def _w3(l1,l2,l3):
    return np.asfarray(wigner_3j(l1,l2,l3,0,0,0))
_w3 = np.vectorize(_w3)


def _check_lmax(wl,lmax):
    """Readjusts and returns wl and lmax.
    Written to be used in MASTER algorithm functions"""
    if lmax is None:
        return wl, len(wl) - 1
    elif len(wl) >= (lmax+1):
        return wl[:lmax+1], lmax
    elif len(wl) < (lmax+1):
        print("lmax > len(wl)-1, using lmax = len(wl) -1 = {} instead".format(len(wl)-1))
        return wl, len(wl) - 1

    
def master(l1,l2,wl,lmax=None):
    """Calculate component (l1,l2) of M matrix.
    > Parameters:
    l1,l2: ints, elements of matrix
    wl: numpy array, power spectrum of mask
    lmax: int, if not give lmax is taken from wl
    > Returns:
    M_l1l2: float, element (l1,l2) of M matrix"""
    wl_in, lmax_in = _check_lmax(wl,lmax)
    l3 = np.arange(lmax_in+1)
    sum_l3 = np.sum((2*l3+1)*wl_in*_w3(l1,l2,l3)**2)
    return (2*l2+1)*sum_l3/(4*np.pi)


def master_matrix(wl,lmax=None, get_inv = True):
    """Calculate M matrix as (lmax+1,lmax+1) ndarray.
    > Parameters:
    wl: numpy array, power spectrum of mask
    lmax: int, if not give lmax is taken from wl
    get_inv: Bool, if True return matrix to transform from pseudo Cl to Cl,
    if False returns transformation from Cl to pseudo Cl.
    > Returns:
    M: ndarray of dimension, the master matrix"""
    wl_in, lmax_in = _check_lmax(wl,lmax)
    def master2(l1,l2):
        return master(l1,l2,wl_in,lmax_in)
    master2 = np.vectorize(master2)
    lrange = np.arange(lmax_in+1)
    matrix = master2(lrange[:,None],lrange[None,:])
    if get_inv == True:
        return np.linalg.inv(matrix)
    else:
        return matrix