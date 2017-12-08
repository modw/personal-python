import healpy as hp
import numpy as np


def alm_coord_change(alm, coord):
    """Rotates alm between coordinate systems.
    coord : a tuple with initial and final coord systems (e.g. ('G','E'))
    Returns >> new rotated alm"""
    angs = hp.rotator.coordsys2euler_zyz(coord)
    lmax = hp.Alm.getlmax(len(alm))
    alm_rot = np.copy(alm)
    hp.rotate_alm(alm_rot, *angs, lmax)
    return alm_rot


def rotate_map(m, coord, lmax):
    """Rotates map by rotating map alms.
    coord : a tuple with initial and final coord systems (e.g. ('G','E'))
    lmax: int, maximum multipole for convertion to alm
    Returns >> new rotated map"""
    nside = hp.get_nside(m)
    map_alm = hp.map2alm(m, lmax=lmax)
    map_alm = alm_coord_change(map_alm, coord)
    return hp.alm2map(map_alm, nside, lmax, verbose=False)
