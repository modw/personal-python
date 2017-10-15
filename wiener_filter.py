# module to calculate Wiener filter in pixel space.
# should only work up to nside = 32.

import numpy as np
import healpy as hp
import scipy
from scipy import special

class WienerFilter:
    """Wiener filter class. Should be initiated with a mask if one is used.
    Use set methods to add signal and noise covariance information.
    Method get_wiener_filter will calculate wiener filter matrix and assign
    it to wiener_filter attribute. Method apply_wiener_filter will call
    get_wiener_filter if not called already and apply it to a data vector."""

    def __init__(self, nside, mask=None):
        self.params = Parameters(nside)
        if mask is not None:
            self.mask = Mask(mask, nside)
        else:
            self.mask = Mask(np.ones(self.params.npix))

    def set_signal_cov(self, cl, lmax=None):
        """ Calculate signal covariance matrix under attribute signal_cov.signal_cov
        from cl with default lmax being 1.5*nside"""
        if lmax is None:
            lmax = 3*self.params.nside//2
        self.signal_cov = SignalCov(cl, lmax, self.params.nside)

    def set_noise_cov(self, noise_cov):
        """Calculate noise covariance matrix and assign to atribute noise_cov."""
        self.noise_cov = NoiseCov(noise_cov, self.mask.mask)

    def get_wiener_filter(self):
        """Calculate Wiener filter from covariance matrices and mask. Assign it
        to wiener_filter attribute."""
        fl = np.dot(self.signal_cov.signal_cov,np.transpose(self.mask.mask_cov))
        fr = np.dot(self.mask.mask_cov,fl) + self.noise_cov.noise_cov
        fr_chol = np.linalg.cholesky(fr)
        fr_inv = np.dot(np.linalg.inv(np.transpose(fr_chol)),np.linalg.inv(fr_chol))
        self.wiener_filter = np.dot(fl,fr_inv)

    def apply_wiener_filter(self,data):
        """Apply Wiener filter to data vector"""
        if self.wiener_filter is None:
            self.get_wiener_filter()
        return np.dot(self.wiener_filter, data[self.mask.good_pix])

#### support classes

class Parameters:
    def __init__(self, nside):
        """Set base parameters"""
        self.nside = nside
        self.npix = hp.nside2npix(nside)

class Mask:
    """Instantiate mask, get good_pix and bad_pix and degrade with 0.9 criteria
    if nside_out is specified."""
    def __init__(self, mask, nside_out=None):
        msk = np.copy(mask)
        if nside_out is not None:
            msk = hp.ud_grade(msk, nside_out)
            msk[msk >= 0.9] = 1
            msk[msk < 0.9] = 0
        self.good_pix = msk == 1
        self.bad_pix = msk == 0
        self.mask = msk
        self.mask_cov = np.diag(msk)[mask == 1]


class SignalCov:
    """Calculate signal covariance matrix"""
    def __init__(self, cl, lmax, nside):
        self.lmax = lmax
        self.cl = cl[:lmax+1]
        self.l = np.arange(lmax+1)
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.get_signal_cov()

    def get_signal_cov(self):
        """Get signal covariance matrix under attribute signal_cov using helper
        methods."""
        cosines = self.get_cosines(self.nside)
        unique, uniqinv = self.get_unique(cosines)
        def s(x):
            return self.two_point(x, self.l, self.cl, self.lmax)
        s = np.vectorize(s)
        s_unique = s(unique)
        s_matrix = s_unique[uniqinv].reshape(self.npix,self.npix)
        self.signal_cov = s_matrix

    def get_cosines(self, nside):
        """Calculate cosines for every par of pixels in the healpix grid."""
        vectors = np.array(hp.pix2vec(nside,np.arange(hp.nside2npix(nside)))).transpose()
        cosines = np.einsum('ik,jk', vectors, vectors)
        cosines[cosines>1] =1
        cosines[cosines<-1]=-1
        return cosines

    def get_unique(self, cosines):
        """Get unique cosines to calculate the two point function from."""
        uniq, uniqindex, uniqinv, uniqcounts = \
        np.unique(cosines.flatten(),return_index=True, return_inverse=True,return_counts=True)
        return uniq, uniqinv

    def two_point(self, x, l, cl, lmax):
        """Two point function on the sphere."""
        return np.sum((2*l+1)*cl*scipy.special.lpn(lmax,x)[0])/(4*np.pi)

class NoiseCov:
    def __init__(self, noise_cov, mask):
        self.noise_cov = np.diag(noise_cov[mask == 1])
