import numpy as np
import healpy as hp


def complex_to_real(complex_alm):
    """
    Transforms alm set in complex basis to real basis.
    To see how output is strutured see source code.
    To transform back to real basis use real_to_complex
    """
    LMAX = hp.Alm.getlmax(len(complex_alm))
    m_zero_bool = hp.sphtfunc.Alm.getlm(LMAX)[1] == 0
    # even and odd idices are separete because of a sign change
    m_even_bool = hp.sphtfunc.Alm.getlm(LMAX)[1] % 2 == 0
    m_odd_bool = hp.sphtfunc.Alm.getlm(LMAX)[1] % 2 != 0

    real_alm = np.zeros(len(complex_alm), dtype=complex)
    # the real part (.real) of the real_alm is for m>0 and the imaginary part (.imag) is for m<0
    real_alm[m_even_bool] = np.sqrt(2)*complex_alm[m_even_bool]
    real_alm[m_odd_bool] = - np.sqrt(2)*complex_alm[m_odd_bool]
    real_alm[m_zero_bool] = complex_alm[m_zero_bool]
    return real_alm


def real_to_complex(real_alm):
    """
    Transforms alm set in real basis to complex basis.
    To see how input is strutured see source code.
    To transform back to complex basis use complex_to_real.
    """
    LMAX = hp.Alm.getlmax(len(real_alm))
    m_zero_bool = hp.sphtfunc.Alm.getlm(LMAX)[1] == 0
    # even and odd idices are separe because of a sign change
    m_even_bool = hp.sphtfunc.Alm.getlm(LMAX)[1] % 2 == 0
    m_odd_bool = hp.sphtfunc.Alm.getlm(LMAX)[1] % 2 != 0

    complex_alm = np.zeros(len(real_alm), dtype=complex)
    complex_alm[m_even_bool] = (real_alm[m_even_bool].real+real_alm[m_even_bool].imag*1j)/(np.sqrt(2))
    complex_alm[m_odd_bool] = -(real_alm[m_odd_bool].real+real_alm[m_odd_bool].imag*1j)/(np.sqrt(2))
    complex_alm[m_zero_bool] = real_alm[m_zero_bool]
    return complex_alm


def constrained_alm(input_alm, cl_auto_in, cl_auto_out, cl_cross):
    """
    Generate constrained realizations of correlated alm sets given input alm,
    the two auto power spectra and the cross power spectrum.
    ATTENTION:
     - The way it is currently written, ell_max is given by length of input_alm.
     - Currently it only works with no monopole or dipole (c0 = c1 = 0 for
     every cl given)
    """
    # maximum l from input alm
    LMAX = hp.Alm.getlmax(len(input_alm))
    # getting the l list for non zero cls (no monopole or dipole)
    l_list = hp.Alm.getlm(LMAX)[0]
    non_zero = l_list > 1
    l_list_nz = l_list[non_zero]
    # crabbing the non zero cls and ordering to multiply by the alms
    cl_auto_in_nz, cl_auto_out_nz, cl_cross_nz = \
        cl_auto_in[l_list_nz], cl_auto_out[l_list_nz], cl_cross[l_list_nz]
    # defining weights to generate the constrained realizations
    W1 = cl_cross_nz/cl_auto_in_nz
    W2 = np.sqrt(cl_auto_out_nz - cl_cross_nz**2/cl_auto_in_nz)
    # calculating the output alm and returning them in complex basis
    input_alm_real = complex_to_real(input_alm)
    output_alm_real = np.zeros(len(input_alm_real), dtype=complex)
    random_complex = np.random.normal(size=len(l_list_nz))\
        + 1j*np.random.normal(size=len(l_list_nz))
    output_alm_real[non_zero] = W1*input_alm_real[non_zero] + W2*random_complex
    return real_to_complex(output_alm_real)


def corr_piece(input_alm, cl_auto_in, cl_cross):
    """
    Gives correlated piece of output alm.
    ATTENTION:
     - The way it is currently written, ell_max is given by length of input_alm.
     - Currently it only works with no monopole or dipole (c0 = c1 = 0 for
     every cl given)
    """
    # maximum l from input alm
    LMAX = hp.Alm.getlmax(len(input_alm))
    # getting the l list for non zero cls (no monopole or dipole)
    l_list = hp.Alm.getlm(LMAX)[0]
    non_zero = l_list > 1
    l_list_nz = l_list[non_zero]
    # crabbing the non zero cls and ordering to multiply by the alms
    cl_auto_in_nz, cl_cross_nz = \
        cl_auto_in[l_list_nz], cl_cross[l_list_nz]
    # defining weights to generate the constrained realizations
    W1 = cl_cross_nz/cl_auto_in_nz
    # calculating the output alm and returning them in complex basis
    input_alm_real = complex_to_real(input_alm)
    output_alm_real = np.zeros(len(input_alm_real), dtype=complex)
    output_alm_real[non_zero] = W1*input_alm_real[non_zero]
    return real_to_complex(output_alm_real)
