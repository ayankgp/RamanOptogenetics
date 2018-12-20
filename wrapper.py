import os
import ctypes
from ctypes import c_int, c_double, POINTER, Structure


__doc__ = """
Python wrapper for RamanOptogenetics.c
Compile with:
gcc -O3 -shared -o RamanOptogenetics.so RamanOptogenetics.c -lm -fopenmp -fPIC
"""


class c_complex(Structure):
    """
    Complex double ctypes
    """
    _fields_ = [
        ('real', c_double),
        ('imag', c_double)
    ]


class Parameters(Structure):
    """
    Parameters structure ctypes
    """
    _fields_ = [

        ('rho_0', POINTER(c_complex)),

        ('nDIM', c_int),
        ('nEXC', c_int),

        ('frequency_A', POINTER(c_double)),
        ('frequency_R', POINTER(c_double)),
        ('freqDIM_A', c_int),
        ('freqDIM_R', c_int),

        ('time_A', POINTER(c_double)),
        ('time_R', POINTER(c_double)),
        ('timeDIM_A', c_int),
        ('timeDIM_R', c_int),

        ('field_amp_A', c_double),
        ('field_amp_R', c_double),

        ('omega_R', c_double),
        ('omega_v', c_double),
        ('omega_e', c_double),

        ('exc_coeff_ratio', c_double),
        ('thread_num', c_int),
        ('mu_guess_A', POINTER(c_double)),
        ('mu_guess_B', POINTER(c_double)),
        ('mu_guess_num', c_int),
        ('freq_points_A', POINTER(c_double)),
        ('freq_points_B', POINTER(c_double)),
        ('reference_spectra', POINTER(c_double)),
        ('Raman_levels_A', POINTER(c_double)),
        ('Raman_levels_B', POINTER(c_double)),
        ('lower_bound', POINTER(c_double)),
        ('upper_bound', POINTER(c_double)),
        ('max_iter', c_int)
    ]


class Molecule(Structure):
    """
    Parameters structure ctypes
    """
    _fields_ = [
        ('nDIM', c_int),
        ('energies', POINTER(c_double)),
        ('gamma_population_decay', POINTER(c_double)),
        ('gamma_pure_dephasing', POINTER(c_double)),
        ('rho_0', POINTER(c_complex)),
        ('mu', POINTER(c_complex)),

        ('field_A', POINTER(c_complex)),
        ('field_R', POINTER(c_complex)),
        ('rho', POINTER(c_complex)),
        ('absorption_spectra', POINTER(c_double)),
        ('Raman_spectra', POINTER(c_double)),

        ('dyn_rho_A', POINTER(c_complex)),
        ('dyn_rho_R', POINTER(c_complex))
    ]


try:
    # Load the shared library assuming that it is in the same directory
    lib1 = ctypes.cdll.LoadLibrary(os.getcwd() + "/RamanOptogenetics.so")
except OSError:
    raise NotImplementedError(
        """
        The library is absent. You must compile the C shared library using the commands:
        gcc -O3 -shared -o RamanOptogenetics.so RamanOptogenetics.c -lm -fopenmp -fPIC
        """
    )

lib1.CalculateSpectra.argtypes = (
    POINTER(Molecule),                  # molecule molA
    POINTER(Parameters),        # parameter field_params
)
lib1.CalculateSpectra.restype = POINTER(c_complex)


def CalculateSpectra(molA, params):
    return lib1.CalculateSpectra(
        molA,
        params
    )


lib1.CalculateControl.argtypes = (
    POINTER(Molecule),                  # molecule molA
    POINTER(Molecule),                  # molecule molB
    POINTER(Parameters),        # parameter field_params
)
lib1.CalculateControl.restype = POINTER(c_complex)


def CalculateControl(molA, molB, params):
    return lib1.CalculateControl(
        molA,
        molB,
        params
    )