import numpy as np
from types import MethodType, FunctionType
from wrapper import *
from ctypes import c_int, c_double, c_char_p, POINTER, Structure
from multiprocessing import cpu_count, Pool


class ADict(dict):
    """
    Dictionary where you can access keys as attributes
    """
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            dict.__getattribute__(self, item)


class RamanControl:
    """
    """

    def __init__(self, params, **kwargs):
        """
        __init__ function call to initialize variables from the
        parameters for the class instance provided in __main__ and
        add new variables for use in other functions in this class.
        """

        for name, value in kwargs.items():
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
                setattr(self, name, value)

        self.time_A = np.linspace(-params.timeAMP_A, params.timeAMP_A, params.timeDIM_A)
        self.time_R = np.linspace(-params.timeAMP_R, params.timeAMP_R, params.timeDIM_R)
        # self.frequency_A = 1./np.linspace(1./params.frequencyMAX_A, 1./params.frequencyMIN_A, params.frequencyDIM_A)
        self.frequency_A = params.frequency_A
        self.frequency_R = 1./np.linspace(1./params.frequencyMAX_R, 1./params.frequencyMIN_R, params.frequencyDIM_R)
        self.field_A = np.zeros(params.timeDIM_A, dtype=np.complex)
        self.field_R = np.zeros(params.timeDIM_R, dtype=np.complex)
        self.gamma_population_decay = np.ascontiguousarray(self.gamma_population_decay)
        self.gamma_pure_dephasingA = np.ascontiguousarray(self.gamma_pure_dephasingA)
        self.gamma_pure_dephasingB = np.ascontiguousarray(self.gamma_pure_dephasingB)
        self.mu_A = np.ascontiguousarray(self.mu)
        self.mu_B = np.ascontiguousarray(self.mu.copy())
        self.rho_0 = np.ascontiguousarray(params.rho_0)
        self.rhoA = np.ascontiguousarray(params.rho_0.copy())
        self.rhoB = np.ascontiguousarray(params.rho_0.copy())
        self.energies_A = np.ascontiguousarray(self.energies_A)
        self.energies_B = np.ascontiguousarray(self.energies_B)

        N = len(self.energies_A)

        self.abs_spectraA = np.ascontiguousarray(np.zeros(len(self.frequency_A)))
        self.abs_spectraB = np.ascontiguousarray(np.zeros(len(self.frequency_A)))
        self.Raman_spectraA = np.ascontiguousarray(np.zeros(len(self.frequency_R)))
        self.Raman_spectraB = np.ascontiguousarray(np.zeros(len(self.frequency_R)))

        self.dyn_rho_R_A = np.ascontiguousarray(np.zeros((N, params.timeDIM_R)), dtype=np.complex)
        self.dyn_rho_R_B = np.ascontiguousarray(np.zeros((N, params.timeDIM_R)), dtype=np.complex)

        self.dyn_rho_A_A = np.ascontiguousarray(np.zeros((N, params.timeDIM_A)), dtype=np.complex)
        self.dyn_rho_A_B = np.ascontiguousarray(np.zeros((N, params.timeDIM_A)), dtype=np.complex)

    def create_molecules(self, molA, molB):
        molA.nDIM = len(self.energies_A)
        molA.energies = self.energies_A.ctypes.data_as(POINTER(c_double))
        molA.gamma_population_decay = self.gamma_population_decay.ctypes.data_as(POINTER(c_double))
        molA.gamma_pure_dephasing = self.gamma_pure_dephasingA.ctypes.data_as(POINTER(c_double))
        molA.rho_0 = self.rho_0.ctypes.data_as(POINTER(c_complex))
        molA.mu = self.mu_A.ctypes.data_as(POINTER(c_complex))

        molA.field_A = self.field_A.ctypes.data_as(POINTER(c_complex))
        molA.field_R = self.field_R.ctypes.data_as(POINTER(c_complex))
        molA.rho = self.rhoA.ctypes.data_as(POINTER(c_complex))
        molA.absorption_spectra = self.abs_spectraA.ctypes.data_as(POINTER(c_double))
        molA.Raman_spectra = self.Raman_spectraA.ctypes.data_as(POINTER(c_double))
        molA.dyn_rho_A = self.dyn_rho_A_A.ctypes.data_as(POINTER(c_complex))
        molA.dyn_rho_R = self.dyn_rho_R_A.ctypes.data_as(POINTER(c_complex))

        molB.nDIM = len(self.energies_B)
        molB.energies = self.energies_B.ctypes.data_as(POINTER(c_double))
        molB.gamma_population_decay = self.gamma_population_decay.ctypes.data_as(POINTER(c_double))
        molB.gamma_pure_dephasing = self.gamma_pure_dephasingB.ctypes.data_as(POINTER(c_double))
        molB.rho_0 = self.rho_0.ctypes.data_as(POINTER(c_complex))

        molB.mu = self.mu_B.ctypes.data_as(POINTER(c_complex))
        molB.field_A = self.field_A.ctypes.data_as(POINTER(c_complex))
        molB.field_R = self.field_R.ctypes.data_as(POINTER(c_complex))
        molB.rho = self.rhoB.ctypes.data_as(POINTER(c_complex))
        molB.absorption_spectra = self.abs_spectraB.ctypes.data_as(POINTER(c_double))
        molB.Raman_spectra = self.Raman_spectraB.ctypes.data_as(POINTER(c_double))
        molB.dyn_rho_A = self.dyn_rho_A_B.ctypes.data_as(POINTER(c_complex))
        molB.dyn_rho_R = self.dyn_rho_R_B.ctypes.data_as(POINTER(c_complex))

    def create_parameters_spectra(self, spectra_params, params):
        spectra_params.rho_0 = self.rho_0.ctypes.data_as(POINTER(c_complex))

        spectra_params.nDIM = len(self.energies_A)
        spectra_params.nEXC = params.nEXC

        spectra_params.frequency_A = self.frequency_A.ctypes.data_as(POINTER(c_double))
        spectra_params.frequency_R = self.frequency_R.ctypes.data_as(POINTER(c_double))
        spectra_params.freqDIM_A = len(self.frequency_A)
        spectra_params.freqDIM_R = len(self.frequency_R)

        spectra_params.time_A = self.time_A.ctypes.data_as(POINTER(c_double))
        spectra_params.time_R = self.time_R.ctypes.data_as(POINTER(c_double))
        spectra_params.timeDIM_A = len(self.time_A)
        spectra_params.timeDIM_R = len(self.time_R)

        spectra_params.field_amp_A = params.field_amp_A
        spectra_params.field_amp_R = params.field_amp_R

        spectra_params.omega_R = params.omega_R
        spectra_params.omega_v = params.omega_v
        spectra_params.omega_e = params.omega_e

        spectra_params.exc_coeff_ratio = params.exc_coeff_ratio
        spectra_params.thread_num = params.num_threads
        spectra_params.mu_guess = params.mu_guess.ctypes.data_as(POINTER(c_double))
        spectra_params.mu_guess_num = len(params.mu_guess)
        spectra_params.freq_points = params.freq_points.ctypes.data_as(POINTER(c_double))
        spectra_params.reference_spectra = params.reference_spectra.ctypes.data_as(POINTER(c_double))
        spectra_params.Raman_levels = params.Raman_levels.ctypes.data_as(POINTER(c_double))

    def calculate_spectra(self, params):
        molA = Molecule()
        molB = Molecule()
        self.create_molecules(molA, molB)
        params_spectra = Parameters()
        self.create_parameters_spectra(params_spectra, params)
        CalculateSpectra(molA, molB, params_spectra)
        return


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import time
    import pandas as pd

    energy_factor = 1. / 27.211385
    time_factor = .02418884 / 1000

    N = 8
    energies_A = np.empty(N)
    energies_B = np.empty(N)
    N_vib = 4
    N_exc = N - N_vib
    
    rho_0 = np.zeros((N, N), dtype=np.complex)
    rho_0[0, 0] = 1. + 0j

    mu = 4.97738 * np.ones_like(rho_0)
    np.fill_diagonal(mu, 0j)
    
    population_decay = 2.418884e-8

    gamma_population_decay = np.ones((N, N)) * population_decay
    np.fill_diagonal(gamma_population_decay, 0.0)
    gamma_population_decay = np.tril(gamma_population_decay).T

    electronic_dephasingA = 1.8*2.418884e-4
    electronic_dephasingB = 2.418884e-4
    vibrational_dephasing = 2.418884e-6
    gamma_pure_dephasingA = np.ones_like(gamma_population_decay) * vibrational_dephasing
    gamma_pure_dephasingB = np.ones_like(gamma_population_decay) * vibrational_dephasing
    np.fill_diagonal(gamma_pure_dephasingA, 0.0)
    np.fill_diagonal(gamma_pure_dephasingB, 0.0)

    for i in range(N_vib):
        for j in range(N_vib, N):
            gamma_pure_dephasingA[i, j] = electronic_dephasingA
            gamma_pure_dephasingA[j, i] = electronic_dephasingA
            gamma_pure_dephasingB[i, j] = electronic_dephasingB
            gamma_pure_dephasingB[j, i] = electronic_dephasingB

    df = pd.read_csv('Cph8_RefSpectra.csv', sep=',')
    wavelengthPR = df.values[:, 0][:600][0:-1:6]
    wavelengthPFR = df.values[:, 0][:700][0:-1:7]

    freqPR = (1239.84 * energy_factor / wavelengthPR)
    freqPFR = (1239.84 * energy_factor / wavelengthPFR)
    absPR = df.values[:, 1][:600][0:-1:6]
    absPFR = df.values[:, 2][:700][0:-1:7]

    absPR *= 100. / absPR.max()
    absPFR *= 100. / absPFR.max()

    mu_factor_A = np.asarray([0.10, 0.15, 0.18, 0.23, 0.31, 0.37, 0.45, 0.5, 0.55, 0.75, 0.8, 0.5][::-1])
    mu_factor_B = np.asarray([0.10, 0.15, 0.18, 0.23, 0.31, 0.37, 0.45, 0.5, 0.55, 0.75, 0.8, 0.5][::-1])
    freq_points_A = np.asarray(1239.84 / np.linspace(505, 690, 4 * len(mu_factor_A))[::-1])
    freq_points_B = np.asarray(1239.84 / np.linspace(540, 742, 4 * len(mu_factor_A))[::-1])
    Raman_levels = np.asarray([0.000, 0.09832, 0.16304, 0.20209])*energy_factor

    params = ADict(

        nEXC=N_exc,
        
        energy_factor=energy_factor,
        time_factor=time_factor,
        rho_0=rho_0,

        timeDIM_A=1000,
        timeAMP_A=5000,
        timeDIM_R=10000,
        timeAMP_R=52000,

        frequencyDIM_A=freqPR.size,
        frequency_A=freqPR,
        frequencyDIM_R=250,
        frequencyMIN_R=0.075*energy_factor,
        frequencyMAX_R=0.21*energy_factor,

        field_amp_R=0.0000185,
        field_amp_A=0.0000018,

        omega_R=0.5*energy_factor,
        omega_v=energies_A[3],
        omega_e=(energies_A[4] - energies_A[3]),
        
        exc_coeff_ratio=.7366,
        num_threads=cpu_count(),
        mu_guess=mu_factor_A,
        freq_points=np.ascontiguousarray(freq_points_A),

        reference_spectra=np.ascontiguousarray(absPR),
        Raman_levels=Raman_levels
    )

    FourLevels = dict(
        energies_A=energies_A,
        energies_B=energies_B,
        gamma_population_decay=gamma_population_decay,
        gamma_pure_dephasingA=gamma_pure_dephasingA,
        gamma_pure_dephasingB=gamma_pure_dephasingB,
        mu=mu,
    )
    
    def render_ticks(axes):
        axes.get_xaxis().set_tick_params(which='both', direction='in', width=1, labelrotation=0, labelsize='x-large')
        axes.get_yaxis().set_tick_params(which='both', direction='in', width=1, labelcolor='r', labelsize='x-large')
        axes.get_xaxis().set_ticks_position('both')
        axes.get_yaxis().set_ticks_position('both')
        axes.grid()

    start = time.time()

    molecule = RamanControl(params, **FourLevels)
    molecule.calculate_spectra(params)

    print(np.linalg.norm((absPR - molecule.abs_spectraA), 1))

    print(time.time() - start)

    fig, axes = plt.subplots(nrows=1, ncols=1)
    axes.plot(energy_factor * 1239.84 / molecule.frequency_A, molecule.abs_spectraA, 'r*-', label='Fitted_A', linewidth=2.)
    axes.plot(wavelengthPR, absPR, 'k*-', label='Experimental_A')
    axes.set_xlabel('Wavelength (in nm)', fontweight='bold')
    axes.set_ylabel('Normalized spectra', fontweight='bold')
    plt.legend()
    render_ticks(axes)

    plt.show()
