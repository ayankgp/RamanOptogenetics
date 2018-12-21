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
        spectra_params.mu_guess_A = params.mu_guess_A.ctypes.data_as(POINTER(c_double))
        spectra_params.mu_guess_B = params.mu_guess_B.ctypes.data_as(POINTER(c_double))
        spectra_params.mu_guess_num = len(params.mu_guess_A)
        spectra_params.freq_points_A = params.freq_points_A.ctypes.data_as(POINTER(c_double))
        spectra_params.freq_points_B = params.freq_points_B.ctypes.data_as(POINTER(c_double))
        spectra_params.reference_spectra = params.reference_spectra.ctypes.data_as(POINTER(c_double))
        spectra_params.Raman_levels_A = params.Raman_levels_A.ctypes.data_as(POINTER(c_double))
        spectra_params.Raman_levels_B = params.Raman_levels_B.ctypes.data_as(POINTER(c_double))
        spectra_params.lower_bound = params.lower_bound.ctypes.data_as(POINTER(c_double))
        spectra_params.upper_bound = params.upper_bound.ctypes.data_as(POINTER(c_double))
        spectra_params.max_iter = params.max_iter

    def calculate_spectra(self, params):
        molA = Molecule()
        molB = Molecule()
        self.create_molecules(molA, molB)
        params_spectra = Parameters()
        self.create_parameters_spectra(params_spectra, params)
        # CalculateSpectra(molA, params_spectra)
        CalculateControl(molA, molB, params_spectra)
        return


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import time
    import pandas as pd

    import numpy as np
    from scipy.interpolate import interp1d

    dfC = pd.read_csv('ChR2.csv', sep=',')
    dfG = pd.read_csv('ChR2.csv', sep=',')
    wavelengthG = dfG.values[:, 0]
    wavelengthC = dfC.values[:, 0]

    energy_factor = 1. / 27.211385
    absG = dfG.values[:, 1]
    absG *= 100. / absG.max()

    absC = dfC.values[:, 1]
    absC *= 100. / absC.max()

    fG = interp1d(wavelengthG, absG, kind='cubic')
    wavelengthG_new = 1. / np.linspace(1. / wavelengthG.max(), 1. / wavelengthG.min(), 100)
    absG_new = fG(wavelengthG_new)
    absG_new *= 100. / absG_new.max()
    freqG = (1239.84 * energy_factor / wavelengthG_new)

    fC = interp1d(wavelengthC, absC, kind='cubic')
    wavelengthC_new = 1. / np.linspace(1. / wavelengthC.max(), 1. / wavelengthC.min(), 100)
    absC_new = fC(wavelengthC_new)
    absC_new *= 100. / absC_new.max()
    freqC = (1239.84 * energy_factor / wavelengthC_new)

    plt.plot(wavelengthG, absG)
    plt.plot(wavelengthC, absC)
    plt.plot(wavelengthG_new, absG_new, 'k')
    plt.plot(wavelengthC_new, absC_new, 'r')

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

    # electronic_dephasingA = 2.5 * 2.418884e-4  # GCaMP
    electronic_dephasingA = 5.0 * 2.418884e-4  # channelrhodopsin
    electronic_dephasingB = 2.0 * 2.418884e-4  # cerulean
    vibrational_dephasing = 0.1 * 2.418884e-5
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

    mu_factor_A = np.asarray([0.99091, 0.957032, 0.844571, 0.782997, 0.719178, 0.571303, 0.471282, 0.363345, 0.273248, 0.188955, 0.246669])  # GCaMP 11
    # mu_factor_A = np.asarray([0.376979, 0.760245, 1, 0.999439, 0.999927, 0.786846, 0.708944, 0.695928, 0.957517, 0.998632, 0.996917, 0.999331])  # channelrhodopsin 12
    mu_factor_B = np.asarray([0.387479, 0.778226, 0.978226, 0.964315, 0.856, 0.743111, 0.692994, 0.725469, 0.750104, 0.763556, 0.726722])  # channelrhodopsin 11
    # mu_factor_B = np.asarray([0.360422, 0.88336, 1, 0.999998, 0.999953, 0.7802, 0.616421, 0.456223, 0.348055, 0.305119, 0.315951, 0.274767])  # cerulean

    lower_bound = np.zeros_like(mu_factor_A)
    upper_bound = np.ones_like(mu_factor_A)

    freq_points_A = np.asarray(1239.84 / np.linspace(400, 507, 4 * len(mu_factor_A))[::-1])*energy_factor  # GCaMP
    freq_points_B = np.asarray(1239.84 / np.linspace(270, 540, 4 * len(mu_factor_A))[::-1])*energy_factor  # channelrhodopsin
    # freq_points_B = np.asarray(1239.84 / np.linspace(320, 480, 4 * len(mu_factor_A))[::-1])*energy_factor  # cerulean

    Raman_levels_A = np.asarray([0.000, 0.09832, 0.16304, 0.20209])*energy_factor
    Raman_levels_B = np.asarray([0.000, 0.09832, 0.16304, 0.19909])*energy_factor

    mu_factor_append_A = np.concatenate((np.asarray([1]), mu_factor_A))
    mu_factor_append_B = np.concatenate((np.asarray([1]), mu_factor_B))
    mu_big_A = np.kron(np.outer(mu_factor_append_A, mu_factor_append_A), np.ones((4, 4), dtype=np.complex))
    mu_big_B = np.kron(np.outer(mu_factor_append_B, mu_factor_append_B), np.ones((4, 4), dtype=np.complex))
    np.fill_diagonal(mu_big_A, 0j)
    print(mu_big_A.real)
    np.fill_diagonal(mu_big_B, 0j)
    print(mu_big_B.real)

    print(1239.84/(freq_points_A[0]/energy_factor))
    params = ADict(

        nEXC=N_exc,
        
        energy_factor=energy_factor,
        time_factor=time_factor,
        rho_0=rho_0,

        timeDIM_A=1000,
        timeAMP_A=5000,
        timeDIM_R=10000,
        timeAMP_R=52000,

        frequencyDIM_A=freqG.size,
        frequency_A=freqG,
        frequencyDIM_R=250,
        frequencyMIN_R=0.075*energy_factor,
        frequencyMAX_R=0.21*energy_factor,

        field_amp_R=0.000032,
        field_amp_A=0.00000145,

        omega_R=0.5*energy_factor,
        omega_v=Raman_levels_A[3],
        omega_e=(freq_points_A[0] - Raman_levels_A[3]),
        # omega_e=freq_points_A[0],
        #
        # omega_v=Raman_levels_B[3],
        # omega_e=(freq_points_B[0] - Raman_levels_B[3]),
        # omega_e=freq_points_B[0],

        exc_coeff_ratio=1,
        num_threads=cpu_count(),
        mu_guess_A=mu_factor_A,
        mu_guess_B=mu_factor_B,
        freq_points_A=np.ascontiguousarray(freq_points_A),
        freq_points_B=np.ascontiguousarray(freq_points_B),

        reference_spectra=np.ascontiguousarray(absG_new),
        Raman_levels_A=Raman_levels_A,
        Raman_levels_B=Raman_levels_B,

        lower_bound=lower_bound,
        upper_bound=upper_bound,

        max_iter=100
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

    # print(np.linalg.norm((absG_new - molecule.abs_spectraA), 1))
    #
    # print(time.time() - start)
    #
    # fig, axes = plt.subplots(nrows=1, ncols=1)
    # axes.plot(energy_factor * 1239.84 / molecule.frequency_A, molecule.abs_spectraA, 'r*-', label='Fitted_A', linewidth=2.)
    # axes.plot(wavelengthG_new, absG_new, 'k', label='Experimental_A')
    # axes.set_xlabel('Wavelength (in nm)', fontweight='bold')
    # axes.set_ylabel('Normalized spectra', fontweight='bold')
    # plt.legend()
    # render_ticks(axes)
    # fig.savefig('fit_chr2.eps', format='eps')

    # fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    #
    # molecule.time_A += molecule.time_R.max() + molecule.time_A.max()
    # time_axis = time_factor * (molecule.time_R.max() + np.concatenate((molecule.time_R, molecule.time_A)))
    #
    # axes[0].plot(time_factor * (molecule.time_R.max() + molecule.time_R), 5.142e9*molecule.field_R.real, 'k', linewidth=1.5)
    # axes[0].plot(time_factor * (molecule.time_R.max() + molecule.time_A), 5.142e9*molecule.field_A.real, 'darkblue', linewidth=1.5)
    #
    # axes[1].plot(time_factor * (molecule.time_R.max() + molecule.time_R), molecule.dyn_rho_R_A[0], 'b', label='g1_A', linewidth=1.)
    # axes[1].plot(time_factor * (molecule.time_R.max() + molecule.time_A), molecule.dyn_rho_A_A[0], 'b', label='g1_A', linewidth=2.5)
    #
    # axes[1].plot(time_factor * (molecule.time_R.max() + molecule.time_R), molecule.dyn_rho_R_A[3], 'r', label='g4_A', linewidth=1.)
    # axes[1].plot(time_factor * (molecule.time_R.max() + molecule.time_A), molecule.dyn_rho_A_A[3], 'r', label='g4_A', linewidth=2.5)
    #
    # axes[1].plot(time_factor * (molecule.time_R.max() + molecule.time_R), molecule.dyn_rho_R_A[4:].sum(axis=0), 'k', label='EXC_A', linewidth=1.)
    # axes[1].plot(time_factor * (molecule.time_R.max() + molecule.time_A), molecule.dyn_rho_A_A[4:].sum(axis=0), 'k', label='EXC_A', linewidth=2.5)
    #
    # axes[2].plot(time_factor * (molecule.time_R.max() + molecule.time_R), molecule.dyn_rho_R_B[0], 'b', label='g1_B', linewidth=1.)
    # axes[2].plot(time_factor * (molecule.time_R.max() + molecule.time_A), molecule.dyn_rho_A_B[0], 'b', label='g1_B', linewidth=2.5)
    #
    # axes[2].plot(time_factor * (molecule.time_R.max() + molecule.time_R), molecule.dyn_rho_R_B[3], 'r', label='g4_B', linewidth=1.)
    # axes[2].plot(time_factor * (molecule.time_R.max() + molecule.time_A), molecule.dyn_rho_A_B[3], 'r', label='g4_B', linewidth=2.5)
    #
    # axes[2].plot(time_factor * (molecule.time_R.max() + molecule.time_R), molecule.dyn_rho_R_B[4:].sum(axis=0), 'k', label='EXC_B', linewidth=1.)
    # axes[2].plot(time_factor * (molecule.time_R.max() + molecule.time_A), molecule.dyn_rho_A_B[4:].sum(axis=0), 'k', label='EXC_B', linewidth=2.5)
    #
    # axes[2].set_xlabel('Time (in ps)', fontweight='bold')
    # axes[0].set_ylabel('Electric field \n (in V/cm)', fontweight='bold')
    # axes[0].ticklabel_format(style='sci', scilimits=(0, 3))
    # axes[1].set_ylabel('Population \n VSFP', fontweight='bold')
    # axes[2].set_ylabel('Population \n ChR2', fontweight='bold')
    # render_ticks(axes[0])
    # render_ticks(axes[1])
    # render_ticks(axes[2])
    # axes[0].yaxis.set_label_position("right")
    # axes[1].yaxis.set_label_position("right")
    # axes[2].yaxis.set_label_position("right")
    #
    # axes[1].legend(loc=6)
    # axes[2].legend(loc=6)
    #
    # fig.subplots_adjust(left=0.15, hspace=0.1)
    #
    # print(molecule.rhoA.diagonal()[:4].sum(), molecule.rhoA.diagonal()[4:].sum())
    # print(molecule.rhoA.diagonal(), molecule.rhoA.diagonal().sum())
    # print()
    # print(molecule.rhoB.diagonal()[:4].sum(), molecule.rhoB.diagonal()[4:].sum())
    # print(molecule.rhoB.diagonal(), molecule.rhoB.diagonal().sum())
    #
    # print(molecule.rhoA.diagonal()[4:].sum() / molecule.rhoB.diagonal()[4:].sum())
    # print(molecule.rhoA.diagonal()[4:].sum() ** 2 / molecule.rhoB.diagonal()[4:].sum())
    #
    # print(molecule.rhoB.diagonal()[4:].sum() / molecule.rhoA.diagonal()[4:].sum())
    # print(molecule.rhoB.diagonal()[4:].sum() ** 2 / molecule.rhoA.diagonal()[4:].sum())

    plt.show()
