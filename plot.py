import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d


def plot_interpol(filename, axes, clr, ex_coeff, legend):
    df = pd.read_csv(filename, sep=',')
    wavelength = df.values[:, 0]
    abs = df.values[:, 1]
    abs *= 1. / abs.max()

    f = interp1d(wavelength, abs, kind='cubic')
    wavelength_new = 1. / np.linspace(1. / wavelength.max(), 1. / wavelength.min(), 100)
    abs_new = f(wavelength_new)
    abs_new *= 1. / abs_new.max()

    axes[0].plot(wavelength_new, ex_coeff*abs_new, clr, label=legend, linewidth=2.0)
    axes[1].semilogy(wavelength_new, ex_coeff*abs_new, clr, label=legend, linewidth=2.0)


def render_ticks(axes):
    axes.get_xaxis().set_tick_params(which='both', direction='in', width=1, labelrotation=0, labelsize='x-large')
    axes.get_yaxis().set_tick_params(which='both', direction='in', width=1, labelcolor='r', labelsize='x-large')
    axes.get_xaxis().set_ticks_position('both')
    axes.get_yaxis().set_ticks_position('both')
    axes.grid()


fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True)
plot_interpol('GCAMP.csv', axes, 'k', 50300, 'GCaMP')
plot_interpol('ChR2.csv', axes, 'b', 45500, 'ChR2')
plot_interpol('cerulean.csv', axes, 'c', 43000, 'VSF3.1')
axes[0].legend(loc=1)
axes[1].legend(loc=1)
axes[0].set_xlim(350., 600.)
axes[1].set_xlim(350., 600.)
axes[1].set_ylim(3e1, 1e5)
axes[0].set_xlabel("Wavelength (in nm)")
axes[1].set_xlabel("Wavelength (in nm)")
render_ticks(axes[0])
render_ticks(axes[1])
fig.set_size_inches((8.5, 6.5), forward=False)
fig.savefig("relative_spectra.eps", format='eps')