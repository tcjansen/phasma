import numpy as np
import matplotlib.pyplot as plt
from bin import bin

t, phase, flux, flux_err = np.genfromtxt('phasecurve.csv', unpack=True,
                                         usecols=(0, 1, 2, 3),
                                         delimiter=',')

bin_phase, bin_flux, bin_flux_err = bin(phase, .001, flux, flux_err)
bin_phase2, bin_flux2, bin_flux_err2 = bin(phase, .02, flux, flux_err)

# plt.scatter(t, flux)
# plt.show()

# plt.errorbar(bin_phase, bin_flux, bin_flux_err, alpha=0.5)
plt.errorbar(bin_phase2, bin_flux2, bin_flux_err2, fmt='o', color='#FF7700')
plt.xlabel('time since secondary eclipse')
plt.show()
