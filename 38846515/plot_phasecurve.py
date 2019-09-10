import numpy as np
import matplotlib.pyplot as plt
from bin import bin
import models

t, phase, flux, flux_err = np.genfromtxt('phasecurve.csv', unpack=True,
                                         usecols=(0, 1, 2, 3),
                                         delimiter=',')

bin_phase, bin_flux, bin_flux_err = bin(phase, .001, flux, flux_err)
bin_phase2, bin_flux2, bin_flux_err2 = bin(phase, .01, flux, flux_err)

# plt.scatter(t, flux)
# plt.show()

plt.errorbar(bin_phase, bin_flux, bin_flux_err, alpha=0.5, color='#C19F9B', zorder=-32)
plt.errorbar(bin_phase2, bin_flux2, bin_flux_err2, fmt='o', color='#491105')
plt.xlabel('time since transit')
plt.ylabel('$(F_{p} + F_{*})F_{*}^{-1}$ [ppm]')
plt.xlim(-0.1, 0.1)

# plt.plot(bin_phase, models.beer(bin_phase, 6.4174055439677486,
#     20.973413394830313,-23.190845159868747, -12.340127524116896, 11.361623988302828), color='#491105')
# plt.plot(bin_phase, models.beer(bin_phase, 6.99016916,
#     29.38123035,-30.5449382, 0,0), color='#491105')
plt.show()
