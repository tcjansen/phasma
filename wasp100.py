import phasma
from fit import Fit
import astropy.units as u
import numpy as np
from bin import bin
from modelselection import ModelSelection
import models
import matplotlib.pyplot as plt

tic = 38846515
period = 2.8493814 * u.day
transit_duration = 3.80088 * u.hr
transit_epoch = 2458426.47329 - 2457000
sectors = range(1, 14)
cadence = (2 * u.min).to(u.day).value
cleaning_window = (0.3 * transit_duration).to(u.day).value

# UNCOMMENT THE BELOW TO GET THE PHASE CURVE
# wasp100 = phasma.TESSPhasecurve(tic, period, transit_duration, transit_epoch,
#                                 sectors, transit_duration_buff=1.1,
#                                 remove_curl=False, remove_fits=False,
#                                 strict_clean=True, plot_clean_lc=True,
#                                 plot_raw_lc=True, transit_at_0=True,
#                                 cleaning_window=cleaning_window, save=True,
#                                 mask_secondary=True)

# import the resulting phase curve
t, phase, flux, flux_err = np.genfromtxt(str(tic) + '/phasecurve.csv',
                                         unpack=True,
                                         usecols=(0, 1, 2, 3),
                                         delimiter=',')
# bin_phase2, bin_flux2, bin_flux_err2 = bin(phase, .01, flux, flux_err)
bin_phase, bin_flux, bin_flux_err = bin(phase, .001, flux, flux_err)

# plt.errorbar(bin_phase, bin_flux, bin_flux_err, alpha=0.5)

plt.errorbar(bin_phase, bin_flux, bin_flux_err, alpha=0.5)
# plt.errorbar(bin_phase2, bin_flux2, bin_flux_err2, fmt='o', color='#FF7700')

# plot the fit
x = np.arange(-0.5, 0.51, 0.01)
therm = models.therm(x, 6.75728961, 30.93984724)
beam = models.beam(x, 6.75728961, 31.26970701)
plt.plot(x, therm)
plt.plot(x, beam)
plt.plot(x, therm + beam)
plt.show()
plt.xlabel('time since secondary eclipse')

plt.show()

# # fit a certain model
# nwalkers = 50
# nsteps = 2000
# theta = [0.0, 100, 100]
# priors = [[-100, 100], [-1000, 1000], [-1000, 1000]]
# wasp100fit = Fit(phase, flux, flux_err, "therm_beam",
#                  nwalkers=nwalkers, nsteps=nsteps)
# samples = wasp100fit.results(tic, theta, priors,
#                              convergenceplot_name=str(tic) +
#                              '/convergence.png',
#                              cornerplot_name=str(tic) +
#                              '/thermbeam_corner.png')
# print(np.median(samples, axis=0))
# print("median = ", np.percentile(samples, 50, axis=0))
# print("lower 1sig = ", np.percentile(samples, 50-34.1, axis=0))
# print("upper 1sig = ", np.percentile(samples, 50+34.1, axis=0))

# ModelSelection(phase, flux, flux_err)