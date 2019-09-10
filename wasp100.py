import phasma
import fit
import astropy.units as u
import numpy as np
from bin import bin
from modelselection import ModelSelection
import matplotlib.pyplot as plt

tic = 38846515
period = 2.8493814 * u.day
transit_duration = 3.80088 * u.hr
transit_epoch = 2458426.47329 - 2457000
sectors = range(1, 14)
cadence = (2 * u.min).to(u.day).value

# UNCOMMENT THE BELOW TO GET THE PHASE CURVE
wasp100 = phasma.TESSPhasecurve(tic, period, transit_duration, transit_epoch,
                                sectors, transit_duration_buff=1.1,
                                remove_curl=False, remove_fits=False,
                                strict_clean=False, plot_clean_lc=False,
                                plot_raw_lc=False, transit_at_0=True,
                                cleaning_window=20 * cadence, save=True,
                                mask_primary=False, mask_secondary=False)

# import the resulting phase curve
t, phase, flux, flux_err = np.genfromtxt(str(tic) + '/phasecurve.csv',
                                         unpack=True,
                                         usecols=(0, 1, 2, 3),
                                         delimiter=',')

# fit a certain model
# nwalkers = 10
# nsteps = 50
# theta = [0.0, 100, 100]
# priors = [[-100, 100], [-1000, 1000], [-1000, 1000]]
# wasp100fit = fit.Fit(phase, flux, flux_err, "therm_beam",
#                  nwalkers=nwalkers, nsteps=nsteps)
# samples = wasp100fit.results(tic, theta, priors,
#                              convergenceplot_name=str(tic) +
#                              '/convergence.png',
#                              cornerplot_name=str(tic) +
#                              '/thermbeam_corner.png')
print(samples.shape)
print(np.median(samples, axis=0))
print("median = ", np.percentile(samples, 50, axis=0),
      "lower 1sig = ", np.percentile(samples, 50-34.1, axis=0),
      "upper 1sig = ", np.percentile(samples, 50+34.1, axis=0))

# write the fit samples to a file
# with open('./' + str(tic) + '/thermbeam-fit_samples.txt', 'w') as f:
#     samples = samples.T
#     f.write('offset, therm-ref, beam' + '\n')
#     for i, j, k in zip(samples[0], samples[1], samples[2]):
#         f.write(str(i) + '\t' + str(j) + '\t' + str(k) + '\n')
# run the beer fit
# fit.beer(tic, period.value, transit_duration.value, transit_epoch)