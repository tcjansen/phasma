import phasma  # make sure this file is in the same directory as phasma.py
import astropy.units as u
import matplotlib.pyplot as plt
import bin

tic = 100100827
period = 0.9414563 * u.day
transit_duration = 2.19264 * u.hr
transit_epoch = 2458381.76006 - 2457000
sectors = [2, 3]

object_of_interest = phasma.Tess(tic, period, transit_duration, transit_epoch,
                                 sectors)
phase, flux, flux_err = (object_of_interest.phase,
                         object_of_interest.flux,
                         object_of_interest.flux_err)

plt.figure()
plt.errorbar(phase, flux, flux_err, fmt='o', alpha=0.5, color='black')
plt.savefig(str(tic) + "/" + str(tic) + "_phasecurve.png")

# optional: you can take a look at the preprocessed light curve by calling:
time, flux, flux_err = (object_of_interest._raw_time,
                        object_of_interest._raw_flux,
                        object_of_interest._raw_flux_err)
plt.figure()
plt.errorbar(time, flux, flux_err, fmt='o', alpha=0.5, color='black')
plt.savefig(str(tic) + "/" + str(tic) + "_lightcurve.png")