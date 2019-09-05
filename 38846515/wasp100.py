import phasma
import astropy.units as u
import numpy as np
from bin import bin

tic = 38846515
period = 2.8493814 * u.day
transit_duration = 3.80088 * u.hr
transit_epoch = 2458426.47329 - 2457000
sectors = range(1, 14)

wasp100 = phasma.TESSPhasecurve(tic, period, transit_duration, transit_epoch,
                                sectors, transit_duration_buff=1.1,
                                remove_curl=False, remove_fits=False,
                                strict_clean=False, plot_clean_lc=True,
                                plot_raw_lc=True, transit_at_0=False,
                                cleaning_window=False, save=True)
