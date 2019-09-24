from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import sys
import os
import astropy.units as u
sys.path.append(os.getcwd() + '/../')

from phasma import Kepler


class TestKepler(object):
    def __init__(self):
        kic = 10748390
        period = 4.887802443 * u.day
        transit_duration = 2.36386 * u.hr
        transit_epoch = 2454957.8131411

        self.hatp11b = Kepler(kic, period, transit_duration,
                              transit_epoch)

    def test_get_raw_lightcurve(self):
        import matplotlib.pyplot as plt
        time, flux, flux_err = self.hatp11b._get_raw_lightcurve()
        plt.errorbar(time, flux, flux_err, fmt='o')
        plt.show()


TestKepler().test_get_raw_lightcurve()
