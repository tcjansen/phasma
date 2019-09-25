from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import sys
import os
import astropy.units as u
sys.path.append(os.getcwd() + '/../')

from phasma import Kepler


class TestKepler(object):
    def __init__(self):
        kic = 10666592
        period = 2.20473540 * u.day
        transit_duration = 4.0398 * u.hr
        transit_epoch = 54.358470 + 2454900 - 2454833

        self.hatp11b = Kepler(kic, period, transit_duration,
                              transit_epoch, transit_at_0=True,
                              transit_duration_buff=1.1,
                              mask_primary=False)
        plt.errorbar(self.hatp11b.phase, self.hatp11b.flux,
                     self.hatp11b.flux_err, fmt='o')
        plt.show()

    def test_get_raw_lightcurve(self):
        import matplotlib.pyplot as plt
        time, flux, flux_err = self.hatp11b._get_raw_lightcurve()
        plt.errorbar(time, flux, flux_err, fmt='o')
        plt.show()


TestKepler().test_get_raw_lightcurve()
