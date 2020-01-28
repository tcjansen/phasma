from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import sys
import os
import astropy.units as u
import matplotlib.pyplot as plt

sys.path.append(os.getcwd() + '/../')
from phasma import Kepler


class TestKepler(object):
    def __init__(self):
        self.kic = 10666592
        self.period = 2.20473540 * u.day
        self.transit_duration = 4.0398 * u.hr
        self.transit_epoch = 54.358470 + 2454900 - 2454833

        self.hatp11b = Kepler(self.kic, self.period,
                              self.transit_duration,
                              self.transit_epoch)

    def test_get_raw_lightcurve(self):

        raw_time, raw_flux, raw_flux_err = (self.hatp11b.raw_time,
                                            self.hatp11b.raw_flux,
                                            self.hatp11b.raw_flux_err)
        assert len(raw_time) == 51324
        assert len(raw_flux) == len(time)
        assert len(raw_flux_err) == len(time)
