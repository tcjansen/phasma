phasma: A moving median detrending algorithm for TESS and Kepler phase curves
-----------------------------------------------------------------------------

I'll write a better README later, but here's something to get started:

First clone this repo in the terminal:

`git clone https://github.com/tcjansen/phasma.git`

Simple example for HAT-P-7b as observed by Kepler:

>>> import phasma
>>> import astropy.units as u
>>> import matplotlib.pyplot as plt
>>>
>>> kic = 10666592
>>> period = 2.20473540 * u.day  # units are a must
>>> transit_duration = 4.0398 * u.hr
>>> transit_epoch = 54.358470 + (2454900 - 2454833)
>>>
>>> hatp7b = phasma.Kepler(kic, period, transit_duration, transit_epoch)
>>> phase, flux, flux_err = (hatp7b.phase, hatp7b.flux, hatp7b.flux_err)
>>>
>>> plt.errorbar(phase, flux, flux_err, fmt='o')

If the above doesn't work please let me know!