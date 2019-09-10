import csv
import sys, os
import numpy as np
import matplotlib.pyplot as plt

ID = sys.argv[1]
period = float(sys.argv[2])
t_14 = float(sys.argv[3])
t_14 /= 24
t0 = float(sys.argv[4])

time, phase, flux, flux_err = np.genfromtxt('../' + ID + '/phasecurve.csv',
	usecols=(0,1,2,3), delimiter=',', unpack=True)
intransit = (phase > -1.1*t_14/period/2) & (phase < 1.1*t_14/period/2)
time[intransit] = np.nan
flux[intransit]= np.nan
flux_err[intransit]= np.nan

inoccultation = (phase > 0.5 - 1.1*t_14/period/2)
time[inoccultation] = np.nan
flux[inoccultation]= np.nan
flux_err[inoccultation]= np.nan

inoccultation = (phase < -0.5 + 1.1*t_14/period/2)
time[inoccultation] = np.nan
flux[inoccultation]= np.nan
flux_err[inoccultation]= np.nan

notnan = (~np.isnan(time) & ~np.isnan(flux) & ~np.isnan(flux_err) \
          & (time != 0) & (flux != 0) & (flux_err != 0))

time = time[notnan]
flux = flux[notnan]
flux_err = flux_err[notnan]

time = time - t0

with open("data.dat", 'w') as beer:
	for i,j in zip(time[:1], flux[:1]):
		beer.write(str(i) + '\t' + str(j))
	for i,j in zip(time[1:], flux[1:]):
		beer.write('\n' + str(i) + '\t' + str(j))