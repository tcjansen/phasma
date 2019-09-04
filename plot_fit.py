import beer
import matplotlib.pyplot as plt 
from operator import itemgetter
import sys, os
import numpy as np

def plot(ID):

	a0, a1, a2, a3, a4 = np.genfromtxt('./targets/' + ID + '/coefficients.txt', usecols=(0,1,2,3,4), dtype='float', unpack=True)
	period, t14, t0 = np.genfromtxt('./targets/' + ID + '/target_info.csv', delimiter=',', usecols=(1,2,3), dtype='float', unpack=True)
	t14 /= 24

	file = os.getcwd() + '/targets/' + str(ID) + '/' + str(ID) + '_phasma_final.csv'
	# binfile = os.getcwd() + '/targets/267263253/267263253_binned.csv'
	binfile = os.getcwd() + '/targets/' + str(ID) + '/' + str(ID) + '_binned.csv'
	time, phase, flux, flux_err = np.genfromtxt(file, usecols=(0,2,3,4), delimiter=',', unpack=True)
	# phase_bin, flux_bin, flux_err_bin = np.genfromtxt(binfile, usecols=(0,1,2), delimiter=',', unpack=True)
	phase_bin, flux_bin, flux_err_bin = np.genfromtxt(binfile, usecols=(0,1,2), delimiter=',', unpack=True)

	all_data = np.vstack([phase, flux, flux_err]).T
	all_data = sorted(all_data, key=itemgetter(0))
	phase = np.array(all_data).T[0]
	flux = np.array(all_data).T[1]
	flux_err = np.array(all_data).T[2]

	intransit = (phase > -1.05*t14/period/2) & (phase < 1.05*t14/period/2)
	phase[intransit] = np.nan
	flux[intransit]= np.nan
	flux_err[intransit]= np.nan

	intransit = (phase_bin > -t14/period/2) & (phase_bin < t14/period/2)
	phase_bin[intransit] = np.nan
	flux_bin[intransit] = np.nan
	flux_err_bin[intransit] = np.nan

	fit = beer.beer(phase, a0, a1, a2, a3, a4)
	plt.errorbar(phase, flux, flux_err, fmt='o', alpha=0.5)
	plt.plot(phase, fit)

	plt.figure()
	plt.plot(phase, (fit - a0)*1e6, color='black', label='total', lw=2)
	plt.plot(phase, (a1 * np.sin(2 * np.pi * phase))*1e6, color='orange', label='beaming', lw=2)
	plt.plot(phase, (a2 * np.cos(2 * np.pi * phase))*1e6, color='red', label='reflection + thermal', lw=2)
	plt.plot(phase, (a3 * np.sin(2 * np.pi * 2 * phase))*1e6, color='green', label='nonreal', lw=2)
	plt.plot(phase, (a4 * np.cos(2 * np.pi * 2 * phase))*1e6, color='blue', label='ellipsoidal', lw=2)
	plt.legend()
	plt.xlim(-0.5, 0.5)
	plt.xlabel('orbital periods since transit')
	plt.ylabel('normalized flux [ppm]')
	plt.savefig('./targets/' + ID + '/' + ID + '_fit_components.png')

targetlist = sys.argv[1]
with open(targetlist) as o:
	for ID in o.readlines()[1:]:
		plot(ID.strip())

