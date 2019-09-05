# Made to work with phasma.TESSPhasecurve object

import numpy as np
import os
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.utils.data import download_file
import astropy.units as u
from astropy.units import cds

import urllib.request
import subprocess
from operator import itemgetter
from scipy.interpolate import interp1d

# LOCAL
import mcmc
import models


def _beer_fit(tic, period, duration, ephemeris):
    with open('BEER/run.sh', 'w') as run:

        run.write('#!/bin/bash' + '\n\n')

        run.write('tic=' + str(tic) + '\n\n')
        run.write('period=' + str(period) + '\n\n')
        run.write('duration=' + str(duration) + '\n\n')
        run.write('ephemeris=' + str(ephemeris) + '\n\n')

        run.write('python make_data-dat.py $tic $period $duration $ephemeris' + '\n')
        run.write('python edit_beersh.py $tic $period' + '\n')
        run.write('./make.sh' + '\n')
        run.write('values=`./beerfit.sh`' + '\n')
        run.write('coefffile=`echo "../"$tic"/coefficients.txt"`' + '\n')
        run.write('echo $values' + '\n')
        run.write('echo $values > $coefffile' + '\n')

        run.close()

    print('Fitting BEER to', str(tic) + "...")
    os.chdir('./BEER')
    subprocess.call(['./run.sh'])
    os.chdir('..')


class Fit(object):

    def __init__(self, phase, flux, flux_err, model, nwalkers=100,
                 nsteps=1000, nbins=21):
        self.phase = phase
        self.flux = flux
        self.flux_err = flux_err
        self.model = model
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.nbins = nbins

    def _get_params(self, model=None):
        if model=="flat":
            return ['offset - 1 [ppm]'], 1
        elif model=="therm":
            return ['offset - 1 [ppm]', r'log10$A_{therm+ref}$'], 2
        elif model=="ellip":
            return ['offset - 1 [ppm]', r'log10$A_{ellip}$'], 2
        elif model=="beam":
            return ['offset - 1 [ppm]', r'log10$A_{beam}$'], 2
        elif model=="shifted_therm":
            return ['offset - 1 [ppm]', r'log10$A_{therm+ref}$', r'$\phi$'], 3
        elif model=="therm_ellip":
            return ['offset - 1 [ppm]', r'log10$A_{therm+ref}$', r'log10$A_{ellip}$'], 3
        elif model=="ellip_beam":
            return ['offset - 1 [ppm]', r'log10$A_{ellip}$', r'log10$A_{beam}$'], 3
        elif model=="therm_beam":
            return ['offset - 1 [ppm]', r'log10$A_{therm+ref}$', r'log10$A_{beam}$'], 3
        else:
            raise ValueError("The correct model type must be specified when calling for the model parameters")

    def _get_pos(self, theta, model=None):
        if model=="flat":
            return [np.array([theta[0]+1e-2*np.random.randn()]) for i in range(self.nwalkers)]
        elif model=="therm":
            return [np.array([theta[0]+1e-2*np.random.randn(), theta[1] - 0.1 * np.random.randn()])
                   for i in range(self.nwalkers)]
        elif model=="ellip":
            return [np.array([theta[0]+1e-2*np.random.randn(), theta[1] - 0.1 * np.random.randn()])
                   for i in range(self.nwalkers)]
        elif model=="beam":
            return [np.array([theta[0]+1e-2*np.random.randn(), theta[1] - 0.1 * np.random.randn()])
                   for i in range(self.nwalkers)]
        elif model=="shifted_therm":
            return [np.array([theta[0]+1e-2*np.random.randn(), theta[1] - 0.1 * np.random.randn(),
                    theta[2]+1e-2*np.random.randn()])
                    for i in range(self.nwalkers)]
        elif model=="therm_ellip":
            return [np.array([theta[0]+1e-2*np.random.randn(), theta[1] - 0.1 * np.random.randn(),
                    theta[2] - 0.1 * np.random.randn()])
                    for i in range(self.nwalkers)]
        elif model=="ellip_beam":
            return [np.array([theta[0]+1e-2*np.random.randn(), theta[1] - 0.1 * np.random.randn(),
                    theta[2] - 0.1 * np.random.randn()])
                    for i in range(self.nwalkers)]
        elif model=="therm_beam":
            return [np.array([theta[0]+1e-2*np.random.randn(), theta[1] - 0.1 * np.random.randn(),
                    theta[2] - 0.1 * np.random.randn()])
                    for i in range(self.nwalkers)]
        else:
            raise ValueError("The correct model type must be specified when calling for the walker starting positions")

    def results(self, tic, theta, priors,
                convergenceplot_name=None, cornerplot_name=None):

        params, ndim = self._get_params(model=self.model)
        pos = self._get_pos(theta, model=self.model)
        samples = mcmc.get_samples(tic, params, pos, ndim, self.nwalkers,
            self.nsteps, theta, getattr(models, self.model), priors,
            self.phase, self.flux, self.flux_err,
            convergenceplot_name=convergenceplot_name,
            cornerplot_name=cornerplot_name)

        return samples

    def plot(self, coefficients, fits=None, plot_name=None, one_plot=False):

        plotting_phase = np.linspace(-0.5, 0.5, len(self.phase))
        plt.figure(figsize=(8,4))
        plt.errorbar(self.bin_phase, (self.bin_flux - 1)*1e6, \
            yerr=self.bin_flux_err*1e6, fmt='o', color='#E71D36')

        if one_plot:

            for i in range(len(fits)):
                if fits[i]=="flat":
                    plt.plot(plotting_phase, (models.flat(plotting_phase, \
                        coefficients[i][0]) - 1)*1e6, lw=2, color='#2EC4B6', ls=None)
                elif fits[i]=="beer":
                    plt.plot(plotting_phase, (models.beer(plotting_phase, \
                        coefficients[i][0], coefficients[i][1], coefficients[i][2], \
                        coefficients[i][3], coefficients[i][4]) - 1)*1e6, \
                        color='#FF9F1C', lw=2, ls='--', label="beer")
                elif fits[i]=="therm":
                    plt.plot(plotting_phase, (models.therm(plotting_phase, \
                        coefficients[i][0], coefficients[i][1]) - 1)*1e6, \
                        color='#FF9F1C', lw=2, ls=None, label="therm + ref")
                elif fits[i]=="ellip":
                    plt.plot(plotting_phase, (models.ellip(plotting_phase, \
                        coefficients[i][0], coefficients[i][1]) - 1)*1e6, \
                        color='#FF9F1C', lw=2, ls=None, label="ellip")
                elif fits[i]=="beam":
                    plt.plot(plotting_phase, (models.beam(plotting_phase, \
                        coefficients[i][0], coefficients[i][1]) - 1)*1e6, \
                        color='#FF9F1C', lw=2, ls=None, label="beam")
                elif fits[i]=="shifted_therm":
                    plt.plot(plotting_phase, (models.shifted_therm(plotting_phase, \
                        coefficients[i][0], coefficients[i][1], coefficients[i][2]) - 1)*1e6, \
                        color='#FF9F1C', lw=2, ls='-.', label="shifted therm + ref")
                elif fits[i]=="therm_ellip":
                    plt.plot(plotting_phase, (models.therm_ellip(plotting_phase, \
                        coefficients[i][0], coefficients[i][1], coefficients[i][2]) - 1)*1e6, \
                        color='#FF9F1C', lw=2, ls='-.', label="therm + ellip")
                elif fits[i]=="ellip_beam":
                    plt.plot(plotting_phase, (models.ellip_beam(plotting_phase, \
                        coefficients[i][0], coefficients[i][1], coefficients[i][2]) - 1)*1e6, \
                        color='#FF9F1C', lw=2, ls='-.', label="ellip + beam")
                elif fits[i]=="therm_beam":
                    plt.plot(plotting_phase, (models.therm_beam(plotting_phase, \
                        coefficients[i][0], coefficients[i][1], coefficients[i][2]) - 1)*1e6, \
                        color='#FF9F1C', lw=2, ls='-.', label="therm + beam")
                plt.ylabel('ppm')
                plt.legend()

        else:

            if len(fits)==1:

                if fits[i]=="flat":
                    plt.plot(plotting_phase, (models.flat(plotting_phase, \
                             coefficients[i][0]) - 1)*1e6, lw=2, color='#2EC4B6', ls=None)
                elif fits[i]=="beer":
                    plt.plot(plotting_phase, (models.beer(plotting_phase, \
                        coefficients[i][0], coefficients[i][1], coefficients[i][2], \
                        coefficients[i][3], coefficients[i][4]) - 1)*1e6, \
                        color='#FF9F1C', lw=2, ls='--', label="beer")
                elif fits[i]=="therm":
                    plt.plot(plotting_phase, (models.therm(plotting_phase, \
                        coefficients[i][0], coefficients[i][1]) - 1)*1e6, \
                        color='#FF9F1C', lw=2, ls=None, label="therm + ref")
                elif fits[i]=="ellip":
                    plt.plot(plotting_phase, (models.ellip(plotting_phase, \
                        coefficients[i][0], coefficients[i][1]) - 1)*1e6, \
                        color='#FF9F1C', lw=2, ls=None, label="ellip")
                elif fits[i]=="beam":
                    plt.plot(plotting_phase, (models.beam(plotting_phase, \
                        coefficients[i][0], coefficients[i][1]) - 1)*1e6, \
                        color='#FF9F1C', lw=2, ls=None, label="beam")
                elif fits[i]=="shifted_therm":
                    plt.plot(plotting_phase, (models.shifted_therm(plotting_phase, \
                        coefficients[i][0], coefficients[i][1], coefficients[i][2]) - 1)*1e6, \
                        color='#FF9F1C', lw=2, ls='-.', label="shifted therm + ref")
                elif fits[i]=="therm+ellip":
                    plt.plot(plotting_phase, (models.therm_ellip(plotting_phase, \
                        coefficients[i][0], coefficients[i][1], coefficients[i][2]) - 1)*1e6, \
                        color='#FF9F1C', lw=2, ls='-.', label="therm + ellip")
                elif fits[i]=="ellip_beam":
                    plt.plot(plotting_phase, (models.ellip_beam(plotting_phase, \
                        coefficients[i][0], coefficients[i][1], coefficients[i][2]) - 1)*1e6, \
                        color='#FF9F1C', lw=2, ls='-.', label="ellip + beam")
                elif fits[i]=="therm_beam":
                    plt.plot(plotting_phase, (models.therm_beam(plotting_phase, \
                        coefficients[i][0], coefficients[i][1], coefficients[i][2]) - 1)*1e6, \
                        color='#FF9F1C', lw=2, ls='-.', label="therm + beam")
                plt.xlabel('orbital periods since transit')

            else:

                fig, axs = plt.subplots(len(fits), 1, sharex=True)
                for i in range(len(fits)):
                    if fits[i]=="flat":
                        axs[i].plot(plotting_phase, (models.flat(plotting_phase, \
                            coefficients[i][0]) - 1)*1e6, lw=2, color='#FF9F1C')
                        axs[i].set_title("flat model")
                        axs[i].set_ylabel("[ppm]")

                    elif fits[i]=="therm":
                        axs[i].plot(plotting_phase, (models.therm(plotting_phase, \
                            coefficients[i][0], coefficients[i][1]) - 1)*1e6, \
                            color='#FF9F1C', lw=2, label="therm + ref")
                        axs[i].set_title("therm + ref model")
                        axs[i].set_ylabel("[ppm]")

                    elif fits[i]=="ellip":
                        axs[i].plot(plotting_phase, (models.ellip(plotting_phase, \
                            coefficients[i][0], coefficients[i][1]) - 1)*1e6, \
                            color='#FF9F1C', lw=2, label="ellip")
                        axs[i].set_title("ellip model")
                        axs[i].set_ylabel("[ppm]")

                    elif fits[i]=="beam":
                        axs[i].plot(plotting_phase, (models.beam(plotting_phase, \
                            coefficients[i][0], coefficients[i][1]) - 1)*1e6, \
                            color='#FF9F1C', lw=2, label="beam")
                        axs[i].set_title("beam model")
                        axs[i].set_ylabel("[ppm]")

                    elif fits[i]=="shifted_therm":
                        axs[i].plot(plotting_phase, (models.shifted_therm(plotting_phase, \
                            coefficients[i][0], coefficients[i][1], coefficients[i][2]) - 1)*1e6, \
                            color='#FF9F1C', lw=2, label="shifted therm + ref")
                        axs[i].set_title("shifted therm + ref model")
                        axs[i].set_ylabel("[ppm]")

                    elif fits[i]=="beer":
                        axs[i].plot(plotting_phase, (models.beer(plotting_phase, \
                            coefficients[i][0], coefficients[i][1], coefficients[i][2], \
                            coefficients[i][3], coefficients[i][4]) - 1)*1e6, \
                            color='#FF9F1C', lw=2, label="beer")
                        axs[i].set_title("beer model")
                        axs[i].set_ylabel("[ppm]")

                    elif fits[i]=="therm_ellip":
                        axs[i].plot(plotting_phase, (models.therm_ellip(plotting_phase, \
                            coefficients[i][0], coefficients[i][1], coefficients[i][2]) - 1)*1e6, \
                            color='#FF9F1C', lw=2, label="therm + ellip")
                        axs[i].set_title("therm + ellip model")
                        axs[i].set_ylabel("[ppm]")

                    elif fits[i]=="ellip_beam":
                        axs[i].plot(plotting_phase, (models.ellip_beam(plotting_phase, \
                            coefficients[i][0], coefficients[i][1], coefficients[i][2]) - 1)*1e6, \
                            color='#FF9F1C', lw=2, label="ellip + beam")
                        axs[i].set_title("ellip + beam model")
                        axs[i].set_ylabel("[ppm]")

                    elif fits[i]=="therm_beam":
                        axs[i].plot(plotting_phase, (models.therm_beam(plotting_phase, \
                            coefficients[i][0], coefficients[i][1], coefficients[i][2]) - 1)*1e6, \
                            color='#FF9F1C', lw=2, label="therm + beam")
                        axs[i].set_title("therm + beam model")
                        axs[i].set_ylabel("[ppm]")

                    axs[i].errorbar(self.bin_phase, (self.bin_flux - 1)*1e6, \
                             yerr=self.bin_flux_err*1e6, fmt='o', color='#E71D36')

                axs[len(fits) - 1].set_xlabel('orbital periods since transit')
                fig.set_size_inches(6, 8)

        plt.xlim(-0.5, 0.5)
        plt.savefig(os.getcwd() + '/' + plot_name)

        return
