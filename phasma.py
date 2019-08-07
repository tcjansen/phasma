import numpy as np
import os
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.utils.data import download_file
import astropy.units as u
from astropy.units import cds
cds.enable()

import urllib.request
import subprocess
from operator import itemgetter
from scipy.interpolate import interp1d

class TESSPhasecurve():
    """
    Returns the phase curve of an object of interest observed by TESS.

    Parameters
    ----------
    sector : list or tuple
        Sector(s) of interest.
    identifier : float or int, optional
        e.g. toi101 (for TOI/KOI), 231663901 (for TIC/KIC id)
    mission : {'tess', 'kepler'}, optional
    keep_lc_fits : bool, optional
        Set to True if you want to save a copy of the raw light curve
        fits file.
    """
    @u.quantity_input(period=u.day, transit_duration=u.hr)
    def __init__(self, tic, period, transit_duration, transit_epoch, sectors,
                 identifier='', mission='', remove_curl=False, remove_fits=False,
                 plot_clean_lc=False, plot_raw_lc=False):
        self.tic = str(tic)
        self.period = period
        self.transit_duration = transit_duration
        self.transit_epoch = transit_epoch
        self.sectors = sectors
        self.identifier = identifier
        self.mission = mission
        self.remove_fits = remove_fits
        self.remove_curl = remove_curl
        self.plot_clean_lc = plot_clean_lc
        self.plot_raw_lc = plot_raw_lc

        (self._raw_time,
         self._raw_flux,
         self._raw_flux_err) = self._get_raw_lightcurve()

        (self.phase,
         self.time,
         self.flux,
         self.flux_err) = self._wrap()

    def write(self):
        """ Writes the phase curve to a fits file. """
        return

    def plot(self, show=True, save=False, file_format='png', bin=False,
             binsize=0.01, alpha=0.5):
        """ Plots the phase curve. """
        phase = self.phase
        flux = self.flux
        flux_err = self.flux_err

        if bin:
            phase, flux, flux_err = _bin(phase, binsize, flux, flux_err)

        plt.figure(figsize=(16, 5))
        plt.errorbar(phase, flux, yerr=flux_err,
                     fmt='o', alpha=alpha, color='black')
        plt.ylabel('ppm')
        plt.xlabel('phase')
        plt.xlim(-0.5, 0.5)

        if show:
            plt.show()

        if save:
            plt.savefig('phasma_phasecurve_' + self.tic + '.' + file_format)

        return

    def _plot_raw_lc(self, show=True, save=False, file_format='png'):
        plt.figure(figsize=(16, 5))
        plt.scatter(self._raw_time, self._raw_flux,
                    color='black')
        if show:
            plt.show()

        if save:
            plt.savefig('rawlc_' + tic + '.' + file_format)

    def _get_raw_lightcurve(self):

        self.actual_sectors = []

        time = np.array([])
        flux = np.array([])
        flux_err = np.array([])
        for sector in self.sectors:

            try:
                # download the curl file for each sector if not in directory
                curl_sh_path = './tesscurl_sector_' + str(sector) + '_lc.sh'
                if not os.path.isfile(curl_sh_path):
                    print("Downloading the light curve curl file for sector " +
                          str(sector) + "...")
                    urllib.request.urlretrieve('https://archive.stsci.edu/missions/' +
                                'tess/download_scripts/sector/tesscurl_' +
                                'sector_' + str(sector) + '_lc.sh', curl_sh_path)

                with open(curl_sh_path) as curl_sh:
                    array_of_curls = np.array(curl_sh.read().splitlines())

                    # search for this toi's curl
                    toi_curls = [curl for curl in array_of_curls
                                if self.tic in curl]

                    # download the fits files if not in directory
                    mast_url = 'https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:TESS/product/'
                    for curl in toi_curls:
                        fits_file = curl[16:71]
                        if not os.path.isfile(fits_file):
                            print('Downloading the fits files for TIC ' + self.tic + "...")
                            urllib.request.urlretrieve(mast_url + fits_file, './' + fits_file)

                # delete the curl files to save space
                if self.remove_curl:
                    os.remove(curl_sh_path)

                # unpack the fits file
                open_fits = fits.open(fits_file)
                fits_data = open_fits[1].data
                raw_time = fits_data.field('TIME')
                raw_flux = fits_data.field('PDCSAP_FLUX')
                raw_flux_err = fits_data.field('PDCSAP_FLUX_ERR')
                data_quality = fits_data.field('QUALITY')

                # remove flagged data
                good_data = (data_quality == 0) & (~np.isnan(raw_flux))
                time = np.append(time, raw_time[good_data])
                # normalize the flux
                flux = np.append(flux, raw_flux[good_data] /
                                       np.median(raw_flux[good_data]))
                flux_err = np.append(flux_err, raw_flux_err[good_data] /
                                               np.median(raw_flux[good_data]))

                # delete the fits file to save space
                if self.remove_fits:
                    os.remove(fits_file)

                self.actual_sectors += [sector]

            except:
                print('TIC ' + self.tic + ' not in sector ' + str(sector))

        return time, flux, flux_err

    def _locate_gaps(self, t):
        # get the indices where the gaps start and end
        true_gap_starts = []
        true_gap_ends = []
        for i in range(len(t)-1):
            if t[i+1] - t[i] > 0.1 * self.period.value:
                true_gap_starts += [i]
                true_gap_ends += [i+1]

        return true_gap_starts, true_gap_ends

    def _split_lc_at_gap(self, true_gap_starts, true_gap_ends):
        # rename for brevity
        t = self._raw_time
        flux = self._raw_flux
        flux_err = self._raw_flux_err
        period = self.period.value
        cadence = t[1] - t[0]

        if len(true_gap_starts) == 0:
            print("No data gaps to split at, continuing...")

        else:

            if len(true_gap_starts) == 1:
                if not len(t[:true_gap_starts[0] + 1]) < 2 * period / cadence:
                    split_time = t[:true_gap_starts[0] + 1]
                    split_flux = flux[:true_gap_starts[0] + 1]
                    split_flux_err = flux_err[:true_gap_starts[0] + 1]

                else:
                    print("Baseline is shorter than twice the length of the period.")

                if not len(t[true_gap_ends[0]:]) < 2 * period / cadence:
                    split_time = t[true_gap_ends[0]:]
                    split_flux = flux[true_gap_ends[0]:]
                    split_flux_err = flux_err[true_gap_ends[0]:]

                else:
                    print("Baseline is shorter than twice the length of the period.")

            elif true_gap_starts[0] != 0:

                split_time = []
                split_flux = []
                split_flux_err = []

                if not len(t[:true_gap_starts[0] + 1]) < 2 * period / cadence:
                    split_time += [list(t[:true_gap_starts[0] + 1])]
                    split_flux += [list(flux[:true_gap_starts[0] + 1])]
                    split_flux_err += list([flux_err[:true_gap_starts[0] + 1]])
                else:
                    print("Baseline is shorter than twice the length of the period.")

                for i in range(len(true_gap_starts)-1):
                    if not len(t[true_gap_ends[i]:true_gap_starts[i+1]]) < 2 * period / cadence:
                        split_time += [list(t[true_gap_ends[i]:true_gap_starts[i+1]])]
                        split_flux += [list(flux[true_gap_ends[i]:true_gap_starts[i+1]])]
                        split_flux_err += [list(flux_err[true_gap_ends[i]:true_gap_starts[i+1]])]
                    else:
                        print("Baseline is shorter than twice the length of the period.")

                if not len(t[true_gap_ends[-1]:]) < 2 * period / cadence:
                    split_time += [list(t[true_gap_ends[-1]:])]
                    split_flux += [list(flux[true_gap_ends[-1]:])]
                    split_flux_err += [list(flux_err[true_gap_ends[-1]:])]
                else:
                    print("Baseline is shorter than twice the length of the period.")

            else:
                split_time = []
                split_flux = []
                split_flux_err = []
                for i in range(len(true_gap_starts) - 1):
                    if not len(t[true_gap_ends[i]:true_gap_starts[i+1]]) < 2 * period / cadence:
                        split_time += [list(t[true_gap_ends[i]:true_gap_starts[i+1]])]
                        split_flux += [list(flux[true_gap_ends[i]:true_gap_starts[i+1]])]
                        split_flux += [list(flux_err[true_gap_ends[i]:true_gap_starts[i+1]])]
                    else:
                        print("Baseline is shorter than twice the length of the period.")

        return split_time, split_flux, split_flux_err

    def _phase(self, time):
        """ Returns the phases corresponding to a given time array. """
        return (time - self.transit_epoch) / self.period.value % 1 - 0.5

    def _fold(self, phase, time, flux, flux_err):
        """
        Folds flux on the period given and returns the
        resulting phase curve sorted by the phase
        """
        transverse_data = np.vstack([phase, time, flux, flux_err]).T

        # sort the data by phase
        (sorted_phase,
         sorted_t,
         sorted_flux,
         sorted_flux_err) = np.array(sorted(transverse_data, key=itemgetter(0))).T

        return sorted_phase, sorted_t, sorted_flux, sorted_flux_err

    def _wrap(self):

        time_all = np.array([])
        flux_all = np.array([])
        flux_err_all = np.array([])

        if self.plot_raw_lc:
            plt.figure(figsize=(16, 5))
            plt.scatter(self._raw_time, self._raw_flux, color='black')

        for sector in self.actual_sectors:

            # split the data by saving to temporary fits files
            true_gap_starts, true_gap_ends = self._locate_gaps(self._raw_time)
            (split_time,
             split_flux,
             split_flux_err) = self._split_lc_at_gap(true_gap_starts,
                                                     true_gap_ends)

            for semisector in range(len(split_time)):

                # clean the semi sectors
                (clean_t,
                 clean_flux,
                 clean_flux_err) = _clean(np.array(split_time[semisector]),
                                          np.array(split_flux[semisector]),
                                          np.array(split_flux_err[semisector]))

                if self.plot_clean_lc:
                    plt.scatter(clean_t, clean_flux, color='red')

                # define the phase corresponding to the cleaned fluxes
                phase = self._phase(clean_t)

                # manipulate arrays such that transit occurs at phase = 0
                midpoint = int(len(clean_flux) / 2)
                first_half = clean_flux[:midpoint]
                last_half = clean_flux[midpoint:]
                clean_flux = np.append(last_half, first_half)
                first_half = clean_flux_err[:midpoint]
                last_half = clean_flux_err[midpoint:]
                clean_flux_err = np.append(last_half, first_half)

                # remove transit from light curve
                transit_phase = float(self.transit_duration / self.period) / 2
                in_transit = (phase >= -transit_phase) & (phase <= transit_phase)

                # clean_t[in_transit] = np.nan
                clean_flux[in_transit] = np.nan
                clean_flux_err[in_transit] = np.nan

                # apply phasma to cleaned data
                (phasma_t,
                 phasma_flux,
                 phasma_flux_err) = _phasma_detrend(self.period.value,
                                                    clean_t,
                                                    clean_flux,
                                                    clean_flux_err)

                # combine semisectors and sectors
                time_all = np.append(time_all, phasma_t)
                flux_all = np.append(flux_all, phasma_flux)
                flux_err_all = np.append(flux_err_all, phasma_flux_err)

            if self.plot_clean_lc or self.plot_raw_lc:
                plt.show()

        # define the phases corresponding to the surviving data
        phase = self._phase(time_all)

        # fold the combined data and return
        return self._fold(phase, time_all, flux_all, flux_err_all)


def _clean(time, flux, flux_err):
    """
    Applies a 20-cadence moving median function and discards outliers
    defined by flux > 2.57 sigma (from the inverse erf function)
    """
    cadence = time[1] - time[0]
    window_size = 20 * cadence

    (trimmed_t,
     trimmed_flux,
     trimmed_flux_err,
     moving_med_func) = _moving_median(time, flux, flux_err, window_size)

    # get the residuals
    res = trimmed_flux - moving_med_func(trimmed_t)

    # remove outliers (> 2.57-sigma from median)
    outliers = abs(res / trimmed_flux_err) > 2.57

    return (trimmed_t[~outliers], trimmed_flux[~outliers],
            trimmed_flux_err[~outliers])


def _moving_median(x, y, y_err, window_size):

    moving_med_x = np.array([])  # x in middle of bins
    moving_med_y = np.array([])
    i = 0
    while x[i] + window_size < x[-1]:
        in_window = (x >= x[i]) & (x < x[i] + window_size)
        moving_med_x = np.append(moving_med_x, np.nanmedian(x[in_window]))
        moving_med_y = np.append(moving_med_y, np.nanmedian(y[in_window]))
        i += 1

    mid_window = window_size / 2
    trim = (x >= x[0] + mid_window) & (x < x[-1] - mid_window)
    moving_med_func = interp1d(moving_med_x, moving_med_y,
                               fill_value='extrapolate')

    return x[trim], y[trim], y_err[trim], moving_med_func


def _phasma_detrend(P, time, flux, flux_err):
    window_size = P
    (trimmed_t,
     trimmed_flux,
     trimmed_flux_err,
     moving_med_func) = _moving_median(time, flux, flux_err, window_size)
    return (trimmed_t,
            (trimmed_flux / moving_med_func(trimmed_t) - 1) * 1e6,
            trimmed_flux_err * 1e6)


def _bin(x, binsize, flux, flux_err):
    # bin the combined data from the two sectors
    bin_start = x[0]
    bin_end = x[-1]
    bin_edges = np.arange(bin_start, bin_end, binsize)
    binned_x = np.arange(bin_start + binsize / 2,
                         bin_end + binsize / 2,
                         binsize)
    bin_indices = np.digitize(x, bin_edges) - 1

    binned_flux = np.array([])
    binned_error = np.array([])
    for i in range(max(bin_indices) + 1):
        bin = bin_indices == i
        errors_to_bin = flux_err[bin]

        if len(errors_to_bin) > 0:
            binned_flux = np.append(binned_flux,
                                    [np.average(flux[bin],
                                                weights=1 / errors_to_bin ** 2)])
            binned_error = np.append(binned_error,
                                     [np.average(errors_to_bin) /
                                                 np.sqrt(len(errors_to_bin))])
        else:
            binned_flux = np.append(binned_flux, np.array([np.nan]))
            binned_error = np.append(binned_error, np.array([np.nan]))

    return binned_x, binned_flux, binned_error
