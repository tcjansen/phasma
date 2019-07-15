# ASTROPY
import astropy.units as u
from astropy.io import fits

# THIRD-PARTY
import os
import subprocess


class Target():
    """
    Defines an object for phasma's use. With this object one can get the
    phasma-detrended phase/light curve and plot it or write it to a file.

    Parameters
    ----------
    tic_id : str or int
        The TESS input catalog number specific to the target
    orbital_period : `~astropy.units.quantity.Quantity`
    transit_epoch : '~astropy.time.core.Time'
        The time of the center of transit
    transit_duration : `~astropy.units.quantity.Quantity`
    sectors : list of int
        List of sectors of interest
    rm_curl_files : bool
        Set to True to delete the curl files after they have been
        used. Default is False.
    """
    def __init__(self, tic_id, orbital_period, transit_epoch,
                 transit_duration, sectors, rm_curl_files=False):
        self.tic_id = tic_id
        self.orbital_period = orbital_period
        self.transit_epoch = transit_epoch
        self.transit_duration = transit_duration
        self.sectors = sectors
        self.rm_curl_files = rm_curl_files

        # dowload the curl file for each sector
        curl_paths = [download_file('https://archive.stsci.edu/' +
                                    'missions/tess/download_scripts/' +
                                    'sector/tesscurl_sector_' +
                                    sector + '_tp.sh')
                      for sector in self.sectors]

        # download the fits files
        for curl_path in curl_paths:
            subprocess.run(curl_path)

        # open the fits files and normalize the light curves

        # concatenate the data across sectors together

        # define light curve objects

    # delete the curl files to save space
    if self.rm_curl_files:
        [os.remove('https://archive.stsci.edu/missions/' +
                   'tess/download_scripts/sector/tesscurl_' +
                   'sector_' + sector + '_tp.sh')
         for sector in self.sectors]
