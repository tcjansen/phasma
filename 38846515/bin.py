import numpy as np


def bin(x, binsize, flux, flux_err):
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
                                    np.nanmean(flux[bin]))
                                    # [np.average(flux[bin],
                                    #             weights=1 / errors_to_bin ** 2)])
            binned_error = np.append(binned_error,
                                     np.nanmean(errors_to_bin) / np.sqrt(len(errors_to_bin)))
                                     # [np.average(errors_to_bin) /
                                     #             np.sqrt(len(errors_to_bin))])
        else:
            binned_flux = np.append(binned_flux, np.array([np.nan]))
            binned_error = np.append(binned_error, np.array([np.nan]))

    return binned_x, binned_flux, binned_error