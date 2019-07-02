import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
import pylab as py

from astropy.io import fits
from scipy import signal
from scipy.stats import f
from scipy.interpolate import interp1d

#load in data
infile = sys.argv[1] #string: name of input tess .fits file
outfile = sys.argv[2] #string: name of master output file for this planet.
t_orbit = float(sys.argv[3]) #period, days
t_14 = (float(sys.argv[4]))/24. #transit duration, hours converted to days
t_midpt = float(sys.argv[5]) #transit ephemeris, days

#other relevant timescales
t_longcadence = 0.02083333333 #average elapsed time between long-cadence data points, in days
t_shortcadence = 0.0013888889 #average elapsed time between short-cadence data points, in days

#open fits file and extract data from HDU 1
fitsdata = fits.open(infile)
data = fitsdata[1].data

t = data.field('TIME')

pdc_flux = data.field('PDCSAP_FLUX') # a more processed version of SAP with artifact mitigation
pdc_flux_err = data.field('PDCSAP_FLUX_ERR')

#ignore nan values (i.e. only use data with values)
notnan = (~np.isnan(t) & ~np.isnan(pdc_flux) & ~np.isnan(pdc_flux_err) \
          & (t != 0) & (pdc_flux != 0) & (pdc_flux_err != 0))
t = t[notnan]
pdc_flux = pdc_flux[notnan]
pdc_flux_err = pdc_flux_err[notnan]

if t[2] - t[1] < 0.01:
  cadence = t_shortcadence
else:
  cadence = t_longcadence


#begin with window = orbital period
window = t_orbit
tstart = np.min(t) + 0.5*window
tend = np.max(t)

# apply the moving median function
j = 0
windowi_end = np.min(t) + window
beta_ms = []
moving_median_fluxes = []
while tstart + (j*cadence) + window/2 <= tend: # move window along point by point
  times_mask = ((t >= tstart + cadence * j - window / 2) & (t < tstart + cadence * j + window / 2))
  selected_times = t[times_mask]
  selected_fluxes = pdc_flux[times_mask]

  if len(selected_times) > 0:
    beta_ms.append(tstart + j * cadence) # mid times at each window
    moving_median_fluxes.append(np.median(selected_fluxes)) # corresponding median values of each window

  j += 1


beta_ms = np.array(beta_ms) #(indicates temporal position of moving median filter along time array)
moving_median_fluxes = np.array(moving_median_fluxes) #array of fluxes after passing a moving median filter of width (P + cadence)

tp = np.min(beta_ms) # new starting time
tq = np.max(beta_ms) # new ending time

times_mask = ((tp <= t) & (t <= tq))

selected_times = t[times_mask] # grabbing moving median times
selected_fluxes = pdc_flux[times_mask] # grabbing fluxes at those times

beta_ms_unique, beta_ms_unique_idxs = np.unique(beta_ms, return_index=True) # avoid repeating values. first thing returned is
                                                                            # the unique value, second is where that value occurs in the array

#interpolate moving median fluxes between new times 
beta_ms = beta_ms[beta_ms_unique_idxs] # "xs" for interpolation. only unique values of x
moving_median_fluxes = moving_median_fluxes[beta_ms_unique_idxs] # "ys" for interpolation. only unique values of y
interpolation = interp1d(x=beta_ms, y=moving_median_fluxes) # the moving median interpolation function

# normalize flux by moving median function (i.e. the median interpolation)
detrended_t = []
detrended_flux = []
for i in range(0, len(selected_times)):
  detrended_t.append(selected_times[i]) # selected times = all times within the moving median, not necessarily unique
  detrended_flux.append(selected_fluxes[i] / interpolation(selected_times[i])) # selected fluxes = all fluxes within the moving median, not necessarily unique

# convert list to array
detrended_t = np.array(detrended_t)
detrended_flux = np.array(detrended_flux)

# sort data by time
detrended_t_sorted = detrended_t[np.argsort(detrended_t)]
detrended_flux_sorted = detrended_flux[np.argsort(detrended_t)]

#shift time into a phase curve, where x axis goes from -period/2 to +period/2
shifted_t = detrended_t_sorted - t_midpt
for i in range(0, len(detrended_t_sorted)):
  st = shifted_t[i]
  while st > 0.5*t_orbit:
    st = st - t_orbit
    shifted_t[i] = st
  while st < -0.5*t_orbit:
    st = st + t_orbit
    shifted_t[i] = st

# take out any nans or empty pixels
notnan = (~np.isnan(detrended_t_sorted) & ~np.isnan(shifted_t) & ~np.isnan(detrended_flux_sorted) \
          & (detrended_flux_sorted != 0))
detrended_t_sorted = detrended_t_sorted[notnan]
shifted_t = shifted_t[notnan]
detrended_flux_sorted = detrended_flux_sorted[notnan]

# flag whether these data were of short cadences or long
if cadence == t_shortcadence:
  cadence_flag_arr = np.ones_like(detrended_t_sorted)
elif cadence == t_longcadence:
  cadence_flag_arr = np.zeros_like(detrended_t_sorted)

selected_fluxerrs = pdc_flux_err[times_mask] # make flux_errs the same length of array. don't need to shift around with 
# the other values because all flux errors are given the same value

# normalize the flux error by the median function as well
detrended_flux_err = []
for i in range(0, len(selected_times)):
	detrended_flux_err.append(selected_fluxerrs[i] / interpolation(selected_times[i]))

# sort and grab only real values of the error
detrended_flux_err = np.array(detrended_flux_err)
detrended_flux_err_sorted = detrended_flux_err[np.argsort(detrended_t)]
detrended_flux_err_sorted = detrended_flux_err_sorted[notnan]
notnan = (~np.isnan(detrended_t_sorted) & ~np.isnan(shifted_t) & ~np.isnan(detrended_flux_sorted) & ~np.isnan(detrended_flux_err_sorted))
detrended_flux_err_sorted = detrended_flux_err_sorted[notnan]

# write everything to a file which can be appended to, i.e. the "intermediate file"
fmt = '%f %f %f %f %d'

alldata = np.vstack((detrended_t_sorted,shifted_t,detrended_flux_sorted,detrended_flux_err_sorted,cadence_flag_arr)).T

with open(outfile,'ab') as outfile_name:
    np.savetxt(outfile_name,alldata,fmt=fmt)


