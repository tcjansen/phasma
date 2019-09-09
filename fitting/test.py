import phasecurve_model as pm 
import time
import numpy as np
import astropy.units as u
import astropy.constants as const


phase = np.linspace(-0.5, 0.5, 100)
Ab = 0.1
eps = 0.5
f = 1.5
Rp = 1 * const.R_jup
Rs = 1.1 * const.R_sun
Ts = 5000 * u.K
semi_a = 1.0 * u.au


start = time.time()
pm.run(phase, Ab, eps, f, None, None, None, Rp, Rs, Ts, semi_a,
       therm=True, s_reflection=True, a_reflection=False)
end = time.time()
print('seconds =', end - start)