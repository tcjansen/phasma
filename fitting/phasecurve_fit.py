import phasecurve_model as pm
import time
import numpy as np
import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt


phase = np.linspace(-0.5, 0.5, 100) * np.pi


def doppler_beaming_amp(alpha_beam, K):
	return alpha_beam * 4 * (K / const.c).to(u.Unit(''))


def thermref_beam_model(Ab, eps, f, alpha_beam, alpha_ellip):
	# draw from the posteriors of known parameters
	Rs = np.random.normal(2.0, 0.3) * const.R_sun
	depth = np.random.normal(0.0076, 0.0005)
	Rp = np.sqrt(depth) * Rs
	Ts = np.random.normal(6900, 120) * u.K
	K = np.random.normal(0.213, 0.008) * u.km / u.s
	Ms = np.random.normal(0.69, 0.06) * const.M_sun
	semi_a = np.random.normal( 0.0457, 0.0010) * u.au
	inc = ?
	mpsini = np.random.normal(2.03, 0.12) * const.M_jup * np.sin(inc)



	therm, ref, thermref_model = pm.run(phase, Ab, eps, f, None, None,
										None, Rp, Rs, Ts, semi_a,
	      							    therm=True, s_reflection=True,
	      							    a_reflection=False)
	beam_model = doppler_beaming_amp(alpha_beam, K) * np.sin(2 * phase)
	ellip_model = alpha_ellip * other stuff

	return thermref_model + beam_model + ellip_model

# set free parameter priors
Ab = [0.0, 1.0]
eps = ?
f = ?

