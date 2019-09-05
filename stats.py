import beer
import matplotlib.pyplot as plt 
import sys, os
import numpy as np

def chi_squared(theta, model, x, y, yerr):
	return np.nansum(((y-model(x, *theta))/yerr)**2)


def lnlike(theta, model, x, y, yerr):
    return -np.nansum(0.5 * np.log([2 * np.pi] * len(y)))\
           -np.nansum(np.log(yerr))\
           -0.5*chi_squared(theta, model, x, y, yerr)

def BIC(theta, model, x, y, yerr):
	k = len(theta)
	N = len(x)
	return chi_squared(theta, model, x, y, yerr) + k * np.log(N)

def AIC(theta, model, x, y, yerr):
	k = len(theta)
	return chi_squared(theta, model, x, y, yerr) + 2 * k

def O12(bic1, bic2):
	return np.exp(-0.5 * (bic1 - bic2))
