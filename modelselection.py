import numpy as np
import matplotlib.pyplot as plt
import sys

# LOCAL
import mcmc
import models
import stats
import beer

# PHASMA
from fit import Fit


class ModelSelection(Fit):
    def __init__(self, thetas, models, model_names):
        self.thetas = thetas
        self.models = models
        self.model_names = model_names
        super().__init__(self)

    def chisquares(self):
        for _ in range(len(self.thetas)):
            chi = stats.chi_squared(self.thetas[_], self.models[_],
                                    self.phase, self.flux, self.flux_err)
            print('chi ' + self.model_names[_] + ' =', chi)

    def reduced_chisquares(self):
        for _ in range(len(self.thetas)):
            chi = stats.chi_squared(self.thetas[_], self.models[_],
                                    self.phase, self.flux, self.flux_err)
            chi_red = chi / (len(self.phase[~np.isnan(self.phase)]) -
                             len(self.thetas[_]))
            print('chi red ' + self.model_names[_] + ' =', chi_red)

    def bics_and_odds(self):
        last_bic = None
        for _ in range(len(self.thetas)):
            bic = stats.BIC(self.thetas[_], self.models[_],
                            self.phase, self.flux, self.flux_err)
            print('bic ' + self.model_names[_] + ' =', bic)

            if _ == 0:
                last_bic = bic

            if _ > 0:
                o12 = stats.O12(last_bic, bic)

                print('delta bic (flat, ' + self.model_names[_] + ') =',
                      last_bic - bic)
                print('odds of flat being a better fit than ' +
                      self.model_names[_] + ' =', o12)
