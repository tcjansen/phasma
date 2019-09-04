import matplotlib.pyplot as plt
import numpy as np
import sys, csv

ID=sys.argv[1]
ID_dir = '../targets/' + str(ID)
time, flux = np.genfromtxt('data.dat', unpack=True, usecols=(0,1))
P, t0 = np.genfromtxt(ID_dir + '/target_info.csv', usecols=(1,3),delimiter=',')
# time = time - t0

# read in fit coefficients
a0 = 0
a1 = 0
a2 = 0
a3 = 0
a4 = 0
# with open("../targets/" + ID + "/coefficients.csv") as c:
# 	reader = csv.DictReader(c)
# 	for row in reader:
# 		a0 += float(row["a0"])
# 		a1 += float(row["a1"])
# 		a2 += float(row["a2"])
# 		a3 += float(row["a3"])
# 		a4 += float(row["a4"])
a0,a1,a2,a3,a4 = np.genfromtxt("../targets/" + ID + "/coefficients.txt")

fit = a0 + a1 * np.sin(2 * np.pi * time/P) \
	+ a2 * np.cos(2 * np.pi * time / P) \
	+ a3 * np.sin(2 * np.pi * time/ (P/2)) \
	+ a4 * np.cos(2 * np.pi * time / (P/2))

plt.scatter(time, flux, color='#77EABA', alpha=0.5)
plt.plot(time, fit, color="#6D7774", lw=2)
plt.show()
