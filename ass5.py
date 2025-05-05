# -*- coding: utf-8 -*-
"""
Created on Sun May  4 17:51:28 2025

@author: maxmi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the data files
paths = [np.genfromtxt("PS20D2NGC45481.DAT", delimiter=None, comments='#', usecols=(1, 2), skip_header = 5),
         np.genfromtxt("PS20D2NGC43211.DAT", delimiter=None, comments='#', usecols=(1, 2), skip_header = 5),
         np.genfromtxt("PS20D2NGC36211.DAT", delimiter=None, comments='#', usecols=(1, 2), skip_header = 5),
         np.genfromtxt("PS20D2NGC25411.DAT", delimiter=None, comments='#', usecols=(1, 2), skip_header = 5),
         np.genfromtxt("PS20D2NGC33511.DAT", delimiter=None, comments='#', usecols=(1, 2), skip_header = 5),
         np.genfromtxt("PS20D2NGC14251.DAT", delimiter=None, comments='#', usecols=(1, 2), skip_header = 5)]

# derive names and v values from data
names = ['NGC45481', 'NGC43211', 'NGC36211', 'NGC25411', 'NGC33511', 'NGC14251']
v_hel = [1512, 559, 778, 805, 1571, 486]
# empty lists for values
H_list = []
d_list = []
sigma_d_list = []

plt.figure(figsize=(10, 6))

# Loop over each dataset and corresponding velocity
for i, (path, v) in enumerate(zip(paths, v_hel)):
# calculate d and mean d for each galaxy   
    P = path[:, 0]
    mv = path[:, 1]
    
    Mv = -2.76 * (np.log10(P) - 1) - 4.22

    d = 10 ** ((mv - Mv + 5) / 5)
    
    mean_d = np.mean(d)

    name = names[i]
    d_list.append(mean_d)
    
# calculate uncertainties
    sigma_modulus = np.std(mv - Mv) / np.sqrt(len(P))
    sigma_d = np.log(10) * mean_d * (1 / 5) * sigma_modulus
    sigma_d_list.append(sigma_d)
    
#plots
    plt.scatter(v, mean_d / 1e6)
    plt.errorbar(v, mean_d / 1e6, yerr=sigma_d / 1e6,
                 alpha=0.6,
                 markersize=2,
                 capsize = 2)

# calucltae H    
    H = v / (mean_d / 1e6)
    H_list.append(H)
    print(f"{name}: {mean_d / 1e6:.3} +- {sigma_d / 1e6:.3} Mpc; H = {H:.2} km/s/Mpc")
# mean H
mean_H = np.mean(H_list)
sigma_H = np.std(H_list) / np.sqrt(len(H_list))


print(f"Mean Hubble constant: {mean_H:.2} km/s/Mpc +- {sigma_H:.2} km/s/Mpc")

# calculate dispersion by calculating differnce in v values, first generate empty list
v_new_list = []

for d in d_list:
    v_new = mean_H * (d / 1e6)
    v_new_list.append(v_new)

residuals = []

for v_h, v_n in zip(v_hel, v_new_list):
    residual = v_h - v_n
    residuals.append(residual)

H_dispersion = np.std(residuals)

print(f"Velocity dispersion: {H_dispersion:.3} km/s")
plt.xlabel('Velocity (km/s)')
plt.ylabel('Distance (Mpc)')
plt.title('Hubble Law: Velocity vs Distance')
plt.show()

plt.figure(figsize = (10, 6))
colors = ['red', 'blue', 'orange', 'purple', 'pink', 'green']
for color, d, sigma_d, residual in zip(colors, d_list, sigma_d_list, residuals):
  
    plt.errorbar(d / 1e6, residual, yerr=sigma_d / 1e6 * mean_H, markersize=2, capsize = 2, color = color)

plt.axhline(0, ls = '-', color = 'black', label = "residual at 0 km/s")
plt.axhline(H_dispersion, ls = '-', color = 'gray', label = "velocity dispersion for the average of H")
plt.axhline(-H_dispersion, ls = '-', color = 'gray', label = "velocity dispersion for the average of H")
plt.title("Residuals of individual velocities")
plt.xlabel("Distance D (Mpc)")
plt.ylabel("Residuals (km/s)")
plt.legend(loc = 'best')
plt.show()


data = np.genfromtxt('Redshift and apparent magnitudes of Type Ia SNe.dat', names = True)

mask_zhel = data["zhel"] < 0.1
zhel_near= data["zhel"][mask_zhel]
mb_near = data["mb"][mask_zhel]
sigma_mb_near = data["dmb"][mask_zhel]

# find fit values, z mostly log scale
x_fit = np.log10(zhel_near)
y_fit = mb_near
w = 1 / sigma_mb_near* 2

# derive varuables a and b for fit with formulas
a = (np.sum(w) * np.sum(w * x_fit * y_fit) - np.sum(w * x_fit) * np.sum(w * y_fit)) / (np.sum(w) * np.sum(w* x_fit ** 2) - (np.sum(w * x_fit)) ** 2)
b = (np.sum(w * y_fit) * np.sum(w * x_fit ** 2) - np.sum(w * x_fit) * np.sum(w* x_fit * y_fit)) / (np.sum(w) * np.sum(w * x_fit ** 2) - (np.sum(w* x_fit)) ** 2)
a_error = np.sqrt(np.sum(w)) / (np.sum(w) * np.sum(w * x_fit ** 2) - (np.sum(w * x_fit)) ** 2)
b_error = np.sqrt(np.sum(w * x_fit ** 2)) / (np.sum(w) * np.sum(w * x_fit ** 2) - (np.sum(w * x_fit)) ** 2)

plt.figure(figsize = (10,6))
plt.plot(x_fit, a * x_fit + b, label = f'm = {a:.3}log(z) + {b:.3}', color = 'black')
plt.errorbar(x_fit, y_fit, fmt = 'o', yerr = sigma_mb_near, markersize = 2, capsize = 2)

plt.title("Hubble estimate for z < 0.1")
plt.xlabel('log(z)')
plt.ylabel("apparent magnitude")
plt.legend(loc = 'best')
plt.show()


c = 299792.5
M = -19.3
H_fit = c * 10 ** ((M+25-b)/5)
error_H = (np.log(10) / 5) * H_fit * np.sqrt(b_error ** 2 + 0.2 **2)

print(f"Hubble constant z<0.1: {H_fit:.3} +- {error_H:.3} km/s/Mpc")

# find values for z > 0.1
mask_zhel_far = data["zhel"] >= 0.1
zhel_far = data["zhel"][mask_zhel_far]
mb_far = data["mb"][mask_zhel_far]
sigma_mb_far = data["dmb"][mask_zhel_far]
x_fit_far = np.log10(zhel_far)
y_fit_far = mb_far

redshift_model = np.logspace(-1, 0.4, num = 150)
h_model = 5 * np.log10(c * redshift_model / H_fit) + 25 + M
h_model_p1s = 5 * np.log10(c * redshift_model / (H_fit + error_H)) + 25 + M
h_model_m1s = 5 * np.log10(c * redshift_model / (H_fit - error_H)) + 25 + M
h_model_cosmological = 5 * np.log10((c * redshift_model / H_fit) * ((1- -0.55 * redshift_model) / 2)) + 25 + M 

plt.figure(figsize = (12,6))

plt.plot(x_fit, a * x_fit + b, label = f'm = {a:.3}log(z) + {b:.3} (z<0.1)', color = 'black')
plt.errorbar(x_fit, y_fit, alpha = 0.2, fmt = 'o', yerr = sigma_mb_near, markersize = 2, capsize = 2, color = 'blue')

plt.plot(x_fit_far, a * x_fit_far + b, label = f'm = {a:.3}log(z) + {b:.3} (z>0.1)', color = 'black')
plt.errorbar(x_fit_far, y_fit_far, alpha =0.2, fmt = 'o', yerr = sigma_mb_far, markersize = 2, capsize = 2, color = 'blue')


plt.plot(np.log10(redshift_model), h_model, color = 'blue', label = "H0 fit")
plt.plot(np.log10(redshift_model), h_model_p1s, color = 'gray', label = "H0 +- 1σ", linestyle = '-.')
plt.plot(np.log10(redshift_model), h_model_m1s, color = 'gray', linestyle = '-.')
plt.plot(np.log10(redshift_model), h_model_cosmological, color = 'red', alpha = 0.6, label = "$H_{0}$ cosmological model")

plt.axvline(-1.0, color = 'grey', linestyle = '--', label = "z = 0.1", alpha = 0.6)
plt.axvline(0.0, color = 'lightcoral', linestyle = '--', label = "z = 0.0", alpha = 0.6)


plt.title("Hubble estimate for near & distant galaxies")
plt.xlabel('log(z)')
plt.ylabel("Apparent magnitude")
plt.grid(True, alpha = 0.3)
plt.legend(loc = 'best')
plt.show()


def residuals_hubblemodel_near(z):
    h = H_fit
    M = -19.3
    m = 5 * np.log10(c * z/ h) + 25 + M    
    return mb_near - m

def residuals_hubblemodel_far(z):
    h = H_fit
    M = -19.3
    m = 5 * np.log10((c * z/ h) * ((1 - -0.55 * z) / 2)) + 25 + M +1.4
    return mb_far - m


fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)


ax[0].plot(x_fit, a * x_fit + b, label = f'm = {a:.3}log(z) + {b:.3} (z<0.1)', color = 'black')
ax[0].errorbar(x_fit, y_fit, alpha = 0.2, fmt = 'o', yerr = sigma_mb_near, markersize = 2, capsize = 2, color = 'blue')

ax[0].plot(x_fit_far, a * x_fit_far + b, label = f'm = {a:.3}log(z) + {b:.3} (z>0.1)', color = 'black')
ax[0].errorbar(x_fit_far, y_fit_far, alpha =0.2, fmt = 'o', yerr = sigma_mb_far, markersize = 2, capsize = 2, color = 'blue')


ax[0].plot(np.log10(redshift_model), h_model, color = 'blue', label = "H0 fit")
ax[0].plot(np.log10(redshift_model), h_model_p1s, color = 'gray', label = "H0 +- 1σ", linestyle = '-.')
ax[0].plot(np.log10(redshift_model), h_model_m1s, color = 'gray', linestyle = '-.')
ax[0].plot(np.log10(redshift_model), h_model_cosmological, color = 'red', alpha = 0.6, label = "H cosmological model")

ax[0].axvline(-1.0, color = 'gray', linestyle = '--', label = "z = 0.1", alpha = 0.8)
ax[0].axvline(0.0, color = 'yellow', linestyle = '--', label = "z = 0.0", alpha = 0.8)


ax[0].set_title("Hubble estimate for near & distant galaxies")
ax[0].set_xlabel('log(z)')
ax[0].set_ylabel("Apparent magnitude")
ax[0].grid(True, alpha = 0.3)
ax[0].legend(loc = 'best')


ax[1].errorbar(np.log10(zhel_near), residuals_hubblemodel_near(zhel_near), yerr = sigma_mb_near,
               fmt = 'o',
               color = 'blue',
               alpha = 0.2,
               label = "residuals z<0.1",
               markersize = 2,
               capsize = 2)

ax[1].errorbar(np.log10(zhel_far), residuals_hubblemodel_far(zhel_far), yerr = sigma_mb_far,
               fmt = 'o',
               color = 'blue',
               alpha = 0.2,
               label = "residuals z>0.1",
               markersize = 2,
               capsize = 2)


ax[1].axvline(-1.0, color = 'grey', linestyle = '--', label = "z = 0.1", alpha = 0.8)
ax[1].axvline(0.0, color = 'yellow', linestyle = '--', label = "z = 0", alpha = 0.8)


ax[1].set_title("Hubble residuals for near & distant galaxies")
ax[1].set_xlabel('log(z)')
ax[1].set_ylabel("difference in magnitude")
ax[1].grid(True, alpha = 0.5)
ax[1].legend(loc = 'best')
plt.show()