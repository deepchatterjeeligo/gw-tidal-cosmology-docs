import json
import pickle

from astropy import units as u, constants as const

import numpy as np
from scipy.integrate import quad
from scipy.stats import gaussian_kde
import scipy
import bilby
import matplotlib.pyplot as plt
import pandas as pd

import corner


plt.style.use('publication.mplstyle')

omega_m = 0.3
omega_lambda = 0.7
integrand = lambda z: 1./np.sqrt(omega_m * (1 + z)**3 + omega_lambda)

def add_hubble(result):
    hubble_vals = np.array([quad(integrand, 0, z_vals)[0] for z_vals in result.posterior.z.values])
    hubble_vals /= result.posterior.luminosity_distance.values
    hubble_vals *= (1 + result.posterior.z.values)
    hubble_vals *= (const.c.to('km/s')/u.Mpc).value
    result.posterior['hubble'] = hubble_vals


filename1 = \
    '/home/deep/work/campus-cluster-runs/pbilby-runs/redshift-injections/ce/gaussian-prior/outdir_gw_tidal_cosmo_dl_1000_CE_incl_45/result/gw_tidal_cosmo_dl_1000_CE_incl_45_0_result.json'
filename2 = \
    '/home/deep/work/campus-cluster-runs/pbilby-runs/redshift-injections/ce/gaussian-prior/outdir_gw_tidal_cosmo_dl_2000_CE_incl_45/result/gw_tidal_cosmo_dl_2000_CE_incl_45_0_result.json'
filename3 = \
    '/home/deep/work/campus-cluster-runs/pbilby-runs/redshift-injections/ce/gaussian-prior/outdir_gw_tidal_cosmo_dl_3000_CE_incl_45/result/gw_tidal_cosmo_dl_3000_CE_incl_45_0_result.json'
filename4 = \
    '/home/deep/work/campus-cluster-runs/pbilby-runs/redshift-injections/ce/gaussian-prior/outdir_gw_tidal_cosmo_dl_4000_CE_incl_45/result/gw_tidal_cosmo_dl_4000_CE_incl_45_0_result.json'
filename5 = \
    '/home/deep/work/campus-cluster-runs/pbilby-runs/redshift-injections/ce/gaussian-prior/outdir_gw_tidal_cosmo_dl_5000_CE_incl_45/result/gw_tidal_cosmo_dl_5000_CE_incl_45_0_result.json'


filename6 = \
    '/home/deep/work/campus-cluster-runs/pbilby-runs/redshift-injections/ce/outdir_gw_tidal_cosmo_dl_1000_CE_incl_45/result/gw_tidal_cosmo_dl_1000_CE_incl_45_0_result.json'
filename7 = \
    '/home/deep/work/campus-cluster-runs/pbilby-runs/redshift-injections/ce/outdir_gw_tidal_cosmo_dl_2000_CE_incl_45/result/gw_tidal_cosmo_dl_2000_CE_incl_45_0_result.json'
filename8 = \
    '/home/deep/work/campus-cluster-runs/pbilby-runs/redshift-injections/ce/outdir_gw_tidal_cosmo_dl_3000_CE_incl_45/result/gw_tidal_cosmo_dl_3000_CE_incl_45_0_result.json'
filename9 = \
    '/home/deep/work/campus-cluster-runs/pbilby-runs/redshift-injections/ce/outdir_gw_tidal_cosmo_dl_4000_CE_incl_45/result/gw_tidal_cosmo_dl_4000_CE_incl_45_0_result.json'
filename10 = \
    '/home/deep/work/campus-cluster-runs/pbilby-runs/redshift-injections/ce/outdir_gw_tidal_cosmo_dl_5000_CE_incl_45/result/gw_tidal_cosmo_dl_5000_CE_incl_45_0_result.json'

result_1000 = bilby.result.read_in_result(filename=filename1)
result_2000 = bilby.result.read_in_result(filename=filename2)
result_3000 = bilby.result.read_in_result(filename=filename3)
result_4000 = bilby.result.read_in_result(filename=filename4)
result_5000 = bilby.result.read_in_result(filename=filename5)

result_1000_fixed_lambda = bilby.result.read_in_result(filename=filename6)
result_2000_fixed_lambda = bilby.result.read_in_result(filename=filename7)
result_3000_fixed_lambda = bilby.result.read_in_result(filename=filename8)
result_4000_fixed_lambda = bilby.result.read_in_result(filename=filename9)
result_5000_fixed_lambda = bilby.result.read_in_result(filename=filename10)

for r in (
        result_1000, result_2000, result_3000, result_4000, result_5000,
        result_1000_fixed_lambda, result_2000_fixed_lambda, result_3000_fixed_lambda,
        result_4000_fixed_lambda, result_5000_fixed_lambda):
    add_hubble(r)

plt.figure()
delta_d_over_d = []
delta_d_over_d_gaussian = []

for r in (result_1000, result_2000, result_3000, result_4000, result_5000):
    five, ninetyfive = np.percentile(r.posterior.luminosity_distance, (5, 95))
    ninety = ninetyfive - five
    delta_d_over_d_gaussian.append(0.5*ninety/r.injection_parameters['luminosity_distance'])
for r in (result_1000_fixed_lambda, result_2000_fixed_lambda,
          result_3000_fixed_lambda, result_4000_fixed_lambda,
          result_5000_fixed_lambda):
    five, ninetyfive = np.percentile(r.posterior.luminosity_distance, (5, 95))
    ninety = ninetyfive - five
    delta_d_over_d.append(0.5*ninety/r.injection_parameters['luminosity_distance'])

delta_z_over_z = []
delta_z_over_z_gaussian = []

for r in (result_1000, result_2000, result_3000, result_4000, result_5000):
    five, ninetyfive = np.percentile(r.posterior.z, (5, 95))
    ninety = ninetyfive - five
    delta_z_over_z_gaussian.append(0.5*ninety/r.injection_parameters['z'])
for r in (result_1000_fixed_lambda, result_2000_fixed_lambda,
          result_3000_fixed_lambda, result_4000_fixed_lambda,
          result_5000_fixed_lambda):
    five, ninetyfive = np.percentile(r.posterior.z, (5, 95))
    ninety = ninetyfive - five
    delta_z_over_z.append(0.5*ninety/r.injection_parameters['z'])

delta_h0_over_h0 = []
delta_h0_over_h0_gaussian = []
for r in (result_1000, result_2000, result_3000, result_4000, result_5000):
    five, ninetyfive = np.percentile(r.posterior.hubble, (5, 95))
    ninety = ninetyfive - five
    delta_h0_over_h0.append(0.5*ninety/70)
for r in (result_1000_fixed_lambda, result_2000_fixed_lambda,
          result_3000_fixed_lambda, result_4000_fixed_lambda,
          result_5000_fixed_lambda):
    five, ninetyfive = np.percentile(r.posterior.hubble, (5, 95))
    ninety = ninetyfive - five
    delta_h0_over_h0_gaussian.append(0.5*ninety/70)

distances = [1000, 2000, 3000, 4000, 5000]

plt.plot(distances, np.array(delta_d_over_d), color='blue', linestyle='dashdot')
plt.plot(distances, np.array(delta_d_over_d_gaussian), color='red', linestyle='dashdot')
plt.plot(distances, np.array(delta_z_over_z), color='blue', linestyle='dashed')
plt.plot(distances, np.array(delta_z_over_z_gaussian), color='red', linestyle='dashed')
plt.plot(distances, np.array(delta_h0_over_h0), color='blue',
         linestyle='dotted', marker='o')
plt.plot(distances, np.array(delta_h0_over_h0_gaussian), color='red',
         linestyle='dotted', marker='o')


plt.xlabel('Distance (Mpc)')
plt.ylabel('Fractional error')
plt.text(3000, 0.17, "$\Delta z/z$; Gaussian", fontsize=17)
plt.text(2000, 0.14, "$\Delta z/z$; Fixed", fontsize=17)
plt.text(2000, 0.35, "$\Delta D_L/D_L$; Gaussian", fontsize=17)
plt.text(2000, 0.4, "$\Delta D_L/D_L$; Fixed", fontsize=17)
plt.text(1000, 0.5, "$\Delta H_0/H_0$; Gaussian", fontsize=17)
plt.text(1000, 0.44, "$\Delta H_0/H_0$; Fixed", fontsize=17)
plt.tight_layout()
# plt.show()
plt.savefig('../figures/gaussian-error-lambda00.pdf')
