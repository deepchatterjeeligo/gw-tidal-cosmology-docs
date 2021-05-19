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
    '/home/deep/work/campus-cluster-runs/pbilby-runs/redshift-injections/ce/gaussian-prior-sys-err/outdir_gw_tidal_cosmo_dl_1000_CE_incl_45/result/gw_tidal_cosmo_dl_1000_CE_incl_45_0_result.json'
filename2 = \
    '/home/deep/work/campus-cluster-runs/pbilby-runs/redshift-injections/ce/gaussian-prior-sys-err/outdir_gw_tidal_cosmo_dl_2000_CE_incl_45/result/gw_tidal_cosmo_dl_2000_CE_incl_45_0_result.json'
filename3 = \
    '/home/deep/work/campus-cluster-runs/pbilby-runs/redshift-injections/ce/gaussian-prior-sys-err/outdir_gw_tidal_cosmo_dl_3000_CE_incl_45/result/gw_tidal_cosmo_dl_3000_CE_incl_45_0_result.json'
filename4 = \
    '/home/deep/work/campus-cluster-runs/pbilby-runs/redshift-injections/ce/gaussian-prior-sys-err/outdir_gw_tidal_cosmo_dl_4000_CE_incl_45/result/gw_tidal_cosmo_dl_4000_CE_incl_45_0_result.json'
filename5 = \
    '/home/deep/work/campus-cluster-runs/pbilby-runs/redshift-injections/ce/gaussian-prior-sys-err/outdir_gw_tidal_cosmo_dl_5000_CE_incl_45/result/gw_tidal_cosmo_dl_5000_CE_incl_45_0_result.json'

filename6 = \
    '/home/deep/work/campus-cluster-runs/pbilby-runs/redshift-injections/ce/gaussian-prior/outdir_gw_tidal_cosmo_dl_1000_CE_incl_45/result/gw_tidal_cosmo_dl_1000_CE_incl_45_0_result.json'
filename7 = \
    '/home/deep/work/campus-cluster-runs/pbilby-runs/redshift-injections/ce/gaussian-prior/outdir_gw_tidal_cosmo_dl_2000_CE_incl_45/result/gw_tidal_cosmo_dl_2000_CE_incl_45_0_result.json'
filename8 = \
    '/home/deep/work/campus-cluster-runs/pbilby-runs/redshift-injections/ce/gaussian-prior/outdir_gw_tidal_cosmo_dl_3000_CE_incl_45/result/gw_tidal_cosmo_dl_3000_CE_incl_45_0_result.json'
filename9 = \
    '/home/deep/work/campus-cluster-runs/pbilby-runs/redshift-injections/ce/gaussian-prior/outdir_gw_tidal_cosmo_dl_4000_CE_incl_45/result/gw_tidal_cosmo_dl_4000_CE_incl_45_0_result.json'
filename10 = \
    '/home/deep/work/campus-cluster-runs/pbilby-runs/redshift-injections/ce/gaussian-prior/outdir_gw_tidal_cosmo_dl_5000_CE_incl_45/result/gw_tidal_cosmo_dl_5000_CE_incl_45_0_result.json'

result_1000 = bilby.result.read_in_result(filename=filename1)
result_2000 = bilby.result.read_in_result(filename=filename2)
result_3000 = bilby.result.read_in_result(filename=filename3)
result_4000 = bilby.result.read_in_result(filename=filename4)
result_5000 = bilby.result.read_in_result(filename=filename5)

result_1000_no_sys = bilby.result.read_in_result(filename=filename6)
result_2000_no_sys = bilby.result.read_in_result(filename=filename7)
result_3000_no_sys = bilby.result.read_in_result(filename=filename8)
result_4000_no_sys = bilby.result.read_in_result(filename=filename9)
result_5000_no_sys = bilby.result.read_in_result(filename=filename10)

add_hubble(result_2000)
add_hubble(result_2000_no_sys)

params = ['luminosity_distance',  'z']
truth = {p: result_2000_no_sys.injection_parameters[p] for p in params}
truth['hubble'] = 70

fig = bilby.result.plot_multiple([
    result_2000, result_2000_no_sys,],
    truth=truth, quantiles=(0.05, 0.95), levels=(0.05, 0.95),
    legends=('$\delta(\\bar{\lambda} - 230)$', '$\delta(\\bar{\lambda} - 200)$'),
    corner_labels=["$D_L$", "$z$", "$H_0$"],
    range=[(1000, 3000), (0.25, 0.5), (40, 150)]
)
axes = fig.get_axes()

dl_lim = axes[0].get_xlim()
z_lim = axes[4].get_xlim()
hubble_lim = axes[8].get_xlim()
axes[3].set_xlim(dl_lim)
axes[3].set_ylim(z_lim)
axes[6].set_xlim(dl_lim)
axes[6].set_ylim(hubble_lim)
axes[7].set_xlim(z_lim)
axes[7].set_ylim(hubble_lim)
plt.savefig('../figures/sys-err-h0.pdf')