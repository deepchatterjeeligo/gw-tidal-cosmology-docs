import glob
import json
import pickle
import os
import re

from astropy import units as u, constants as const
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad

import bilby

plt.style.use('publication.mplstyle')

filenames = glob.glob(
    '/home/deep/work/campus-cluster-runs/pbilby-runs/redshift-injections/'
    'ce/outdir_gw_tidal_cosmo_dl_[1234567]000_CE_incl_[13467][0|5]/result/*.json'
)
filenames = sorted(filenames)

pattern = re.compile('.*_dl_([0-9]+)_.*_incl_([0-9]+).*')
results_master = pd.DataFrame(
    data=[
        (*re.match(pattern, os.path.basename(f)).groups(),
         bilby.result.read_in_result(filename=f)) for f in filenames],
    columns=('distance', 'inclination', 'bilby_result')
)
results_master.distance = results_master.distance.astype(float)
results_master.inclination = results_master.inclination.astype(float)

omega_m = 0.3
omega_lambda = 0.7
integrand = lambda z: 1./np.sqrt(omega_m * (1 + z)**3 + omega_lambda)

def add_hubble(result):
    hubble_vals = np.array([
        quad(integrand, 0, z_vals)[0] for z_vals in result.posterior.z.values
    ])
    hubble_vals /= result.posterior.luminosity_distance.values
    hubble_vals *= (1 + result.posterior.z.values)
    hubble_vals *= (const.c.to('km/s')/u.Mpc).value
    result.posterior['hubble'] = hubble_vals

for idx, r in results_master.iterrows():
    add_hubble(r.bilby_result)

fig = bilby.result.plot_multiple(
    results_master.loc[
        (results_master.distance==3000) &
        (results_master.inclination != 75)
    ].bilby_result.values,
    quantiles=(0.05, 0.95), levels=(0.5, 0.9), bins=20,
    save=False,
    legends=('$\iota = 15^{\circ}$', '$\iota = 30^{\circ}$',
             '$\iota = 45^{\circ}$', '$\iota = 60^{\circ}$'),
    truth={'luminosity_distance': 3000, 'z': 0.52441811062841, 'hubble': 70},
    titles=False, plot_datapoints=True,
    corner_labels=["$D_L$", "$z$", "$H_0$"],
    range=[(500, 6000), (0.35, 0.65), (20, 180)],
)
axes = fig.get_axes()

# very hacky
dl_lim = axes[0].get_xlim()
z_lim = axes[4].get_xlim()
hubble_lim = axes[8].get_xlim()
axes[3].set_xlim(dl_lim)
axes[3].set_ylim(z_lim)
axes[6].set_xlim(dl_lim)
axes[6].set_ylim(hubble_lim)
axes[7].set_xlim(z_lim)
axes[7].set_ylim(hubble_lim)

plt.savefig('../figures/appendix-z-hubble-distance.pdf')
