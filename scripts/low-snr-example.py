import glob
import json
import pickle
import os
import re

from astropy import units as u, constants as const
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import gaussian_kde
from scipy.integrate import quad

import bilby
plt.style.use('publication.mplstyle')

filenames = glob.glob(
    '/home/deep/work/campus-cluster-runs/pbilby-runs/redshift-injections/O5/'
    'low-snr-injections/outdir_gw_tidal_cosmo_dl_2000_O5_incl_45/result/*.json'
)
filenames = sorted(filenames)

pattern = re.compile('.*_dl_([0-9]+)_.*_incl_([0-9]+).*')
results_master = pd.DataFrame(
    data=[(*re.match(pattern, os.path.basename(f)).groups(), bilby.result.read_in_result(filename=f))
          for f in filenames],
    columns=('distance', 'inclination', 'bilby_result')
)
results_master.distance = results_master.distance.astype(float)
results_master.inclination = results_master.inclination.astype(float)


omega_m = 0.3
omega_lambda = 0.7
integrand = lambda z: 1./np.sqrt(omega_m * (1 + z)**3 + omega_lambda)

def add_hubble(result):
    hubble_vals = np.array([quad(integrand, 0, z_vals)[0] for z_vals in result.posterior.z.values])
    hubble_vals /= result.posterior.luminosity_distance.values
    hubble_vals *= (1 + result.posterior.z.values)
    hubble_vals *= (const.c.to('km/s')/u.Mpc).value
    result.posterior['h0'] = hubble_vals


for idx, r in results_master.iterrows():
    if hasattr(r.bilby_result.posterior, 'z'):
        add_hubble(r.bilby_result)


def add_reweighted_hubble(row, num_prior_points=10000,
                          bin_start=-10, bin_step=20,
                          num_bins=15, plot=True):
    result = row.bilby_result
    hubble_posterior = result.posterior.h0.values
    try:
        hubble_prior = result.priors['h0'].sample(num_prior_points)
    except KeyError:
        z_prior = result.priors['z'].sample(num_prior_points)
        distance_prior = result.priors['luminosity_distance'].sample(num_prior_points)
        hubble_prior = np.array([quad(integrand, 0, z_vals)[0] for z_vals in z_prior])
        hubble_prior /= distance_prior
        hubble_prior *= (1 + z_prior)
        hubble_prior *= (const.c.to('km/s')/u.Mpc).value

    # cut the prior at max value of the bin
    bins = [bin_start + n*bin_step for n in range(num_bins)]
    accept_mask = hubble_prior < (max(bins) - bin_step)

    #distance_prior = distance_prior[accept_mask]
    hubble_prior = hubble_prior[accept_mask]
    # cut posterior values at 99 percentile
    hubble_posterior = hubble_posterior[
        hubble_posterior < np.percentile(
            hubble_posterior, 99.4
        )
    ]
    counts, bb = np.histogram(hubble_prior, bins=bins, density=True)
    bin_centers = 0.5*(bb[1:] + bb[:-1])

    weights_interpolant  = sp.interpolate.interp1d(
        bin_centers, counts, kind='cubic',
        fill_value=(0,  0), bounds_error=False
    )

    wts = 1./weights_interpolant(hubble_prior)
    wts /= np.sum(wts)

    print(hubble_prior[np.where(np.isnan(wts))])
    print(hubble_prior[np.where(np.isinf(wts))])

    posterior_wts = 1./weights_interpolant(hubble_posterior)
    posterior_wts /= np.sum(posterior_wts)

    print("Bad posterior values")
    print(hubble_posterior[np.where(np.isnan(posterior_wts))])
    print(hubble_posterior[np.where(np.isinf(posterior_wts))])

    if plot:
        _, _b, _p = plt.hist(hubble_posterior, bins=15, density=True, alpha=0.5,
                             label='Orig. Posterior')
        _, _b, _p = plt.hist(hubble_prior, histtype='step', lw=3,
                             bins=_b, density=True, alpha=0.5, label='Orig. Prior')
        _, _b, _p = plt.hist(hubble_posterior, weights=posterior_wts, bins=_b,
                             density=True, facecolor='none', edgecolor='black',
                             hatch='/', alpha=0.5, label='Reweighted Posterior')
        _, _b, _p = plt.hist(hubble_prior, histtype='step', lw=3, linestyle='dashed',
                             weights=wts, bins=_b, density=True, alpha=0.5,
                             label='Reweighted Prior')
        
        reweighted_kde = sp.stats.gaussian_kde(
            hubble_posterior, weights=posterior_wts
        )

        fiducial_hubble_vals = np.linspace(0, 150, 1000)
        reweighted_kde_hubble_vals = reweighted_kde(fiducial_hubble_vals)
        #plt.plot(fiducial_hubble_vals, reweighted_kde_hubble_vals)
        #plt.title(f"Inclination: {row.inclination:.1f}, Distance: {row.distance}")
        plt.legend()
        plt.xlabel("$H_0$")
        plt.ylabel("$p(H_0)$")
        plt.show()
    return hubble_posterior, posterior_wts


posteriors = []
weights = []
for idx, row in results_master.iterrows():
    print(row.distance, row.inclination)
    posterior, weight = add_reweighted_hubble(
        row, bin_start=-5, num_prior_points=5000,
        bin_step=10, num_bins=16
    )
    posteriors.append(posterior)
    weights.append(weight)

results_master['hubble_reweighting'] = posteriors
results_master['weights'] = weights

results_master['reweighted_hubble_kde'] = [
    gaussian_kde(
        r.hubble_reweighting,
        weights=r.weights
    )
    for idx, r in results_master.iterrows()
]
