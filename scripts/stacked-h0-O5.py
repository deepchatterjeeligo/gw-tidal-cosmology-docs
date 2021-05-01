import glob
import json
import pickle
import os
import re
from functools import reduce
from operator import mul


from astropy import units as u, constants as const
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import gaussian_kde
from scipy.integrate import quad

import bilby

plt.style.use('publication.mplstyle')

# luminosity distance integral
omega_m = 0.3
omega_lambda = 0.7
integrand = lambda z: 1./np.sqrt(omega_m * (1 + z)**3 + omega_lambda)

def add_hubble(result):
    hubble_vals = np.array(
        [quad(integrand, 0, z_vals)[0] for z_vals in result.posterior.z.values]
    )
    hubble_vals /= result.posterior.luminosity_distance.values
    hubble_vals *= (1 + result.posterior.z.values)
    hubble_vals *= (const.c.to('km/s')/u.Mpc).value
    result.posterior['h0'] = hubble_vals


def add_reweighted_hubble(row, num_prior_points=10000,
                          bin_start=-10, bin_step=20,
                          num_bins=15, plot=False):
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
            hubble_posterior, 99
        )
    ]

    counts, bb = np.histogram(hubble_prior, bins=bins, density=True)
    bin_centers = 0.5*(bb[1:] + bb[:-1])

    weights_interpolant  = sp.interpolate.interp1d(
        bin_centers, counts, kind='cubic',
        fill_value=(0.,  0.), bounds_error=False
    )

    wts = 1./weights_interpolant(hubble_prior)
    wts /= np.sum(wts)

    #print(hubble_prior[np.where(np.isnan(wts))])
    #print(hubble_prior[np.where(np.isinf(wts))])

    posterior_wts = 1./weights_interpolant(hubble_posterior)
    posterior_wts /= np.sum(posterior_wts)

    #print("Bad posterior values")
    #print(hubble_posterior[np.where(np.isnan(posterior_wts))])
    #print(hubble_posterior[np.where(np.isinf(posterior_wts))])

    if plot:
        _, _b, _p = plt.hist(hubble_posterior, weights=posterior_wts,
                             bins=20, density=True, alpha=0.5, label='reweighted')
        _, _b, _p = plt.hist(hubble_posterior,
                             bins=_b, density=True, alpha=0.5, label='original')
        reweighted_kde = sp.stats.gaussian_kde(
            hubble_posterior, weights=posterior_wts
        )

        fiducial_hubble_vals = np.linspace(0, 200, 100)
        reweighted_kde_hubble_vals = reweighted_kde(fiducial_hubble_vals)
        plt.plot(fiducial_hubble_vals, reweighted_kde_hubble_vals, label='KDE')
        plt.title(f"Inclination: {row.inclination:.1f}, Distance: {row.distance}")
        plt.legend()
        plt.show()
    return hubble_posterior, posterior_wts


filenames = glob.glob(
    '/home/deep/work/campus-cluster-runs/pbilby-runs/redshift-injections/O5/'
    'outdir_gw_tidal_cosmo_dl_[1234567]00_CE_incl_[13467][0|5]/result/*.json'
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

for idx, r in results_master.iterrows():
    if hasattr(r.bilby_result.posterior, 'z'):
        add_hubble(r.bilby_result)

# attach posterior for hubble
results_master['hubble_kde'] = [
    gaussian_kde(r.bilby_result.posterior.h0.values)
    for idx, r in results_master.iterrows()
]
# attach re-weighted hubble
posteriors = []
weights = []
for idx, row in results_master.iterrows():
    posterior, weight = add_reweighted_hubble(
        row, bin_start=-10, bin_step=20,
        num_bins=16, plot=False
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
# load population weights
with open('/home/deep/github/gw-tidal-cosmology-docs/data/O5/dl-iota-kde.pickle-O5', 'rb') as f:
    kde = pickle.load(f)

dist_vals = np.linspace(50, 800, 100)
incl_vals = np.linspace(0, np.pi, 20)

X, Y = np.meshgrid(dist_vals, incl_vals)
positions = np.vstack([X.ravel(), Y.ravel()])
Z = np.reshape(kde(positions).T, X.shape)

plt.figure()
plt.contourf(X, Y, Z, cmap='bone')
plt.colorbar()
plt.xlabel(r'Distance (Mpc)')
plt.ylabel(r'Inclination (Rad.)')
plt.ylim((0, 1.57))
plt.scatter(
    results_master.distance, results_master.inclination/180*np.pi,
    s=50,
    c='r',

)
plt.savefig('../figures/p-det-heatmap-O5.pdf')
plt.close()

pop_wts = [kde([r.distance, r.inclination/180*np.pi])[0] for idx, r in results_master.iterrows()]

results_master['pop_wt'] = np.array(pop_wts)

prior_wts_incl = np.array(
    [np.sin(r.inclination/180*np.pi) for idx, r in results_master.iterrows()]
)
prior_wts_incl /= prior_wts_incl.sum()

results_master['prior_wt_incl'] = prior_wts_incl

prior_wts_dist = np.array([r.distance**2 for idx, r in results_master.iterrows()])
prior_wts_dist /= prior_wts_dist.sum()

results_master['prior_wt_dist'] = prior_wts_dist

# net-prior weighting
prior_wt = prior_wts_dist*prior_wts_incl
prior_wt /= prior_wt.sum()
results_master['prior_wt'] = prior_wt

relative_pop_wts = 10*(results_master.pop_wt/np.sum(results_master.pop_wt))

# Stacked H0
stacked_h0 = lambda x: reduce(
    mul, [
        (prior_wt * k(x))**pop_wt
        for k, prior_wt, pop_wt in zip(results_master.reweighted_hubble_kde.values,
                                       results_master.prior_wt.values,
                                       relative_pop_wts)
    ]
)
norm = quad(stacked_h0, 0, 300)[0]
norm_stacked_h0 = lambda x: stacked_h0(x)/norm

plt.figure()
h0_vals = np.linspace(20, 150, 150)

for idx, r in results_master.iterrows():
    plt.plot(h0_vals, r.reweighted_hubble_kde(h0_vals),
             label=f'DL = {r.distance:.1e}; iota = {r.inclination:.1e}',
             color='black',
             linestyle='dotted')


plt.plot(h0_vals, norm_stacked_h0(h0_vals), linewidth=3)
plt.xlim((20, 150))
plt.axvline(x=70, c='r')
plt.xlabel('$H_0$ (km s$^{-1}/$Mpc)')
plt.ylabel('$p(H_0)$')
#plt.show()
plt.savefig('../figures/stacked-h0-O5.pdf')
