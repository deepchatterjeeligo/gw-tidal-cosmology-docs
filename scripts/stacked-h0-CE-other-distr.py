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
from scipy.integrate import quad, cumtrapz

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
    result.posterior['hubble'] = hubble_vals


def add_reweighted_hubble(row, num_prior_points=10000,
                          bin_start=-10, bin_step=20,
                          num_bins=15, plot=True):
    result = row.bilby_result
    hubble_posterior = result.posterior.hubble.values

    z_prior = result.priors['z'].sample(num_prior_points)
    distance_prior = result.priors['luminosity_distance'].sample(num_prior_points)

    hubble_prior = np.array([quad(integrand, 0, z_vals)[0] for z_vals in z_prior])

    hubble_prior /= distance_prior
    hubble_prior *= (1 + z_prior)
    hubble_prior *= (const.c.to('km/s')/u.Mpc).value

    # cut the prior at max value of the bin
    bins = [bin_start + n*bin_step for n in range(num_bins)]
    accept_mask = hubble_prior < (max(bins) - bin_step)

    z_prior = z_prior[accept_mask]
    distance_prior = distance_prior[accept_mask]
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

    # print(hubble_prior[np.where(np.isnan(wts))])
    # print(hubble_prior[np.where(np.isinf(wts))])

    posterior_wts = 1./weights_interpolant(hubble_posterior)
    posterior_wts /= np.sum(posterior_wts)

    # print("Bad posterior values")
    # print(hubble_posterior[np.where(np.isnan(posterior_wts))])
    # print(hubble_posterior[np.where(np.isinf(posterior_wts))])

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
        plt.legend()
        plt.show()
    return hubble_posterior, posterior_wts


filenames = glob.glob(
    '/home/deep/work/campus-cluster-runs/pbilby-runs/redshift-injections/ce/'
    'outdir_gw_tidal_cosmo_dl_[1234567]000_CE_incl_[13467][0|5]/result/*.json'
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
    add_hubble(r.bilby_result)

# attach posterior for hubble
results_master['hubble_kde'] = [
    gaussian_kde(r.bilby_result.posterior.hubble.values)
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
with open('/home/deep/github/tidal-cosmology-selections/dl-iota-kde-md.pickle', 'rb') as f:
    kde_md = pickle.load(f)

with open('/home/deep/github/tidal-cosmology-selections/dl-iota-kde-comoving.pickle', 'rb') as f:
    kde_comoving = pickle.load(f)

pop_wts_md = [kde_md([r.distance, r.inclination/180*np.pi])[0] for idx, r in results_master.iterrows()]
pop_wts_comoving = [kde_comoving([r.distance, r.inclination/180*np.pi])[0] for idx, r in results_master.iterrows()]

results_master['pop_wt_md'] = np.array(pop_wts_md)
results_master['pop_wt_comoving'] = np.array(pop_wts_comoving)

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

dist_vals = np.linspace(100, 8000, 200)
incl_vals = np.linspace(0, np.pi, 80)

X, Y = np.meshgrid(dist_vals, incl_vals)
positions = np.vstack([X.ravel(), Y.ravel()])
Z_md = np.reshape(kde_md(positions).T, X.shape)
Z_comoving = np.reshape(kde_comoving(positions).T, X.shape)

plt.figure(figsize=(8, 5))
plt.contourf(
    X, Y, Z_md, cmap='bone', linestyles='solid',
    levels=(1e-7, 1e-6, 4e-6, 1e-5, 2e-5, 3e-5, 5e-5, 8e-5, 1e-4, 2e-4)
)
c_md = plt.colorbar(orientation='vertical')
plt.contour(X, Y, Z_comoving, cmap='hot_r', linestyles='solid', linewidths=2,
            levels=(1e-7, 1e-6, 4e-6, 1e-5, 2e-5, 3e-5, 5e-5, 8e-5, 1e-4, 2e-4))
c_comoving = plt.colorbar(orientation='vertical')
c_md.set_label("S.F.H.")
c_comoving.set_label("Unif. Com.")
plt.xlabel(r'Distance (Mpc)')
plt.ylabel(r'Inclination (Rad.)')
plt.ylim((0, 1.57))
plt.scatter(
    results_master.distance, results_master.inclination/180*np.pi,
    s=50,
    c='r',
)
plt.savefig('../figures/p-det-heatmap-ce-other-distr.pdf')
# plt.show()
plt.close()

num_events = 70
relative_pop_wts_md = num_events * (
    results_master.pop_wt_md/np.sum(results_master.pop_wt_md))
relative_pop_wts_comoving = num_events * (
    results_master.pop_wt_comoving/np.sum(results_master.pop_wt_comoving))

# Stacked H0
stacked_h0_md = lambda x: reduce(
    mul, [
        k(x)**pop_wt
        for k, pop_wt in zip(results_master.reweighted_hubble_kde.values,
                             relative_pop_wts_md)
    ]
)
stacked_h0_comoving = lambda x: reduce(
    mul, [
        k(x)**pop_wt
        for k, pop_wt in zip(results_master.reweighted_hubble_kde.values,
                             relative_pop_wts_comoving)
    ]
)
norm_md = quad(stacked_h0_md, 40, 100)[0]
norm_stacked_h0_md = lambda x: stacked_h0_md(x)/norm_md
norm_comoving = quad(stacked_h0_comoving, 40, 100)[0]
norm_stacked_h0_comoving = lambda x: stacked_h0_comoving(x)/norm_comoving

h0_vals_md = np.linspace(50, 90, num=1000)
p_h0_vals_md = norm_stacked_h0_md(h0_vals_md)
h0_vals_comoving = np.linspace(50, 90, num=1000)
p_h0_vals_comoving = norm_stacked_h0_comoving(h0_vals_comoving)

h0_cdf_md = cumtrapz(p_h0_vals_md, h0_vals_md)
h0_vals_md = 0.5*(h0_vals_md[1:] + h0_vals_md[:-1])
five_md, ninety_five_md = h0_vals_md[np.argmax(h0_cdf_md > 0.05)], h0_vals_md[np.argmax(h0_cdf_md > 0.95)]
print("Madau-Dickinson population")
print("5 percent/ 95 percent/ confidence interval = {:.2f}/{:.2f}/{:.2f}".format(
    five_md, ninety_five_md, ninety_five_md - five_md
))
print(f"With 1/sqrt(N) scaling, width for 1000 events = {(ninety_five_md - five_md)*(num_events/1000)**0.5:.2f}")

h0_cdf_comoving = cumtrapz(p_h0_vals_comoving, h0_vals_comoving)
h0_vals_comoving = 0.5*(h0_vals_comoving[1:] + h0_vals_comoving[:-1])
five_comoving, ninety_five_comoving = h0_vals_comoving[
    np.argmax(h0_cdf_comoving > 0.05)], h0_vals_comoving[np.argmax(h0_cdf_comoving > 0.95)]
print("Uniform in comoving volume population")
print("5 percent/ 95 percent/ confidence interval = {:.2f}/{:.2f}/{:.2f}".format(
    five_comoving, ninety_five_comoving, ninety_five_comoving - five_comoving
))
print(f"With 1/sqrt(N) scaling, width for 1000 events = {(ninety_five_comoving - five_comoving)*(num_events/1000)**0.5:.2f}")

plt.figure()
h0_vals = np.linspace(20, 150, 150)
colors = {'15.0': 'y', '20.0':'y', '30.0': 'y', '37.5': 'y', '45.0': 'm', '60.0': 'm', '75.0': 'm'}

for idx, r in results_master.iterrows():
    plt.plot(h0_vals, r.reweighted_hubble_kde(h0_vals),
             color='black',
             linestyle='dotted')

plt.plot(h0_vals_md, norm_stacked_h0_md(h0_vals_md), linewidth=3,
         color='C0', label='S.F.H.')
plt.plot(h0_vals_comoving, norm_stacked_h0_comoving(h0_vals_comoving),
         linewidth=2, linestyle='dashed', color='C1', label='Unif. Com.')
plt.legend()
plt.xlim((20, 150))
plt.axvline(x=70, c='r')
plt.xlabel('$H_0$ (km s$^{-1}/$Mpc)')
plt.ylabel('$p(H_0)$')
# plt.show()
plt.savefig('../figures/stacked-h0-ce-other-distr.pdf')
