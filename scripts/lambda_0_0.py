import numpy as np
import scipy as sp
import bilby

import matplotlib.pyplot as plt
plt.style.use('publication.mplstyle')

import t_cosmo

r_sys_err = bilby.result.read_in_result('../data/GW170817/GW170817_syserr_result.json')
r_no_sys_err = bilby.result.read_in_result('../data/GW170817/lambda-00-gw170817/lambda-00-gw170817_result.json')

p = sp.stats.gaussian_kde(r_no_sys_err.posterior.lambda_0_0)
q = sp.stats.gaussian_kde(r_sys_err.posterior.lambda_0_0)
lambda_vals = np.linspace(1, 1000, 10000)
p_vals = p(lambda_vals)
q_vals = q(lambda_vals)
kl_div_1 = sp.stats.entropy(p_vals, qk=q_vals)
kl_div_2 = sp.stats.entropy(q_vals, qk=p_vals)
print(f"KL divergences: {kl_div_1:.3f}/{kl_div_2:.3f}")

fig_ax = r_no_sys_err.plot_single_density(
    key='lambda_0_0',
    quantiles=(0.05, 0.95),
    prior=r_no_sys_err.priors['lambda_0_0'],
    bins=50,
    label_fontsize=18,
    save=False,
    histtype='stepfilled', color='C1',
    alpha=0.7, label="w/o Residuals: "
)
fig_ax = r_sys_err.plot_single_density(
    key='lambda_0_0',
    quantiles=(0.05, 0.95),
    prior=None, bins=50,
    label_fontsize=18, save=True,
    histtype='step', hist_lw=2,
    color='C0', label="Inc. Residuals: ",
    fig_ax=fig_ax, file_base_name="../figures/"
)
plt.show()

fig = plt.figure(figsize=(10, 10))
plt.figure(figsize=(10, 10))
_ = bilby.result.plot_multiple(
    [r_no_sys_err, r_sys_err],
    parameters=[
        'mass_1', 'mass_2', 'a_1', 'a_2',
        'luminosity_distance', 'theta_jn', 'lambda_0_0'],
    quantiles=(0.1, 0.9), levels=(0.5, 0.9), save=True,
    legends=('Using Fit', 'Fit+Residuals'),
    filename='../figures/corner-plots.pdf'
)
