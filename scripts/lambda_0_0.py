import bilby

import matplotlib.pyplot as plt
plt.style.use('publication.mplstyle')

r_sys_err = bilby.result.read_in_result('../data/GW170817/GW170817_syserr_result.json')
r_no_sys_err = bilby.result.read_in_result('../data/GW170817/GW170817_no_sys_error.json')

r_sys_err.plot_marginals(
    parameters=['lambda_0_0'],
    quantiles=(0.05, 0.95),
    priors=True, bins=50,
    label_fontsize=18
)
plt.show()

fig = plt.figure(figsize=(10, 10))
#plt.figure(figsize=(10, 10))
#_ = bilby.result.plot_multiple(
#    [r_sys_err, r_no_sys_err],
#    parameters=[
#        'mass_1', 'mass_2', 'a_1', 'a_2',
#        'luminosity_distance', 'theta_jn', 'lambda_0_0'],
#    quantiles=(0.1, 0.9), levels=(0.5, 0.9), save=False,
#)
r_sys_err.plot_corner(
    parameters=[
        'mass_1', 'mass_2', 'luminosity_distance',
        'theta_jn', 'lambda_0_0'],
    quantiles=(0.05, 0.95), levels=(0.5, 0.9), save=True,
    filename='../figures/corner-plots.pdf',
    fig=fig
)

