import matplotlib.pyplot as plt

import numpy as np
from t_cosmo import lambda_k_relations
import bilby

plt.style.use('publication.mplstyle')

r_no_sys_err = bilby.result.read_in_result('../data/GW170817/lambda-00-gw170817/lambda-00-gw170817_result.json')

mass1_prior_source = r_no_sys_err.priors['mass_1'].sample(10000)/(1 + 0.0099)
mass2_prior_source = r_no_sys_err.priors['mass_2'].sample(10000)/(1 + 0.0099)
lambda_0_0_prior = r_no_sys_err.priors['lambda_0_0'].sample(10000)

lambda1_prior = lambda_k_relations.get_lambda_from_mass(mass1_prior_source, lambda_0_0_prior, M0=1.4)
lambda2_prior = lambda_k_relations.get_lambda_from_mass(mass2_prior_source, lambda_0_0_prior, M0=1.4)

finite_mask = np.isfinite(lambda1_prior) * np.isfinite(lambda2_prior)

lambda1_prior = lambda1_prior[finite_mask]
lambda2_prior = lambda2_prior[finite_mask]
mass1_prior_source = mass1_prior_source[finite_mask]
mass2_prior_source = mass2_prior_source[finite_mask]

lambda_tilde_prior = bilby.gw.conversion.lambda_1_lambda_2_to_lambda_tilde(
    lambda1_prior, lambda2_prior,
    mass1_prior_source, mass2_prior_source
)

plt.hist2d(
    lambda1_prior,
    lambda2_prior,
    cmap='cividis_r',
    density=True,
    bins=15,
    range=((0, 2000),(0, 2000)),
)
plt.xlabel("$\\bar{\lambda}_1$")
plt.ylabel("$\\bar{\lambda}_2$")
plt.colorbar()
plt.savefig('../figures/appendix-lambda-12-prior-using-lambda00.pdf')

# bins = [0 + 100*n for n in range(25)]
# plt.hist(lambda_tilde_prior, bins=bins, density=True)
# plt.hist(lambda_0_0_prior, density=True, bins=bins, lw=3, histtype='step')
# plt.hist(lambda1_prior, density=True, label='$\\bar{\lambda}_1$', facecolor='none',
#          edgecolor='black', hatch='/', alpha=0.5, bins=bins)
# plt.hist(lambda2_prior, density=True, label='$\\bar{\lambda}_2$', bins=bins,
#          histtype='stepfilled', alpha=0.5)
# plt.legend()
# plt.show()
# plt.xlabel("$\\bar{\Lambda}$ Prior")
# plt.xlabel("$\\bar{\lambda}_{1,2}$ Prior")
# plt.savefig('../figures/appendix-lambda-tilde-prior.pdf')
# plt.savefig('../figures/appendix-lambda-12-prior.pdf')


# compare with LVK EoS insensitive results
a = 0.07550
b = [[-2.235, 0.8474], [10.45, -3.251], [-15.70, 13.61]]
c = [[-2.048, 0.5976], [7.941, 0.5658], [-7.360, -1.320]]

n = 0.743

f_n_q = lambda q: (1 - q**(10/(3 - n)))/(1 + q**(10/(3 - n)))
def get_lambda_asym_from_sym(lambda_sym, q):
    num = 0
    den = 0
    for i in range(3):
        for j in range(2):
            num += b[i][j] * q**(j+1) * lambda_sym**(-(i + 1)/5)
            den += c[i][j] * q**(j+1) * lambda_sym**(-(i + 1)/5)
    return f_n_q(q) * lambda_sym * num / den

qs = mass2_prior_source/mass1_prior_source
lambda_syms = np.random.uniform(0, 2000, len(qs))
lambda_asyms =  np.array(
    [get_lambda_asym_from_sym(lambda_sym, q)
     for lambda_sym, q in zip(lambda_syms, qs)]
)

lambda_1s = lambda_syms + lambda_asyms
lambda_2s = lambda_syms - lambda_asyms

mask = lambda_1s > lambda_2s

lambda_1s = lambda_1s[mask]
lambda_2s = lambda_2s[mask]
# note the difference in convention used by YY17 and LVK
# Lambda_1 for YY17 is the larger one
plt.close('all')
plt.figure()
plt.hist2d(
    lambda_2s,
    lambda_1s,
    cmap='cividis_r',
    density=True,
    bins=15,
    range=((0, 2000),(0, 2000)),
)
plt.xlabel("$\\bar{\lambda}_1$")
plt.ylabel("$\\bar{\lambda}_2$")
plt.colorbar()
plt.savefig('../figures/appendix-lambda-12-prior-using-lambdasymm.pdf')