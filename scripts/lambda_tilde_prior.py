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
bins = [0 + 100*n for n in range(25)]
plt.hist(lambda_tilde_prior, bins=bins, density=True)
plt.hist(lambda_0_0_prior, density=True, bins=bins, lw=3, histtype='step')
# plt.hist(lambda1_prior, density=True, label='$\\bar{\lambda}_1$', facecolor='none',
#          edgecolor='black', hatch='/', alpha=0.5, bins=bins)
# plt.hist(lambda2_prior, density=True, label='$\\bar{\lambda}_2$', bins=bins,
#          histtype='stepfilled', alpha=0.5)
# plt.legend()
# plt.show()
plt.xlabel("$\\bar{\Lambda}$ Prior")
# plt.xlabel("$\\bar{\lambda}_{1,2}$ Prior")
# plt.savefig('../figures/appendix-lambda-tilde-prior.pdf')
plt.savefig('../figures/appendix-lambda-12-prior.pdf')