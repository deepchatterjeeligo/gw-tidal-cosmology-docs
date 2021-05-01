import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import matplotlib.ticker as tkr
import bilby
import corner
from astropy.coordinates import SkyCoord
import glob
import math
import pickle
plt.style.use('publication.mplstyle')


with open('/home/abhi/Projects/pbilby_runs/dl-iota-kde-O8.pickle', 'rb') as f:
    dist_population = pickle.load(f)
def h0samples(data):
    return data.posterior['h0']

datafiles = sorted(glob.glob('/home/abhi/Projects/pbilby_runs/Voyager-Runs/Runs-to-Redo/Data-Cluster/json_files_completed_runs/*.json',  
                   recursive = True))
datasamples = list(bilby.result.read_in_result(filename) for filename in datafiles)

massratiodat = np.array([sample.posterior['mass_ratio'] for sample in datasamples])
dlpost = np.array([sample.posterior['luminosity_distance'] for sample in datasamples])
iotapost = np.array([sample.posterior['theta_jn'] for sample in datasamples])

def colorcode(iota):
    if iota<0.7 :
        return "orange"
    else :
        return "purple"


dlvalsinj = np.array([sample.injection_parameters['luminosity_distance'] for sample in datasamples])
iotavalsinj = np.array([sample.injection_parameters['theta_jn'] for sample in datasamples])


h0data = list(h0samples(dat) for dat in datasamples)


def meannetsnr(dat):
    labels =list(['A1_optimal_snr','H1_optimal_snr','K1_optimal_snr',
                 'L1_optimal_snr','V1_optimal_snr']) 
    snr = np.array([np.mean(dat.posterior[label]) for label in labels])
    return np.round((np.sum(snr*snr))**(1/2))

def wvals(dl,iota):
    widthy = 0.2 if dl == 250 else 0.15
    widthx = 100
    return dist_population.integrate_box([dl-100,iota-widthy], [dl+100,iota+widthy])
    
#Plotting The Stacked Posterior
weights = np.array([wvals(dlvalsinj[j],iotavalsinj[j]) for j in range(len(dlvalsinj))])
weights = (50/np.sum(weights))*weights

weights1 = np.array([dist_population([dlvalsinj[j],iotavalsinj[j]])[0] for j in range(len(dlvalsinj))])

def disth0(data):
    dist = sp.stats.gaussian_kde(data)
    return dist



distsforstacking = list(disth0(sample)(20) for sample in h0data)

def stacked(xx):
    values = np.array([(((disth0(h0data[j])(xx)))**((weights[j]))) for j in range(len(h0data))])
    return np.prod(values)

x = np.linspace(0,300,550)
y = np.array([stacked(j) for j in x])

norm = sp.integrate.quad(stacked,0,300)[0]




fig,ax = plt.subplots()
ax.set_xlim(0, 150)
for j in range(len(h0data)):
    sns.kdeplot(data=h0data[j],ax=ax,linestyle="dotted",color='black',linewidth=1)
    #sns.kdeplot(data=massratiodat[j],ax=ax[1],linestyle="--",color = colorcode(iotavalsinj[j]))
ax.set_xlabel('$H_0$')
ax.set_ylabel('$p(H_0)$')
ax.axvline(x=70,color='r')    
ax.plot(x,(1/norm)*y,linewidth=3)
ax.axvline(x=70,c='r')
plt.xlim((20, 150))
plt.xlabel('$H_0$ (km s$^{-1}/$Mpc)')
plt.ylabel('$p(H_0)$')
#ax[1].axvline(x=0.9,c='r')
#plt.title('50 Events',fontsize=20)
#plt.savefig('/home/abhi/Desktop/Plots/stacked-h0-O8-v1.pdf')
plt.show()