import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import matplotlib.ticker as tkr
import pickle
import math
import matplotlib.ticker
formatter = tkr.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
plt.style.use('~/Projects/Tidal-Cosmology/gw-tidal-cosmology-docs/scripts/publication.mplstyle')

with open('/home/abhi/Projects/pbilby_runs/dl-iota-kde-O8.pickle', 'rb') as f:
    kde = pickle.load(f)
    
dist_vals = np.linspace(100, 2000, 100)
incl_vals = np.linspace(0, np.pi, 20)

X, Y = np.meshgrid(dist_vals, incl_vals)
positions = np.vstack([X.ravel(), Y.ravel()])
Z = np.reshape(kde(positions).T, X.shape)
fig, ax = plt.subplots(1, 1)
cf=ax.contourf(X, Y, Z, cmap='bone')  
ax.set_xlabel('Distance (Mpc)',fontsize=16) 
ax.set_ylabel('Inclination (Rad.)',fontsize=16) 
#plt.contourf(X, Y, Z, cmap='bone')
#plt.colorbar()
#plt.xlabel('Distance (Mpc)')
#plt.ylabel('Inclination (Rad.)')
plt.ylim((0, 1.57))
plt.scatter([250,250,250,250],[0.35,0.65,1,1.4],s=50,c='r')
plt.scatter([450,450,450,450,450],[0.3,0.6,0.9,1.2,1.5],s=50,c='r')
plt.scatter([650,650,650,650,650],[0.15,0.45,0.75,1.05,1.35],s=50,c='r')
plt.scatter([850,850,850,850],[0.1,0.4,0.7,1],s=50,c='r')
plt.scatter([1050,1050,1050,1050],[0.2,0.5,0.8,1.1],s=50,c='r')
plt.scatter([1250,1250,1250,1250],[0.1,0.4,0.7,1],s=50,c='r')
plt.scatter([1450,1450,1450],[0.1,0.4,0.7],s=50,c='r')
plt.scatter([1650,1650,1650],[0.05,0.35,0.65],s=50,c='r')
fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
fmt.set_powerlimits((0, 0))
fig.colorbar(cf,ax=ax,format=fmt)
plt.grid(True)
plt.savefig('/home/abhi/Desktop/Plots/p-det-heatmap-O8.pdf')
plt.show()
