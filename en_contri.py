import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
import ConfigParser, argparse
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from offaxispos import powerlaw
from offaxispos import band
from offaxispos import model
from offaxispos import simulated_dph
from offaxispos import data_bkgd_image
from offaxispos import get_configuration
from offaxispos import resample

parser = argparse.ArgumentParser()
parser.add_argument("configfile", nargs="?", help="Name of configuration file", type=str)
parser.add_argument('--noloc', dest='noloc', action='store_true')
parser.set_defaults(noloc=False)
args = parser.parse_args()
runconf = get_configuration(args)

grbdir = "GRB160623A"#runconf['l2file'][0:10]
pre_tstart = runconf['bkg1start']
pre_tend = runconf['bkg1end']
trigtime = runconf['trigtime']
grb_tstart = runconf['transtart']
grb_tend = runconf['tranend']
post_tstart = runconf['bkg2start']
post_tend = runconf['bkg2end']
t_src = grb_tend - grb_tstart
t_tot = (pre_tend-pre_tstart)+(post_tend-post_tstart)
ra_tran = runconf['ra']
dec_tran = runconf['dec']
lc_bin = runconf['lc_bin']
alpha = runconf['alpha']
beta = runconf['beta']
E0 = runconf['E0']
A = runconf['A']
sim_scale = t_src
pixbin = int(runconf['pixsize'])
comp_bin = int(runconf['comp_bin'])
typ = runconf['typ']

plotfile = "plots/"+grbdir+"_en_contri.eps"

def band(E, alpha = -1.08, beta = -1.75, E0 = 189,  A = 5e-3):
    if (alpha - beta)*E0 >= E:
        return A*(E/100)**alpha*np.exp(-E/E0)
    elif (alpha - beta)*E0 < E:
        return A*((alpha - beta)*E0/100)**(alpha - beta)*np.exp(beta - alpha)*(E/100)**beta

out = np.arange(100, 205.1, 5)

flist = glob.glob(grbdir + "/MM_out/*")
flist.sort()
resp = np.zeros((21, len(flist)))

for i,f in enumerate(flist):
	E = i*5 + 100
	if ((i == (235 - 100) / 5) or (i == (465 - 100)/5)):
		resp[:,i] = .8*resp[:,i-1]
	else:
		data = fits.getdata(f + "/SingleEventFile.fits")
		spec = data.sum(0)
		n, bins = np.histogram(np.arange(5, 261,.5), weights=spec, bins = out)
		resp[:,i] = n/55.*1.0#band(E, alpha, beta, E0,A)
                 
out = (out[:-1] + out[1:]) / 2.

e = np.array([125, 150, 175])
index = (e - 100) / 5 

print flist[-1]

col = np.array(["y-","b-","r-"])
cumsum_flat = np.cumsum(resp, 1)

cumsum_flat = cumsum_flat/np.max(cumsum_flat, 1)[:,np.newaxis]

frac_flat = (1 - cumsum_flat.T)

fig = plt.figure()
plt.style.use("dark_background")
ax = fig.add_subplot(111,projection="3d")

#plt.legend(plt.plot(np.arange(100, 205.1, 5), frac_flat[:,index],[col[i].astype(str) for i in range(e.size)]), [e[i].astype(str) + " keV" for i in range(e.size)], loc='best')

#for i in range(len(e)):
#    plt.plot(np.arange(100, 205.1, 5),frac_flat[:,index[i]],col[i], label=str(e[i])+" keV")
#    plt.legend()

#plt.xlabel("Input E")
#plt.ylabel("Fractional Contribution")
#plt.title("Fractional Contribution to Detected Energies \n (125 keV, 150 keV, 175 keV) from Simulated Energies")
#plt.axhline(0.1,c="w",linewidth=0.3)
#plt.text(100, .08, "10 %")
#plt.axhline(.3,c="w",linewidth=0.3)
#plt.text(100, .25, "30 %")
#plt.axvline(200,c="w",linewidth=0.3)
#plt.grid()
#plt.yscale("log")
#plt.savefig(plotfile)

print out
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import spline
ax.set_zscale("log") 
#### Fisrt energy ###########
#y = 0.5*np.ones(len(resp[:,1]))
x = np.arange(100,200.1,5)
z = resp[:,5]

xnew = np.linspace(min(x),max(x),300)
zsmooth = spline(x,z,xnew)
y = 0.5*np.ones(len(xnew))

zsmooth[np.where(zsmooth<0)[0][0]:]=0

ax.plot(xnew,y,zsmooth,"y-",linewidth=0.1,label=str(out[5]-2.5)+" keV")
verts = [(xnew[i],y[i],zsmooth[i]) for i in range(len(xnew))] + [(xnew.max(),0.5,0),(xnew.min(),0.5,0)]
ax.add_collection3d(Poly3DCollection([verts],color='y'))

#### Second enrgy #########
#y = 0.4*np.ones(len(resp[:,1]))
x = np.arange(100,200.1,5)
z = resp[:,10]

xnew = np.linspace(min(x),max(x),300)
zsmooth = spline(x,z,xnew)
y = 0.4*np.ones(len(xnew))

zsmooth[np.where(zsmooth<0)[0][0]:]=0

ax.plot(xnew,y,zsmooth,"r-",linewidth=2,label=str(out[10]-2.5)+" keV")
verts = [(xnew[i],y[i],zsmooth[i]) for i in range(len(xnew))] + [(xnew.max(),0.4,0),(xnew.min(),0.4,0)]
ax.add_collection3d(Poly3DCollection([verts],color='r'))
#### Third energy ##########
#y = 0.3*np.ones(len(resp[:,1]))
x = np.arange(100,200.1,5)
z = resp[:,15]

xnew = np.linspace(min(x),max(x),300)
zsmooth = spline(x,z,xnew)
y = 0.3*np.ones(len(xnew))

zsmooth[np.where(zsmooth<0)[0][0]:]=0

ax.plot(xnew,y,zsmooth,"b-",linewidth=2,label=str(out[15]-2.5)+" keV")
verts = [(xnew[i],y[i],zsmooth[i]) for i in range(len(xnew))] + [(xnew.max(),0.3,0),(xnew.min(),0.3,0)]
ax.add_collection3d(Poly3DCollection([verts],color='b'))
### Fourth energy #########
#y = 0.1*np.ones(len(resp[:,1]))
x = np.arange(100,200.1,5)
z = resp[:,-2]

xnew = np.linspace(min(x),max(x),300)
zsmooth = spline(x,z,xnew)
y = 0.1*np.ones(len(xnew))

#zsmooth[np.where(zsmooth<0)[0][0]:]=0

ax.plot(xnew,y,zsmooth,"g-",linewidth=0.7,label="500 keV")
verts = [(xnew[i],y[i],zsmooth[i]) for i in range(len(xnew))] + [(xnew.max(),0.1,0),(xnew.min(),0.1,0)]
ax.add_collection3d(Poly3DCollection([verts],color='g'))
### Fifth energy ##########
#y = 0.0*np.ones(len(resp[:,1]))
x = np.arange(100,200.1,5)
z = resp[:,-1] 
xnew = np.linspace(min(x),max(x),300)
zsmooth = spline(x,z,xnew)
y = 0.0*np.ones(len(xnew))
ax.plot(xnew,y,zsmooth,"w-",linewidth=0.7,label="1000 keV")
verts = [(xnew[i],y[i],zsmooth[i]) for i in range(len(xnew))] + [(xnew.max(),0.0,0),(xnew.min(),0.0,0)]
ax.add_collection3d(Poly3DCollection([verts],color='w'))
ax.grid(False)
ax.set_yticks(np.array([0.0,0.1,0.3,0.4,0.5]))
ax.set_yticklabels(["{a:0.0f}".format(a=1000),"{a:0.0f}".format(a=500),"{a:0.0f}".format(a=out[15]-2.5),"{a:0.0f}".format(a=out[10]-2.5),"{a:0.0f}".format(a=out[5]-2.5)])
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
ax.set_ylabel("Energy of incident photons (keV)")

ax.set_xlabel("Energy of detected photons (keV)")
ax.set_zlabel("log(counts)")
ax.set_zlim(10**(-3),0.35*10**(1))
ax.set_zticks(np.linspace(10**(0),0.35*10**(1),2))
ax.set_zticklabels(["{a:0.1f}x10$^{b:0.0f}$".format(a=i%10,b=np.log10(i)) for i in np.linspace(10**(0),0.35*10**(1),2)],rotation=270)
plt.tick_params(axis='z', which='minor')
ax.zaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
#plt.legend(loc="upper left",prop={"size":6})
plt.show()
