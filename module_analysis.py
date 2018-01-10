import matplotlib.pyplot as plt
import numpy as np
import ConfigParser, argparse
from matplotlib.backends.backend_pdf import PdfPages

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

grbdir = runconf['l2file'][0:10]
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
pixbin = 16 #int(runconf['pixsize'])
comp_bin = int(runconf['comp_bin'])
typ = runconf['typ']

imsize = 128/runconf['pixsize']

bad_ratio_sel = 5

plotfile = "plots/module_analysis/"+grbdir+"_mod_analysis.pdf"

# This code finds the bad modules and checks pixel wise what is wrong

pdf_file = PdfPages(plotfile)

grb_flat,bkgd_flat,grb_dph,bkgd_dph,t_src,t_total = data_bkgd_image(grbdir,pre_tstart,pre_tend,grb_tstart,grb_tend,post_tstart,post_tend)

sim_flat,sim_dph,badpix_mask,sim_err_dph = simulated_dph(grbdir,typ,t_src,alpha,beta,E0,A)

src_dph = grb_dph-bkgd_dph*t_src/t_tot
src_flat = src_dph.flatten()

sim_copy = np.copy(sim_dph)
grb_copy = np.copy(grb_dph)
bkgd_copy = np.copy(bkgd_dph)
src_copy = np.copy(src_dph)

sim_dph = sim_dph*badpix_mask
sim_err_dph = sim_err_dph*badpix_mask
grb_dph = grb_dph*badpix_mask
bkgd_dph = bkgd_dph*badpix_mask

sim_bin = resample(sim_dph,pixbin)
sim_err_bin = np.sqrt(resample(sim_err_dph**2,pixbin))
grb_bin = resample(grb_dph,pixbin)
bkgd_bin = resample(bkgd_dph,pixbin)
src_bin = resample(src_dph*badpix_mask,pixbin)

sim_flat_bin = sim_bin.flatten()
sim_err_flat_bin = sim_err_bin.flatten()
grb_flat_bin = grb_bin.flatten()
bkgd_flat_bin = bkgd_bin.flatten()


model = sim_flat_bin

bkgd = bkgd_flat_bin*t_src/t_tot
src = grb_flat_bin

data = src - bkgd

err_src = np.sqrt(src)
err_bkgd = np.sqrt(bkgd_flat_bin)
err_model = sim_err_flat_bin
err_data = np.sqrt(((err_src)**2) + ((err_bkgd)**2)*(t_src/t_total)**2)

ratio = data/model
err_ratio = ratio*np.sqrt(((err_data/data)**2) + ((err_model/model)**2))

bad_modules = np.ones(len(model))*np.nan


for i in range(len(data)):
	if (ratio[i] > bad_ratio_sel) :
		bad_modules[i] = 1
	if (ratio[i] < 1/bad_ratio_sel):
		bad_modules[i] = 1

bad_mod_dph = np.reshape(bad_modules,(8,8))

index = np.where(bad_mod_dph != 1)

pix_bad_modules = np.ones((128,128))

for i in range(len(index[0])):
	sim_copy[16*index[0][i]:16*index[0][i]+16,16*index[1][i]:16*index[1][i]+16] = np.nan 
	src_copy[16*index[0][i]:16*index[0][i]+16,16*index[1][i]:16*index[1][i]+16] = np.nan

# Plotting the dph with the chosen bad modules

f,((ax0,ax1),(ax2,ax3)) = plt.subplots(2,2)
plt.suptitle(r'Bad_ratio_sel = {brs:0.1f}'.format(brs=bad_ratio_sel))

im = ax0.imshow(sim_bin,interpolation='none',vmin=0)
ax0.set_title('Sim')
ax0.set_xlim(-1,128/pixbin -0.5)
ax0.axvline(x=-0.75,ymin=0,ymax=64,linewidth=5,color='k')
ax0.spines['left'].set_position(('data',-0.5))
ax0.set_yticklabels([])
ax0.xaxis.set_ticks(np.arange(0,(128/pixbin),16/pixbin))
ax0.set_xticklabels(np.arange(0,128,16))
ax0.text(-1.5,2,'Radiator Plate',rotation=90)
f.colorbar(im,ax=ax0,fraction=0.046, pad=0.04)

im = ax1.imshow(src_bin,interpolation='none',vmin=0)
ax1.set_title('Src')
ax1.set_xlim(-1,128/pixbin -0.5)
ax1.axvline(x=-0.75,ymin=0,ymax=64,linewidth=5,color='k')
ax1.spines['left'].set_position(('data',-0.5))
ax1.set_yticklabels([])
ax1.xaxis.set_ticks(np.arange(0,(128/pixbin),16/pixbin))
ax1.set_xticklabels(np.arange(0,128,16))
ax1.text(-1.5,2,'Radiator Plate',rotation=90)
f.colorbar(im,ax=ax1,fraction=0.046, pad=0.04)

im = ax2.imshow(bad_mod_dph,interpolation='none',vmin=0)
ax2.set_title('Bad Modules')
ax2.set_xlim(-1,128/pixbin -0.5)
ax2.axvline(x=-0.75,ymin=0,ymax=64,linewidth=5,color='k')
ax2.spines['left'].set_position(('data',-0.5))
ax2.set_yticklabels([])
ax2.xaxis.set_ticks(np.arange(0,(128/pixbin),16/pixbin))
ax2.set_xticklabels(np.arange(0,128,16))
ax2.text(-1.5,2,'Radiator Plate',rotation=90)
f.colorbar(im,ax=ax2,fraction=0.046, pad=0.04)

im = ax3.imshow(src_copy,interpolation='none',vmin=0)
ax3.set_title('Bad Modules')
ax3.set_xlim(-1,128 -0.5)
ax3.axvline(x=-0.75,ymin=0,ymax=64,linewidth=5,color='k')
ax3.spines['left'].set_position(('data',-0.5))
ax3.set_yticklabels([])
ax3.xaxis.set_ticks(np.arange(0,128,16))
f.colorbar(im,ax=ax3,fraction=0.046, pad=0.04)

pdf_file.savefig(f)

pdf_file.close()
