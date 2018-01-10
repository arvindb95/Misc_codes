import numpy as np, matplotlib.pyplot as plt
from astropy.io import fits
from scipy.integrate import simps
from astropy.table import Table
from astropy.stats import sigma_clip, sigma_clipped_stats
from scipy.optimize import curve_fit
import astropy.coordinates as coo
import astropy.units as u
import argparse
from matplotlib.backends.backend_pdf import PdfPages
import glob
import validation_func as vf
import offaxispos as oap

# Setting all the initial parameters
parser = argparse.ArgumentParser()
parser.add_argument("grbdir", help="The directory of the GRB",type=str)
parser.add_argument("trigger_time",help="Trigger time of GRB",type=float)
parser.add_argument("t_90",help="Width at 90 percent of the GRB peak",type=float)
parser.add_argument("alpha",help="alpha parameter for simulated_dph",type=float)
parser.add_argument("beta",help="beta parameter for simulated_dph",type=float)
parser.add_argument("E0",help="E0 parameter for the simulated_dph",type=float)
parser.add_argument("A",help="A parameter for the simulated_dph",type=float)
parser.add_argument("--dt_background_left",help="Time interval for pre GRB Background",type=float,default=500)
parser.add_argument("--dt_background_right",help="Time interval for post GRB Background",type=float,default=500)
parser.add_argument("--pixbin",help="Binning number for plotting the comparison graphs",type=int,default=1)
parser.add_argument("--sim_scale",help="Scaling factor for simulation",type=int,default=-1)
parser.add_argument("--lc_bin",help="Bin for the light curve",type=int,default=5)
args = parser.parse_args()

pre_tstart = args.trigger_time - 5 - args.dt_background_left
pre_tend = args.trigger_time - 5
grb_tstart = args.trigger_time
grb_tend = args.trigger_time + args.t_90
post_tstart = grb_tend + 5
post_tend = grb_tend + 5 + args.dt_background_right
t_src = grb_tend - grb_tstart
t_tot = (pre_tend-pre_tstart)+(post_tend-post_tstart)

if (args.sim_scale == -1):
	sim_scale = args.t_90
else :
	sim_scale = args.sim_scale

thetax = float(glob.glob(args.grbdir + "/MM_out/*")[0][30:35])
thetay = float(glob.glob(args.grbdir + "/MM_out/*")[0][38:])


# Calling the functions for the dphs 
grb_flat,bkgd_flat,grb_dph,bkgd_dph,t_src,t_total = vf.data_bkgd_image(args.grbdir,pre_tstart,pre_tend,grb_tstart,grb_tend,post_tstart,post_tend)

sim_flat,sim_dph,badpix_mask = vf.simulated_dph(args.grbdir,args.alpha,args.beta,args.E0,args.A)

src_dph = grb_dph-bkgd_dph*t_src/t_total

#########################################################################################################
pdf_file = PdfPages(args.grbdir + '/comparison_'+args.grbdir+'.pdf')

# Plotting the light curves for all four quadrants
f,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
quad_clean_file = glob.glob(args.grbdir+"/*quad_clean.evt")[0]
clean_file = fits.open(quad_clean_file)
plt.suptitle('Light curves for '+args.grbdir + "\n" + r"$\theta_x$={tx:0.1f} and $\theta_y$={ty:0.1f} ".format(tx=thetax,ty=thetay))
	# quad0
quad0 = clean_file[1].data
ax1.set_title('quad A',fontsize=10)
ax1.hist(quad0['time'], bins=np.arange(quad0['time'][0],quad0['time'][-1],args.lc_bin),histtype='step')
ax1.set_xlim(pre_tstart,post_tend)
ax1.axvline(grb_tstart,color='k',linewidth=0.5)
ax1.axvline(grb_tend,color='k',linewidth=0.5)
	# quad1
quad1 = clean_file[2].data
ax2.set_title('quad B',fontsize=10)
ax2.hist(quad1['time'], bins=np.arange(quad1['time'][0],quad1['time'][-1],args.lc_bin),histtype='step')
ax2.set_xlim(pre_tstart,post_tend)
ax2.axvline(grb_tstart,color='k',linewidth=0.5)
ax2.axvline(grb_tend,color='k',linewidth=0.5)
	# quad2
quad2 = clean_file[3].data
ax3.set_title('quad C',fontsize=10)
ax3.hist(quad2['time'], bins=np.arange(quad2['time'][0],quad2['time'][-1],args.lc_bin),histtype='step')
ax3.set_xlim(pre_tstart,post_tend)
ax3.axvline(grb_tstart,color='k',linewidth=0.5)
ax3.axvline(grb_tend,color='k',linewidth=0.5)
	# quad3
quad3 = clean_file[4].data
ax4.set_title('quad D',fontsize=10)
ax4.hist(quad3['time'], bins=np.arange(quad3['time'][0],quad3['time'][-1],args.lc_bin),histtype='step')
ax4.set_xlim(pre_tstart,post_tend)
ax4.axvline(grb_tstart,color='k',linewidth=0.5)
ax4.axvline(grb_tend,color='k',linewidth=0.5)
f.set_size_inches([7,7])
pdf_file.savefig(f,orientation = 'portrait')  # saves the current figure into a pdf_file page

# Plotting all the graphs before badpix masking
f,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
plt.suptitle('DPHs before badpix correction for '+args.grbdir + "\n" + r"$\theta_x$={tx:0.1f} and $\theta_y$={ty:0.1f} ".format(tx=thetax,ty=thetay))
	# Sim
im = ax3.imshow(sim_dph*sim_scale,interpolation='none')
ax3.set_title('Sim DPH',fontsize=10)
ax3.set_xlim(-1,128 - 0.5)
ax3.axvline(x=-0.75,ymin=0,ymax=64,linewidth=5,color='k')
ax3.spines['left'].set_position(('data',-0.5))
ax3.set_yticklabels([])
f.colorbar(im,ax=ax3,fraction=0.046, pad=0.04)

	# Source 
im = ax4.imshow(src_dph,interpolation='none',vmin=0)
ax4.set_title('Src DPH (bkg subtracted)',fontsize=10)
ax4.set_xlim(-1,128 -0.5)
ax4.axvline(x=-0.75,ymin=0,ymax=64,linewidth=5,color='k')
ax4.spines['left'].set_position(('data',-0.5))
ax4.set_yticklabels([])
f.colorbar(im,ax=ax4,fraction=0.046, pad=0.04)

	# Source + Background
im = ax1.imshow(grb_dph,interpolation='none')
ax1.set_title('Src + Bkg DPH',fontsize=10)
ax1.set_xlim(-1,128 -0.5)
ax1.axvline(x=-0.75,ymin=0,ymax=64,linewidth=5,color='k')
ax1.spines['left'].set_position(('data',-0.5))
ax1.set_yticklabels([])
f.colorbar(im,ax=ax1,fraction=0.046, pad=0.04)

	# Background
im = ax2.imshow(bkgd_dph*t_src/t_total,interpolation='none')
ax2.set_title('Bkg DPH',fontsize=10)
ax2.set_xlim(-1,128 -0.5)
ax2.axvline(x=-0.75,ymin=0,ymax=64,linewidth=5,color='k')
ax2.spines['left'].set_position(('data',-0.5))
ax2.set_yticklabels([])
f.colorbar(im,ax=ax2,fraction=0.046, pad=0.04)
f.set_size_inches([7,10])

pdf_file.savefig(f,orientation = 'portrait')  # saves the current figure into a pdf_file page


# Plotting the Badpix mask
fig = plt.figure()
ax = plt.subplot(111)
plt.title('Badpix Mask for '+args.grbdir+ "\n" + r"$\theta_x$={tx:0.1f} and $\theta_y$={ty:0.1f} ".format(tx=thetax,ty=thetay))
im = ax.imshow(badpix_mask,interpolation='none')
ax.set_xlim(-9,128 -0.5)
ax.axvline(x=-5.,ymin=0,ymax=64,linewidth=5,color='k')
ax.spines['left'].set_position(('data',-0.5))
fig.colorbar(im,ax=ax,fraction=0.046, pad=0.04)
fig.set_size_inches([7,10])

pdf_file.savefig(fig,orientation = 'portrait')  # saves the current figure into a pdf_file page

# Plotting badpix masked graphs
f,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
plt.suptitle('DPHs after badpix correction for '+args.grbdir+ "\n" + r"$\theta_x$={tx:0.1f} and $\theta_y$={ty:0.1f} ".format(tx=thetax,ty=thetay))
	# Sim
im = ax3.imshow(sim_dph*badpix_mask*sim_scale,interpolation='none')
ax3.set_title('Sim DPH',fontsize=10)
ax3.set_xlim(-1,128 -0.5)
ax3.axvline(x=-0.75,ymin=0,ymax=64,linewidth=5,color='k')
ax3.spines['left'].set_position(('data',-0.5))
ax3.set_yticklabels([])
f.colorbar(im,ax=ax3,fraction=0.046, pad=0.04)

	# Source 
im = ax4.imshow(src_dph*badpix_mask,interpolation='none',vmin=0)
ax4.set_title('Src DPH (bkg subtracted)',fontsize=10)
ax4.set_xlim(-1,128 -0.5)
ax4.axvline(x=-0.75,ymin=0,ymax=64,linewidth=5,color='k')
ax4.spines['left'].set_position(('data',-0.5))
ax4.set_yticklabels([])
f.colorbar(im,ax=ax4,fraction=0.046, pad=0.04)

	# Source + Background
im = ax1.imshow(grb_dph*badpix_mask,interpolation='none')
ax1.set_title('Src + Bkg DPH',fontsize=10)
ax1.set_xlim(-1,128 -0.5)
ax1.axvline(x=-0.75,ymin=0,ymax=64,linewidth=5,color='k')
ax1.spines['left'].set_position(('data',-0.5))
ax1.set_yticklabels([])
f.colorbar(im,ax=ax1,fraction=0.046, pad=0.04)

	# Background
im = ax2.imshow(bkgd_dph*badpix_mask*t_src/t_total,interpolation='none')
ax2.set_title('Bkg DPH',fontsize=10)
ax2.set_xlim(-1,128 -0.5)
ax2.axvline(x=-0.75,ymin=0,ymax=64,linewidth=5,color='k')
ax2.spines['left'].set_position(('data',-0.5))
ax2.set_yticklabels([])
f.colorbar(im,ax=ax2,fraction=0.046, pad=0.04)
f.set_size_inches([7,10])
pdf_file.savefig(f,orientation = 'portrait')  # saves the current figure into a pdf_file page

# Plotting badpix masked graphs
f,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
plt.suptitle('DPHs after badpix correction for '+args.grbdir+ "\n" + r"$\theta_x$={tx:0.1f} and $\theta_y$={ty:0.1f} ".format(tx=thetax,ty=thetay))
	# Sim
im = ax3.imshow(oap.resample(sim_dph*badpix_mask*sim_scale,args.pixbin),interpolation='none')
ax3.set_title('Sim DPH',fontsize=10)
ax3.set_xlim(-1,128/args.pixbin -0.5)
ax3.axvline(x=-0.75,ymin=0,ymax=64,linewidth=5,color='k')
ax3.spines['left'].set_position(('data',-0.5))
ax3.set_yticklabels([])
f.colorbar(im,ax=ax3,fraction=0.046, pad=0.04)

	# Source 
im = ax4.imshow(oap.resample(src_dph*badpix_mask,args.pixbin),interpolation='none',vmin=0)
ax4.set_title('Src DPH (bkg subtracted)',fontsize=10)
ax4.set_xlim(-1,128/args.pixbin -0.5)
ax4.axvline(x=-0.75,ymin=0,ymax=64,linewidth=5,color='k')
ax4.spines['left'].set_position(('data',-0.5))
ax4.set_yticklabels([])
f.colorbar(im,ax=ax4,fraction=0.046, pad=0.04)

	# Source + Background
im = ax1.imshow(oap.resample(grb_dph*badpix_mask,args.pixbin),interpolation='none')
ax1.set_title('Src + Bkg DPH',fontsize=10)
ax1.set_xlim(-1,128/args.pixbin -0.5)
ax1.axvline(x=-0.75,ymin=0,ymax=64,linewidth=5,color='k')
ax1.spines['left'].set_position(('data',-0.5))
ax1.set_yticklabels([])
f.colorbar(im,ax=ax1,fraction=0.046, pad=0.04)

	# Background
im = ax2.imshow(oap.resample(bkgd_dph*badpix_mask*t_src/t_total,args.pixbin),interpolation='none')
ax2.set_title('Bkg DPH',fontsize=10)
ax2.set_xlim(-1,128/args.pixbin -0.5)
ax2.axvline(x=-0.75,ymin=0,ymax=64,linewidth=5,color='k')
ax2.spines['left'].set_position(('data',-0.5))
ax2.set_yticklabels([])
f.colorbar(im,ax=ax2,fraction=0.046, pad=0.04)
f.set_size_inches([7,10])

pdf_file.savefig(f,orientation = 'portrait')  # saves the current figure into a pdf_file page



# Plotting the comparison graphs with equal bins

sim_bin = oap.resample(sim_dph*badpix_mask,args.pixbin)
grb_bin = oap.resample(grb_dph*badpix_mask,args.pixbin)
bkgd_bin = oap.resample(bkgd_dph*badpix_mask,args.pixbin)

sim_flat_bin = sim_bin.flatten()
grb_flat_bin = grb_bin.flatten()
bkgd_flat_bin = bkgd_bin.flatten()

#module_order = np.flip(np.argsort(sim_flat_bin),0)
#sim_flat_bin = sim_flat_bin[module_order]
#grb_flat_bin = grb_flat_bin[module_order]
#bkgd_flat_bin = bkgd_flat_bin[module_order]

	# Defining model background and data
model = sim_flat_bin*sim_scale
bkgd = bkgd_flat_bin*t_src/t_tot
src = grb_flat_bin

data = src - bkgd

err_src = np.sqrt(src)
err_bkgd = np.sqrt(bkgd_flat_bin)
err_model = np.sqrt(model)
err_data = np.sqrt(((err_src)**2) + ((err_bkgd)**2)*(t_src/t_total)**2)

	# PLotting the comparison plots
fig = plt.figure(7)
plt.title("Comparison between simulated and real data for "+args.grbdir+ "\n" + r"$\theta_x$={tx:0.1f} and $\theta_y$={ty:0.1f} ".format(tx=thetax,ty=thetay))
plt.errorbar(np.arange(0,(len(data))),data,yerr=err_data,fmt='b.',markersize=2,label="Data",elinewidth=0.5)
plt.errorbar(np.arange(0,(len(model))),model,yerr=err_model,fmt='r.',markersize=2,label="Simulation",elinewidth=0.5)
plt.legend()
fig.set_size_inches([7,7])
pdf_file.savefig(fig,orientation = 'portrait')  # saves the current figure into a pdf_file page
pdf_file.close()

