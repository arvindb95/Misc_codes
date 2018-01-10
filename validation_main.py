# Main programme for validation
import numpy as np, matplotlib.pyplot as plt
from astropy.io import fits
from offaxispos import resample
from scipy.integrate import quad
from astropy.table import Table
from astropy.io import fits
from astropy.stats import sigma_clip, sigma_clipped_stats
from scipy.optimize import curve_fit
import astropy.coordinates as coo
import astropy.units as u
import argparse
from matplotlib.backends.backend_pdf import PdfPages
import glob
import validation_func as vf
import offaxispos as oap
# All the required parameters

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
parser.add_argument("--pixbin",help="Binning number for plotting the comparison graphs",type=int,default=16)
args = parser.parse_args()

pre_tstart = args.trigger_time - 5 - args.dt_background_left
pre_tend = args.trigger_time - 5
grb_tstart = args.trigger_time
grb_tend = args.trigger_time + args.t_90 
post_tstart = grb_tend + 5 
post_tend = grb_tend + 5 + args.dt_background_right
t_src = grb_tend - grb_tstart
t_tot = (pre_tend-pre_tstart)+(post_tend-post_tstart)

# Calling functions and getting the required flat and dph arrays

grb_flat,bkgd_flat,grbdph,bkgddph,t_src,t_total = vf.data_bkgd_image(args.grbdir,pre_tstart,pre_tend,grb_tstart,grb_tend,post_tstart,post_tend)

sim_flat,sim_dph,badpix_mask = vf.simulated_dph(args.grbdir,args.alpha,args.beta,args.E0,args.A)

sim_dph = sim_dph*badpix_mask

# Sorting the simulated dph and ordering the other flats in the same order
sim_bin = oap.resample(sim_dph,args.pixbin)
grb_bin = oap.resample(grbdph,args.pixbin)
bkgd_bin = oap.resample(bkgddph,args.pixbin)

sim_flat_bin = sim_bin.flatten()
grb_flat_bin = grb_bin.flatten()
bkgd_flat_bin = bkgd_bin.flatten()

module_order = np.flip(np.argsort(sim_flat_bin),0)
sim_flat_bin = sim_flat_bin[module_order]
grb_flat_bin = grb_flat_bin[module_order]
bkgd_flat_bin = bkgd_flat_bin[module_order]


# Defining model background and data
model = sim_flat_bin*t_src
bkgd = bkgd_flat_bin*t_src/t_tot
src = grb_flat_bin

data = src - bkgd

err_src = np.sqrt(src)
err_bkgd = np.sqrt(bkgd_flat_bin)
err_model = np.sqrt(model)
err_data = np.sqrt(((err_src)**2) + ((err_bkgd)**2)*(t_src/t_total)**2)
# PLotting the comparison plots
plt.figure(2)
plt.title("Comparison between simulated and real data")
plt.errorbar(np.arange(0,(len(data))),data,yerr=err_data,fmt='bo',label="Data")
plt.errorbar(np.arange(0,(len(model))),model,yerr=err_model,fmt='ro',label="Simulation")
plt.legend()
plt.show()

# Binning Statistics 
#module_order = np.flip(np.argsort(sim_flat),0)
#sim_flat = sim_flat[module_order]
#grb_flat = grb_flat[module_order]
#bkgd_flat = bkgd_flat[module_order]
#bin_boundaraies = np.digitize(np.linspace(0,max(np.cumsum(sim_flat)),15),np.cumsum(sim_flat),right=True)
#x= bin_boundaraies
#no_in_bin_sim = np.array([sim_flat[x[i]:x[i+1]].sum() for i in range(14)])*t_src
#no_in_bin_src = np.array([grb_flat[x[i]:x[i+1]].sum() for i in range(14)])
#no_in_bin_bkgd = np.array([bkgd_flat[x[i]:x[i+1]].sum() for i in range(14)])
#no_in_bin_data = no_in_bin_src - no_in_bin_bkgd*t_src/t_tot

#plt.figure(1)
#plt.title("Binning Statistics")
#plt.errorbar(np.arange(1,15,1),no_in_bin_sim,np.sqrt(no_in_bin_sim),fmt='ro',label='Simulation')
#plt.errorbar(np.arange(1,15,1),no_in_bin_data,np.sqrt((no_in_bin_src) + (no_in_bin_bkgd)*(t_src/t_tot)**2),fmt='bo',label='Data')
#plt.xlabel("Bin No.")
#plt.ylabel("No. of photons in each bin")
#plt.legend()
#plt.show()


