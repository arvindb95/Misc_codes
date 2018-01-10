from astropy.table import Table
from astropy.io import fits
from astropy.stats import sigma_clip, sigma_clipped_stats
from scipy.optimize import curve_fit
import astropy.coordinates as coo
import astropy.units as u
import argparse
import numpy as np
import healpy as hp

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#Define the required parameters
parser = argparse.ArgumentParser()
parser.add_argument("--infile", help="The raw observed file",type=str,default='AS1CZT_GRB151006A_quad_clean.evt')
parser.add_argument("--badpixelfile", help="The bad pixel file",type=str,default='AS1P01_003T01_9000000002_00123cztM0_level2_quad_badpix.fits')
parser.add_argument("--pre_tstart",help="Start time for pre-GRB background",type=float,default=181821000)
parser.add_argument("--pre_tend",help="End time for pre-GRB background",type=float,default=181821250)
parser.add_argument("--grb_tstart",help="Start time for GRB",type=float,default=181821280)
parser.add_argument("--grb_tend",help="Start time for GRB ",type=float,default=181821400)
parser.add_argument("--post_tstart",help="Start time for post-GRB background",type=float,default=181821450)
parser.add_argument("--post_tend",help="Start time for post-GRB background",type=float,default=181821700)

args = parser.parse_args()

pixbin = 16

# Functions taken from Sir's code ----------------------------------------------

def evt2image(infile, tstart, tend):
    """
    Read an events file (with or without energy), and return a combined
    DPH for all 4 quadrants. 
    If tstart and tend are given, use data only in that time frame. The 
    default values are set to exceed expected bounds for Astrosat.
    """
    hdu = fits.open(infile)
    pixel_edges = np.arange(-0.5, 63.6)
    
    e_low = 100
    e_high = 150	
    data = hdu[1].data[np.where( (hdu[1].data['Time'] >= tstart) & (hdu[1].data['Time'] <= tend) )]
    sel = (data['energy'] > e_low) & (data['energy'] < e_high)
    data = data[sel]
    im1 = np.histogram2d(data['detx'], data['dety'], bins=(pixel_edges, pixel_edges))
    im1 = np.transpose(im1[0])
    data = hdu[2].data[np.where( (hdu[2].data['Time'] >= tstart) & (hdu[2].data['Time'] <= tend) )]
    sel = (data['energy'] > e_low) & (data['energy'] < e_high)
    data = data[sel]
    im2 = np.histogram2d(data['detx'], data['dety'], bins=(pixel_edges, pixel_edges))
    im2 = np.transpose(im2[0])
    data = hdu[3].data[np.where( (hdu[3].data['Time'] >= tstart) & (hdu[3].data['Time'] <= tend) )]
    sel = (data['energy'] > e_low) & (data['energy'] < e_high)
    data = data[sel]
    im3 = np.histogram2d(data['detx'], data['dety'], bins=(pixel_edges, pixel_edges))
    im3 = np.transpose(im3[0])
    data = hdu[4].data[np.where( (hdu[4].data['Time'] >= tstart) & (hdu[4].data['Time'] <= tend) )]
    sel = (data['energy'] > e_low) & (data['energy'] < e_high)
    data = data[sel] 
    im4 = np.histogram2d(data['detx'], data['dety'], bins=(pixel_edges, pixel_edges))
    im4 = np.transpose(im4[0])

    image = np.zeros((128,128))
    image[0:64,0:64] = im4
    image[0:64,64:128] = im3
    image[64:128,0:64] = im1
    image[64:128,64:128] = im2

    image = np.flip(image,0)

    #plt.imshow(image, origin="lower")
    return image

def resample(image, pixsize):
    """
    Take a 128 x 128 pixel image, and rebin it such that
    new pixels = pixsize x pixsize old pixels
    """
    assert pixsize in [1, 2, 4, 8, 16] # return error and exit otherwise
    imsize = 128/pixsize
    newimage = np.zeros((imsize, imsize))
    for xn, x in enumerate(np.arange(0, 128, pixsize)):
        for yn, y in enumerate(np.arange(0, 128, pixsize)):
            newimage[xn, yn] = np.nansum(image[x:x+pixsize, y:y+pixsize]) # Nansum is important as sum of masked array can be nan
    return newimage


# Making the mask ------------------------------------------------------------------

badpic = fits.open(args.badpixelfile)
dphmask = np.ones((128,128))

badq0 = badpic[1].data # Quadrant 0
badpixmask = (badq0['PIX_FLAG']!=0)
dphmask [(63 - badq0['PixY'][badpixmask]),badq0['PixX'][badpixmask]] = 0

badq1 = badpic[2].data # Quadrant 1
badpixmask = (badq1['PIX_FLAG']!=0)
dphmask[(63 - badq1['PixY'][badpixmask]),(badq1['PixX'][badpixmask]+64)] = 0

badq2 = badpic[3].data # Quadrant 2
badpixmask = (badq2['PIX_FLAG']!=0)
dphmask[(127 - badq2['PixY'][badpixmask]),(badq2['PixX'][badpixmask]+64)] = 0

badq3 = badpic[4].data # Quadrant 3
badpixmask = (badq3['PIX_FLAG']!=0)
dphmask[(127 - badq3['PixY'][badpixmask]),badq3['PixX'][badpixmask]] = 0

# Making raw dph for pre-grb ,grb and post grb phases -----------------------------

predph = evt2image(args.infile,args.pre_tstart,args.pre_tend)
grbdph = evt2image(args.infile,args.grb_tstart,args.grb_tend)
postdph = evt2image(args.infile,args.post_tstart,args.post_tend)


# Subtracting the sdded background from the raw grbdph -------------------------------

bkgddph = predph+postdph

# The exposures required-------------------------------------------------

t_src = args.grb_tend-args.grb_tstart
t_total = ((args.pre_tend-args.pre_tstart)+(args.post_tend-args.post_tstart))

# The images required---------------------------------------------------

bkgddph = bkgddph*dphmask
grbdph = grbdph*dphmask

def data_binner(pixbin):
	grbdph_binned = resample(grbdph,pixbin)
	bkgddph_binned = resample(bkgddph,pixbin)
	return grbdph_binned,bkgddph_binned


grbdph = np.reshape(grbdph,(np.product(grbdph.shape)))
bkgddph = np.reshape(bkgddph,(np.product(bkgddph.shape)))

grbdph,bkgddph = data_binner(pixbin) # Binning the data and background

# Importing Sujay's code for the simulation

import sim_dph as smd

simdph = smd.simulated_dph(1)
simdph = np.reshape(simdph,(np.product(simdph.shape)))

simdph = resample(simdph,pixbin) # Binning the simulated data

# Getting all the required parameters for plotting------------------------------------------------------------------------ 

module_order = np.flip(np.argsort(simdph),0)
model = simdph[module_order]*t_src
src = grbdph[module_order]
bkgd = bkgddph[module_order] * t_src/t_total
data = src - bkgd

err_src = np.sqrt(src)
err_bkgd = np.sqrt(bkgddph)
err_data = np.sqrt(((err_src)**2) + ((err_bkgd)**2)*(t_src/t_total)**2)

err_model = np.sqrt(model)

# Plotting the grophs-----------------------------------
plt.errorbar(np.arange(1,65),data,yerr=err_data,fmt='bo',label="Data")
plt.errorbar(np.arange(1,65),model,yerr=err_model,fmt='ro',label="Simulation")
plt.legend()
plt.show()

