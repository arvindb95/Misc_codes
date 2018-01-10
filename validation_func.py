# All functions required for validation of the model

import numpy as np, matplotlib.pyplot as plt
from astropy.io import fits
from offaxispos import resample
from scipy.integrate import simps
from astropy.table import Table
from astropy.io import fits
from astropy.stats import sigma_clip, sigma_clipped_stats
from scipy.optimize import curve_fit
import astropy.coordinates as coo
import astropy.units as u
import argparse
from matplotlib.backends.backend_pdf import PdfPages
import glob
# Makes the simulated curve of the right form

def band(E, alpha = -1.08, beta = -1.75, E0 = 189,  A = 5e-3):
	if (alpha - beta)*E0 >= E:
		return A*(E/100)**alpha*np.exp(-E/E0)
	elif (alpha - beta)*E0 < E:
		return A*((alpha - beta)*E0/100)**(alpha - beta)*np.exp(beta - alpha)*(E/100)**beta

# Returns the Simulated 1D array

def simulated_dph(grbdir,alpha,beta,E0,A):
	filenames = glob.glob(grbdir + "/MM_out/*")
	badpixfile = glob.glob(grbdir + "/*badpix.fits")[0]
	filenames.sort()
	pix_cnts = np.zeros((16384,len(filenames)))
	en 	 = np.arange(5, 261., .5)
	sel  = (en>=100) & (en <= 150)
	E = np.array([])
	for i,f in enumerate(filenames):
		data = fits.getdata(f + "/SingleEventFile.fits")
		E = np.append(E, float(f[20:26]))
		data = data * band(E[i], alpha, beta, E0, A)/55.5
		data[:,~sel] = 0.
		pix_cnts[:,i] = data.sum(1)
	
	pix_cnts_total = np.zeros((16384,))
	for i in range(16384):
		pix_cnts_total[i] = simps(pix_cnts[i,:], E)

	quad0pix = pix_cnts_total[:4096]
	quad1pix = pix_cnts_total[4096:2*4096]
	quad2pix = pix_cnts_total[2*4096:3*4096]
	quad3pix = pix_cnts_total[3*4096:]
	
	
	quad0 =  np.reshape(quad0pix, (64,64), 'F')
	quad1 =  np.reshape(quad1pix, (64,64), 'F')
	quad2 =  np.reshape(quad2pix, (64,64), 'F')
	quad3 =  np.reshape(quad3pix, (64,64), 'F')
	
	fig, ax = plt.subplots(1,1)
	
	sim_DPH = np.zeros((128,128), float)
	
	sim_DPH[:64,:64] = np.flip(quad0, 0)
	sim_DPH[:64,64:] = np.flip(quad1, 0)
	sim_DPH[64:,64:] = np.flip(quad2, 0)
	sim_DPH[64:,:64] = np.flip(quad3, 0)
	
	badpix = fits.open(badpixfile)
	dphmask = np.ones((128,128))
	
	badq0 = badpix[1].data # Quadrant 0
	badpixmask = (badq0['PIX_FLAG']!=0)
	dphmask[(63 - badq0['PixY'][badpixmask]) ,badq0['PixX'][badpixmask]] = 0
	
	badq1 = badpix[2].data # Quadrant 1
	badpixmask = (badq1['PIX_FLAG']!=0)
	dphmask[(63 - badq1['PixY'][badpixmask]), (badq1['PixX'][badpixmask]+64)] = 0
	
	badq2 = badpix[3].data # Quadrant 2
	badpixmask = (badq2['PIX_FLAG']!=0)
	dphmask[(127 - badq2['PixY'][badpixmask]), (badq2['PixX'][badpixmask]+64)] = 0
	
	badq3 = badpix[4].data # Quadrant 3
	badpixmask = (badq3['PIX_FLAG']!=0)
	dphmask[(127 - badq3['PixY'][badpixmask]), badq3['PixX'][badpixmask]] = 0
	
	#sim_DPH = dphmask*sim_DPH
	oneD_sim = (sim_DPH*dphmask).flatten()
	
	return oneD_sim,sim_DPH,dphmask

# Function to get image from evt

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

# Function to get 1D arrays for data and background

def data_bkgd_image(grbdir,pre_tstart,pre_tend,grb_tstart,grb_tend,post_tstart,post_tend):
# Creating badpixmask
	
	badpixelfile = glob.glob(grbdir + "/*badpix.fits")[0]
	badpic = fits.open(badpixelfile)
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

	infile = glob.glob(grbdir + "/*quad_clean.evt")[0]
	predph = evt2image(infile,pre_tstart,pre_tend)
	grbdph = evt2image(infile,grb_tstart,grb_tend)
	postdph = evt2image(infile,post_tstart,post_tend)

	bkgddph = predph+postdph

	oneD_grbdph = grbdph.flatten()
	oneD_bkgddph = bkgddph.flatten()
	t_src = grb_tend - grb_tstart
	t_total = (pre_tend-pre_tstart)+(post_tend-post_tstart)

	return oneD_grbdph,oneD_bkgddph,grbdph,bkgddph,t_src,t_total

	
