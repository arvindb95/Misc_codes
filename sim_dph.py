#!/usr/bin/python

import numpy as np, matplotlib.pyplot as plt
from astropy.io import fits
from offaxispos import resample
from scipy.integrate import quad

def band(E, alpha = -1.08, beta = -1.75, E0 = 189,  A = 5e-3):
    if (alpha - beta)*E0 >= E:
        return A*(E/100)**alpha*np.exp(-E/E0)
    elif (alpha - beta)*E0 < E:
        return A*((alpha - beta)*E0/100)**(alpha - beta)*np.exp(beta - alpha)*(E/100)**beta



def simulated_dph(pixbin):
	pix_cnts = np.zeros((16384,))
	for i in range(100,205,5):
		thx = 34.4 #float(raw_input('Theta X = '))
		thy = 58.9 #float(raw_input('Theta_Y = '))
		data = fits.getdata('E_%06.1f_TX%06.1f_TY%06.1f/SingleEventFile.fits' % (i, thx, thy) )
		en 	 = np.arange(5, 261., .5)
		sel  = (en>=100) & (en <= 150)
		data[:,~sel] = 0.
		pix_cnts = pix_cnts + data.sum(1)* quad(band, i-5, i+5, args=(-1.2, -1.8, 189, 5e-3))[0] /55.5
	
	quad0pix = pix_cnts[:4096]
	quad1pix = pix_cnts[4096:2*4096]
	quad2pix = pix_cnts[2*4096:3*4096]
	quad3pix = pix_cnts[3*4096:]
	
	
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
	
	badpix = fits.open('AS1P01_003T01_9000000002_00123cztM0_level2_quad_badpix.fits')
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
	
	fig.suptitle('THX = %3.1f, THY = %3.1f' % (thx, thy), fontsize=35)
	
	sim_DPH = dphmask*sim_DPH
	image = resample(sim_DPH, pixbin)
	
	im=plt.imshow(image, interpolation='none')
	
	fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.75])
	plt.colorbar(mappable=im, cax =cbar_ax)
	plt.show()
	return image
