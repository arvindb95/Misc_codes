#!/usr/bin/env python2.7
import os
from scipy import integrate
from scipy.interpolate import griddata
from astropy.io import fits 
from astropy.table import Table
from astropy.time import Time
import astropy.coordinates as coo
import astropy.units as u

import numpy as np, healpy as hp

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap

import argparse

#Define Powerlaw and Band
def powerlaw(E, alpha):
	return E**alpha

def band(E, alpha, beta, Ep):
	mask = (alpha - beta)*Ep >= E
	return np.append((E[mask]/100)**alpha*np.exp(-E[mask]/Ep), ((alpha - beta)*Ep/100)**(alpha - beta)*np.exp(beta - alpha)*(E[~mask]/100)**beta)

#Define function to select spectral type
def spectra(typ, E, alpha, beta, Ep):
	if typ == 'powerlaw':
		return powerlaw(E, alpha)
	else:
		return band(E, alpha, beta, Ep)

#Function to calculate Total counts for each Thx and Thy
def countrate(typ, alpha, beta, Ep, qmask,inbase):
	En = np.append(np.arange(30, 90, 10.), np.arange(100, 201,20.))
	angles = np.loadtxt(inbase + 'eff_area_files/angle_list.txt')
	
	data = {En[k]: np.array([]) for k in range(En.size)}
	
	for i in range(En.size):
		data[En[i]] = np.loadtxt( inbase + 'eff_area_files/angle_eff_area_%d.txt' % En[i])
	
	K = np.zeros((angles.shape[0],3))
	K[:,:2] = angles
	
	for i in range(angles.shape[0]):
		eff_area = np.zeros((En.size, 4))
		for j in range(En.size):
			eff_area[j,:] =  data[En[j]][np.where(np.logical_and(data[En[j]][:,0]==angles[i,0], data[En[j]][:,1]==angles[i,1]))][:,2:]
		eff_area[:,qmask] = 0
		K[i,2] = integrate.simps(eff_area[:,:].sum(1)*spectra(typ, En, alpha, beta, Ep) ,En)
	np.savetxt('countrate_thx_thy.txt', K, '%3.1f\t\t%3.1f\t\t%2.3f', header = ' Thx\t\tThy\t\tN_Total')
	return K, data, angles

#Function to calculate fluxlimits for each Thx, Thy
def calc_fluxlimit(K, data, angles, tbin, typ, alpha, beta, Ep, qmask, inbase, far):
	En = np.append(np.arange(30, 90, 10.), np.arange(100, 201,20.))
	
	hdu = fits.open( inbase + 'rates/far{far:1.1f}_{tbin:s}_hist_tbin{tbin:s}.fits'.format(tbin=tbin, far=far))
	quad = np.array(['A', 'B', 'C', 'D'])
	quad = np.delete(quad, qmask)
	cutoff_rate = 0
	for i in range(quad.size):
		cutoff_rate = cutoff_rate + hdu[1].header['RATE_%s' %quad[i]]
	norm = cutoff_rate / K[:,2]
	
	F = np.zeros((angles.shape[0],3))
	F[:,:2] = angles
	
	for i in range(angles.shape[0]):
		F[i,2] = integrate.simps(spectra(typ, En, alpha, beta, Ep)*En,En)*(norm[i]*u.keV.to(u.erg))
	np.savetxt('Fluxlimit_{tbin:s}.txt'.format(tbin=tbin), F, '%3.3f\t\t%3.3f\t\t%2.3e', header = ' ThX\t\tThY\t\tFlux')

	return F

#Function to get Thx, Thy, earth angle and roll, rot of given ra,dec
def get_txty(mkfdata, trigtime, ra, dec, window=10):
	"""
	Calculate earth ra-dec and satellite ponting using astropy
	Use pitch, roll and yaw information from the MKF file
	"""
	# x = -yaw
	# y = +pitch
	# z = +roll

	# Read in the MKF file
	sel = abs(mkfdata['time'] - trigtime) < window

	# Get pitch, roll, yaw
	# yaw is minus x
	pitch = coo.SkyCoord( np.median(mkfdata['pitch_ra'][sel]) * u.deg, np.median(mkfdata['pitch_dec'][sel]) * u.deg )
	roll = coo.SkyCoord( np.median(mkfdata['roll_ra'][sel]) * u.deg, np.median(mkfdata['roll_dec'][sel]) * u.deg )
	roll_rot = np.median(mkfdata['roll_rot'][sel])
	yaw_ra  = (180.0 + np.median(mkfdata['yaw_ra'][sel]) ) % 360
	yaw_dec = -np.median(mkfdata['yaw_dec'][sel])
	minus_yaw = coo.SkyCoord( yaw_ra * u.deg, yaw_dec * u.deg )
	
	# Earth - the mkffile has satellite xyz 
	earthx = np.median(mkfdata['posx'][sel]) * u.km
	earthy = np.median(mkfdata['posy'][sel]) * u.km
	earthz = np.median(mkfdata['posz'][sel]) * u.km
	earth = coo.SkyCoord(-earthx, -earthy, -earthz, frame='icrs', representation='cartesian')
	
	# Transient:
	transient = coo.SkyCoord(ra * u.deg, dec * u.deg)
	
	# Angles from x, y, z axes are:
	ax = minus_yaw.separation(transient)
	ay = pitch.separation(transient)
	az = roll.separation(transient)
	
	# the components are:
	cx = np.cos(ax.radian) # The .radian is not really needed, but anyway...
	cy = np.cos(ay.radian)
	cz = np.cos(az.radian)
	
	# Thetax = angle from z axis in ZX plane
	# lets use arctan2(ycoord, xcoord) for this
	thetax = u.rad * np.arctan2(cx, cz)
	thetay = u.rad * np.arctan2(cy, cz)
	
	return thetax.to(u.deg).value, thetay.to(u.deg).value, earth, roll, roll_rot

#Visibility mask , this excludes angles covered by SXT, UVVIT and LAXPC
def visible(thetax, thetay):
	"""
	Return a boolean mask based on what should be considered "visible" for CZTI
	"""
	mask = np.repeat(True, len(thetax))
	# Angles outside +-90 are invalid
	mask[ abs(thetax) > 90 ] = False
	mask[ abs(thetay) > 90 ] = False
	# This is based on a 30 degree cutoff on SXT/UVIT side
	# If thetax < -22, thetay must be < -thetax
	mask[ (thetax < -22.0) & (thetay >= -thetax) ] = False
	# For thetax between -22 and 22, thetay < sqrt(30**2 - thetax**2)
	mask[ (abs(thetax) <= 22.0) & (thetay > np.sqrt(30.**2 - thetax**2)) ] = False
	# Based on 40 degree cutoff on LAXPC side
	# If thetay < -30, thetax must be < -thetay
	mask[ (thetay < -30) & (thetax >= -thetay) ] = False
	# If thetay between -30 and 30, thetax < sqrt(40.**2 - thetay**2)
	# Note - slightly conservative cut
	mask[ (abs(thetay) <= 30) & (thetax > np.sqrt(40.**2 - thetay**2)) ] = False
	# Line things up between laxpc and xuv:
	# if thetax > 22 then thetay < thetax
	mask[ (thetax > 22.0) & (thetay >= thetax) ] = False
	# if thetay > 30 then thetax<thetay
	mask[ (thetay > 30.0) & (thetax >= thetay) ] = False

	return mask

#plorring function
def plot_mollview(data, title, czti_theta, czti_phi, roll_rot, plotfile, cmap, cticks=None, log=False, cmax=None):
	RA = np.arange(30, 360, 30)
	Dec = np.arange(-75, 75.1,15)
	
	plt.figure(0,figsize=(16,9))
	#with 180 deg rotation (LIGO Map Compatible)
	if log:
		data = np.log(data)
	if cmax==None:
		hp.mollview(data, title = '', min= np.nanmin(data), max= np.nanmax(data) ,rot=(180,0), cmap=cmap)
	else:
		hp.mollview(data, title = '', min= np.nanmin(data), max= cmax ,rot=(180,0), cmap=cmap)
	hp.projscatter(czti_theta, czti_phi,color = 'r' ,marker='x', s=80)
	hp.projtext(czti_theta, czti_phi,'CZTI', fontsize=17)
	for i in range(RA.size):
		hp.projtext(RA[i],0, '%2d$^h$' %(RA[i]/15), lonlat= True, fontsize=15)
		hp.projtext(10,Dec[i], '%2d$^\circ$' %Dec[i], lonlat= True, fontsize=15)
	hp.graticule(15, 30)
	fig = plt.gcf()
	fig.suptitle(title, fontsize=10)
	ax = fig.axes[0]
	cb = ax.images[0].colorbar
	clow, chigh = cb.get_clim()
	if cticks == None:
		ticks = np.linspace(clow, chigh, 5)
		if log:
			ticknames = ["{:0.1e}".format(np.exp(x)) for x in ticks]
		else:
			ticknames = ["{:0.1e}".format(x) for x in ticks]
	else:
		ticks = np.linspace(clow, chigh, len(cticks))
		ticknames = cticks
	cb.set_ticks(ticks)
	cb.set_ticklabels(ticknames)
	cb.set_label("Flux Density ( erg cm$^{-2}$ s$^{-1}$ )", fontsize=15)
	cb.ax.tick_params(labelsize=15) 
	plotfile.savefig()
	
	#With CZTI Boresightat centre
	if cmax==None:
		hp.mollview(data, title= '', min=np.nanmin(data), max= np.nanmax(data), rot=(np.rad2deg(czti_phi), 90 - np.rad2deg(czti_theta), roll_rot), cmap=cmap)
	else:
		hp.mollview(data, title= '', min=np.nanmin(data), max= cmax, rot=(np.rad2deg(czti_phi), 90 - np.rad2deg(czti_theta), roll_rot), cmap=cmap)
	hp.projscatter(czti_theta, czti_phi,color = 'r' ,marker='x')
	hp.projtext(czti_theta, czti_phi,'CZTI')
	for i in range(RA.size):
		hp.projtext(RA[i],0, '%2d$^h$' %(RA[i]/15), lonlat= True, fontsize=10)
		hp.projtext(0,Dec[i], '%2d$^\circ$' %Dec[i], lonlat= True, fontsize=10)
	hp.graticule(15, 30)
	fig = plt.gcf()
	fig.suptitle(title + ' (CZTI frame)', fontsize=10)
	ax = fig.axes[0]
	cb = ax.images[0].colorbar
	clow, chigh = cb.get_clim()
	if cticks == None:
		ticks = np.linspace(clow, chigh, 5)
		if log:
			ticknames = ["{:0.1e}".format(np.exp(x)) for x in ticks]
		else:
			ticknames = ["{:0.1e}".format(x) for x in ticks]
	else:
		ticks = np.linspace(clow, chigh, len(cticks))
		ticknames = cticks
	cb.set_ticks(ticks)
	cb.set_ticklabels(ticknames)
	plotfile.savefig()
	return


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()

	parser.add_argument("mkffile", type = str, help = " Level 1 mkf file")
	parser.add_argument("locmap", type = str, help = "LIGO baystar file")
	parser.add_argument("trigtime", type = str, help = "Trigtime in YYYY-MM-DDTHH:MM:SS format")	
	parser.add_argument("--only_map", dest='only_map', action='store_true') # If option not given, args.plotlc becomes False
	parser.add_argument("--mask_quad", nargs='+',type = int, help = 'List of bad quadrants if any', default=[] )
	parser.add_argument("--spectra", type = str, help = " Spectra type (powerlaw or band), default is Powerlaw with alpha = -1", default = 'powerlaw')
	parser.add_argument("--alpha", type = float, help = "Photon powerlaw index for powerlaw function and band function, default = -1 ", default = -1)
	parser.add_argument("--beta", type = float, help = " Beta value for Band function", default = -1)
	parser.add_argument("--E_peak", type = float, help = "Epeak value for Band function", default = 150)
	parser.add_argument("--inbase", type = str, help = "Base directroy for pixarea and angle files, default is current dir", default = './')
	parser.add_argument("--far", type = str, help = "False alarm rate, default is 0.1", default = 0.1)
	parser.add_argument("--outbase", type = str, help = "Output pdf file prefix, default is none", default = '')
	

	args = parser.parse_args()
	#Load mkf file and convert Trig time
	print "Loading mkf data and calculating trigger time in AstroSat seconds...\n"
	mkfdata = fits.getdata(args.mkffile, 1)
	mission_time = Time(args.trigtime) - Time('2010-01-01 00:00:00')
	trigtime = mission_time.sec
		
	#Load sky localisation map
	print "Loading localisaton map...\n"
	probmap = hp.read_map(args.inbase + args.locmap)
	NSIDE = fits.open(args.inbase + args.locmap)[1].header['NSIDE']
	
	print "Calculating Thx, Thy values and visibility masks...\n"
	#Calculate ThetaX ThetaY values.
	theta, phi = hp.pix2ang(NSIDE, np.arange(0,hp.nside2npix(NSIDE),1)) 
	ra = phi
	dec = np.pi/2 - theta
	thx, thy, earth, czti_z, roll_rot= get_txty(mkfdata, trigtime, np.rad2deg(ra), np.rad2deg(dec), 10)
		
	#Caculate visibility mask
	vismask = visible(thx, thy)	
	
	#CZTI pointing
	czti_ra  = czti_z.fk5.ra.rad
	czti_dec = czti_z.fk5.dec.rad
	czti_theta = np.pi/2 - czti_dec
	czti_phi   = czti_ra
	
	#Earth and Focal Plane view
	ref_map = np.zeros(hp.nside2npix(NSIDE))
	#Earth and focal plane mask
	mask = np.repeat(False, probmap.size)
	
	earth_ra   = earth.fk5.ra.rad
	earth_dec  = earth.fk5.dec.rad
	earth_dist = earth.fk5.distance.km
	earth_theta = np.pi/2 - earth_dec
	earth_phi   = earth_ra
	earth_occult = np.arcsin(6378./earth_dist)
	earth_vec = hp.ang2vec(earth_theta, earth_phi)
	earthmask = hp.query_disc(NSIDE, earth_vec, earth_occult)
	ref_map[earthmask] = 1
	front_vec  = hp.ang2vec(czti_theta , czti_phi)
	front = hp.query_disc(NSIDE, front_vec, np.pi/2)
	mask[front] = True
	ref_map[~mask] = ref_map[~mask] + 2
	mask[earthmask] = False
	
	plotfile = PdfPages(args.outbase + 'Fluxlimits.pdf')
	
	print "Plotting visibility plots...\n"
	#Colormap
	colors = [(1, 1, 0), (.5, 0, 0), (.1, 0, 0)]
	cmap = LinearSegmentedColormap.from_list('VisMap', colors, N=4)
	cmap.set_under("W")
	
	plot_mollview(ref_map, 'CZTI Visibility', czti_theta, czti_phi, roll_rot, plotfile, cmap, ['Visible', 'Earth', 'Behind', 'Behind + Earth'])
	
	#Colormap
	cmap = plt.cm.YlOrRd
	cmap.set_under("w")
	
	plot_mollview(probmap, 'GW Localisation Map', czti_theta, czti_phi, roll_rot, plotfile, cmap)
	
	skymap = np.copy(probmap)
	skymap[~vismask] = np.nan
	skymap[~mask] = np.nan
	
	vis_prob = np.nansum(skymap)
	
	print "\n Total Visible Probability =  %6.2e" % (vis_prob*100) , "\n"
	
	plot_mollview(skymap, "", czti_theta, czti_phi, roll_rot, plotfile, cmap, cmax = probmap.max())
	
	if args.only_map:
		plotfile.close()
		quit()

	# Calculate counterate from energy files.
	
	if not os.path.isdir("./eff_area_files"):
		os.symlink("/home/cztipoc/czti/trunk/users/sujay/eff_area_files", "./eff_area_files")
	
	print "Calculating countrate and fluxlimits...\n"
	K, data, angles = countrate(args.spectra, args.alpha, args.beta, args.E_peak, args.mask_quad,args.inbase)
	
	for tbin in ['0.1', '1', '10']:
		F = calc_fluxlimit(K, data, angles, tbin, args.spectra, args.alpha, args.beta, args.E_peak, args.mask_quad,args.inbase, args.far)
		grid_tx, grid_ty = np.mgrid[-90:90.1:1, -90.:90.1:1]
		points = F[:,:2]
		grid_area = griddata(points, F[:,2], (grid_tx, grid_ty), method='nearest')
		pixflux = np.zeros((thx.size))
		pixflux[vismask] = grid_area[np.int16(np.round(thx[vismask] - 91)), np.int16(np.round(thy[vismask] - 91))]
		pixflux[~mask] = np.nan
		pixflux[pixflux==0] = np.nan
		
		#Calclate weighted flux
		fluxlim = np.nansum(pixflux * skymap)/ np.nansum(skymap)
		#Colormap
		cm = plt.cm.jet
		cm.set_under("w")
		#Plot
		title= "{tbin:s} s binning, effective fluence limit = {fluence:0.2e} $ergs/cm^2$, flux limit  {flux:0.2e} $ergs/cm^2/sec$".format(tbin=tbin, fluence=fluxlim, flux=fluxlim/float(tbin))
		plot_mollview(pixflux, "", czti_theta, czti_phi, roll_rot, plotfile, cm, log=True)
		print "\n At {tbin:s} s binning, Effective limit = {fluence:0.2e} ergs/cm^2 = {flux:0.3e} ergs/cm^2/sec \n".format(tbin=tbin, fluence=fluxlim, flux=fluxlim/float(tbin))
	
	
	plotfile.close()
