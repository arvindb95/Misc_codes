#!/usr/bin/env python2.7

from astropy.table import Table
from astropy.io import fits
from astropy.time import Time
import astropy.coordinates as coo
import astropy.units as u

import numpy as np, healpy as hp

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from offaxispos import get_configuration

def get_angles(mkfdata, trigtime, ra_tran, dec_tran, window=10):
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
    yaw_ra  = (180.0 + np.median(mkfdata['yaw_ra'][sel]) ) % 360
    yaw_dec = -np.median(mkfdata['yaw_dec'][sel])
    minus_yaw = coo.SkyCoord( yaw_ra * u.deg, yaw_dec * u.deg )

    # Earth - the mkffile has satellite xyz 
    earthx = np.median(mkfdata['posx'][sel]) * u.km
    earthy = np.median(mkfdata['posy'][sel]) * u.km
    earthz = np.median(mkfdata['posz'][sel]) * u.km
    earth = coo.SkyCoord(-earthx, -earthy, -earthz, frame='icrs', representation='cartesian')

    # Sun coordinates:
#    sunra = np.median(mkfdata['sun_ra'][sel]) * u.deg
#    sundec = np.median(mkfdata['sun_dec'][sel]) * u.deg
#    sun = coo.SkyCoord(sunra, sundec)
    # Transient:
    transient = coo.SkyCoord(ra_tran * u.deg, dec_tran * u.deg)

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

    return earth, roll

def creat_map(mkffile, ra, dec, time, outfile):
	
	mkfdata = fits.getdata(mkffile, 1)
	
	#Calculate Earth and CZTI Coords
	earth_coo, czti_z = get_angles(mkfdata, time, ra, dec, window=10)
		
	plotfile = PdfPages(outfile)
	
	#Calculate Earth RA DEC Dist
	earth_ra   = earth_coo.fk5.ra.rad
	earth_dec  = earth_coo.fk5.dec.rad
	earth_dist = earth_coo.fk5.distance.km
	
	#CZTI RA DEC
	czti_ra  = czti_z.fk5.ra.rad
	czti_dec = czti_z.fk5.dec.rad
	
	
#Load Bayestar File
	NSIDE = 512
	no_pix = hp.nside2npix(NSIDE)
	prob = np.zeros(no_pix) 
#	prob = hp.read_map(loc_map)
#	NSIDE = fits.open(args.loc_map)[1].header['NSIDE']
	prob2 = np.copy(prob)
	prob3 = np.copy(prob)
#	
#	#ColorMap
	cmap = plt.cm.YlOrRd
	cmap.set_under("w")
#	
#	hp.mollview(prob, title='Complete Localisation Map', rot=(180, 0), cmap=cmap)
#	
#	hp.graticule()
#	plotfile.savefig()
	
	#Earth Occult

	czti_theta = np.pi/2 - czti_dec
	czti_phi   = czti_ra
	earth_theta = np.pi/2 - earth_dec
	earth_phi   = earth_ra
	earth_occult = np.arcsin(6378./earth_dist)
	earth_vec = hp.ang2vec(earth_theta, earth_phi)
	earth = hp.query_disc(NSIDE, earth_vec, earth_occult)
		
	prob[earth] = np.nan
	not_occult = np.nansum(prob)
	
	# Add the GRB now 
	grb_theta = np.pi/2 - dec
	grb_phi = ra

	hp.mollview(prob, title="Earth Occulted Localisation Map (Remaining Probability = {not_occult:2.3f})".format(not_occult = not_occult),rot=(earth_theta, earth_phi), cmap =cmap)
	hp.graticule()
	hp.projscatter(czti_theta, czti_phi,color = 'r' ,marker='x')	
	hp.projtext(czti_theta, czti_phi, 'CZTI')
	hp.projscatter(grb_theta, grb_phi,color = 'k',marker='o')
	hp.projtext(grb_theta, grb_phi, grbdir)
	hp.projscatter(earth_theta, earth_phi,color = 'g',marker='^')
	hp.projtext(earth_theta, earth_phi, 'Earth')

	plotfile.savefig()
	
	
	
	#Above Focal Plane
	back_vec  = hp.ang2vec(np.pi - czti_theta , czti_phi)
	front = hp.query_disc(NSIDE, back_vec, np.pi/2)
	
	prob2[front] = np.nan
	front_prob = np.nansum(prob2)
	
	hp.mollview(prob2, title="Above the Focal Plane Localisation Map (Remaining Probability = {front_prob:2.3f})".format(front_prob = front_prob),rot=(180, 0), cmap=cmap)
	hp.graticule()
	hp.projscatter(czti_theta, czti_phi, color = 'r' ,marker='x')
	hp.projtext(czti_theta, czti_phi, 'CZTI')
	hp.projscatter(grb_theta, grb_phi,color = 'k',marker='o')
	hp.projtext(grb_theta, grb_phi, grbdir)
	hp.projscatter(earth_theta, earth_phi,color = 'g',marker='^')
	hp.projtext(earth_theta, earth_phi, 'Earth')

	plotfile.savefig()
	
	#Earth Occult and Above Focal Plane
	prob2[earth] = np.nan
	final_prob = np.nansum(prob2)
	
	hp.mollview(prob2, title="Earth Occult + Above Focal Plane (Remaining Probability = {final_prob:2.3f})".format(final_prob = final_prob),rot=(180, 0), cmap=cmap)
	hp.graticule()
	hp.projscatter(czti_theta, czti_phi, color = 'r' ,marker='x')
	hp.projtext(czti_theta, czti_phi, 'CZTI')
	hp.projscatter(grb_theta, grb_phi,color = 'k',marker='o')
	hp.projtext(grb_theta, grb_phi, grbdir)
	hp.projscatter(earth_theta, earth_phi,color = 'g',marker='^')
	hp.projtext(earth_theta, earth_phi, 'Earth')

	plotfile.savefig()
	
	plotfile.close()

	return

#Main Code

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("configfile", help="Config file for the transient in question", type=str)
parser.add_argument("--window", type=float, default= 50,help="Localisation and Coverage T sec Before and After Tigger Default = 50 sec")
args = parser.parse_args()
runconf = get_configuration(args)

grbdir = runconf['name']
ra = runconf['ra']
dec = runconf['dec']
mkffile = runconf['mkffile']

mkfdata = fits.getdata(mkffile, 1)
#locmap = args.loc_map
delta = args.window
#mission_time = Time(args.trigtime) - Time('2010-01-01 00:00:00')
trigtime = runconf['trigtime']


#creat_map(mkfdata, ra, dec,  trigtime - delta, "Localisation_and_CZTI_Coverage_Before_{T:3.1f}.pdf".format(T=delta))
creat_map(mkffile, ra, dec,  trigtime, "plots/earth_occult/"+grbdir+"_earth_occult.pdf")
#creat_map(mkfdata, ra, dec,  trigtime + delta, "Localisation_and_CZTI_Coverage_After_{T:3.1f}.pdf".format(T=delta))

