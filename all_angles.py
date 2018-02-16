from astropy.table import Table
from astropy.io import fits
from astropy.time import Time
import astropy.coordinates as coo
import astropy.units as u
import ConfigParser, argparse
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from offaxispos import get_configuration

#------------------------------------------------------------------------
# Define functions required for processing

def get_angles(mkffile, trigtime, ra_tran, dec_tran, window=10):
    """
    Calculate thetax, thetay using astropy
    Use pitch, roll and yaw information from the MKF file
    """
    # x = -yaw
    # y = +pitch
    # z = +roll

    # Read in the MKF file
    mkfdata = fits.getdata(mkffile, 1)
    sel = abs(mkfdata['time'] - trigtime) < window

    # Get pitch, roll, yaw
    # yaw is minus x
    pitch = coo.SkyCoord( np.median(mkfdata['pitch_ra'][sel]) * u.deg, np.median(mkfdata['pitch_dec'][sel]) * u.deg )
    roll = coo.SkyCoord( np.median(mkfdata['roll_ra'][sel]) * u.deg, np.median(mkfdata['roll_dec'][sel]) * u.deg )
    yaw_ra  = (180.0 + np.median(mkfdata['yaw_ra'][sel]) ) % 360
    yaw_dec = -np.median(mkfdata['yaw_dec'][sel])
    minus_yaw = coo.SkyCoord( yaw_ra * u.deg, yaw_dec * u.deg )

    # Earth - the mkffile has satellite xyz 
    earthx = int(np.median(mkfdata['posx'][sel])* 1000) * u.m
    earthy = int(np.median(mkfdata['posy'][sel])* 1000) * u.m
    earthz = int(np.median(mkfdata['posz'][sel])* 1000) * u.m
    earth = coo.SkyCoord(-earthx, -earthy, -earthz, frame='icrs', representation='cartesian')
    # Sun coordinates:
    #sunra = np.median(mkfdata['sun_ra'][sel]) * u.deg
    #sundec = np.median(mkfdata['sun_dec'][sel]) * u.deg
    #sun = coo.SkyCoord(sunra, sundec)
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

    phi = u.rad * np.arctan2(cy, cx) # phi is the angle of the transient from the x axis

    if (phi.to(u.deg).value < 0 ):
        phi_new = 360 + phi.to(u.deg).value
    else:
        phi_new = phi.to(u.deg).value
    return  minus_yaw, pitch, roll, transient, earth

parser = argparse.ArgumentParser()
parser.add_argument("configfile", nargs="?", help="Name of configuration file", type=str)
parser.add_argument('--noloc', dest='noloc', action='store_true')
parser.set_defaults(noloc=False)
args = parser.parse_args()
runconf = get_configuration(args)

grbdir = runconf['l2file'][0:10]
mkffile = runconf['mkffile']
trigtime = runconf['trigtime']
ra_tran = runconf['ra']
dec_tran = runconf['dec']
plot_file = "plots/all_angles/"+grbdir+"_all_angles.pdf"

x, y, z, t, earth = get_angles(mkffile, trigtime, ra_tran, dec_tran, window=10)

all_coords = np.array([x,y,z,t,earth])

names = ["CZT-X","CZT-Y","CZT-Z","Transient","Earth"]

data = np.ndarray((len(names),len(names)))

for i in range(len(names)):
	for j in range(len(names)):
		data[i][j] = "{a:0.2f}".format(a=all_coords[i].separation(all_coords[j]).value)
		print "{a:0.2f}".format(a=all_coords[i].separation(all_coords[j]).value)
print all_coords

plt.figure(figsize=(4,2))
plt.title("All angles for : "+grbdir)
plt.axis('off')
plt.table(cellText=data,rowLabels=names,colLabels=names,colWidths=np.ones(len(names))*0.14,loc='center')
plt.tight_layout()
plt.savefig(plot_file)


