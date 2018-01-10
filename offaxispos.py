#!/usr/bin/env python2.7

"""
    offaxispos.py

    Aim: Find the position of an off-axis transient

    **** NOTE: not yet a generalized code ****
    Needs the following files to be present in the same folder:
        nonoise_grb_AS1P01_003T01_9000000002cztM0_level2.events
        czti_Aepix.out (compiled from czti_Aepix.f and czti_pixarea.f)

    Algorithm steps: 
      Make a grid of responses around Ra0, Dec0
      Fit data = const + mult * resp, find chi^2
      See which position gives least chi^2

    Version  : $Rev: 1186 $
    Last Update: $Date: 2016-07-18 11:13:39 +0530 (Mon, 18 Jul 2016) $

"""

# v1.0  : First code added to SVN, works only for GRB151006A
# v1.1  : Updated to work with new .out files containing detx,dety. Actually works!
# v2.0  : Major upgrade: use healpy to get well spaced grid, read configuration from files, make pdf plots, etc
# v2.1  : Added fancy plots
# v2.2  : Added the "noloc" keyword for making plots without doing localisation calculations
# v2.3  : Changes in plotting order, handling zero count cases correctly
# v2.4  : Made it easier for external codes to write a config file
version = "2.4"

import subprocess, os, shutil
import ConfigParser, argparse
from astropy.table import Table
from astropy.io import fits
from astropy.stats import sigma_clip, sigma_clipped_stats
from scipy.optimize import curve_fit
import astropy.coordinates as coo
import astropy.units as u

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

parser = argparse.ArgumentParser(epilog="""

    Version  : $Rev: 1186 $
    Last Update: $Date: 2016-07-18 11:13:39 +0530 (Mon, 18 Jul 2016) $

""")
parser.add_argument("configfile", nargs="?", help="Name of configuration file", type=str)
parser.add_argument('--noloc', dest='noloc', action='store_true')
parser.set_defaults(noloc=False)


#------------------------------------------------------------------------
# Define functions required for processing

def plot_map(data, pixlist, nside, runconf, title, unit):
    """
    Use healpy to make a plot and add coordinate grid
    """
    hpmap = np.repeat(np.nan, hp.pixelfunc.nside2npix(nside))
    hpmap[pixlist] = data
    plot_lon_cen, plot_lat_cen = runconf['ra'], runconf['dec']
    if plot_lon_cen > 180:
        plot_lon_cen -= 360
    plot_bounds = (1.0 + 1.1 * runconf['radius']) * np.array([-1.0, 1.0])
    
    hp.cartview(hpmap, rot=(plot_lon_cen, plot_lat_cen, 0), 
            lonra=plot_bounds, latra=plot_bounds,
            notext=True, unit=unit, title=title, 
            min=min(data), max=max(data), coord='C', flip='astro')

    dec0 = np.round(runconf['dec'])
    dec_spacing = runconf['radius']/2.0
    decs = dec0 + np.arange(-2*runconf['radius'], 2.1*runconf['radius'], dec_spacing)
    decs_min, decs_max = min(decs), max(decs)

    ra0 = np.round(runconf['ra'])
    cosdelt =np.cos(np.deg2rad(dec0)) 
    ra_spacing = runconf['radius']/cosdelt / 2.0
    ras = ra0 + np.arange(-2.0*runconf['radius']/cosdelt, 2.1*runconf['radius']/cosdelt, ra_spacing)
    ras_min, ras_max = min(ras), max(ras)
    #num_ras = np.ceil(1.0 * runconf['radius'] / grid_spacing / np.cos(np.deg2rad(min(decs))) )

    line_dec = np.linspace(decs_min, decs_max, 100)
    line_ra = np.linspace(ras_min, ras_max, 100)
    for ra in ras:
        hp.projplot(np.repeat(ra, 100), line_dec, lonlat=True, ls='dashed', color='black')
        hp.projtext(ra, dec0, r"{ra:0.1f}$^\circ$".format(ra=ra), lonlat=True, clip_on=True)
    for dec in decs:
        hp.projplot(line_ra, np.repeat(dec, 100), lonlat=True, ls='dashed', color='black')
        hp.projtext(ra0, dec, r"{dec:0.1f}$^\circ$".format(dec=dec), lonlat=True, clip_on=True, rotation=90)
    return

def rd2txy(ra0, dec0, twist, ra_tran, dec_tran):
    """
    Call Dipankar's radec2txty fortran code to get the theta_x, theta_y of any
    point on the sky.
    In tests, the run time was 7 ms/call
    """
    tcalc = subprocess.Popen([runconf['txycode']],  stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = tcalc.communicate("{ra0} {dec0}\n{twist}\n{ra_tran} {dec_tran}".format(ra0=ra0, dec0=dec0, twist=twist, ra_tran=ra_tran, dec_tran=dec_tran))
    junk, tx, ty = out.rsplit(None, 2)
    return float(tx), float(ty)

def get_twist(mkffile, trigtime, time_range = 100.0):
    """
    Open the MKF file, return roll angle to trigtime and return
    To safegaurd against data jumps, I use a median value from '+-time_range'
    around the trigger time
    """
    mkfdata = fits.getdata(mkffile, 1)
    sel = abs(mkfdata['time'] - trigtime) < 100
    return 0.0 - np.median(mkfdata['roll_rot'][sel])

def txy(mkffile, trigtime, ra_tran, dec_tran):
    """
    Calculate thetax, thetay using astropy
    Use pitch, roll and yaw information from the MKF file
    """
    # x = -yaw
    # y = +pitch
    # z = +roll

    # Read in the MKF file
    mkfdata = fits.getdata(mkffile, 1)
    sel = abs(mkfdata['time'] - trigtime) < 100

    # Get pitch, roll, yaw
    # yaw is minus x
    pitch = coo.SkyCoord( np.median(mkfdata['pitch_ra'][sel]) * u.deg, np.median(mkfdata['pitch_dec'][sel]) * u.deg )
    roll = coo.SkyCoord( np.median(mkfdata['roll_ra'][sel]) * u.deg, np.median(mkfdata['roll_dec'][sel]) * u.deg )
    yaw_ra  = (180.0 + np.median(mkfdata['yaw_ra'][sel]) ) % 360
    yaw_dec = -np.median(mkfdata['yaw_dec'][sel])
    minus_yaw = coo.SkyCoord( yaw_ra * u.deg, yaw_dec * u.deg )

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

    return thetax.to(u.deg).value, thetay.to(u.deg).value, minus_yaw, pitch, roll, transient

def plot_xyzt(ax, x, y, z, t):
    """
    Make a subplot that shows X, Y, Z axes and a transient vector
    The input coordinates are astropy.coordinate.SkyCoord objects
    """
    colors = ['blue', 'green', 'red', 'black']
    names = ['X', 'Y', 'Z', 'T']
    zdirs = ['x', 'y', 'z', None]

    ax.set_xlim(-1.2,1.2)
    ax.set_ylim(-1.2,1.2)
    ax.set_zlim(-1.2,1.2)

    for count, dirn in enumerate([x, y, z, t]):
        xx, yy, zz = dirn.cartesian.x.value, dirn.cartesian.y.value, dirn.cartesian.z.value
        ax.quiver(xx, yy, zz, xx, yy, zz, color=colors[count])
        ax.text(xx, yy, zz, names[count], zdirs[count])

    #ax.set_xlabel("RA = 0")
    #ax.set_zlabel("Pole")
    return

def add_satellite(ax, coo_x, coo_y, coo_z):
    """
    Add a basic version of the satellite outline to the plots
    """
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    tr = np.transpose(np.vstack((coo_x.cartesian.xyz.value, coo_y.cartesian.xyz.value, coo_z.cartesian.xyz.value)))

    alpha_czti = 0.5
    alpha_radiator = 0.5
    alpha_sat = 0.3

    color_czti = 'yellow'
    color_radiator = 'black'
    color_sat = 'green'

    c_w2 = 0.15 # czti half-width
    c_h  = 0.30 # czti height
    c_hr = 0.40 # czti radiator height
    sat_w = 0.6

    # For each surface, do the following:
    # verts = []
    # verts.append([tuple(tr.dot(np.array[cx, cy, cz]))])
    # surf = Poly3DCollection(verts)
    # surf.set_alpha()
    # surf.set_color()
    # ax.add_collection3d(surf)
    
    # +x rect
    verts = []
    verts.append(tuple(tr.dot(np.array([c_w2, c_w2, 0]))))
    verts.append(tuple(tr.dot(np.array([c_w2, c_w2, c_h]))))
    verts.append(tuple(tr.dot(np.array([c_w2, -c_w2, c_h]))))
    verts.append(tuple(tr.dot(np.array([c_w2, -c_w2, 0]))))
    surf = Poly3DCollection([verts])
    surf.set_alpha(alpha_czti)
    surf.set_color(color_czti)
    ax.add_collection3d(surf)
    
    # +y rect
    verts = []
    verts.append(tuple(tr.dot(np.array([c_w2, c_w2, 0]))))
    verts.append(tuple(tr.dot(np.array([c_w2, c_w2, c_h]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, c_w2, c_h]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, c_w2, 0]))))
    surf = Poly3DCollection([verts])
    surf.set_alpha(alpha_czti)
    surf.set_color(color_czti)
    ax.add_collection3d(surf)

    # -y rect
    verts = []
    verts.append(tuple(tr.dot(np.array([c_w2, -c_w2, 0]))))
    verts.append(tuple(tr.dot(np.array([c_w2, -c_w2, c_h]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, -c_w2, c_h]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, -c_w2, 0]))))
    surf = Poly3DCollection([verts])
    surf.set_alpha(alpha_czti)
    surf.set_color(color_czti)
    ax.add_collection3d(surf)
    
    # -x radiator plate
    verts = []
    verts.append(tuple(tr.dot(np.array([-c_w2, c_w2, 0]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, c_w2, c_hr]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, -c_w2, c_hr]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, -c_w2, 0]))))
    surf = Poly3DCollection([verts])
    surf.set_alpha(alpha_radiator)
    surf.set_color(color_radiator)
    ax.add_collection3d(surf)

    # # Bottom CZTI only
    # verts = []
    # verts.append(tuple(tr.dot(np.array([c_w2, c_w2, 0]))))
    # verts.append(tuple(tr.dot(np.array([-c_w2, c_w2, 0]))))
    # verts.append(tuple(tr.dot(np.array([-c_w2, -c_w2, 0]))))
    # verts.append(tuple(tr.dot(np.array([c_w2, -c_w2, 0]))))
    # surf = Poly3DCollection([verts])
    # surf.set_alpha(alpha_czti)
    # surf.set_color(color_czti)
    # ax.add_collection3d(surf)

    # Satellite top
    verts = []
    verts.append(tuple(tr.dot(np.array([sat_w-c_w2, sat_w-c_w2, 0]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, sat_w-c_w2, 0]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, -c_w2, 0]))))
    verts.append(tuple(tr.dot(np.array([sat_w-c_w2, -c_w2, 0]))))
    surf = Poly3DCollection([verts])
    surf.set_alpha(alpha_sat)
    surf.set_color(color_sat)
    ax.add_collection3d(surf)

    # Satellite bottom
    verts = []
    verts.append(tuple(tr.dot(np.array([sat_w-c_w2, sat_w-c_w2, -sat_w]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, sat_w-c_w2, -sat_w]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, -c_w2, -sat_w]))))
    verts.append(tuple(tr.dot(np.array([sat_w-c_w2, -c_w2, -sat_w]))))
    surf = Poly3DCollection([verts])
    surf.set_alpha(alpha_sat)
    surf.set_color(color_sat)
    ax.add_collection3d(surf)

    # Satellite back (radiator side)
    verts = []
    verts.append(tuple(tr.dot(np.array([-c_w2, sat_w-c_w2, 0]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, sat_w-c_w2, -sat_w]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, -c_w2, -sat_w]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, -c_w2, 0]))))
    surf = Poly3DCollection([verts])
    surf.set_alpha(alpha_sat)
    surf.set_color(color_sat)
    ax.add_collection3d(surf)

    # Satellite front (opposite radiator side)
    verts = []
    verts.append(tuple(tr.dot(np.array([sat_w-c_w2, sat_w-c_w2, 0]))))
    verts.append(tuple(tr.dot(np.array([sat_w-c_w2, sat_w-c_w2, -sat_w]))))
    verts.append(tuple(tr.dot(np.array([sat_w-c_w2, -c_w2, -sat_w]))))
    verts.append(tuple(tr.dot(np.array([sat_w-c_w2, -c_w2, 0]))))
    surf = Poly3DCollection([verts])
    surf.set_alpha(alpha_sat)
    surf.set_color(color_sat)
    ax.add_collection3d(surf)

    # Satellite right (-y, common to czti)
    verts = []
    verts.append(tuple(tr.dot(np.array([sat_w-c_w2, -c_w2, 0]))))
    verts.append(tuple(tr.dot(np.array([sat_w-c_w2, -c_w2, -sat_w]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, -c_w2, -sat_w]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, -c_w2, 0]))))
    surf = Poly3DCollection([verts])
    surf.set_alpha(alpha_sat)
    surf.set_color(color_sat)
    ax.add_collection3d(surf)

    # Satellite left (+y)
    verts = []
    verts.append(tuple(tr.dot(np.array([sat_w-c_w2, sat_w-c_w2, 0]))))
    verts.append(tuple(tr.dot(np.array([sat_w-c_w2, sat_w-c_w2, -sat_w]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, sat_w-c_w2, -sat_w]))))
    verts.append(tuple(tr.dot(np.array([-c_w2, sat_w-c_w2, 0]))))
    surf = Poly3DCollection([verts])
    surf.set_alpha(alpha_sat)
    surf.set_color(color_sat)
    ax.add_collection3d(surf)

    return


def visualize_3d(x, y, z, t, thetax, thetay, name):
    """
    Make a plot that allows us to visualize the transient location in 3d
    Use matplotlib Axes3D
    Uses the helper function plot_xyzt
    """
    # Set ax.azim and ax.elev to ra, dec
    global runconf

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    plt.suptitle(r"Visualisation of {name} in 3d:$\theta_x$={tx:0.1f},$\theta_y$={ty:0.1f}".format(name=name, tx=thetax, ty=thetay))
    # Z
    ax = plt.subplot(2, 2, 1, projection='3d')
    plot_xyzt(ax, x, y, z, t)
    ax.azim = z.ra.deg
    ax.elev = z.dec.deg
    add_satellite(ax, x, y, z)
    ax.set_title("View from CZTI pointing (z)")

    # Transient
    ax = plt.subplot(2, 2, 2, projection='3d')
    plot_xyzt(ax, x, y, z, t)
    ax.azim = t.ra.deg
    ax.elev = t.dec.deg
    add_satellite(ax, x, y, z)
    ax.set_title("View from nominal transient direction")

    # X
    ax = plt.subplot(2, 2, 3, projection='3d')
    plot_xyzt(ax, x, y, z, t)
    ax.azim = x.ra.deg
    ax.elev = x.dec.deg
    add_satellite(ax, x, y, z)
    ax.set_title("View from CZTI X axis")

    # Z
    ax = plt.subplot(2, 2, 4, projection='3d')
    plot_xyzt(ax, x, y, z, t)
    ax.azim = y.ra.deg
    ax.elev = y.dec.deg
    add_satellite(ax, x, y, z)
    ax.set_title("View from CZTI Y axis")

    return

def get_default_configuration():
    """
    Return dicts with the default values for config files
    """
    # Pre-configured default values for various parameters:
    default_config = {
            "name":"Transient",
            "auto":True,
            "ra":0.0,
            "dec":0.0,
            "radius":10.0,
            "resolution":1.8,
            "energy":70.0,
            "pixsize": 16,
            "respcode":"czti_Aepix.out",
            "txycode":"radec2txty.out",
            "resppath":"pixarea",
            "plotfile":"plots/localize.pdf",
            "verbose":True,
            "do_fit":True
            }
    required_config = {
            'l2file':"_level2.evt",
            'infile':"file.evt",
            'mkffile':"file.mkf",
            'trigtime':0.00,
            'transtart':0.00,
            'tranend':0.00,
            'bkg1start':0.00,
            'bkg1end':0.00,
            'bkg2start':0.00,
            'bkg2end':0.00
            }
    return default_config, required_config

def write_configuration(creator, creator_version, default_values, required_values, filename=None):
    """
    Print default config file on screen, or optionally save it to a file
    """
    default_config, required_config = get_default_configuration()
    default_config.update(default_values)
    required_config.update(required_values)
    printdict = {"prog":creator, "version": creator_version}
    printdict.update(default_config)
    printdict.update(required_config)
    configuration = """
# Sample configuration file for {prog} version {version}
# SVN revision $Rev: 1186 $
# Last updated $Date: 2016-07-18 11:13:39 +0530 (Mon, 18 Jul 2016) $
# Comment lines starting with # are ignored
# Do not add extra spaces!
#
#
[Config]
#------------------------------------------------------------------------
# Event-specific configuration
#
# Event name (for plots etc)
name:{name}
#
# Level2 file for reprocessing
l2file:{l2file}
# Input file for analysis
infile:{infile}
#
# MKF file for getting rotation
mkffile:{mkffile}
#
# Energy (keV)
energy:{energy}
# Trigger time in czti seconds
# Seconds since 1/1/2010, 00:00:00 UTC
trigtime:{trigtime}
#
# Start and end time for data to use for localisation
transtart:{transtart}
tranend:{tranend}
#
# Define two windows for estimating background to subtract for localisation
bkg1start:{bkg1start}
bkg1end:{bkg1end}
bkg2start:{bkg2start}
bkg2end:{bkg2end}
# 
# Transient location in decimal degrees
# Set auto:True if search is to be centered on spacecraft pointing
auto:{auto}
ra:{ra:0.2f}   ; Ignored if auto=True
dec:{dec:0.2f}  ; Ignored if auto=True
# 
# Transient search radius (degrees) and approximate resolution (degrees)
radius:{radius:0.2f}
resolution:{resolution:0.2f}
# actual resolution is the nearest healpix resolution available
# Some of the supported values are 7.33, 3.66, 1.83, 0.92, 0.46, 0.23, 0.11
# ANY value is allowed, the closest supported value will actually be used
#
#------------------------------------------------------------------------
# Generic configuration parameters
#
# Grouping pixels: group n x n pixels into a "superpixel"
# May be 1, 2, 4, 8, 16
pixsize:{pixsize}
#
# Use fitting for calculating resposnse, or just scale?
# If True, best-fit "flux" is calculated for image = background + flux * source
# If False, "flux" is simply (np.sum(source_image) - np.sum(bkg_image)) / np.sum(response)
do_fit:True
#
# Codes for calculating responses, and theta_x, theta_y from ra,dec
# Must be executable
respcode:{respcode}
txycode:{txycode}
#
# Location of response files
resppath:{resppath}
#
# Output plot pdf path
plotfile:{plotfile}
# 
# Give verbose output: True / False
verbose:{verbose}
""".format(**printdict)
    if filename is not None:
        with open(filename, 'w') as thefile:
            thefile.write(configuration)
    else:
        print configuration



def get_configuration(args):
    """
    Read the configuration file specified.
    Exit with error if any required parameter is missing.

    Dump default configuration file if no file specified on command line.
    """

    #------------------------------------------------------------------------
    # If configfile is not specified, dump a configfile on screen
    default_config, required_config = get_default_configuration()
    if args.configfile is None:
        write_configuration(parser.prog, version, default_config, required_config)
        raise SystemExit

    #------------------------------------------------------------------------
    # If you are here, then a configfile was given. Try to parse it
    userconf = ConfigParser.SafeConfigParser()
    userconf.read(args.configfile)
    runconf = {}

    if userconf.has_option('Config', 'verbose'): 
        runconf['verbose'] = userconf.getboolean('Config', 'verbose')
    else:
        runconf['verbose'] = default_config['verbose']

    if runconf['verbose']: print "Configuration for this run: "
    for key in required_config.keys():
        try:
            runconf[key] = userconf.getfloat('Config', key)
            if runconf['verbose']: print "  > {key} = {val}".format(key=key, val=runconf[key])
        except ConfigParser.NoOptionError:
            if runconf['verbose']: print "\nError: Required parameter {key} missing from config file!!".format(key=key)
            if runconf['verbose']: print "Update the file {configfile} and try again\n".format(configfile=args.configfile)
            raise SystemExit
        except ValueError:
            runconf[key] = userconf.get('Config', key)
            if runconf['verbose']: print "  > {key} = {val}".format(key=key, val=runconf[key])

    for key in default_config.keys():
        if key == 'verbose': continue
        try:
            runconf[key] = userconf.getfloat('Config', key)
        except ConfigParser.NoOptionError:
            runconf[key] = default_config[key]
            if runconf['verbose']: print "Using default value for {key}".format(key=key)
        except ValueError:
            runconf[key] = userconf.get('Config', key)
        if runconf['verbose']: print "  > {key} = {val}".format(key=key, val=runconf[key])

    # Now convert the bool values: verbose is already processed
    boolkeys = ['auto', 'do_fit']
    for b_key in boolkeys:
        #print b_key, runconf[b_key]
        test_string = "{b_key}".format(b_key = runconf[b_key])
        #print test_string
        if (test_string[0].upper() == 'T') or (test_string[0] == '1'):
            runconf[b_key] = True
        else:
            runconf[b_key] = False
        #print b_key, runconf[b_key]

    # Configuaration finally parsed!
    # At some future time, I should validate it...
    return runconf

def clean_mask(image, sig=5, iters=3):
    """
    Mask out input image by removing 5-sigma outliers and zero pixels.
    Return cleaned image and mask.
    """

    mean, median, stddev = sigma_clipped_stats(image[image>0], sigma=sig, iters=iters)

    mask_bad = (np.abs(image - median) > sig * stddev) | (image == 0)
    image_ret = np.copy(image)
    image_ret[mask_bad] = 0
    
    return image_ret, mask_bad

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

def f2image(infile, maskit=True):
    """
    Convert the ".out" file from Dipankar's code into an image
    """
    global master_mask

    tab = Table.read(infile, format="ascii", names=("quadrant", "detx", "dety", "area"), comment="#")
    pixel_edges = np.arange(-0.5, 63.6)
    image = np.zeros((128,128))
    im_sub = np.zeros((64,64))

    data = tab[tab["quadrant"] == 0]
    im_sub[data['detx'], data['dety']] = data['area']
    im_sub = np.transpose(im_sub)
    image[64:128,0:64] = np.copy(im_sub)

    data = tab[tab["quadrant"] == 1]
    im_sub[data['detx'], data['dety']] = data['area']
    im_sub = np.transpose(im_sub)
    image[64:128,64:128] = np.copy(im_sub)

    data = tab[tab["quadrant"] == 2]
    im_sub[data['detx'], data['dety']] = data['area']
    im_sub = np.transpose(im_sub)
    image[0:64,64:128] = np.copy(im_sub)

    data = tab[tab["quadrant"] == 3]
    im_sub[data['detx'], data['dety']] = data['area']
    im_sub = np.transpose(im_sub)
    image[0:64,0:64] = np.copy(im_sub)

    if maskit:
        image[master_mask] = 0

    return image

def calc_resp(energy, theta_x, theta_y):
    """
    Call Dipankar's czti_Aepix fortran code to get the 
    detector response for a given angle. Note that the code 
    has been modified to give filenames in this format. If an
    output file of that name already exists, it is not created 
    again by this subroutine.
    """
    global runconf
    respfile_name = "pixarea_{energy:0d}_{theta_x:0d}_{theta_y:0d}.out".format(energy=int(energy), theta_x=int(theta_x), theta_y=int(theta_y))
    respfile_full = "{resppath}/{respname}".format(resppath=runconf['resppath'], respname=respfile_name)
    if not os.path.exists(respfile_full):
        respmake = subprocess.Popen([runconf['respcode']], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        respmake.communicate("{energy} {theta_x} {theta_y}".format(energy=int(energy), theta_x=int(theta_x), theta_y=int(theta_y)))
        shutil.move(respfile_name, respfile_full)
    else:
        # The fotran output file already exists
        pass
    return respfile_full

def plot_resp(energy, theta_x, theta_y):
    """
    Save a plot of the response calculated from Dipankar's codes
    """
    global runconf
    respfile_f = calc_resp(energy, theta_x, theta_y)
    response = f2image(respfile_f)
    plotfile = "{outdir}/pixarea_{energy:0d}_{theta_x:0d}_{theta_y:0d}.png".format(outdir=runconf['outdir'], 
            energy=int(energy), theta_x=int(theta_x), theta_y=int(theta_y))
    plt.clf()
    plt.imshow(response, origin="lower")
    plt.title(r"Energy: {energy:0d} keV, $\theta_x$ = {theta_x:0d}, $\theta_y$ = {theta_y:0d}".format(energy=int(energy), theta_x=int(theta_x), theta_y=int(theta_y)))
    plt.colorbar()
    plt.savefig(plotfile)
    return

def inject(energy, theta_x, theta_y, flux=1, noise=0):
    """
    Create a fake image, using response from a certain angle
    Add poission noise
    """
    respfile_f = calc_resp(energy, theta_x, theta_y)
    image = f2image(respfile_f) * flux + np.random.poisson(noise, (128, 128))
    return image

def evt2image(infile, tstart=0, tend=1e20):
    """
    Read an events file (with or without energy), and return a combined
    DPH for all 4 quadrants. 
    If tstart and tend are given, use data only in that time frame. The 
    default values are set to exceed expected bounds for Astrosat.
    """
    hdu = fits.open(infile)
    pixel_edges = np.arange(-0.5, 63.6)

    data = hdu[1].data[np.where( (hdu[1].data['Time'] >= tstart) & (hdu[1].data['Time'] <= tend) )]
    im1 = np.histogram2d(data['detx'], data['dety'], bins=(pixel_edges, pixel_edges))
    im1 = np.transpose(im1[0])
    data = hdu[2].data[np.where( (hdu[2].data['Time'] >= tstart) & (hdu[2].data['Time'] <= tend) )]
    im2 = np.histogram2d(data['detx'], data['dety'], bins=(pixel_edges, pixel_edges))
    im2 = np.transpose(im2[0])
    data = hdu[3].data[np.where( (hdu[3].data['Time'] >= tstart) & (hdu[3].data['Time'] <= tend) )]
    im3 = np.histogram2d(data['detx'], data['dety'], bins=(pixel_edges, pixel_edges))
    im3 = np.transpose(im3[0])
    data = hdu[4].data[np.where( (hdu[4].data['Time'] >= tstart) & (hdu[4].data['Time'] <= tend) )]
    im4 = np.histogram2d(data['detx'], data['dety'], bins=(pixel_edges, pixel_edges))
    im4 = np.transpose(im4[0])

    image = np.zeros((128,128))
    image[0:64,0:64] = im4
    image[0:64,64:128] = im3
    image[64:128,0:64] = im1
    image[64:128,64:128] = im2

    #plt.imshow(image, origin="lower")
    return image

def fitbkg(x, flux):
    global bkg_image
    return bkg_image.flatten() + abs(flux) * x

#------------------------------------------------------------------------

#energies = [70]
#thetas_x = [25, 35, 45]
#thetas_y = [40, 50, 60, 70]
#thetas_x = [25, 30, 35, 40, 45]
#thetas_y = [50, 55, 60, 65, 70]

#thetas_x = np.arange(15, 56, 3)
#thetas_y = np.arange(35, 76, 3)
#thetas_x = np.arange(25, 45, 3)
#thetas_y = np.arange(48, 68, 3)

# GRB160119A
#     runconf['infile'] = 'GRB160119A.evt'
#     #runconf['infile'] = "test_dx30_dy60.evt"
#     src_image = evt2image(runconf['infile'], 190868870., 190868970.)   # t_trig+100 to +200 - note that peak is trig+150
#     crude_source_image, mask_src = clean_mask(src_image)
#     print "Source photons    : {sp:0d}".format(sp=int(np.sum(crude_source_image)))
#     
#     # Background image
#     bkg_image1 = evt2image(runconf['infile'], 190868970., 190868670.)   # -100 to -400
#     bkg_image2 = evt2image(runconf['infile'], 190869270., 190869570.)   # +500 to +800
#     
# GRB160119A is at approx 40, -30, peak is at 190868770+150 sec
# thetas_x = np.arange(20, 60, 5)
# thetas_y = np.arange(-50, -10, 5)
#
# source_image = inject(60, 50, 40, 15, 5)
# GRB151006A is at 34, 58
#   # Source image:
#   infile = 'AS1P01_003T01_9000000002cztM0_level2.evt_grb_gt60kev'
#   #infile = "test_dx30_dy60.evt"
#   src_image = evt2image(runconf['infile'], 181821200., 181821400.)
#   crude_source_image, mask_src = clean_mask(src_image)
#   print "Source photons    : {sp:0d}".format(sp=int(np.sum(crude_source_image)))
#   
#   # Background image
#   bkg_image1 = evt2image(runconf['infile'], 181821000., 181821200.)
#   bkg_image2 = evt2image(runconf['infile'], 181821500., 181821700.)
#   bkg_image0 = (bkg_image1 + bkg_image2) / 2.0
#   crude_bkg_image, mask_bkg = clean_mask(bkg_image0)
#   print "Background photons: {bp:0d}".format(bp=int(np.sum(crude_bkg_image)))


#------------------------------------------------------------------------
# Main code begins:

if __name__ == "__main__":
    # Parse arguments to get config file
    args = parser.parse_args()
    runconf = get_configuration(args)
    imsize = 128/runconf['pixsize']

    #------------------------------------------------------------------------
    # Begin calculations

    # Source image:
    #runconf['infile'] = "test_dx30_dy60.evt"
    src_image = evt2image(runconf['infile'], runconf['transtart'], runconf['tranend'])   # t_trig+100 to +200 - note that peak is trig+150
    crude_source_image, mask_src = clean_mask(src_image)
    if runconf['verbose']: print "Source photons    : {sp:0d}".format(sp=int(np.sum(crude_source_image)))

    # Background image
    bkg_image1 = evt2image(runconf['infile'], runconf['bkg1start'], runconf['bkg1end'])   # Example: -100 to -400
    bkg_image2 = evt2image(runconf['infile'], runconf['bkg2start'], runconf['bkg2end'])   # Example: +500 to +800
    bkg_image0 = 1.0 * (bkg_image1 + bkg_image2) * (runconf['tranend'] - runconf['transtart']) / (runconf['bkg2end'] - runconf['bkg2start'] + runconf['bkg1end'] - runconf['bkg1start'])
    crude_bkg_image, mask_bkg = clean_mask(bkg_image0)
    if runconf['verbose']: print "Background photons (scaled by time): {bp:0d}".format(bp=int(np.sum(crude_bkg_image)))


    # now apply same mask to source and background image, then resample them
    # Mask is True for outliers
    master_mask = mask_src | mask_bkg
    crude_source_image[master_mask] = 0
    source_image = resample(crude_source_image, runconf['pixsize'])
    crude_bkg_image[master_mask] = 0
    bkg_image = resample(crude_bkg_image, runconf['pixsize'])

    # Plot source, background and S-B images
    if runconf['verbose']: print "Making source / background plots"
    plotfile = PdfPages(runconf['plotfile'])
    plt.figure()

    # source
    plt.clf()
    plt.imshow(source_image, interpolation='nearest', origin='lower')
    col = plt.colorbar()
    col.set_label("Counts")
    plt.title("Image for source: T={transtart:0.2f} to {tranend:0.2f}".format(**runconf))
    plt.suptitle("{name} - Data file: {infile}".format(**runconf))
    plotfile.savefig()

    # background
    plt.clf()
    plt.imshow(bkg_image, interpolation='nearest', origin='lower')
    col = plt.colorbar()
    col.set_label("Counts")
    plt.title("Image for scaled background: T={bkg1start:0.2f} to {bkg1end:0.2f},\nand T={bkg2start:0.2f} to {bkg2end:0.2f}".format(**runconf))
    #plt.suptitle("{name} - Data file: {infile}".format(**runconf))
    plotfile.savefig()
    plt.clf()

    # background-subtracted
    plt.imshow(source_image - bkg_image, interpolation='nearest', origin='lower')
    col = plt.colorbar()
    col.set_label("Counts")
    plt.title("Background-subtracted source image".format(**runconf))
    plt.suptitle("{name} - Data file: {infile}".format(**runconf))
    plotfile.savefig()
    plt.clf()

    if runconf['verbose']: print "Source / background plots complete"
    # First set of plots done. 
    
    # ------------------------------------------------------------------------
    # Now make RA Dec grid
    # First find out the pointing:
    data_header = fits.getheader(runconf['infile'], 0)
    ra_pnt, dec_pnt = data_header['ra_pnt'], data_header['dec_pnt']

    # Now see if ra/dec were "auto", in which case edit them
    if runconf['auto'] == True:
        runconf['ra'] = ra_pnt
        runconf['dec'] = dec_pnt

    # Option A:
    # # Now get the roll angle
    twist = get_twist(runconf['mkffile'], runconf['trigtime'])

    # # Calculate tx, ty for nominal transient position
    # transient_thetax, transient_thetay = rd2txy(ra_pnt, dec_pnt, twist, runconf['ra'], runconf['dec'])

    # Option B: calculate thetax, thetay directly
    transient_thetax, transient_thetay, coo_x, coo_y, coo_z, coo_transient = txy(runconf['mkffile'], runconf['trigtime'], runconf['ra'], runconf['dec'])

    if runconf['verbose']: 
        print "RA and Dec for satellite pointing are {ra:0.2f}, {dec:0.2f}".format(ra=ra_pnt, dec=dec_pnt)
        print "Twist angle is {twist:0.2f}".format(twist=twist)
        print "RA and Dec for nominal transient location are {ra:0.2f}, {dec:0.2f}".format(ra=runconf['ra'], dec=runconf['dec'])
        print "Theta_x and theta_y for nominal transient location are {tx:0.2f}, {ty:0.2f}".format(tx=transient_thetax, ty=transient_thetay)


    #------------------------------------------------------------------------
    # Plot the theoretical response at nominal direction
    try:
        respfile_f = calc_resp(runconf['energy'], transient_thetax, transient_thetay)
        response = resample(f2image(respfile_f), runconf['pixsize']) # this also applies master_mask
        plt.imshow(response, interpolation='nearest', origin='lower')
        col = plt.colorbar()
        col.set_label("Amplitude")
        plt.title(r"Calculated response at nominal $\theta_x$={tx:0.1f},$\theta_y$={ty:0.1f}".format(tx=transient_thetax, ty=transient_thetay))
        plt.suptitle("{name} - Data file: {infile}".format(**runconf))
        plotfile.savefig()
        plt.clf()
    except:
        # In case of any error, don't plot
        pass


    #------------------------------------------------------------------------
    # 3D visualisation of the satellite and transient
    visualize_3d(coo_x, coo_y, coo_z, coo_transient, transient_thetax, transient_thetay, runconf["name"])
    plotfile.savefig()
    plt.clf()

    #------------------------------------------------------------------------
    # Actual calculation on thetax thetay grid
    if not args.noloc:
        log2_nside = np.round(np.log2(1.02332670795 / np.deg2rad(runconf['resolution'])))
        nside = int(2**log2_nside)
        # hp.ang2vec takes theta, phi in radians
        # theta goes from 0 at NP to pi at SP: hence np.deg2rad(90-dec)
        # phi is simply RA
        trans_vec = hp.ang2vec(np.deg2rad(90.0 - runconf['dec']), np.deg2rad(runconf['ra']))
        pixlist = hp.query_disc(nside, trans_vec, np.deg2rad(runconf['radius']), inclusive=True)
        num_pix = len(pixlist)
        thetas, phis = hp.pix2ang(nside, pixlist)
        dec_calc = 90.0 - np.rad2deg(thetas)
        ra_calc = np.rad2deg(phis)
        if runconf['verbose']: print "RA-dec grid ready, with {num_pix} points".format(num_pix=num_pix)

        thetas_x, thetas_y = np.zeros(len(pixlist)), np.zeros(len(pixlist))
        for count in range(num_pix):
            thetas_x[count], thetas_y[count], junk1, junk2, junk3, junk4 = txy(runconf['mkffile'], runconf['trigtime'], ra_calc[count], dec_calc[count])

        redchi = np.zeros( num_pix )
        fluxes = np.zeros( num_pix )
        resp_strength = np.zeros( num_pix )

        if runconf['verbose']: print " # /Tot   T_x   T_y   |      chisq       redchi |    Flux   Bkgrnd"
        #tot = len(thetas_x) * len(thetas_y)
        for count in range(num_pix):
            theta_x, theta_y = thetas_x[count], thetas_y[count]
            # Calculate response:
            respfile_f = calc_resp(runconf['energy'], theta_x, theta_y)
            # Now read the response file and create an image
            response = resample(f2image(respfile_f), runconf['pixsize'])
            resp_strength[count] = np.sum(response)
            sf = source_image.flatten()
            rf = response.flatten()
            if runconf['do_fit']:
                dof = 128*128/runconf['pixsize']/runconf['pixsize'] - 2
                out = curve_fit(fitbkg, rf, sf) # curve_fit(function, x, y)
                fitvals = out[0]
                mf = fitbkg(rf, *fitvals)
                deltasq = (sf - mf)**2 / ((sf + 1e-8)**2)
            else:
                # out = flux * resp + bkg
                dof = 128*128/runconf['pixsize']/runconf['pixsize'] - 1
                calcflux = (np.sum(source_image) - np.sum(bkg_image)) / np.sum(response)
                fitvals = [calcflux]
                mf = fitbkg(rf, calcflux)
                deltasq = (sf - mf)**2 / ((sf + 1e-8)**2)
            chisq = np.sum(deltasq)
            redchi[count] = chisq
            fluxes[count] = fitvals[0]
            if runconf['verbose']: print "{count:3d}/{tot:3d} {tx:6.2f} {ty:6.2f} | {chisq:12.2f} {redchi:9.4f} | {flux:8.2f} {bkg:7.2f}".format(count=count+1, tot=num_pix, tx=theta_x, ty=theta_y, chisq=chisq, redchi=chisq/dof, flux=fitvals[0], bkg=0)

        plot_map(redchi, pixlist, nside, runconf, r"$\chi^2$ for fits", r"Reduced $\chi^2$")
        plotfile.savefig()

        plot_map(fluxes, pixlist, nside, runconf, "Approximate flux for fits", "Flux (amplitude)")
        plotfile.savefig()

        plot_map(resp_strength, pixlist, nside, runconf, "Total transmission as a function of direction", "Relative strength")
        plotfile.savefig()
    # endif not args.noloc - the actual localisation calculations end here

    plotfile.close()


    #for ra in ra0 + runconf['radius'] * 0.9 / np.cos(np.deg2rad(runconf['dec'])) * np.arange(-1, 1.1):
    #    hp.projtext(ra, dec0, "{ra:0.1f},{dec:0.1f}".format(ra=ra,dec=dec0), lonlat=True)
    #    hp.projscatter(ra, dec0, lonlat=True)
    #for dec in dec0 + runconf['radius'] * np.arange(-1, 1.1):
    #    hp.projtext(ra0, dec, "{ra:0.1f},{dec:0.1f}".format(ra=ra0,dec=dec), lonlat=True)
    #    hp.projscatter(ra0, dec, lonlat=True)


    #  fig2 = plt.figure()
    #  plt.contourf(thetas_y, thetas_x, redchi*dof)
    #  plt.xlabel(r"$\theta_y$")
    #  plt.ylabel(r"$\theta_x$")
    #  plt.title(r"$\chi^2$")
    #  plt.colorbar()
    #  xlims = plt.xlim()
    #  ylims = plt.ylim()
    #  plt.scatter(58, 34, s=30, color='black', marker='x', linewidths=3)
    #  plt.xlim(xlims)
    #  plt.ylim(ylims)
    #  plt.show()

    #  fig3 = plt.figure()
    #  plt.contourf(thetas_y, thetas_x, fluxes)
    #  plt.xlabel(r"$\theta_y$")
    #  plt.ylabel(r"$\theta_x$")
    #  plt.title(r"Flux")
    #  plt.colorbar()
    #  plt.show()
