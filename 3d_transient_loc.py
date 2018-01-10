import subprocess, os, shutil
import ConfigParser, argparse
from astropy.table import Table
from astropy.io import fits
from astropy.stats import sigma_clip, sigma_clipped_stats
from scipy.optimize import curve_fit
import astropy.coordinates as coo
import astropy.units as u
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import offaxispos as oap

parser = argparse.ArgumentParser()

parser.add_argument("grbdir",help="The main directory of the grb",type=str)
parser.add_argument("trigtime",help="Trigger time of the grb",type=float)
parser.add_argument("ra_tran",help="Transient RA",type=float)
parser.add_argument("dec_tran",help="Transient Dec",type=float)
parser.add_argument("theta",help="Transient theta",type=float)
parser.add_argument("phi",help="Transient phi",type=float)
args = parser.parse_args()

# Parameters required 
mkf_file = glob.glob(args.grbdir+"/*.mkf")[0]
trigtime = args.trigtime
ra_tran = args.ra_tran
dec_tran = args.dec_tran

# Calling txy function from offaxispos to calculate thetax and thetay 

thetax,thetay,minus_yaw,pitch,roll,transient = oap.txy(mkf_file, trigtime, ra_tran, dec_tran)

x = coo.SkyCoord(x=1,y=0,z=0,representation='cartesian')#minus_yaw
y = coo.SkyCoord(x=0,y=1,z=0,representation='cartesian')#pitch
z = coo.SkyCoord(x=0,y=0,z=1,representation='cartesian')#roll
sin_t = np.sin(np.deg2rad(args.theta))
cos_t = np.cos(np.deg2rad(args.theta))
sin_p = np.sin(np.deg2rad(args.phi))
cos_p = np.cos(np.deg2rad(args.phi))
t = coo.SkyCoord(z=cos_t, y=sin_t * sin_p, x=sin_t * cos_p, representation='cartesian')
#t = coo.SkyCoord(z=np.sin(np.deg2rad(args.theta))*np.cos(args.phi*m.pi/180),x=np.sin(args.theta*m.pi/180)*np.sin(args.phi*m.pi/180),y=np.cos(args.theta*m.pi/180),representation='cartesian')#transient

print r"For {name} :  $\theta_x$={tx:0.1f},$\theta_y$={ty:0.1f}".format(name=args.grbdir,tx = thetax,ty = thetay)
pdf_file = PdfPages(args.grbdir + '/3d_plot_'+args.grbdir+'.pdf')

def plot_xyzt(ax, x, y, z, t):
    """
    Make a subplot that shows X, Y, Z axes and a transient vector
    The input coordinates are astropy.coordinate.SkyCoord objects
    """
    colors = ['blue', 'green', 'red', 'black']
    names = ['X', 'Y', 'Z', args.grbdir]
    zdirs = ['x', 'y', 'z', None]

    ax.set_xlim(-1.2,1.2)
    ax.set_ylim(-1.2,1.2)
    ax.set_zlim(-1.2,1.2)

    for count, dirn in enumerate([x, y, z, t]):
        xx, yy, zz = dirn.cartesian.x.value, dirn.cartesian.y.value, dirn.cartesian.z.value
        ax.quiver(0, 0, 0, xx, yy, zz, color=colors[count])
        ax.text(xx, yy, zz, names[count], zdirs[count])

    #ax.set_xlabel("RA = 0")
    #ax.set_zlabel("Pole")
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
    plt.suptitle(r"Visualisation of {name} in 3d:$\theta$={t:0.1f},$\phi$={p:0.1f}".format(name=name, t=args.theta, p=args.phi))
    # Z
    ax = plt.subplot(2, 2, 1, projection='3d')
    plot_xyzt(ax, x, y, z, t)
    ax.azim = z.fk5.ra.deg
    ax.elev = z.fk5.dec.deg
    oap.add_satellite(ax, x, y, z)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_title("View from CZTI pointing (z)",fontsize=10)

    # Transient
    ax = plt.subplot(2, 2, 2, projection='3d')
    plot_xyzt(ax, x, y, z, t)
    ax.azim = t.fk5.ra.deg
    ax.elev = t.fk5.dec.deg
    oap.add_satellite(ax, x, y, z)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])   
    ax.set_title("View from nominal transient direction",fontsize=10)

    # X
    ax = plt.subplot(2, 2, 3, projection='3d')
    plot_xyzt(ax, x, y, z, t)
    ax.azim = x.fk5.ra.deg
    ax.elev = x.fk5.dec.deg
    oap.add_satellite(ax, x, y, z)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_title("View from CZTI X axis",fontsize=10)

    # Z
    ax = plt.subplot(2, 2, 4, projection='3d')
    plot_xyzt(ax, x, y, z, t)
    ax.azim = y.fk5.ra.deg
    ax.elev = y.fk5.dec.deg
    oap.add_satellite(ax, x, y, z)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])   
    ax.set_title("View from CZTI Y axis",fontsize=10)
    
    pdf_file.savefig(fig,orientation = 'portrait')
    return

#calling visualize_3d from oap to get the image of astrosat
visualize_3d(x,y,z, t, thetax, thetay, args.grbdir)

pdf_file.close()
