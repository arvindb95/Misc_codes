import numpy as np
import healpy as hp
from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from healpy.pixelfunc import get_interp_val

def eff_area(theta,phi,energy):
    """
    Takes the fits output from the simulation and calculates eff_area for 
    that particular energy and direction

    Inputs:
    theta = theta value (deg)
    phi = phi value (deg)
    energy = energy value (keV)

    Output:
    eff_area (cm^2)
    """

    path_file = open("path.txt","r")
    master_path = path_file.read()[:-1]

    fits_file = master_path+"/T{thf:06.2f}_P{phf:06.2f}/T{th:06.2f}_P{ph:06.2f}_E{e:07.2f}_SingleEventFile.fits".format(thf=theta,phf=phi,th=theta,ph=phi,e=energy)

    data = fits.getdata(fits_file)

    eff_area_per_pixel = data.sum(1)/55.5

    eff_area = eff_area_per_pixel.sum()
    return eff_area

theta_tab = Table.read("final_theta.txt",format="ascii")
theta_arr = theta_tab["theta"].data
phi_tab = Table.read("final_phi.txt",format="ascii")
phi_arr = phi_tab["phi"].data

#tab = Table.read("E0100.00_runtimes.txt",format='ascii')

#theta_arr = tab['theta'].data
#phi_arr = tab['phi'].data

# Enter the energy for which the sensitivity must be determined

energy = 120.0

#eff_area_arr = np.zeros(len(theta_arr))

#for i in range(len(theta_arr)):
#    eff_area_arr[i] = eff_area(theta_arr[i],phi_arr[i],energy)


#eff_area_file = open("eff_area_E{e:07.2f}.txt".format(e=energy),"w")
eff_area_tab = Table.read("eff_area_E{e:07.2f}.txt".format(e=energy),format="ascii")
#eff_area_tab.write(eff_area_file,format="ascii",overwrite=True)

eff_area_arr = eff_area_tab["eff_area"].data

NSIDE = 8

sky_map = -np.ones(hp.nside2npix(NSIDE))

print len(sky_map)

new_theta_arr = np.zeros(len(theta_arr))

for i in range(len(theta_arr)):
    if (theta_arr[i] <= 90):
        new_theta_arr[i] = 90 - theta_arr[i]
    else :
        new_theta_arr[i] = -(theta_arr[i] - 90)

new_phi_arr = np.zeros(len(phi_arr))

for i in range(len(phi_arr)):
    if (phi_arr[i] <= 180):
        new_phi_arr[i] = -phi_arr[i]
    else:
        new_phi_arr[i] = 360 - phi_arr[i]

pix = np.zeros(len(new_theta_arr))
for i in range(len(new_theta_arr)):
    pix[i] = hp.ang2pix(NSIDE,new_phi_arr[i],new_theta_arr[i],lonlat=True)
    sky_map[int(pix[i])] = eff_area_arr[i]

NSIDE_NEW = 64

all_pix_ids = np.arange(0,hp.nside2npix(NSIDE_NEW))

all_theta, all_phi = hp.pix2ang(NSIDE_NEW, all_pix_ids, lonlat=True)

new_map = get_interp_val(sky_map, all_theta, all_phi, lonlat=True)

print len(new_map)
hp.mollview(new_map,rot=(0,90),cmap="gist_earth_r",cbar=False)
hp.graticule()
hp.projtext(0,0,"X",fontsize=15,lonlat=True)
hp.projtext(-90,0,"Y",fontsize=15,lonlat=True)
plt.title("All sky sensitivity at 120keV")
fig = plt.gcf()
ax = plt.gca()
image = ax.get_images()[0]
cbar = fig.colorbar(image, ax=ax,orientation="horizontal",fraction=0.046, pad=0.04)
cbar.set_label(r"Effective Area (cm$^{2}$)",labelpad=-45, rotation=0)
plt.show()

