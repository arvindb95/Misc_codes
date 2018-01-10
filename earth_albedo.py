import numpy as np
import matplotlib.pyplot as plt
import astropy.coordinates as coo
from astropy.table import Table

# Making the circle to represent the earth ------------------
step = np.pi/1000.0
theta = np.arange(0, 2*np.pi + step, step)

r_earth = 6371e3 

x = r_earth*np.cos(theta)
y = r_earth*np.sin(theta)

plt.plot(x,y,'.')

# Putting the satellite on the x axis -----------------------

r_orbit = 7020e3

sat_x = r_orbit
sat_y = 0.0

for i in range(60,180,10):

	earth_transient_angle = np.deg2rad(i)
	angle = np.pi - earth_transient_angle

	#plt.plot(sat_x,sat_y,'s')

#plt.quiver(sat_x,sat_y,sat_x*np.cos(np.deg2rad(60)),sat_x*np.sin(np.deg2rad(60)),scale=3.0e7)
#plt.quiver(0,0,sat_x*np.cos(angle),sat_x*np.sin(angle),scale=3.0e7)
#plt.quiver(0,0,sat_x,sat_y,scale=3.0e7)
#plt.xlim(-r_orbit,10000000)
	sat = coo.SkyCoord(sat_x,sat_y,0,frame='icrs',representation='cartesian')
	tran = coo.SkyCoord(sat_x*np.cos(angle),sat_x*np.sin(angle),0,frame='icrs',representation='cartesian')

	theta_sub = np.arange(0, angle + 10*step, step)
	x_sub = r_earth*np.cos(theta_sub)
	y_sub = r_earth*np.sin(theta_sub)
	inc_ang = []
	ref_ang = []
	angles_file = "angle_test"+str(np.rad2deg(earth_transient_angle))+".txt"

	for i in range(len(theta_sub)):
		normal = coo.SkyCoord(x_sub[i],y_sub[i],0,frame='icrs',representation='cartesian')
		inc_ang.append(normal.separation(tran).value)
		sat_normal = coo.SkyCoord(sat_x-x_sub[i],sat_y-y_sub[i],0,frame='icrs',representation='cartesian')
		ref_ang.append(normal.separation(sat_normal).value)
	
	t = Table([np.around(inc_ang,decimals=3),np.around(ref_ang,decimals=3)],names=["i","r"])
	Table.write(t,angles_file,format='ascii')

#plt.show()
