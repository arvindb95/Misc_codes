import numpy as np
import astropy.coordinates as coo
import matplotlib.pyplot as plt
from astropy.table import Table

t1 = Table.read("loc_data_new_grid.txt",names=["tx","ty","cs_wo_sco","cs_w_sca"],format="ascii")

theta_x = (input("Enter the value of theta_x : "))
theta_y = (input("Enter the value of theta_y : "))
#chisq_wo_sca = t1['cs_wo_sco'].data
#chisq_w_sca = t1['cs_w_sca'].data

def tx_ty_2_th_ph(theta_x,theta_y):

	thetax = np.deg2rad(theta_x)
	thetay = np.deg2rad(theta_y)
	
	phi_temp = np.arctan2(np.tan(thetay),np.tan(thetax))
	theta = np.arccos(1.0/np.sqrt((np.tan(thetax))**2 + (np.tan(thetay))**2 + 1.0))

	if (np.rad2deg(phi_temp) < 0):
		phi = 360 +  np.rad2deg(phi_temp)

	else :
		phi = np.rad2deg(phi_temp)
	return np.rad2deg(theta),phi

#theta = np.zeros(len(theta_x))
#phi = np.zeros(len(theta_x))

#for i in range(len(theta_x)):

	#theta[i],phi[i] = tx_ty_2_th_ph(theta_x[i],theta_y[i])

theta,phi = tx_ty_2_th_ph(theta_x,theta_y)

print "Theta values : ",theta
print "Phi values : ",phi

#plt.scatter(theta,phi)
#for i in range(len(theta_x)):
#	plt.text(theta[i],phi[i],"{tx:0.2f},".format(tx=theta_x[i])+"{ty:0.2f}".format(ty=theta_y[i]))
#plt.show()

#t = Table([theta,phi,chisq_wo_sca,chisq_w_sca])

#t.write("test.txt",format='ascii')


