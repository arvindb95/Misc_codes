import numpy as np
import astropy.coordinates as coo
import matplotlib.pyplot as plt
from astropy.table import Table
import argparse

from astropy.table import Table

parser = argparse.ArgumentParser()
parser.add_argument("theta",help="Theta for the GRB",type=float)
parser.add_argument("phi",help="Phi for the GRB",type=float)
parser.add_argument("--grid_spacing",help="The spacing between points in a grid",type=float,default=10.0)
args = parser.parse_args()

theta_arr = np.zeros(9)
phi_arr = np.zeros(9)

for i in [0,1,2]:
	theta_arr[i] = args.theta - args.grid_spacing	 
	theta_arr[i+3] = args.theta
	theta_arr[i+6] = args.theta + args.grid_spacing
	phi_arr[3*i:3*i +3] = [args.phi - args.grid_spacing, args.phi, args.phi + args.grid_spacing]

print theta_arr

print phi_arr
	
def th_ph_2_tx_ty(theta,phi):
	
	theta = np.deg2rad(theta)
	phi = np.deg2rad(phi)
	cz = np.cos(theta)
	cx = np.sin(theta) * np.cos(phi)
	cy = np.sin(theta) * np.sin(phi)

	thetax = np.rad2deg(np.arctan2(cx,cz))
	thetay = np.rad2deg(np.arctan2(cy,cz)) 
	
	return thetax,thetay

thetax = np.zeros(len(theta_arr))
thetay = np.zeros(len(theta_arr))

for i in range(len(theta_arr)):
	thetax[i], thetay[i] = th_ph_2_tx_ty(theta_arr[i],phi_arr[i]) 

t = Table([theta_arr,phi_arr,np.round(thetax,decimals=2),np.round(thetay,decimals=2),np.zeros(len(thetay)),np.zeros(len(thetay))],names=('theta','phi','thetax','thetay','chisq_wo_sca','chisq_w_sca'))

t.write("test.txt",format='ascii',overwrite=True)
