from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from offaxispos import simulated_dph

transtart = 191924430
tranend = 191924500
bkg1start=191924225
bkg1end=191924425
bkg2start=191924505
bkg2end=191924745
grbdir = "GRB160131A"
typ = "band"
alpha=-0.93
beta=-1.56
E0=1152
A=0.005

t_src = tranend - transtart
t_tot = (bkg1end-bkg1start) + (bkg2end-bkg2start)

quanteff_im = np.ndarray((128,128))
quanteff_bkg = np.ndarray((128,128))

hdu0 = fits.open("GRB160131A/test/GRB160131A_q0.dpi")
q0 = np.flip(hdu0[1].data,0)
hdu0_bkg = fits.open("GRB160131A/test/GRB160131A_q0_bkg.dpi")
q0_bkg = np.flip(hdu0_bkg[1].data,0)

hdu1 = fits.open("GRB160131A/test/GRB160131A_q1.dpi")
q1 = np.flip(hdu1[2].data,0)
hdu1_bkg = fits.open("GRB160131A/test/GRB160131A_q1_bkg.dpi")
q1_bkg = np.flip(hdu1_bkg[2].data,0)

hdu2 = fits.open("GRB160131A/test/GRB160131A_q2.dpi")
q2 = np.flip(hdu2[3].data,0)
hdu2_bkg = fits.open("GRB160131A/test/GRB160131A_q2_bkg.dpi")
q2_bkg = np.flip(hdu2_bkg[3].data,0)

hdu3 = fits.open("GRB160131A/test/GRB160131A_q3.dpi")
q3 = np.flip(hdu3[4].data,0)
hdu3_bkg = fits.open("GRB160131A/test/GRB160131A_q3_bkg.dpi")
q3_bkg = np.flip(hdu3_bkg[4].data,0)

quanteff_im[:64,:64] = q0
quanteff_im[:64,64:] = q1
quanteff_im[64:,64:] = q2
quanteff_im[64:,:64] = q3

quanteff_bkg[:64,:64] = q0_bkg
quanteff_bkg[:64,64:] = q1_bkg
quanteff_bkg[64:,64:] = q2_bkg
quanteff_bkg[64:,:64] = q3_bkg

quanteff_data = quanteff_im - quanteff_bkg*t_src/t_tot

plt.figure()
plt.imshow(quanteff_im)
plt.figure()
plt.imshow(quanteff_bkg)
plt.figure()
plt.imshow(quanteff_data)
plt.show()
