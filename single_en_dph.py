import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob

files = glob.glob("*.fits.gz")

pix_counts = np.zeros((len(files),128*128))

for i in range(len(files)):
    data = fits.getdata(files[i])
    pix_counts[i,:] = data.sum(1)
    quad0pix = pix_counts[i,:4096]
    quad1pix = pix_counts[i,4096:2*4096]
    quad2pix = pix_counts[i,2*4096:3*4096]
    quad3pix = pix_counts[i,3*4096:]
    quad0 =  np.reshape(quad0pix, (64,64), 'F')
    quad1 =  np.reshape(quad1pix, (64,64), 'F')
    quad2 =  np.reshape(quad2pix, (64,64), 'F')
    quad3 =  np.reshape(quad3pix, (64,64), 'F')
    sim_DPH = np.zeros((128,128), float)
    sim_DPH[:64,:64] = np.flip(quad0, 0)
    sim_DPH[:64,64:] = np.flip(quad1, 0)
    sim_DPH[64:,64:] = np.flip(quad2, 0)
    sim_DPH[64:,:64] = np.flip(quad3, 0)
    pixsize = 16
    imsize = 128/pixsize
    new_image = np.zeros((imsize,imsize))
    for xn, x in enumerate(np.arange(0, 128, pixsize)):
        for yn, y in enumerate(np.arange(0, 128, pixsize)):
            new_image[xn, yn] = np.nansum(sim_DPH[x:x+pixsize, y:y+pixsize])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(files[i])
    im = ax.imshow(new_image,interpolation='none',vmin=0)
    fig.colorbar(im,ax=ax,fraction=0.046, pad=0.04)
    plt.savefig("test_plots/"+files[i]+".png")

