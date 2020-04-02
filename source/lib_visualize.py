import h5py
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from lib_dist import VonMisesFisherDistribution, KentDistribution


def visualize_signal(signal):
    """ Visualize a given signal """

    plt.plot(signal[0], alpha=0.7, label='H1')
    plt.plot(signal[1], alpha=0.7, label='L1')
    plt.plot(signal[2], alpha=0.7, label='V1')
    plt.xlabel('timesteps (1 sec = 4096 steps)')
    plt.ylabel('signal')
    plt.legend()
    plt.show()


def visualize_path(path: str, signal_nb: int):
    """ A function to inspect the signals in the hdf5  """

    # Extract signals and visualize
    hdf5_file = h5py.File(name=path, mode='r')
    if "noise" in path:
        signals = hdf5_file['noise']
    else:
        signals = hdf5_file['signals']
    visualize_signal(signals[signal_nb])


def visualize_vmf(kappa: float, vec: np.ndarray, tar: np.ndarray=None, significance: int=-1, resolution: int=64):
    """ Visualize the Von Mises Fisher distribution - based Healpytools code"""
    # Setup the distribution
    vmf = VonMisesFisherDistribution(kappa, vec)

    # Select the reach, know that sigma ~ 1 / sqrt(kappa) if kappa is not small.
    radius =  significance * 1 / np.sqrt(kappa)
    if significance > 0:
        pixels = np.array(hp.query_disc(resolution, vec, radius))
    else:
        pixels = np.arange(hp.nside2npix(resolution))

    # Go over the pixels
    if len(pixels) == 0:
        pixels = np.array(hp.vec2pix(resolution, vec))
        weights = np.array([1.])
    else:
        pixel_vecs = hp.pix2vec(resolution, pixels)
        pixel_vecs = np.array([[x, y, z] for x, y, z in zip(pixel_vecs[0], pixel_vecs[1], pixel_vecs[2])])
        weights = vmf.pdf(pixel_vecs)

    # Normalize the weights
    pdf = np.zeros(hp.nside2npix(resolution))
    pdf[pixels] = weights / sum(weights)

    # Visualize the pdf
    cmap = plt.get_cmap('PuRd')
    cmap.set_under('w')

    hp.mollview(pdf, cmap=cmap, title='VMF', cbar=False, min=0)
    hp.graticule(dpar=30, dmer=60)
    hp.projscatter([hp.vec2ang(tar)[1] * 180/np.pi], [hp.vec2ang(tar)[0] * 180/np.pi], lonlat=True, color='black')
    plt.show()


def visualize_kent(kappa: float, beta: float, gamma1: np.ndarray, gamma2: np.ndarray, tar: np.ndarray=None, significance: int=-1, resolution: int=64):
    """ Visualize the Von Mises Fisher distribution - based Healpytools code"""
    # Setup the distribution
    kent = KentDistribution(kappa, beta, gamma1, gamma2)

    # Select the reach, know that sigma ~ 1 / sqrt(kappa) if kappa is not small.
    radius = significance * 1 / np.sqrt(kappa)
    if significance > 0:
        pixels = np.array(hp.query_disc(resolution, gamma1, radius))
    else:
        pixels = np.arange(hp.nside2npix(resolution))

    # Go over the pixels
    if len(pixels) == 0:
        pixels = np.array(hp.vec2pix(resolution, gamma1))
        weights = np.array([1.])
    else:
        pixel_vecs = hp.pix2vec(resolution, pixels)
        pixel_vecs = np.array([[x, y, z] for x, y, z in zip(pixel_vecs[0], pixel_vecs[1], pixel_vecs[2])])
        weights = kent.pdf(pixel_vecs)

    # Normalize the weights
    pdf = np.zeros(hp.nside2npix(resolution))
    pdf[pixels] = weights / sum(weights)

    # Visualize the pdf
    cmap = plt.get_cmap('PuRd')
    cmap.set_under('w')

    hp.mollview(pdf, cmap=cmap, title='Kent', cbar=False, min=0)
    hp.graticule(dpar=30, dmer=60)
    hp.projscatter([hp.vec2ang(tar)[1] * 180/np.pi], [hp.vec2ang(tar)[0] * 180/np.pi], lonlat=True, color='black')
    plt.show()


if __name__ == '__main__':
    # visualize_vmf(500, 1 / ((-8) ** 2 + (-4) ** 2 + 9 ** 2) ** 0.5 * np.array([-8, -4, 9]), tar=1 / ((-8) ** 2 + (-4) ** 2 + 9 ** 2) ** 0.5 * np.array([-8, -4, 9]))
    visualize_kent(300, 63, 1 / np.sqrt(3) * np.array([1, -1, 1]), 3 / np.sqrt(6) * np.array([1 / 3, 2 / 3, 1 / 3]), tar=1 / np.sqrt(3) * np.array([1, -1, 1]))