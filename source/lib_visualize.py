import h5py
import matplotlib.pyplot as plt


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

if __name__ == '__main__':
    visualize_path('../moredata/noise_10000/detector_noise.hdf5',9873)