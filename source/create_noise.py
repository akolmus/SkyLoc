""" The reason we generate a noise file, instead of generating it on the fly is due to a conflict between Torch and the LAL suite / pycbc """

import yaml
import time
import h5py
import math
import numpy as np
import multiprocessing as mp
import pycbc.psd
import pycbc.noise

from tqdm import tqdm
from pathlib import Path


def noise2dataset(cfg_noise: dict):
    """ Fill a dataset with noise signals from the detectors [H1, L1, V1]  """

    # Create or load the hdf5 file
    if Path(f'../moredata/{cfg_noise["name"]}/detector_noise.hdf5').exists():
        print(f'There already exists a file named: ../moredata/{cfg_noise["name"]}/detector_noise.hdf5')
        hdf5_file = h5py.File(name=f'../moredata/{cfg_noise["name"]}/detector_noise.hdf5', mode='r+')
        noise = hdf5_file['noise']
    else:
        length = cfg_noise['sampling_rate'] * cfg_noise['duration']
        hdf5_file = h5py.File(name=f'../moredata/{cfg_noise["name"]}/detector_noise.hdf5', mode='w')
        hdf5_file.create_dataset('noise', shape=(0, 3, length), maxshape=(None, 3, length), dtype=np.float_)
        noise = hdf5_file['noise']

    # Check if the last 100 datapoints are filled
    for index in range(max(noise.shape[0] - 100, 0), noise.shape[0]):
        if np.sum(np.abs(noise[index])) == 0:
            noise.resize((index, 3, noise.shape[2]))
            break

    # Check if we finished the entire setting
    if cfg_noise['nb_samples'] > noise.shape[0]:
        print(f'Generating the {cfg_noise["nb_samples"] - noise.shape[0]} samples in steps of a 100 datapoints')
        time.sleep(0.1)

        # Create the noise profiles
        aligo_psd = pycbc.psd.aLIGOZeroDetHighPower(int(cfg_noise['frequency_upper'] * 2) + 1, 1.0 / 2, cfg_noise['frequency_lower'])
        virgo_psd = pycbc.psd.AdvVirgo(int(cfg_noise['frequency_upper'] * 2) + 1, 1.0 / 2, cfg_noise['frequency_lower'])

        pbar = tqdm(total=math.ceil((cfg_noise['nb_samples'] - noise.shape[0]) / 100))
        while cfg_noise['nb_samples'] > noise.shape[0]:
            # Obtain start and stop position
            start = noise.shape[0]
            stop = min(start + 100, cfg_noise['nb_samples'])

            # Get the detector signals using multiprocessing
            pool = mp.Pool(processes=cfg_noise["nb_processes"])

            # Obtain output
            output = [pool.apply_async(func=generate_noise, args=(cfg_noise, aligo_psd, virgo_psd, index)) for index in range(start, stop)]

            # Create space for the signals
            noise.resize((noise.shape[0] + (stop - start), 3, noise.shape[2]))

            # Insert the detector output in the hdf5 file
            for i, dectector_output in enumerate(output):
                detector_noise, index = dectector_output.get()
                noise[index] = detector_noise

            # Close the multiprocessing session
            pool.close()
            pool.join()

            # Update the tqdm bar
            pbar.update(1)

        # Close up the bar
        pbar.close()

    # Close up the file
    hdf5_file.close()


def generate_noise(cfg_noise: dict, aligo_psd, virgo_psd, index: int):
    """ Use the pycbc pipeline to generate noise signals """

    t_samples = int(cfg_noise['duration'] * cfg_noise['sampling_rate'])
    combined_signal = np.zeros(shape=(3, t_samples))
    combined_signal[0] = pycbc.noise.noise_from_psd(t_samples, delta_t=1.0 / cfg_noise['sampling_rate'], psd=aligo_psd, seed=index * np.random.randint(low=1, high=1000)).numpy()
    combined_signal[1] = pycbc.noise.noise_from_psd(t_samples, delta_t=1.0 / cfg_noise['sampling_rate'], psd=aligo_psd, seed=index * np.random.randint(low=1001, high=2000)).numpy()
    combined_signal[2] = pycbc.noise.noise_from_psd(t_samples, delta_t=1.0 / cfg_noise['sampling_rate'], psd=virgo_psd, seed=index * np.random.randint(low=2001, high=3000)).numpy()

    return combined_signal, index

if __name__  == '__main__':
    # Open the configuration file
    with open('config_noise.yml', 'r') as config_file:
        cfg_noise = yaml.safe_load(config_file)

    # Write the config file to the destination
    if not Path(f'../moredata/{cfg_noise["name"]}/').is_dir():
        Path(f'../moredata/{cfg_noise["name"]}/').mkdir(parents=True)

    with open(f'../moredata/{cfg_noise["name"]}/config.yml', 'w') as out_file:
        yaml.dump(cfg_noise, out_file)

    # Set the numpy seed
    np.random.seed(cfg_noise['seed'])

    # Generate the noise
    noise2dataset(cfg_noise)
