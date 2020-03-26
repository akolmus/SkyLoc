import math
import time
import h5py
import numpy as np
import pandas as pd
import multiprocessing as mp
import pycbc.psd
import pycbc.noise as noise

from lal import LIGOTimeGPS
from pycbc.types import TimeSeries
from pycbc.detector import Detector
from pycbc.waveform import get_td_waveform
from pycbc.filter import sigma
from scipy.signal.windows import tukey
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset


def tukey_fade_on(timeseries: TimeSeries, alpha: float) -> TimeSeries:
    """ Apply a one-sided Tukey window - with as goal to remove discontinuities in the amplitude """

    # Create the one-sided Tukey window
    window = tukey(int(timeseries.get_duration() * timeseries.get_sample_rate()), alpha)
    window[int(0.5 * len(window)):] = 1

    # Apply the window
    numpy_ts = window * timeseries.numpy()

    # Return the numpy array as a timeseries object
    return TimeSeries(initial_array=numpy_ts,
                      delta_t=timeseries.get_delta_t(),
                      epoch=timeseries.start_time,
                      dtype='double')


class GravitationalWavesPrior(object):
    """ The priors for each of the parameters, this object is simply a collection of the marginals """
    def __init__(self, nb_samples: int, prior:dict):
        self.nb_samples = nb_samples
        self.prior = prior

    def mass1(self):
        if self.prior['mass1_min'] >= self.prior['mass1_max']:
            return self.prior['mass1_min']
        else:
            return np.random.randint(low=self.prior['mass1_min'], high=self.prior['mass1_max'], size=(self.nb_samples,))

    def mass2(self):
        if self.prior['mass2_min'] >= self.prior['mass2_max']:
            return self.prior['mass2_min']
        else:
            return np.random.randint(low=self.prior['mass2_min'], high=self.prior['mass2_max'], size=(self.nb_samples,))

    def spin1z(self):
        if self.prior['spin1z_min'] >= self.prior['spin1z_max']:
            return self.prior['spin1z_min']
        else:
            return np.random.randint(low=self.prior['spin1z_min'], high=self.prior['spin1z_max'], size=(self.nb_samples,))

    def spin2z(self):
        if self.prior['spin2z_min'] >= self.prior['spin2z_max']:
            return self.prior['spin2z_min']
        else:
            return np.random.randint(low=self.prior['spin2z_min'], high=self.prior['spin2z_max'], size=(self.nb_samples,))

    def coa_phase(self):
        return self.prior['coa_phase'] * np.random.uniform(low=0, high=2 * np.pi, size=(self.nb_samples))

    def inclination(self):
        return self.prior['inclination'] * np.random.uniform(low=0, high=np.pi, size=(self.nb_samples,))

    def right_ascension(self):
        return self.prior['right_ascension'] * np.random.uniform(low=0, high=2 * np.pi, size=(self.nb_samples,))

    def declination(self):
        return self.prior['declination'] * np.arcsin(1 - 2 * np.random.random(size=(self.nb_samples)))

    def polarization(self):
        return self.prior['polarization'] * np.random.uniform(low=0, high= 2 * np.pi, size=(self.nb_samples))

    def endtime(self):
        return 1192529720 + self.prior['endtime'] * np.random.randint(low=0, high=24*3600, size=(self.nb_samples,))


class GravitationalWavesGenerator(object):
    """ Generates the parameters and allows for generating the samples """
    def __init__(self, name:str, nb_samples: int, nb_processes: int):
        self.name = name
        self.nb_samples = nb_samples
        self.nb_processes = nb_processes

    def generate_parameters(self, prior: dict):
        """ Create the parameters for all the samples """
        if Path(f'../moredata/{self.name}/parameters.csv').exists():
            print(f'There already exists a file named: ../moredata/{self.name}/parameters.csv')

        else:
            # Setup the parameter dataframe and the prior
            df_para = pd.DataFrame(data=np.zeros(shape=(self.nb_samples, 10)),columns=['mass1', 'mass2', 'spin1z', 'spin2z', 'coa_phase', 'inclination', 'right_ascension', 'declination', 'polarization', 'endtime'])
            gw_prior = GravitationalWavesPrior(nb_samples=self.nb_samples, prior=prior)

            # Use the prior
            df_para['mass1'] = gw_prior.mass1()
            df_para['mass2'] = gw_prior.mass2()
            df_para['spin1z'] = gw_prior.spin1z()
            df_para['spin2z'] = gw_prior.spin2z()
            df_para['coa_phase'] = gw_prior.coa_phase()
            df_para['inclination'] = gw_prior.inclination()
            df_para['right_ascension'] = gw_prior.right_ascension()
            df_para['declination'] = gw_prior.declination()
            df_para['polarization'] = gw_prior.polarization()
            df_para['endtime'] = gw_prior.endtime()

            # Add parameters that are a function of the others, the + np.pi / 2 is to account for the celestial coord.
            df_para['h1_start_time'] = 0
            df_para['l1_start_time'] = 0
            df_para['v1_start_time'] = 0
            df_para['declination_x'] = np.cos(df_para['declination'])
            df_para['declination_y'] = np.sin(df_para['declination'])
            df_para['right_ascension_x'] = np.cos(df_para['right_ascension'])
            df_para['right_ascension_y'] = np.sin(df_para['right_ascension'])
            df_para['x'] = np.sin(df_para['declination'] + np.pi / 2) * np.cos(df_para['right_ascension'])
            df_para['y'] = np.sin(df_para['declination'] + np.pi / 2) * np.sin(df_para['right_ascension'])
            df_para['z'] = np.cos(df_para['declination'] + np.pi / 2)

            # Save the parameters
            df_para.to_csv(path_or_buf=f'../moredata/{self.name}/parameters.csv', index=False)

    def parameters2waveform2signal(self, general_properties: dict, sample_parameters: dict, row_nb: int):
        """ Pass the parameters to pycbc / LAL algorithms to generate the signals, enables parallelization """

        # Get the waveform, expressed in the plus and cross polarization
        h_plus, h_cross = get_td_waveform(approximant=general_properties['approximant'],
                                          mass1=sample_parameters['mass1'],
                                          mass2=sample_parameters['mass2'],
                                          spin1z=sample_parameters['spin1z'],
                                          spin2z=sample_parameters['spin2z'],
                                          inclination=sample_parameters['inclination'],
                                          coa_phase=sample_parameters['coa_phase'],
                                          delta_t=1.0 / general_properties['sampling_rate'],
                                          f_lower=general_properties['f_lower'])

        # Apply a tukey fade on the waveform
        h_plus = tukey_fade_on(h_plus, general_properties['tukey_alpha'])
        h_cross = tukey_fade_on(h_cross, general_properties['tukey_alpha'])

        # Set the arrival time for the waveform
        h_plus.start_time += sample_parameters['endtime']
        h_cross.start_time += sample_parameters['endtime']

        # Give the waveform to a detector
        detectors = {'H1': Detector('H1'),
                     'L1': Detector('L1'),
                     'V1': Detector('V1')}

        signal = {}
        for detertor_nm, detector in detectors.items():
            signal[detertor_nm] = detector.project_wave(hp=h_plus,
                                                        hc=h_cross,
                                                        longitude=sample_parameters['right_ascension'],
                                                        latitude=sample_parameters['declination'],
                                                        polarization=sample_parameters['polarization'])


        # Ensure that the timing window is correctly set, the H1 signal is the event center
        center = int(general_properties['sec_before_event'] * general_properties['sampling_rate'])
        time_diff_l1 = int(round(float(signal['L1'].start_time - signal['H1'].start_time) * general_properties['sampling_rate']))
        time_diff_v1 = int(round(float(signal['V1'].start_time - signal['H1'].start_time) * general_properties['sampling_rate']))

        # Create the signal structure
        combined_signal = np.zeros(shape=(3, int(general_properties['sec_before_event'] + general_properties['sec_after_event']) * general_properties['sampling_rate']))
        combined_signal[0, center:center + len(signal['H1'])] += signal['H1']
        combined_signal[1, center + time_diff_l1:center + time_diff_l1 + len(signal['L1'])] += signal['L1']
        combined_signal[2, center + time_diff_v1:center + time_diff_v1 + len(signal['V1'])] += signal['V1']

        return combined_signal, row_nb, signal['H1'].start_time, signal['L1'].start_time, signal['V1'].start_time

    def parameter2dataset(self, general_properties: dict):
        """ Transform the parameters to dataset using multiprocessing to speed up the process """

        # Create or load the hdf5 file
        if Path(f'../moredata/{self.name}/detector_signals.hdf5').exists():
            print(f'There already exists a file named: ../moredata/{self.name}/detector_signals.hdf5')
            hdf5_file = h5py.File(name=f'../moredata/{self.name}/detector_signals.hdf5', mode='r+')
            signals = hdf5_file['signals']
        else:
            length = general_properties['sampling_rate'] * (general_properties['sec_before_event'] + general_properties['sec_after_event'])
            hdf5_file = h5py.File(name=f'../moredata/{self.name}/detector_signals.hdf5', mode='w')
            hdf5_file.create_dataset('signals', shape=(0, 3, length), maxshape=(None, 3, length), dtype=np.float_)
            signals = hdf5_file['signals']

        # Load the parameter file
        df_para = pd.read_csv(filepath_or_buffer=f'../moredata/{self.name}/parameters.csv')

        # Check if the last 100 datapoints are filled
        for index in range(max(signals.shape[0] - 100, 0), signals.shape[0]):
            if np.sum(np.abs(signals[index])) == 0:
                signals.resize((index, 3, signals.shape[2]))
                break

        # Check if we finished the entire setting
        if df_para.shape[0] > signals.shape[0]:
            print(f'Generating the {df_para.shape[0] - signals.shape[0]} samples in steps of a 100 datapoints')
            time.sleep(0.1)

            pbar = tqdm(total=math.ceil((df_para.shape[0] - signals.shape[0]) / 100))
            while df_para.shape[0] > signals.shape[0]:
                # Obtain start and stop position
                start = signals.shape[0]
                stop = min(start + 100, df_para.shape[0])

                # Get the detector signals using multiprocessing
                pool = mp.Pool(processes=self.nb_processes)

                # Obtain output
                output = [pool.apply_async(func=self.parameters2waveform2signal, args=(general_properties, row, row_nb)) for row_nb, row in df_para.iloc[start:stop].iterrows()]

                # Create space for the signals
                signals.resize((signals.shape[0] + (stop - start), 3, signals.shape[2]))

                # Insert the detector output in the hdf5 file
                for i, dectector_output in enumerate(output):
                    detector_signals, row_nb, h1_start_time, l1_start_time, v1_start_time = dectector_output.get()
                    signals[row_nb] = detector_signals
                    df_para.loc[row_nb, 'h1_start_time'] = h1_start_time
                    df_para.loc[row_nb, 'l1_start_time'] = l1_start_time
                    df_para.loc[row_nb, 'v1_start_time'] = v1_start_time

                # Close the multiprocessing session
                pool.close()
                pool.join()

                # Update the tqdm bar
                pbar.update(1)

            # Save the parameters
            df_para.to_csv(path_or_buf=f'../moredata/{self.name}/parameters.csv', index=False)

            # Close the bar
            pbar.close()

        # Close up the file
        hdf5_file.close()


class GravitationalWavesDataset(Dataset):
    """ Reads the waveform from a hdf5 file + simulate the detector and then scale the signals for the correct SNR """

    def __init__(self, parameters: pd.DataFrame, cfg_data:dict , cfg_train: dict):
        # Save the targets
        self.target = parameters[['x', 'y', 'z']]
        self.parameters = parameters

        # Save the settings
        self.snr = cfg_train['data']['SNR']
        self.frequency_lower = cfg_train['data']['frequency_lower']
        self.frequency_upper = cfg_train['data']['frequency_upper']
        self.sampling_rate = cfg_data['general_properties']['sampling_rate']
        self.duration = cfg_data['general_properties']['sec_before_event'] + cfg_data['general_properties']['sec_after_event']
        self.whiten_duration = cfg_train['data']['whitening_segment_duration']
        self.whiten_max_filter = cfg_train['data']['whitening_max_filter_duration']
        self.bandpass_lower = cfg_train['data']['bandpass_lower']
        self.bandpass_upper = cfg_train['data']['bandpass_upper']
        self.noise_nm = f'../moredata/{cfg_train["data"]["noise"]}/detector_noise.hdf5'
        self.signal_nm = f'../moredata/{cfg_train["data"]["name"]}/detector_signals.hdf5'

    def __len__(self):
        return self.parameters.shape[0]

    def __getitem__(self, idx):
        """ Add the noise and transform to match the required SNR - code is heavily based on https://github.com/timothygebhard/ggwd """

        # Translate the index to the dataframe
        idx = self.parameters.index.values[idx]

        # Open the hdf5 signal file and extract the signals
        hdf5_file_signal = h5py.File(self.signal_nm, 'r')
        detector_signal = hdf5_file_signal['signals'][idx]

        # TODO: for now we keep the signal stationary this makes it easier for to learn, later we should add this

        # Transform the signals to pycbc timeseries to enable the
        h1_tseries = TimeSeries(detector_signal[0], delta_t=1.0 / self.sampling_rate, epoch=self.parameters.loc[idx, 'h1_start_time'], dtype=np.float_)
        l1_tseries = TimeSeries(detector_signal[1], delta_t=1.0 / self.sampling_rate, epoch=self.parameters.loc[idx, 'l1_start_time'], dtype=np.float_)
        v1_tseries = TimeSeries(detector_signal[2], delta_t=1.0 / self.sampling_rate, epoch=self.parameters.loc[idx, 'v1_start_time'], dtype=np.float_)

        # Open the hdf5 noise file and extract the noise
        hdf5_file_noise = h5py.File(self.noise_nm, 'r')
        detector_noise = hdf5_file_noise['noise']
        detector_noise = detector_noise[np.random.randint(low=0, high=detector_noise.shape[0])]

        # Combine noise and signal
        if detector_signal.shape[1] > detector_noise.shape[1]:
            raise ValueError(f'The shape of the detector signal {detector_signal.shape} doesnt fit in the detector noise {detector_noise.shape}')
        else:
            start = np.random.randint(low=0, high=(detector_noise.shape[1] - detector_signal.shape[1]))
            strain = detector_signal + detector_noise[:, start:start + detector_signal.shape[1]]
            h1_strain = TimeSeries(strain[0], delta_t=1.0 / self.sampling_rate, epoch=self.parameters.loc[idx, 'h1_start_time'], dtype=np.float_)
            l1_strain = TimeSeries(strain[1], delta_t=1.0 / self.sampling_rate, epoch=self.parameters.loc[idx, 'l1_start_time'], dtype=np.float_)
            v1_strain = TimeSeries(strain[2], delta_t=1.0 / self.sampling_rate, epoch=self.parameters.loc[idx, 'v1_start_time'], dtype=np.float_)

        # Calculate the SNR for each detector
        h1_psd = h1_strain.psd(self.duration)
        h1_snr = sigma(h1_tseries, h1_psd, self.frequency_lower)
        l1_psd = l1_strain.psd(self.duration)
        l1_snr = sigma(l1_tseries, l1_psd, self.frequency_lower)
        v1_psd = v1_strain.psd(self.duration)
        v1_snr = sigma(v1_tseries, v1_psd, self.frequency_lower)

        # Calculate the collective snr
        nomf_snr = np.sqrt(h1_snr ** 2 + l1_snr ** 2 + v1_snr ** 2)
        scale_factor = self.snr / nomf_snr

        # Rescale the signal to match the correct SNR
        strain = scale_factor * detector_signal + detector_noise[:, start:start + detector_signal.shape[1]]
        # h1_strain = TimeSeries(strain[0], delta_t=1.0 / self.sampling_rate, epoch=self.parameters.loc[idx, 'h1_start_time'], dtype=np.float_)
        # l1_strain = TimeSeries(strain[1], delta_t=1.0 / self.sampling_rate, epoch=self.parameters.loc[idx, 'l1_start_time'], dtype=np.float_)
        # v1_strain = TimeSeries(strain[2], delta_t=1.0 / self.sampling_rate, epoch=self.parameters.loc[idx, 'v1_start_time'], dtype=np.float_)
        #
        #
        # TODO: the whiting / bandpass filtering gives weird results, for now we will ignore this
        # # Whiten the signal
        # h1_strain = h1_strain.whiten(self.whiten_duration, self.whiten_max_filter, remove_corrupted=False)
        # l1_strain = l1_strain.whiten(self.whiten_duration, self.whiten_max_filter, remove_corrupted=False)
        # v1_strain = v1_strain.whiten(self.whiten_duration, self.whiten_max_filter, remove_corrupted=False)
        #
        # # Bandpass filter
        # if self.bandpass_lower > 0:
        #     h1_strain.highpass_fir(self.bandpass_lower, remove_corrupted=False, order=512)
        #     l1_strain.highpass_fir(self.bandpass_lower, remove_corrupted=False, order=512)
        #     v1_strain.highpass_fir(self.bandpass_lower, remove_corrupted=False, order=512)
        #
        # if self.bandpass_upper < self.sampling_rate:
        #     h1_strain.lowpass_fir(self.bandpass_upper, remove_corrupted=False, order=512)
        #     l1_strain.lowpass_fir(self.bandpass_upper, remove_corrupted=False, order=512)
        #     v1_strain.lowpass_fir(self.bandpass_upper, remove_corrupted=False, order=512)
        #
        # # Convert the TimeSeries to a numpy array
        # strain[0] = h1_strain.numpy()
        # strain[1] = l1_strain.numpy()
        # strain[2] = v1_strain.numpy()

        # TODO: For now we multiply the signal with a factor of 10 ** 20, we should normalize using a cleaner method
        return strain * np.power(10, 20), self.target.loc[idx].values