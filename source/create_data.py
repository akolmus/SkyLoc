import yaml
import numpy as np

from pathlib import Path
from lib_data import GravitationalWavesGenerator


if __name__  == '__main__':
    # Open the configuration file
    with open('config_data.yml', 'r') as config_file:
        cfg_data = yaml.safe_load(config_file)

    # Write the config file to the destination
    if not Path(f'../moredata/{cfg_data["name"]}/').is_dir():
        Path(f'../moredata/{cfg_data["name"]}/').mkdir(parents=True)

    with open(f'../moredata/{cfg_data["name"]}/config.yml', 'w') as out_file:
        yaml.dump(cfg_data, out_file)

    # Set the numpy seed
    np.random.seed(cfg_data['seed'])

    # Create the Data Generator
    gen = GravitationalWavesGenerator(name=cfg_data['name'], nb_samples=cfg_data['nb_samples'], nb_processes=cfg_data['nb_processes'])

    # Generate the parameters for each event
    gen.generate_parameters(prior=cfg_data['prior'])

    # Create and fill the hdf5 storage file
    gen.parameter2dataset(general_properties=cfg_data['general_properties'])

