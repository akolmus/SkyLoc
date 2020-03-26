import sys
import time
import yaml
import torch
import logging
import pandas as pd

from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from lib_data import GravitationalWavesDataset
from lib_models import VonMisesFisher, Kent


def training(cfg_data: dict, cfg_train: dict, id_time:str) -> torch.nn.Module:
    """ The training procedure for a simple model """

    # Get logger
    logger = logging.getLogger()
    logger.info(f'Model being trained on the loss function: {cfg_train["model"]["loss"]}')
    logger.info(f'SNR: {cfg_train["data"]["SNR"]}')

    # Load the parameters
    parameters = pd.read_csv(f'../moredata/{cfg_train["data"]["name"]}/parameters.csv')

    # Acquire the correct model
    if cfg_train['model']['loss'] == 'VMF':
        model = VonMisesFisher(id_time)
        model = model.double()
        model = model.to(cfg_train['training']['device'])
    elif cfg_train['model']['loss'] == 'Kent':
        model = Kent(id_time)
        model = model.double()
        model = model.to(cfg_train['training']['device'])
    else:
        raise ValueError(f'The specified loss {cfg_train["model"]["loss"]} is unknown')

    # Setup the data for now we do a simple 90:10 split
    split_idx = int(0.9 * parameters.shape[0])
    trn_data = GravitationalWavesDataset(parameters.iloc[:split_idx], cfg_data, cfg_train)
    val_data = GravitationalWavesDataset(parameters.iloc[split_idx:], cfg_data, cfg_train)
    trn_load = DataLoader(trn_data, batch_size=cfg_train['training']['batch_size'], shuffle=True, num_workers=cfg_train['training']['nb_workers'])
    val_load = DataLoader(val_data, batch_size=cfg_train['training']['batch_size'], shuffle=False, num_workers=cfg_train['training']['nb_workers'])

    # Small summary
    logger.info(f'number of training data points: {split_idx}')
    logger.info(f'number of validation data points: {parameters.shape[0] - split_idx}')
    time.sleep(0.1)

    # Train the model
    for epoch_nb in range(cfg_train['training']['nb_epochs']):
        # To keep the overview nice
        logger.info(f'Epoch number: {epoch_nb}')
        time.sleep(0.1)

        # The actual training is defined inside the model
        model.epoch(cfg_train, trn_load, val_load, epoch_nb)


if __name__ == '__main__':
    # Open the configuration files
    with open('config_data.yml') as config_file:
        cfg_data = yaml.safe_load(config_file)

    with open('config_train.yml') as config_file:
        cfg_train = yaml.safe_load(config_file)

    # Current time as identifier and create directory
    id_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    Path(f'../output/{cfg_train["data"]["name"]}/{id_time}/').mkdir(parents=True)

    # Setup the logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(f'../output/{cfg_train["data"]["name"]}/{id_time}/logger.log'),
            logging.StreamHandler(sys.stdout)
        ])

    # Start the training
    model = training(cfg_data, cfg_train, id_time)

    # # Save the model
    # torch.save(model.state_dict(), f'../output/{cfg_train["data"]["name"]}/{id_time}/model_para.pck')
