import logging
import math
import time
import utils
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

class VonMisesFisher(nn.Module):
    """
    A convolutional neural network that parametrizes a Von Mises-Fisher distribution over the surface of a sphere. The
    goal is to estimate the origin of a gravitational wave via a probability distribution. This distribution is the
    Von Mises-Fisher distribution - a simple Gaussian over the surface of a sphere.
    """

    def __init__(self, id_time: str):
        """
        The initialization of the network

        :param id_time: an identifier for the network
        """
        super().__init__()

        # Set the id_time, this enables us to save the images
        self.id_time = id_time

        # Setup the convolutional layers
        self.conv = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=16, dilation=1),
            nn.MaxPool1d(kernel_size=2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv1d(16, 16, kernel_size=16, dilation=2),
            nn.MaxPool1d(kernel_size=4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv1d(16, 32, kernel_size=32, dilation=2),
            nn.MaxPool1d(kernel_size=4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv1d(32, 32, kernel_size=64, dilation=2),
        )

        # Setup the dense layers
        self.dense = nn.Sequential(
            nn.Linear(in_features=3584, out_features=256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(in_features=256, out_features=3),
        )

        # Set the optimizer
        self.optimizer = torch.optim.Adam(params=self.parameters())

    def forward(self, x):
        """ The forward pass through the network """
        # The actual forward pass
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        out = self.dense(x)

        # Normalize and extract the kappa coefficient - this is also done in PYRO
        kappa = torch.norm(out, p=2, dim=-1).view(-1, 1)
        out = 1 / kappa * out

        return kappa, out

    @staticmethod
    def vmf_loss(kappa, out, tar):
        """ The negative loglikelihood for the Von Mises Fisher distribution for p=3 """
        return -torch.sum(torch.log(kappa + 1e-10) - torch.log(1 - torch.exp(-2*kappa)) - kappa - math.log(2 * math.pi) + (kappa * out * tar).sum(axis=1, keepdims=True))

    def epoch(self, cfg_train: dict, trn_load: DataLoader, val_load: DataLoader, epoch_nb: int):
        """ A single training epoch """
        logger = logging.getLogger()

        # Setup the metrics
        trn_lss = 0
        val_lss = 0
        trn_inner = 0
        val_inner = 0
        trn_maae = 0
        val_maae = 0

        # Setup the stock
        trn_kappa = np.zeros((0, 1), dtype=np.float_)
        val_kappa = np.zeros((0, 1), dtype=np.float_)

        # Training
        self.train(mode=True)
        for x, tar in tqdm(trn_load):
            # Setup for the forward pass
            x = x.to(cfg_train['training']['device'])
            tar = tar.to(cfg_train['training']['device'])
            self.optimizer.zero_grad()

            # Forward pass
            kappa, out = self.forward(x)
            lss = self.vmf_loss(kappa, out, tar)

            # Backward pass
            lss.backward()
            self.optimizer.step()

            # Calculate metrics
            mae, mse, inner, maae = utils.vmf_metrics(kappa.detach().cpu().numpy(),
                                                       out.detach().cpu().numpy(),
                                                       tar.detach().cpu().numpy())

            # Update metrics
            trn_lss += float(lss.item())
            trn_inner += inner.sum()
            trn_maae += np.abs(maae * 180 / np.pi).sum()

            # Update stock
            trn_kappa = np.vstack((trn_kappa, kappa.detach().cpu().numpy()))

        # Reset for tqdm
        time.sleep(0.1)

        # Validation
        self.train(mode=False)
        for x, tar in tqdm(val_load):
            # Setup for the forward pass
            x = x.to(cfg_train['training']['device'])
            tar = tar.to(cfg_train['training']['device'])
            self.optimizer.zero_grad()

            # Forward pass
            kappa, out = self.forward(x)
            lss = self.vmf_loss(kappa, out, tar)

            # Calculate metrics
            mae, mse, inner, maae = utils.vmf_metrics(kappa.detach().cpu().numpy(),
                                                       out.detach().cpu().numpy(),
                                                       tar.detach().cpu().numpy())

            # Update metrics
            val_lss += float(lss.item())
            val_inner += inner.sum()
            val_maae += np.abs(maae * 180 / np.pi).sum()

            # Update stock
            val_kappa = np.vstack((val_kappa, kappa.detach().cpu().numpy()))

        # TODO: we can collect the kappa of the validation and make plots (as we did in the last version)
        logger.info(f'General: '
                    f'trn_lss: {trn_lss / len(trn_load.dataset):.2f} - '
                    f'val_lss: {val_lss / len(val_load.dataset):.2f}')
        logger.info(f'Trn: '
                    f'kappa_mean: {trn_kappa.sum() / len(trn_load.dataset):.2f} - '
                    f'inner_product: {trn_inner / len(trn_load.dataset):.2f} - '
                    f'maae: {trn_maae / len(trn_load.dataset):.2f}')
        logger.info(f'Val: '
                    f'kappa_mean: {val_kappa.sum() / len(val_load.dataset):.2f} - '
                    f'inner_product: {val_inner / len(val_load.dataset):.2f} - '
                    f'maae: {val_maae / len(val_load.dataset):.2f}')

        # Small waiting step for nicer formatting with tqdm
        time.sleep(0.1)

    def evaluate(self, cfg_train: dict, _load: DataLoader):
        """ Evaluate given data """
        logger = logging.getLogger()

        # Setup metrics
        _lss = 0
        _inner = 0
        _maae = 0

        # Setup stock
        _kappa = np.zeros((0, 1), dtype=np.float_)
        _beta = np.zeros((0, 1), dtype=np.float_)

        # Validation
        self.train(mode=False)
        for x, tar in tqdm(_load):
            # Setup for the forward pass
            x = x.to(cfg_train['training']['device'])
            tar = tar.to(cfg_train['training']['device'])
            self.optimizer.zero_grad()

            # Forward pass
            kappa, out = self.forward(x)
            lss = self.vmf_loss(kappa, out, tar)

            # Calculate metrics
            mae, mse, inner, maae = utils.vmf_metrics(kappa.detach().cpu().numpy(),
                                                      out.detach().cpu().numpy(),
                                                      tar.detach().cpu().numpy())

            # Update metrics
            _lss += float(lss.item())
            _inner += inner.sum()
            _maae += np.abs(maae * 180 / np.pi).sum()

            # Update stock
            _kappa = np.vstack((_kappa, kappa.detach().cpu().numpy()))

        logger.info(f'General: '
                    f'_lss: {_lss / len(_load.dataset):.2f}')
        logger.info(f'Metrics: '
                    f'kappa_mean: {_kappa.sum() / len(_load.dataset):.2f} - '
                    f'beta_mean: {_beta.sum() / len(_load.dataset):.2f} - '
                    f'inner_product: {_inner / len(_load.dataset):.2f} - '
                    f'maae: {_maae / len(_load.dataset):.2f}')

    def predict_known(self, cfg_train: dict, _load: DataLoader):
        """ Predict on known data - returns (output, tar)"""

        # Validation
        self.train(mode=False)
        for x, tar in tqdm(_load):
            # Setup for the forward pass
            x = x.to(cfg_train['training']['device'])
            tar = tar.to(cfg_train['training']['device'])
            self.optimizer.zero_grad()

            # Forward pass
            kappa, out = self.forward(x)


class Kent(nn.Module):
    """
    A convolutional neural network that parametrizes a Kent distribution over the surface of a sphere. The goal is to
    estimate the origin of a gravitational wave via a probability distribution. This distribution is the Kent
    distribution - a Gaussian over the surface of a sphere where the two axis are not necessarily identical. It can
    generate ellipsoid probability distributions.
    """

    def __init__(self, id_time: str):
        """
        The initialization of the network

        :param id_time: an identifier for the network
        """
        super().__init__()

        # Set the id_time, this enables us to save the images
        self.id_time = id_time

        # Setup the convolutional layers
        self.conv = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=16, dilation=1),
            nn.MaxPool1d(kernel_size=2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv1d(16, 16, kernel_size=16, dilation=2),
            nn.MaxPool1d(kernel_size=4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv1d(16, 32, kernel_size=32, dilation=2),
            nn.MaxPool1d(kernel_size=4),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv1d(32, 32, kernel_size=64, dilation=2),
        )

        # Setup the dense layers
        self.dense = nn.Sequential(
            nn.Linear(in_features=3584, out_features=256),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(in_features=256, out_features=8),
        )

        # Set the optimizer
        self.optimizer = torch.optim.Adam(params=self.parameters())

    def forward(self, x):
        """ The forward pass through the network """
        # The actual forward pass
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        vecs = self.dense(x)

        # split the vectors into three components
        vec1, vec2 = vecs[:, 2:5], vecs[:, 5:8]
        kappa, beta = vecs[:, 0:1], vecs[:, 1:2]
        kappa = torch.nn.functional.softplus(kappa) + 1e-5
        beta = torch.nn.functional.softplus(beta)
        beta = torch.min(beta, kappa / 2 - 1e-10)
        # beta = torch.min(beta + 0.01, kappa / 2 - 1e-10)

        # Gram-Schmidt orthogonalization + the cross product generate an orthogonal major and minor axis
        vec2 = vec2 - (vec2 * vec1).sum(axis=1, keepdim=True) / (vec1 * vec1).sum(axis=1, keepdim=True) * vec1
        vec3 = torch.cross(vec1, vec2, dim=1)

        # Calculate, we could extract kappa from this norm - but the performance is still shit
        norm_vec1 = torch.norm(vec1, p=2, dim=-1).view(-1, 1)
        norm_vec2 = torch.norm(vec2, p=2, dim=-1).view(-1, 1)
        norm_vec3 = torch.norm(vec3, p=2, dim=-1).view(-1, 1)

        # Normalize the output to have radius equal to 1
        vec1 = 1 / norm_vec1 * vec1
        vec2 = 1 / norm_vec2 * vec2
        vec3 = 1 / norm_vec3 * vec3

        return kappa, beta, vec1, vec2, vec3

    @staticmethod
    def kent_loss(kappa, beta, vec1, vec2, vec3, tar):
        """ This an approximation of the loglikelihood for the Kent distribution - see page 3 of: Kent, J. T. (1982). The Fisher‐Bingham distribution on the sphere. Journal of the Royal Statistical Society: Series B (Methodological), 44(1), 71-80. """
        return -torch.sum(- math.log(2 * math.pi) - kappa + 0.5 * torch.log(kappa - 2 * beta) + 0.5 * torch.log(kappa + 2 * beta) + kappa * (vec1 * tar).sum(axis=1, keepdim=True) + beta * ((vec2 * tar).sum(axis=1, keepdim=True) ** 2 - (vec3 * tar).sum(axis=1, keepdim=True) ** 2))

    def epoch(self, cfg_train: dict, trn_load: DataLoader, val_load: DataLoader, epoch_nb: int):
        """ A single training epoch """
        logger = logging.getLogger()

        # Setup the metrics
        trn_lss = 0
        val_lss = 0
        trn_inner = 0
        val_inner = 0
        trn_maae = 0
        val_maae = 0

        # Setup the stock
        trn_kappa = np.zeros((0, 1), dtype=np.float_)
        val_kappa = np.zeros((0, 1), dtype=np.float_)
        trn_beta = np.zeros((0, 1), dtype=np.float_)
        val_beta = np.zeros((0, 1), dtype=np.float_)

        # Training
        self.train(mode=True)
        for x, tar in tqdm(trn_load):
            # Setup for the forward pass
            x = x.to(cfg_train['training']['device'])
            tar = tar.to(cfg_train['training']['device'])
            self.optimizer.zero_grad()

            # Forward pass
            kappa, beta, vec1, vec2, vec3 = self.forward(x)
            lss = self.kent_loss(kappa, beta, vec1, vec2, vec3, tar)

            # Backward pass
            lss.backward()
            self.optimizer.step()

            # Calculate metrics
            mae, mse, inner, maae = utils.kent_metrics(kappa.detach().cpu().numpy(),
                                                        beta.detach().cpu().numpy(),
                                                        vec1.detach().cpu().numpy(),
                                                        tar.detach().cpu().numpy())

            # Update metrics
            trn_lss += float(lss.item())
            trn_inner += inner.sum()
            trn_maae += np.abs(maae * 180 / np.pi).sum()

            # Update stock
            trn_kappa = np.vstack((trn_kappa, kappa.detach()))
            trn_beta = np.vstack((trn_beta, beta.detach()))

        # Reset for tqdm
        time.sleep(0.1)

        # Validation
        self.train(mode=False)
        for x, tar in tqdm(val_load):
            # Setup for the forward pass
            x = x.to(cfg_train['training']['device'])
            tar = tar.to(cfg_train['training']['device'])
            self.optimizer.zero_grad()

            # Forward pass
            kappa, beta, vec1, vec2, vec3 = self.forward(x)
            lss = self.kent_loss(kappa, beta, vec1, vec2, vec3, tar)

            # Calculate metrics
            mae, mse, inner, maae = utils.kent_metrics(kappa.detach().cpu().numpy(),
                                                        beta.detach().cpu().numpy(),
                                                        vec1.detach().cpu().numpy(),
                                                        tar.detach().cpu().numpy())

            # Update metrics
            val_lss += float(lss.item())
            val_inner += inner.sum()
            val_maae += np.abs(maae * 180 / np.pi).sum()

            # Update stock
            val_kappa = np.vstack((val_kappa, kappa.detach().cpu().numpy()))
            val_beta = np.vstack((val_beta, beta.detach().cpu().numpy()))

        # TODO: we can collect the kappa of the validation and make plots (as we did in the last version)
        logger.info(f'General: '
                    f'trn_lss: {trn_lss / len(trn_load.dataset):.2f} - '
                    f'val_lss: {val_lss / len(val_load.dataset):.2f}')
        logger.info(f'Trn: '
                    f'kappa_mean: {trn_kappa.sum() / len(trn_load.dataset):.2f} - '
                    f'beta_mean: {trn_beta.sum() / len(trn_load.dataset):.2f} - '
                    f'inner_product: {trn_inner / len(trn_load.dataset):.2f} - '
                    f'maae: {trn_maae / len(trn_load.dataset):.2f}')
        logger.info(f'Val: '
                    f'kappa_mean: {val_kappa.sum() / len(val_load.dataset):.2f} - '
                    f'beta_mean: {val_beta.sum() / len(val_load.dataset):.2f} - '
                    f'inner_product: {val_inner / len(val_load.dataset):.2f} - '
                    f'maae: {val_maae / len(val_load.dataset):.2f}')

        # Small waiting step for nicer formatting with tqdm
        time.sleep(0.1)

    def evaluate(self, cfg_train: dict, _load: DataLoader):
        """ Evaluate given data """
        logger = logging.getLogger()

        # Setup metrics
        _lss = 0
        _inner = 0
        _maae = 0

        # Setup stock
        _kappa = np.zeros((0, 1), dtype=np.float_)
        _beta = np.zeros((0, 1), dtype=np.float_)

        # Validation
        self.train(mode=False)
        for x, tar in tqdm(_load):
            # Setup for the forward pass
            x = x.to(cfg_train['training']['device'])
            tar = tar.to(cfg_train['training']['device'])
            self.optimizer.zero_grad()

            # Forward pass
            kappa, beta, vec1, vec2, vec3 = self.forward(x)
            lss = self.kent_loss(kappa, beta, vec1, vec2, vec3, tar)

            # Calculate metrics
            mae, mse, inner, maae = utils.kent_metrics(kappa.detach().cpu().numpy(),
                                                       beta.detach().cpu().numpy(),
                                                       vec1.detach().cpu().numpy(),
                                                       tar.detach().cpu().numpy())

            # Update metrics
            _lss += float(lss.item())
            _inner += inner.sum()
            _maae += np.abs(maae * 180 / np.pi).sum()

            # Update stock
            _kappa = np.vstack((_kappa, kappa.detach().cpu().numpy()))
            _beta = np.vstack((_beta, beta.detach().cpu().numpy()))

        logger.info(f'General: '
                    f'_lss: {_lss / len(_load.dataset):.2f}')
        logger.info(f'Metrics: '
                    f'kappa_mean: {_kappa.sum() / len(_load.dataset):.2f} - '
                    f'beta_mean: {_beta.sum() / len(_load.dataset):.2f} - '
                    f'inner_product: {_inner / len(_load.dataset):.2f} - '
                    f'maae: {_maae / len(_load.dataset):.2f}')

    def predict_known(self, cfg_train: dict, _load: DataLoader):
        """ Predict on known data - returns (output, tar)"""

        # Validation
        self.train(mode=False)
        for x, tar in tqdm(_load):
            # Setup for the forward pass
            x = x.to(cfg_train['training']['device'])
            tar = tar.to(cfg_train['training']['device'])
            self.optimizer.zero_grad()

            # Forward pass
            kappa, beta, vec1, vec2, vec3 = self.forward(x)
