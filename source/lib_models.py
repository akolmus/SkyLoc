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
        """ The negative loglikelihood for the Von Mises Fisher distribution for p=3, for large kappa we approximate the log(sinh(kappa)) """
        return -torch.sum(torch.log(kappa + 1e-10) - kappa - math.log(2 * math.pi) + (kappa * out * tar).sum(axis=1, keepdims=True))

    def epoch(self, cfg_train: dict, trn_load: DataLoader, val_load: DataLoader, epoch_nb: int):
        """ A single training epoch """
        logger = logging.getLogger()

        # Setup the metrics
        trn_lss = 0
        trn_kappa = 0
        trn_inner = 0
        trn_angle = 0

        val_lss = 0
        val_kappa = 0
        val_inner = 0
        val_angle = 0

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
            mae, mse, inner, angle = utils.vmf_metrics(kappa.detach().cpu().numpy(),
                                                       out.detach().cpu().numpy(),
                                                       tar.detach().cpu().numpy())

            # Update metrics
            trn_lss += float(lss.item())
            trn_kappa += float(torch.sum(kappa))
            trn_inner += inner.sum()
            trn_angle += np.abs(angle).sum()

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
            mae, mse, inner, angle = utils.vmf_metrics(kappa.detach().cpu().numpy(),
                                                       out.detach().cpu().numpy(),
                                                       tar.detach().cpu().numpy())

            # Update metrics
            val_lss += float(lss.item())
            val_kappa += float(torch.sum(kappa))
            val_inner += inner.sum()
            val_angle += np.abs(angle).sum()

        # TODO: we can collect the kappa of the validation and make plots (as we did in the last version)
        logger.info(f'General: '
                    f'trn_lss: {trn_lss / len(trn_load.dataset):.2f} - '
                    f'val_lss: {val_lss / len(val_load.dataset):.2f}')
        logger.info(f'Trn: '
                    f'kappa_mean: {trn_kappa / len(trn_load.dataset):.2f} - '
                    f'inner_product: {trn_inner / len(trn_load.dataset):.2f} - '
                    f'inner_angle: {trn_angle / len(trn_load.dataset):.2f}')
        logger.info(f'Val: '
                    f'kappa_mean: {val_kappa / len(val_load.dataset):.2f} - '
                    f'inner_product: {val_inner / len(val_load.dataset):.2f} - '
                    f'inner_angle: {val_angle / len(val_load.dataset):.2f}')

        # Small waiting step for nicer formatting with tqdm
        time.sleep(0.1)


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
            nn.Linear(in_features=256, out_features=11),
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
        vec1, vec2, vec3 = vecs[:, 0:3], vecs[:, 3:6], vecs[:, 6:9]
        kappa, beta = vecs[:, 9:10], vecs[:, 10:11]
        kappa = torch.nn.functional.softplus(kappa) + 1e-5
        beta = torch.nn.functional.softplus(beta)
        beta = torch.min(beta, kappa / 2 - 1e-10)

        # Gram-Schmidt orthogonalization
        vec2 = vec2 - (vec2 * vec1).sum(axis=1, keepdim=True) / (vec1 * vec1).sum(axis=1, keepdim=True) * vec1
        vec3 = vec3 - (vec3 * vec1).sum(axis=1, keepdim=True) / (vec1 * vec1).sum(axis=1, keepdim=True) * vec1 - (vec3 * vec2).sum(axis=1, keepdim=True) / (vec2 * vec2).sum(axis=1, keepdim=True) * vec2

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
        trn_kappa = 0
        trn_beta = 0
        trn_inner = 0
        trn_angle = 0

        val_lss = 0
        val_beta = 0
        val_kappa = 0
        val_inner = 0
        val_angle = 0

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
            mae, mse, inner, angle = utils.kent_metrics(kappa.detach().cpu().numpy(),
                                                        beta.detach().cpu().numpy(),
                                                        vec1.detach().cpu().numpy(),
                                                        tar.detach().cpu().numpy())

            # Update metrics
            trn_lss += float(lss.item())
            trn_kappa += float(torch.sum(kappa))
            trn_beta += float(torch.sum(beta))
            trn_inner += inner.sum()
            trn_angle += np.abs(angle).sum()

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
            mae, mse, inner, angle = utils.kent_metrics(kappa.detach().cpu().numpy(),
                                                        beta.detach().cpu().numpy(),
                                                        vec1.detach().cpu().numpy(),
                                                        tar.detach().cpu().numpy())

            # Update metrics
            val_lss += float(lss.item())
            val_kappa += float(torch.sum(kappa))
            trn_beta += float(torch.sum(beta))
            val_inner += inner.sum()
            val_angle += np.abs(angle).sum()

        # TODO: we can collect the kappa of the validation and make plots (as we did in the last version)
        logger.info(f'General: '
                    f'trn_lss: {trn_lss / len(trn_load.dataset):.2f} - '
                    f'val_lss: {val_lss / len(val_load.dataset):.2f}')
        logger.info(f'Trn: '
                    f'kappa_mean: {trn_kappa / len(trn_load.dataset):.2f} - '
                    f'beta_mean: {trn_beta / len(trn_load.dataset):.2f} - '
                    f'inner_product: {trn_inner / len(trn_load.dataset):.2f} - '
                    f'inner_angle: {trn_angle / len(trn_load.dataset):.2f}')
        logger.info(f'Val: '
                    f'kappa_mean: {val_kappa / len(val_load.dataset):.2f} - '
                    f'beta_mean: {val_beta / len(val_load.dataset):.2f} - '
                    f'inner_product: {val_inner / len(val_load.dataset):.2f} - '
                    f'inner_angle: {val_angle / len(val_load.dataset):.2f}')

        # Small waiting step for nicer formatting with tqdm
        time.sleep(0.1)