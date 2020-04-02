import numpy as np

from scipy.special import gamma, iv

class VonMisesFisherDistribution(object):
    """ A Von Mises-Fisher distribution """

    def __init__(self, kappa: float, mu: np.ndarray):
        """  """
        self.kappa = kappa
        self.mu = mu

    def pdf(self, vec: np.ndarray, with_constant=False) -> np.ndarray:
        """ Return the probability density function value for vec """

        if with_constant:
            if self.kappa > 100:
                return self.kappa / (2 * np.pi * (1 - np.exp(-2 * self.kappa))) * np.exp(self.kappa * (vec @ self.mu - 1))
            else:
                return self.kappa / (2 * np.pi * (np.exp(self.kappa) - np.exp(self.kappa))) * np.exp(self.kappa * (vec @ self.mu))
        else:
            if self.kappa > 100:
                return np.exp(self.kappa * (vec @ self.mu - 1))
            else:
                return np.exp(self.kappa * vec @ self.mu)

    def logpdf(self, vec: np.ndarray) -> np.ndarray:
        """ Return the loglikelihood  for vec """
        return np.log(self.kappa + 1e-10) - np.log(1 - np.exp(-2 * self.kappa)) - self.kappa - np.log(2 * np.pi) + (self.kappa * vec @ self.mu)

    def rvs(self, nb_samples: int):
        """ Generate random samples from this distribution - using rejection sampling"""
        pdf_max = self.pdf(self.mu)
        accepted_vecs = np.zeros((0, 3))

        # Iterate until we have enough samples
        while accepted_vecs.shape[0] < nb_samples:
            vecs = np.random.normal(0, 1, size=(1000, 3))
            vecs = np.divide(vecs, np.linalg.norm(vecs, axis=1, keepdims=True))
            pvalues = self.pdf(vecs)
            accepted_vecs = np.vstack((accepted_vecs, vecs[np.random.uniform(0, pdf_max, size=(nb_samples,)) < pvalues]))

        return accepted_vecs[:nb_samples]


class KentDistribution(object):
    """ A Kent distribution """

    def __init__(self, kappa: float, beta: float, gamma1: np.ndarray, gamma2: np.ndarray):
        """  """
        # Set parameters
        self.kappa = kappa
        self.beta = beta
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = np.cross(gamma1, gamma2)

        # Check if the basis is orthonormal
        if np.abs(self.gamma1 @ self.gamma2) > 1e-10 or np.abs(np.linalg.norm(self.gamma1) - 1) > 1e-10 or np.abs(np.linalg.norm(self.gamma2) - 1) > 1e-10:
            raise ValueError(f'The basis is not orthonormal! {np.abs(self.gamma1 @ self.gamma2)}, {np.abs(np.linalg.norm(self.gamma1) - 1)}, {np.abs(np.linalg.norm(self.gamma2) - 1)}')

        # Calculate the normalization constant
        self.norm_constant = self.calc_constant()

    def calc_constant(self) -> np.float:
        """ Calculate the normalization constant, not necessary but fun regardless """
        total = 0
        step = 0
        j = 0
        while step / (total + 1e-10) > 1e-10 and j < 10:
            step = gamma(j + 0.5) / gamma(j + 1) * np.exp(np.log(self.beta) * 2 * j + np.log(0.5 * self.kappa) * (-2 * j - 0.5)) * iv(2 * j + 0.5, self.kappa)
            total = total + step
            j = j + 1

        return total

    def pdf(self, vec: np.ndarray, with_constant=False) -> np.ndarray:
        """ Return the probability density function value for vec """

        if with_constant:
            return 1 / self.norm_constant * np.exp(self.kappa * vec @ self.gamma1 + self.beta * ((vec @ self.gamma2) ** 2 - (vec @ self.gamma3) ** 2))
        else:
            return np.exp(self.kappa * vec @ self.gamma1 + self.beta * ((vec @ self.gamma2) ** 2 - (vec @ self.gamma3) ** 2))

    def logpdf(self, vec: np.ndarray, with_constant=False) -> np.ndarray:
        """ Return the loglikelihood for vec """
        if with_constant:
            return self.kappa * vec @ self.gamma1 + self.beta * ((vec @ self.gamma2) ** 2 - (vec @ self.gamma3) ** 2) - np.log(self.norm_constant)
        else:
            return self.kappa * vec @ self.gamma1 + self.beta * ((vec @ self.gamma2) ** 2 - (vec @ self.gamma3) ** 2)

    def rvs(self, nb_samples: int):
        """ Generate random samples from this distribution - using rejection sampling"""
        pdf_max = self.pdf(self.gamma1)
        accepted_vecs = np.zeros((0, 3))

        # Iterate until we have enough samples
        while accepted_vecs.shape[0] < nb_samples:
            vecs = np.random.normal(0, 1, size=(1000, 3))
            vecs = np.divide(vecs, np.linalg.norm(vecs, axis=1, keepdims=True))
            pvalues = self.pdf(vecs)
            accepted_vecs = np.vstack((accepted_vecs, vecs[np.random.uniform(0, pdf_max, size=(nb_samples,)) < pvalues]))

        return accepted_vecs[:nb_samples]

