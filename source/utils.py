import numpy as np


def vmf_metrics(kappa: np.ndarray, out: np.ndarray, tar: np.ndarray):
    """ Calculate some metrics """

    # Simple metrics on the difference between output and target
    mae = np.sum(np.abs(out - tar), axis=1, keepdims=True)
    mse = np.sum((out - tar) ** 2, axis=1, keepdims=True)
    inner_product = np.einsum('ij,ij->i', out, tar)
    inner_angle = np.arccos(inner_product)

    # TODO: define an insightful metric for kappa

    return mae, mse, inner_product, inner_angle

def kent_metrics(kappa: np.ndarray, beta: np.ndarray, out: np.ndarray, tar: np.ndarray):
    """ Calculate some metrics """

    # Simple metrics on the difference between output and target
    mae = np.sum(np.abs(out - tar), axis=1, keepdims=True)
    mse = np.sum((out - tar) ** 2, axis=1, keepdims=True)
    inner_product = np.einsum('ij,ij->i', out, tar)
    inner_angle = np.arccos(inner_product)

    # TODO: define an insightful metric for kappa

    # TODO: define an insightful metric for Beta

    return mae, mse, inner_product, inner_angle