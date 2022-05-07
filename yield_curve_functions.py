import numpy as np


def nelson_siegel(T, tau, b0, b1, b2):
    x = T/tau
    return b0 + b1* (1-np.exp(-x))/x + b2*((1-np.exp(-x))/x - np.exp(-x))