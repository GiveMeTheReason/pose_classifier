import typing as tp

import numpy as np

from config.base_config import ConfigBaseClass

##################################################

delta_t = 1 / 30
sigma_u = 20.0 * 500
sigma_z = 20.0

init_F = np.array([
    [1, delta_t],
    [0, 1],
])
init_H = np.array([
    [1, 0],
])
init_P = np.array([
    [20 ** 2, 0],
    [0, 0.1 ** 2],
])
init_R = np.array([
    [1],
]) * (sigma_z ** 2)
init_Q = np.array([
    [1/4 * delta_t ** 2, 1/2 * delta_t],
    [1/2 * delta_t, 1],
]) * (delta_t ** 2) * (sigma_u ** 2)

class InitParams(ConfigBaseClass):
    delta_t: float = delta_t
    sigma_u: float = sigma_u
    sigma_z: float = sigma_z
    init_F: np.ndarray = init_F
    init_H: np.ndarray = init_H
    init_P: np.ndarray = init_P
    init_R: np.ndarray = init_R
    init_Q: np.ndarray = init_Q

##################################################

def md_sigma(md: float) -> float:
    return 1 + 1 * md ** 2

sigma_z_heuristic = md_sigma

class Heuristics(ConfigBaseClass):
    sigma_z_heuristic: tp.Callable[[float], float] = sigma_z_heuristic

##################################################

class KALMAN_FILTER_CONFIG(ConfigBaseClass):
    init_params: InitParams = InitParams()
    heuristics: Heuristics = Heuristics()
