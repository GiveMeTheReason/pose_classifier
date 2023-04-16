import typing as tp

import numpy as np

import filterpy.kalman as kalman


def identity(md: float) -> float:
    return md


def md_sigma(md: float) -> float:
    return 1 + 1 * md ** 2


class KalmanFilter:
    def __init__(
        self,
        delta_t: float,
        sigma_u: float,
        sigma_z: float,
        init_F: np.ndarray,
        init_H: np.ndarray,
        init_P: np.ndarray,
        init_R: np.ndarray,
        init_Q: np.ndarray,
        sigma_z_heuristic: tp.Callable[[float], float] = identity,
    ) -> None:
        self.delta_t = delta_t
        self.sigma_u = sigma_u
        self.sigma_z = sigma_z

        self.init_F = init_F
        self.init_H = init_H
        self.init_P = init_P
        self.init_R = init_R
        self.init_Q = init_Q

        self.sigma_z_heuristic = sigma_z_heuristic

        self.filter = kalman.KalmanFilter(
            dim_x=init_F.shape[0],
            dim_z=init_H.shape[0],
            dim_u=0,
        )
        self.reset()

    def reset(
        self,
        init_state: tp.Optional[np.ndarray] = None,
    ) -> None:
        if init_state is None:
            init_state = np.zeros((self.init_F.shape[0], 1))
        self.filter.x = init_state
        self.filter.F = self.init_F
        self.filter.H = self.init_H
        self.filter.P = self.init_P
        self.filter.R = self.init_R
        self.filter.Q = self.init_Q

    def predict(
        self,
        u: tp.Optional[np.ndarray] = None,
        B: tp.Optional[np.ndarray] = None,
        F: tp.Optional[np.ndarray] = None,
        Q: tp.Optional[np.ndarray] = None,
    ) -> np.ndarray:
        self.filter.predict(u, B, F, Q)
        return self.x

    def update(
        self,
        z: np.ndarray,
        R: tp.Optional[np.ndarray] = None,
        H: tp.Optional[np.ndarray] = None,
        use_heuristic: bool = False,
    ) -> np.ndarray:
        if use_heuristic:
            distance = self._mahalanobis(z)
            if R is None:
                R = self.R
            R = R * (self.sigma_z_heuristic(distance) ** 2)
        self.filter.update(z, R, H)
        return self.x

    def _mahalanobis(
        self,
        z: tp.Optional[np.ndarray] = None,
    ) -> float:
        if z is None:
            z = self.z
        residual: np.ndarray = self.filter.residual_of(z)
        S_matrix = self.H @ self.P @ self.H.T + self.R
        distance = np.sqrt(residual.T @ np.linalg.inv(S_matrix) @ residual)
        return float(distance)

    @property
    def F(self) -> np.ndarray:
        return self.filter.F

    @property
    def H(self) -> np.ndarray:
        return self.filter.H

    @property
    def x(self) -> np.ndarray:
        return self.filter.x

    @property
    def z(self) -> np.ndarray:
        return self.filter.z

    @property
    def P(self) -> np.ndarray:
        return self.filter.P

    @property
    def R(self) -> np.ndarray:
        return self.filter.R

    @property
    def Q(self) -> np.ndarray:
        return self.filter.Q

    @property
    def K(self) -> np.ndarray:
        return self.filter.K


class KalmanFilters:
    def __init__(self, kalman_filters: tp.Sequence[KalmanFilter]) -> None:
        self.filters = kalman_filters

    def reset(self, init_states: tp.Sequence[np.ndarray]) -> None:
        for init_state, kalman_filter in zip(init_states, self.filters):
            kalman_filter.reset(init_state)

    def predict(
        self,
        projection: tp.Optional[tp.Union[int, tp.Sequence[int]]] = None,
    ) -> tp.Union[tp.List[float], tp.List[np.ndarray]]:
        for kalman_filter in self.filters:
            kalman_filter.predict()

        if projection is None:
            return [kalman_filter.x[:, 0] for kalman_filter in self.filters]
        return [kalman_filter.x[projection, 0] for kalman_filter in self.filters]

    def update(
        self,
        z: np.ndarray,
        use_heuristic: bool = False,
        projection: tp.Optional[tp.Union[int, tp.Sequence[int]]] = None,
    ) -> tp.Union[tp.List[float], tp.List[np.ndarray]]:
        for observation, kalman_filter in zip(z, self.filters):
            kalman_filter.update(observation, use_heuristic=use_heuristic)

        if projection is None:
            return [kalman_filter.x[:, 0] for kalman_filter in self.filters]
        return [kalman_filter.x[projection, 0] for kalman_filter in self.filters]
