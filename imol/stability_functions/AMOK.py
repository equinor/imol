from collections.abc import Callable

import numpy as np
from scipy.integrate import quad


class AMOK:
    psim: Callable[[float], float]
    psih: Callable[[float], float]

    def __init__(self, params: dict[str, float] | None = None) -> None:
        # Fuga defaults; using these enables the analytical psim/psih path
        self.ams = 5
        self.nms = 1
        self.amu = -19.3
        self.nmu = -4
        self.ahs = 7.8
        self.nhs = 1
        self.ahu = -12
        self.nhu = -2
        if params is not None:
            for key, val in params.items():
                setattr(self, key, val)

        if (self.ams == 5) & (self.nms == 1) & (self.amu == -19.3) & (self.nmu == -4):
            self.psim = self.apsim
        else:
            self.psim = self.spsim

        if (self.ahs == 7.8) & (self.nhs == 1) & (self.ahu == -12) & (self.nhu == -2):
            self.psih = self.apsih
        else:
            self.psih = self.spsih

    def phi(self, a: float, n: float, z: float) -> float:
        return (1 + a * z) ** (1 / n)

    def phim(self, z: float) -> float:
        if z >= 0:
            a = self.ams
            n = self.nms
            return self.phi(a, n, z)
        else:
            a = self.amu
            n = self.nmu
            return self.phi(a, n, z)

    def phih(self, z: float) -> float:
        if z >= 0:
            a = self.ahs
            n = self.nhs
            return self.phi(a, n, z)
        else:
            a = self.ahu
            n = self.nhu
            return self.phi(a, n, z)

    def spsim(self, z: float) -> float:
        if z >= 0:
            return quad(lambda x: (1 - self.phim(x)) / x, 0, z)[0]
        else:
            return -quad(lambda x: (1 - self.phim(x)) / x, z, 0)[0]

    def apsim(self, z: float) -> float:
        if z >= 0:
            a = self.ams
            return -a * z
        else:
            a = self.amu
            return (
                0.5 * np.pi
                - 2 * np.arctan((1 + a * z) ** 0.25)
                + np.log(
                    (1 + (1 + a * z) ** 0.25) ** 2 * ((1 + (1 + a * z) ** 0.5) / 8)
                )
            )

    def spsih(self, z: float) -> float:
        if z >= 0:
            return quad(lambda x: (1 - self.phih(x)) / x, 0, z)[0]
        else:
            return -quad(lambda x: (1 - self.phih(x)) / x, z, 0)[0]

    def apsih(self, z: float) -> float:
        if z >= 0:
            a = self.ahs
            return -a * z
        else:
            a = self.ahu
            return 2 * np.log(1 + (1 + a * z) ** 0.5) - np.log(4)
