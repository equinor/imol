import numpy as np
from scipy.integrate import quad


class AMOK:
    def __init__(self, params: dict = None):
        # Stability function constants
        # These are the Fuga defaults, but may be changed by the user
        # Using the defailts will speed up the calculations as an analytical
        #  solution to the stability function integral is provided
        default_params = {
            "ams": 5,
            "nms": 1,
            "amu": -19.3,
            "nmu": -4,
            "ahs": 7.8,
            "nhs": 1,
            "ahu": -12,
            "nhu": -2,
        }
        actual_params = (
            default_params if params is None else {**default_params, **params}
        )
        for key, val in actual_params.items():
            setattr(self, key, val)

        if (self.ams == 5) & (self.nms == 1) & (self.amu == -19.3) & (self.nmu == -4):
            self.psim = self.apsim
        else:
            self.psim = self.spsim

        if (self.ahs == 7.8) & (self.nhs == 1) & (self.ahu == -12) & (self.nhu == -2):
            self.psih = self.apsih
        else:
            self.psih = self.spsih

    def phi(self, a, n, z):
        return (1 + a * z) ** (1 / n)

    def phim(self, z):
        if z >= 0:
            a = self.ams
            n = self.nms
            return self.phi(a, n, z)
        else:
            a = self.amu
            n = self.nmu
            return self.phi(a, n, z)

    def phih(self, z):
        if z >= 0:
            a = self.ahs
            n = self.nhs
            return self.phi(a, n, z)
        else:
            a = self.ahu
            n = self.nhu
            return self.phi(a, n, z)

    def spsim(self, z):
        if z >= 0:
            return quad(lambda x: (1 - self.phim(x)) / x, 0, z)[0]
        else:
            return -quad(lambda x: (1 - self.phim(x)) / x, z, 0)[0]

    def apsim(self, z):
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

    def spsih(self, z):
        if z >= 0:
            return quad(lambda x: (1 - self.phih(x)) / x, 0, z)[0]
        else:
            return -quad(lambda x: (1 - self.phih(x)) / x, z, 0)[0]

    def apsih(self, z):
        if z >= 0:
            a = self.ahs
            return -a * z
        else:
            a = self.ahu
            return 2 * np.log(1 + (1 + a * z) ** 0.5) - np.log(4)
