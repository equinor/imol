import numpy as np


class CB05:
    def __init__(self, params: dict = None):
        # Stability function constants
        # CB05 (Cheng Y. & Brutsaert W., 2005) stability function parameters (only stable conditions)
        default_params = {
            "amu": -19.3,
            "nmu": -4,
            "ahu": -12,
            "nhu": -2,
            "as_cb": 6.1,
            "bs_cb": 2.5,
            "cs_cb": 5.3,
            "ds_cb": 1.1,
        }

        actual_params = (
            default_params if params is None else {**default_params, **params}
        )
        for key, val in actual_params.items():
            setattr(self, key, val)

    def phi(self, a, n, z):
        return (1 + a * z) ** (1 / n)

    def phim(self, z):
        if z >= 0:
            a = self.as_cb
            b = self.bs_cb
            return 1 + a * (
                (z + z**b * (1 + z**b) ** ((1 - b) / b)) / (z + (1 + z**b) ** (1 / b))
            )
        else:
            a = self.amu
            n = self.nmu
            return self.phi(a, n, z)

    def phih(self, z):
        if z >= 0:
            c = self.cs_cb
            d = self.ds_cb
            return 1 + c * (
                (z + z**d * (1 + z**d) ** ((1 - d) / d)) / (z + (1 + z**d) ** (1 / d))
            )
        else:
            a = self.ahu
            n = self.nhu
            return self.phi(a, n, z)

    def psim(self, z):
        if z >= 0:
            a = self.as_cb
            b = self.bs_cb
            return -a * np.log(z + (1 + z**b) ** (1 / b))
        else:
            a = self.amu
            return (
                0.5 * np.pi
                - 2 * np.arctan((1 + a * z) ** 0.25)
                + np.log(
                    (1 + (1 + a * z) ** 0.25) ** 2 * ((1 + (1 + a * z) ** 0.5) / 8)
                )
            )

    def psih(self, z):
        if z >= 0:
            c = self.cs_cb
            d = self.ds_cb
            return -c * np.log(z + (1 + z**d) ** (1 / d))
        else:
            a = self.ahu
            return 2 * np.log(1 + (1 + a * z) ** 0.5) - np.log(4)
