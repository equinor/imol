import numpy as np


class HB88:
    def __init__(self, params: dict = None):
        # Stability function constants with default values
        # HB88 () stability function parameters (only stable conditions)
        default_params = {
            "amu": -19.3,
            "nmu": -4,
            "ahu": -12,
            "nhu": -2,
            "as_hb": 1,
            "bs_hb": 2 / 3,
            "cs_hb": 5,
            "ds_hb": 0.35,
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
            a = self.as_hb
            b = self.bs_hb
            c = self.cs_hb
            d = self.ds_hb
            return 1 - z * (
                -a + b * d * (-(c / d) + z) * np.exp(-d * z) - b * np.exp(-d * z)
            )
        else:
            a = self.amu
            n = self.nmu
            return self.phi(a, n, z)

    def phih(self, z):
        if z >= 0:
            a = self.as_hb
            b = self.bs_hb
            c = self.cs_hb
            d = self.ds_hb
            return 1 - z * (
                -a * ((2 / 3) * a * z + 1) ** 0.5
                + b * d * (-(c / d) + z) * np.exp(-d * z)
                - b * np.exp(-d * z)
            )
        else:
            a = self.ahu
            n = self.nhu
            return self.phi(a, n, z)

    def psim(self, z):
        if z >= 0:
            a = self.as_hb
            b = self.bs_hb
            c = self.cs_hb
            d = self.ds_hb
            return -b * (z - c / d) * np.exp(-d * z) - a * z - (b * c) / d
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
            a = self.as_hb
            b = self.bs_hb
            c = self.cs_hb
            d = self.ds_hb
            return (
                -b * (z - c / d) * np.exp(-d * z)
                - (1 + (2 / 3) * a * z) ** 1.5
                - (b * c) / d
                + 1
            )
        else:
            a = self.ahu
            return 2 * np.log(1 + (1 + a * z) ** 0.5) - np.log(4)
