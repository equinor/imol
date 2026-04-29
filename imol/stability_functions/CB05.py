import numpy as np


class CB05:
    def __init__(self, params: dict[str, float] | None = None) -> None:
        # CB05 (Cheng Y. & Brutsaert W., 2005) stability function parameters (only stable conditions)
        self.amu = -19.3
        self.nmu = -4
        self.ahu = -12
        self.nhu = -2
        self.as_cb = 6.1
        self.bs_cb = 2.5
        self.cs_cb = 5.3
        self.ds_cb = 1.1
        if params is not None:
            for key, val in params.items():
                setattr(self, key, val)

    def phi(self, a: float, n: float, z: float) -> float:
        return (1 + a * z) ** (1 / n)

    def phim(self, z: float) -> float:
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

    def phih(self, z: float) -> float:
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

    def psim(self, z: float) -> float:
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

    def psih(self, z: float) -> float:
        if z >= 0:
            c = self.cs_cb
            d = self.ds_cb
            return -c * np.log(z + (1 + z**d) ** (1 / d))
        else:
            a = self.ahu
            return 2 * np.log(1 + (1 + a * z) ** 0.5) - np.log(4)
