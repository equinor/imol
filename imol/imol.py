import warnings

import numpy as np
import xarray as xr
from scipy.optimize import root

from imol.stability_functions import AMOK, CB05, HB88


class iMOL:
    # gravitational constant - m**2/s
    g = 9.81
    # von Karman constant
    k = 0.4
    # Charnok constsnt
    Ac = 0.012
    # Laminar flow correction constant in z0 expression
    Bc = 0.12
    # Dry adiabatic lapse rate - K/m
    lr = 9.78e-3
    # Density of air at T=15 degC
    rho = 1.225
    # Reference height - m (can be anything, makes Rb non-dimensional)
    zref = 10
    # Limit on z0
    z0_min = 1e-5
    # Limit on us; us>U/usN
    usN = 100

    def __init__(
        self,
        z0: float = 1e-4,
        calmth: float = 3.0,
        tol: float = np.finfo(float).eps,
        maxiter: int = 50,
        stab_func: str | None = None,
        stab_func_params: dict[str, float] | None = None,
    ) -> None:
        self.z0 = z0
        self.calmth = calmth
        self.tol = tol
        self.maxiter = maxiter
        self.stab_func = self.get_stability_function(stab_func, stab_func_params)
        self.phim = self.stab_func.phim
        self.phih = self.stab_func.phih
        self.psim = self.stab_func.psim
        self.psih = self.stab_func.psih

    def __call__(
        self,
        ds: xr.Dataset,
        zt: float = 2.0,
        zu: float = 10.0,
        method: str = "root",
        tol: float = np.finfo(float).eps,
        maxiter: int = 50,
    ) -> xr.Dataset:
        # assert(ds == xr.core.dataset.Dataset, 'Input has to be a xarray dataset with coordinate time')
        if "zt" in list(ds.attrs.keys()):
            zt = ds.attrs["zt"]
        else:
            warnings.warn(f"zt not in dataset attributes, assumed to be {zt} m.")

        if "zu" in list(ds.attrs.keys()):
            zu = ds.attrs["zu"]
        else:
            warnings.warn(f"zu not in dataset attributes, assumed to be {zu} m.")

        if "loc" in list(ds.attrs.keys()):
            loc = ds.attrs["loc"]
        else:
            loc = ""

        return self.calcInvL(
            ds["Ts"].values,
            zt,
            ds["Ta"].values,
            zu,
            ds["ws"].values,
            ds["wd"].values,
            time=ds["time"].values,
            loc=loc,
            method=method,
            tol=tol,
            maxiter=maxiter,
        )

    def get_stability_function(
        self,
        stab_func: str | None,
        params: dict[str, float] | None,
    ) -> AMOK | HB88 | CB05:
        stab_funcs = {"AMOK": AMOK, "HB88": HB88, "CB05": CB05}
        if stab_func in stab_funcs:
            return stab_funcs[stab_func](params=params)
        elif stab_func is None:
            return AMOK(params=params)
        else:
            raise ValueError(
                f"{stab_func} is not a valid stability function. Choose {', '.join(stab_funcs.keys())}, or leave it as None to use the default AMOK."
            )

    def f(
        self,
        x: np.ndarray,
        invL: float,
        zu: float,
        U: float,
        zt: float,
        Ts: float,
        pt: float,
        k: float,
        v: float,
    ) -> tuple[float, float]:
        us, ths = x
        z1 = zu * invL
        z2 = zt * invL
        z0 = self.get_z0(us, v)
        if z0 <= 0:
            return (np.nan, np.nan)
        return (
            U - (us / k) * (np.log(zu / z0) - self.psim(z1)),
            pt - Ts - (ths / k) * (np.log(zt / z0) - self.psih(z2)),
        )

    def df(
        self,
        x: np.ndarray,
        invL: float,
        zu: float,
        U: float,
        zt: float,
        Ts: float,
        pt: float,
        k: float,
        _: float,
    ) -> list[list[float]]:
        us, ths = x
        z1 = zu * invL
        z2 = zt * invL
        return [
            [U / zu - (us / k) * ((1 / zu) - invL * ((1 - self.phim(z1)) / z1)), 0],
            [
                0,
                (pt - Ts) / zt
                - (ths / k) * ((1 / zt) - invL * ((1 - self.phih(z2)) / z2)),
            ],
        ]

    def invLiter(
        self,
        Ts: float,
        zt: float,
        T: float,
        zu: float,
        U: float,
        tol: float = 1e-7,
        maxiter: int = 50,
    ) -> tuple[float, float, float, float, float, int, int, float]:
        # Potential temperature
        pt = T + self.lr * zt
        icalm = 0
        # Calm-case sentinel defaults; overwritten below when icalm == 0
        invL = -9.999
        us = 0.0
        ths = -9.999
        z0 = 0.0
        n = 1
        err = -9.9999e-99
        usold = 0.0
        if U < self.calmth:
            icalm = 1
        if icalm == 0:
            v = self.get_v(T)
            # Bulk Richardson number
            Rb = self.get_Rb(Ts, U, pt)
            z0 = self.z0
            # Initial guess on invL (assuming neutral conditions)
            invL = self.get_invL(zt, zu, z0, Rb)
            err = np.inf
            n = 0
            while (err > tol) & (n <= maxiter):
                if n:
                    invL = self.get_invL(
                        zt, zu, z0, Rb, self.psim(zu * invL), self.psih(zt * invL)
                    )
                    # err = np.abs(1/invL-1/invLold)
                    usold = us
                us = self.get_us(zu, U, invL, z0)  # +self.psim(z0*invL0)))
                if n:
                    err = np.abs(us - usold)
                ths = self.get_ths(Ts, zt, pt, invL, z0)  # +self.psih(z0*invL0)))
                z0 = self.get_z0(us, v)
                if z0 < self.z0_min or us < U / self.usN:
                    icalm = 3
                    (invL, us, ths, z0, n, err) = (-9.999, 0, -9.999, 0, 1, -9.9999e-99)
                    break
                n += 1
        if (n > maxiter) & (err > tol):
            icalm = 2
            (invL, us, ths, z0) = (-9.999, 0, -9.999, 0)
        if icalm == 0:
            invL = (self.k * self.g * ths) / (Ts * us**2)
        return pt, invL, us, ths, z0, icalm, n - 1, err

    def invLroot(
        self,
        Ts: float,
        zt: float,
        T: float,
        zu: float,
        U: float,
        tol: float = np.finfo(float).eps,
        maxiter: int = 50,
    ) -> tuple[float, float, float, float, float, int, int, float]:
        # Potential temperature
        pt = T + self.lr * zt
        icalm = 0
        (invL, us, ths, z0, fev, fse) = (
            -9.999,
            0,
            -9.999,
            0,
            0,
            -9.9999e-99,
        )

        if U < self.calmth:
            icalm = 1
        elif np.isnan(U):
            icalm = 4

        if icalm == 0:
            v = self.get_v(T)
            # Bulk Richardson number
            Rb = self.get_Rb(Ts, U, pt)
            z0 = self.z0
            # Initial guess on invL (assuming neutral conditions)
            invL = self.get_invL(zt, zu, z0, Rb)
            us = self.get_us(zu, U, invL, z0, self.psim(z0 * invL))
            ths = self.get_ths(Ts, zt, pt, invL, z0, self.psih(z0 * invL))
            z0 = self.get_z0(us, v)
            # Update invL with stabilty functions included
            invL = self.get_invL(
                zt, zu, z0, Rb, self.psim(zu * invL), self.psih(zt * invL)
            )
            us = self.get_us(zu, U, invL, z0, self.psim(z0 * invL))
            ths = self.get_ths(Ts, zt, pt, invL, z0, self.psih(z0 * invL))
            z0 = self.get_z0(us, v)
            if z0 < self.z0_min or us < U / self.usN:
                icalm = 3
                (invL, us, ths, z0, fev, fse) = (-9.999, 0, -9.999, 0, 0, -9.9999e-99)
            else:
                idict = root(
                    self.f,
                    [us, ths],
                    args=(invL, zu, U, zt, Ts, pt, self.k, v),
                    jac=self.df,
                    options={"col_deriv": True, "xtol": tol, "maxfev": maxiter},
                )
                if idict.success:
                    us, ths = idict.x
                    z0 = self.get_z0(us, v)
                    invL = (self.k * self.g * ths) / (Ts * us**2)
                    fev = idict["nfev"]
                    fse = np.abs(idict["fun"][0])  # np.max(np.abs(idict['fun']))
                else:
                    # icalm=2
                    # (invL, us, ths, z0) = (-9.999, 0, -9.999, 0)
                    # Try iteration method if root iteration failed
                    pt, invL, us, ths, z0, icalm, fev, fse = self.invLiter(
                        Ts, zt, T, zu, U, tol=tol, maxiter=maxiter
                    )

        return pt, invL, us, ths, z0, icalm, fev, fse

    def get_Rb(self, Ts: float, U: float, pt: float) -> float:
        return self.g * self.zref * (pt - Ts) / (Ts * U**2)

    def get_invL(
        self,
        zt: float,
        zu: float,
        z0: float,
        Rb: float,
        x: float = 0,
        y: float = 0,
    ) -> float:
        return (Rb / self.zref) * ((np.log(zu / z0) - x) ** 2) / (np.log(zt / z0) - y)

    def get_ths(
        self,
        Ts: float,
        zt: float,
        pt: float,
        invL: float,
        z0: float,
        x: float = 0,
    ) -> float:
        return (pt - Ts) * (self.k / (np.log(zt / z0) - self.psih(zt * invL) + x))

    def get_us(
        self,
        zu: float,
        U: float,
        invL: float,
        z0: float,
        x: float = 0,
    ) -> float:
        return U * (self.k / (np.log(zu / z0) - self.psim(zu * invL) + x))

    def get_v(self, t: float) -> float:
        # Kinematic viscosity of air - m**2/s
        return (2.791e-7 * t**0.7355) / self.rho  # 1.48e-5

    def get_z0(self, us: float, v: float) -> float:
        return self.Ac * us**2 / self.g + self.Bc * v / us

    def calcInvL(
        self,
        Ts: np.ndarray,
        zt: float,
        T: np.ndarray,
        zu: float,
        U: np.ndarray,
        D: np.ndarray | None = None,
        time: np.ndarray | None = None,
        loc: str | None = None,
        method: str = "root",
        tol: float | None = None,
        maxiter: int | None = None,
    ) -> xr.Dataset:
        if time is None:
            Ts = np.array([Ts])
            T = np.array([T])
            U = np.array([U])
            time = np.arange(len(U))
            calc = True
        else:
            calc = False
        if tol is None:
            tol = self.tol
        if maxiter is None:
            maxiter = self.maxiter

        Ts = Ts + 273.15
        T = T + 273.15
        pt = np.ndarray(time.shape)
        invL = np.ndarray(time.shape)
        us = np.ndarray(time.shape)
        ths = np.ndarray(time.shape)
        z0 = np.ndarray(time.shape)
        icalm = np.ndarray(time.shape)
        fev = np.ndarray(time.shape)
        fse = np.ndarray(time.shape)
        for n, t in enumerate(time):
            if method == "root":
                (pt[n], invL[n], us[n], ths[n], z0[n], icalm[n], fev[n], fse[n]) = (
                    self.invLroot(Ts[n], zt, T[n], zu, U[n], tol=tol, maxiter=maxiter)
                )
            elif method == "iter":
                (pt[n], invL[n], us[n], ths[n], z0[n], icalm[n], fev[n], fse[n]) = (
                    self.invLiter(Ts[n], zt, T[n], zu, U[n], tol=tol, maxiter=maxiter)
                )
            else:
                raise ValueError(f"No method {method}")
        if calc:
            print(
                "Ws    : {:4.1f} m/s\nTair  : {:4.1f} degC".format(U[0], T[0] - 273.15)
            )
            print(
                "Tsea  : {:4.1f} degC\ndTh   : {:5.3f} degC".format(
                    Ts[0] - 273.15, pt[0] - Ts[0]
                )
            )
            print(
                "1/L   : {:5.3e} 1/m\nL     : {:5.1f} m\nu*    : {:7.4f} m/s".format(
                    invL[0], 1 / invL[0], us[0]
                )
            )
            print(
                "Theta*: {:7.4f} degC\nz0    : {:5.3e} m\nzeta0 : {:5.3e}".format(
                    ths[0], z0[0], z0[0] * invL[0]
                )
            )
            print(
                "iCalm : {:1d}\nfev  : {:3d}\nerr   : {:5.3e}".format(
                    int(icalm[0]), int(fev[0]), fse[0]
                )
            )
        return xr.Dataset(
            data_vars=dict(
                ws=(
                    "time",
                    U,
                    {"description": "Wind velocity at height zu.", "unit": "m/s"},
                ),
                wd=(
                    "time",
                    D,
                    {"description": "Wind direction at height zu.", "unit": "deg."},
                ),
                Ta=(
                    "time",
                    T,
                    {"description": "Air temperature at height zt.", "unit": "K"},
                ),
                Ts=(
                    "time",
                    Ts,
                    {"description": "Sea surface temperature.", "unit": "K"},
                ),
                pT=(
                    "time",
                    pt,
                    {"description": "Potential temperature.", "unit": "K"},
                ),
                invL=(
                    "time",
                    invL,
                    {
                        "description": "Inverse Monin-Obukhov length, 1/L.",
                        "unit": "1/m",
                    },
                ),
                us=(
                    "time",
                    us,
                    {"description": "Friction velocity, u*.", "unit": "-"},
                ),
                ths=(
                    "time",
                    ths,
                    {
                        "description": "Characteristic temperature scale, theta*.",
                        "unit": "K",
                    },
                ),
                z0=(
                    "time",
                    z0,
                    {"description": "Surface roughness length.", "unit": "m"},
                ),
                iCalm=(
                    "time",
                    icalm,
                    {
                        "description": "Calmness flag = {0, 1, 2, 3, 4}. Valid solution when iCalm = 0."
                    },
                ),
                fev=(
                    "time",
                    fev,
                    {"description": "Function evaluations or number of iterations."},
                ),
                fse=(
                    "time",
                    fse,
                    {"description": "Friction velocity function error."},
                ),
            ),
            coords=dict(
                time=time,
            ),
            attrs=dict(
                desctiption=f"Inverse Monin-Obukhov length calculated with temperature at z={zt} m and wind speed at z={zu} m.",
                zt=zt,
                zu=zu,
                loc=loc,
            ),
        )
