import numpy.testing
import pandas as pd
import xarray as xr

from pymol import pyMOL


class Test_pymol:
    fnm = "tests/input_data/Test_input.txt"
    df = pd.read_csv(
        fnm,
        sep=r"\s+",
        skiprows=2,
        header=None,
        names=["ws", "wd", "Ta", "Ts", "time"],
        index_col="time",
        parse_dates=True,
    )
    ds = df.to_xarray()
    ds.attrs = dict(zt=13, zu=15)

    def test_default_stabf_root(self, var="invL"):
        pm = pyMOL(stab_func="AMOK")
        ds_out = pm(self.ds)

        fnm = "tests/input_data/Test_output1.txt"
        df = pd.read_csv(fnm, sep=r"\s+", header=1)
        time = [pd.Timestamp(str(t)).to_datetime64() for t in df["Time"]]
        df["time"] = time
        df.drop("Time", axis=1, inplace=True)
        df.set_index("time", inplace=True)
        ds_test = df.to_xarray()
        ds_test = ds_test.rename({"1/L": "invL"})

        numpy.testing.assert_allclose(
            ds_out[var]
            .where((ds_out["iCalm"] == 0) & (ds_test["iCalm"] == 0), drop=True)
            .values,
            ds_test[var]
            .where((ds_out["iCalm"] == 0) & (ds_test["iCalm"] == 0), drop=True)
            .values,
            rtol=0.02,
        )

    def test_default_stabf_iter(self, var="invL"):
        pm = pyMOL(stab_func="AMOK")
        ds_out = pm(self.ds, method="iter")

        fnm = "tests/input_data/Test_output2.txt"
        df = pd.read_csv(fnm, sep=r"\s+", header=1)
        time = [pd.Timestamp(str(t)).to_datetime64() for t in df["Time"]]
        df["time"] = time
        df.drop("Time", axis=1, inplace=True)
        df.set_index("time", inplace=True)
        ds_test = df.to_xarray()
        ds_test = ds_test.rename({"1/L": "invL"})

        numpy.testing.assert_allclose(
            ds_out[var]
            .where((ds_out["iCalm"] == 0) & (ds_test["iCalm"] == 0), drop=True)
            .values,
            ds_test[var]
            .where((ds_out["iCalm"] == 0) & (ds_test["iCalm"] == 0), drop=True)
            .values,
            rtol=0.02,
        )

    def test_HB88_stabf(self, var="invL"):
        pm = pyMOL(stab_func="HB88")
        ds_out = pm(self.ds)

        fnm = "tests/input_data/Test_HB88_output.nc"
        ds_test = xr.load_dataset(fnm)

        numpy.testing.assert_allclose(
            ds_out[var]
            .where((ds_out["iCalm"] == 0) & (ds_test["iCalm"] == 0), drop=True)
            .values,
            ds_test[var]
            .where((ds_out["iCalm"] == 0) & (ds_test["iCalm"] == 0), drop=True)
            .values,
            rtol=0.04,
        )

    def test_HC05_stabf(self, var="invL"):
        pm = pyMOL(stab_func="CB05")
        ds_out = pm(self.ds)

        fnm = "tests/input_data/Test_CB05_output.nc"
        ds_test = xr.load_dataset(fnm)

        numpy.testing.assert_allclose(
            ds_out[var]
            .where((ds_out["iCalm"] == 0) & (ds_test["iCalm"] == 0), drop=True)
            .values,
            ds_test[var]
            .where((ds_out["iCalm"] == 0) & (ds_test["iCalm"] == 0), drop=True)
            .values,
            rtol=0.02,
        )
