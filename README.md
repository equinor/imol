# imol

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A method for estimating the Monin-Obukhov length from bulk parameters has been developed following the documentation of AMOK (see documentation in the Fuga 
documentation folder).

The necessary observations are wind speed, $U$, at height $z_u$, air temperature, $T_a$, at height $z_a$, and sea surface temperature, $T_s$.
The observations of wind speed and air temperature should be made in the surface layer, and thus preferably lower than the typical hub height of a wind turbine, 10 m observations are ideal.

An [example notebook](https://github.com/equinor/imol/blob/main/examples/imol_example.ipynb) is available.

## Installation guide:
 Activate your virtual Python environment of choice and install [uv](https://docs.astral.sh/uv/getting-started/installation/).

Then install imol by running
```sh
git clone https://github.com/equinor/imol.git
cd imol
uv sync
```
To check the installation was successful, execute
```sh
uv run pytest tests
```
