# muscopy

Microscopy analysis utilities for quantitative phase imaging, digital holography,
optical diffraction tomography, intensity diffraction tomography, phasor analysis,
and related workflows.

## How to install

First, clone this repository onto your local machine.

```sh
git clone git@github.com:Ideguchi-Lab/muscopy.git
cd muscopy
```

Then, install the package in editable mode.

```sh
(YOUR_VIRTUAL_ENV) pip install -e .
```

If you are willing to contribute, install the development dependencies.

```sh
(YOUR_VIRTUAL_ENV) pip install -e .[dev]
```

## Linter guide

The linter and type checker configuration is in `pyproject.toml`.

```sh
ruff check ./muscopy    # code style check
ruff format ./muscopy   # code formatter
mypy ./muscopy    # static type analysis
pyright ./muscopy    # static type analysis
```

## Docs

You can build the documentation with the following command.

```sh
pip install -e .[doc]
cd ./docs
make html
```

Then, you can find the documentation in `build/html/index.html`.
