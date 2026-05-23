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

## Public API compatibility

For v1.0, muscopy's stable import contract is submodule-based. Import stable
functions and classes from their documented modules, for example:

```python
from muscopy.qpi import qpi
from muscopy.odt import ODTConfig, odt
from muscopy.idt import IDTConfig, compute_idt
```

The stable v1.0 modules are `muscopy.cfg`, `muscopy.dh`, `muscopy.qpi`,
`muscopy.qpi_utils`, `muscopy.odt`, `muscopy.idt`, `muscopy.phasor`,
`muscopy.phase_noise`, and `muscopy.dir_parser`. Public names listed in each
module's `__all__` are covered by compatibility guarantees.

`muscopy.image_checker` is experimental and is not covered by the v1.0
compatibility guarantee. Private names beginning with `_`, examples, generated
documentation files, and implementation constants are also outside the stable
API.

After v1.0, removing or changing a stable public API requires a deprecation
notice, migration guidance, and a changelog entry before removal in a future
major release. Experimental APIs may change with changelog notice.

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
