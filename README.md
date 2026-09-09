# muscopy

`muscopy` is a Python library for label-free microscopy analysis, such as quantitative phase imaging (QPI), digital holography,
optical diffraction tomography, intensity diffraction tomography, phasor analysis,
and related workflows.
The package has been developed and maintained by the [Ideguchi-Lab](https://www.ideguchi.ipst.s.u-tokyo.ac.jp/) at the University of Tokyo, Japan.

## How to install

Install from PyPI (Python 3.11–3.14).

```sh
pip install muscopy
```

For a compatible NVIDIA GPU and CUDA 12 environment:

```sh
pip install "muscopy[cuda]"
```

For source development, clone the repository and use uv.

```sh
git clone https://github.com/Ideguchi-Lab/muscopy.git
cd muscopy
uv sync
```

If you are willing to contribute, install the development dependencies.

```sh
uv sync --extra dev
```

Install documentation dependencies when building docs locally.

```sh
uv sync --extra doc
```

## Optional MLB simulation examples

`examples/odt_with_mlb_simulation.py` and
`examples/idt_with_mlb_simulation.py` require `muscopy-mlbsim`, a **private**
package available only to users with access to the Ideguchi-Lab repository.
It is not included in muscopy or its PyPI extras. These two examples skip
execution when the simulator is unavailable; muscopy's reconstruction APIs
and the other examples do not require it.

See the [separate MLB installation guide](https://github.com/Ideguchi-Lab/muscopy/blob/main/docs/source/mlb_installation.rst)
for access requirements, installation commands, and how to run these examples.
This replaces the former `uv sync --extra extra` installation method.

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
uv run ruff check ./muscopy ./tests ./examples
uv run ruff format ./muscopy ./tests ./examples
uv run mypy
uv run pyright
```

## Docs

You can build the documentation with the following command.

```sh
uv sync --extra doc
uv run sphinx-build -W docs/source docs/build
```

Then, you can find the documentation in `docs/build/index.html`.

## Authors and license

Authors: Masato Fukushima, Kohki Horie, and Takuro Ideguchi.

Copyright (c) 2024–2026 Ideguchi-Lab.
Starting with version 1.0.2, muscopy is distributed under the [MIT License](LICENSE).

## Acknowledgments

We thank Zinan Zhou and Atsuko Price for their contributions to muscopy.
