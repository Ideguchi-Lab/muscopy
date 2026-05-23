# muscopy

Microscopy analysis utilities for quantitative phase imaging, digital holography,
optical diffraction tomography, intensity diffraction tomography, phasor analysis,
and related workflows.

## How to install

Install from source with uv.

```sh
git clone git@github.com:Ideguchi-Lab/muscopy.git
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

Install the optional MLB simulation integration when needed.

```sh
uv sync --extra extra
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
