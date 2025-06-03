# microscopy_converters

A repository containg microscopy converters and test sets.

## How to install

First, clone this repository into your local machine

```sh
git clone git@github.com:Ideguchi-Lab/muscopy.git
```

Then, you can install this library as usual. I recommend to install it in editable mode because if you find a bug, you can fix it without unintall.

```sh
(YOUR_VIRTUAL_ENV) pip install -e ./muscopy
```

If you are willing to contribute, you can install development toolkit with the following installation.

```sh
(YOUR_VIRTUAL_ENV) pip install -e ./muscopy[dev]
```

## Linter guide

I described linter config in pyproject.toml file, so you can lint and check codes through typing

```sh
ruff check ./muscopy    # code style check
ruff format ./muscopy   # code formatter
mypy ./muscopy    # static type analysis
pyright ./muscopy    # static type analysis
```

## Docs

You can build the documentation with the following command.

```sh
pip install -e ./muscopy[docs]
cd ./docs
make html
```

Then, you can find the documentation in `build/html/index.html`.
