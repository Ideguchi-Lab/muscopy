# microscopy_converters
A repository containg microscopy converters and test sets.

## How to install

First, clone this repository into your local machine
```sh
git clone git@github.com:Ideguchi-Lab/microscopy_converters.git
```

Then, you can install this library as usual. I recommend to install it in editable mode because if you find a bug, you can fix it without unintall.
```sh
(YOUR_VIRTUAL_ENV) pip install -e microscopy_converters
```

If you are willing to contribute, you can install development toolkit with the following installation.
```sh
(YOUR_VIRTUAL_ENV) pip install -r microscopy_converters/requirements-dev.txt
```

## Linter guide

I described linter config in pyproject.toml file, so you can lint and check codes through typing
```sh
black ./muscopy   # lint code
isort ./muscopy   # properly reorder package imoort sentences 
pflake8 ./muscopy # check PEP regulations
mypy ./muscopy    # static type analysis
```
