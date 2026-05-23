"""Helper functions to parse directories and return lists of files.

This module provides:

- `numpy_parser`: Returns a list of numpy files in a directory.
- `png_parser`: Returns a list of png files in a directory.
- `tiff_parser`: Returns a list of tiff files in a directory.
- `recursive_numpy_parser`: Returns a list of numpy files in a directory and its subdirectories.
- `recursive_file_parser`: Returns a list of files in a directory and its subdirectories.
"""

import pathlib

__all__ = [
    "numpy_parser",
    "png_parser",
    "recursive_file_parser",
    "recursive_numpy_parser",
    "tiff_parser",
]


def numpy_parser(dir_path: str) -> list[str]:
    r"""Return list of numpy files in a directory.

    Parameters
    ----------
    dir_path : `str`
        Path to the directory to parse.

    Returns
    -------
    `list`\[`str`\]
        List of numpy files in the directory.
    """
    return [str(file) for file in pathlib.Path(dir_path).iterdir() if file.is_file() and file.suffix == ".npy"]


def png_parser(dir_path: str) -> list[str]:
    r"""Return list of png files in a directory.

    Parameters
    ----------
    dir_path : `str`
        Path to the directory to parse.

    Returns
    -------
    `list`\[`str`\]
        List of png files in the directory.
    """
    return [str(file) for file in pathlib.Path(dir_path).iterdir() if file.is_file() and file.suffix == ".png"]


def tiff_parser(dir_path: str) -> list[str]:
    r"""Return list of tiff files in a directory.

    Parameters
    ----------
    dir_path : `str`
        Path to the directory to parse.

    Returns
    -------
    `list`\[`str`\]
        List of tiff files in the directory.
    """
    return [str(file) for file in pathlib.Path(dir_path).iterdir() if file.is_file() and file.suffix == ".tiff"]


def recursive_numpy_parser(dir_path: str) -> list[str]:
    r"""Return list of numpy files in a directory.

    Parameters
    ----------
    dir_path : `str`
        Path to the directory to parse.

    Returns
    -------
    `list`\[`str`\]
        List of numpy files in the directory.
    """
    p = pathlib.Path(dir_path)
    return [str(path) for path in p.glob("**/*.npy")]


def recursive_file_parser(dir_path: str, file_suffix: str) -> list[str]:
    r"""Return list of files in a directory.

    Parameters
    ----------
    dir_path : `str`
        Path to the directory to parse.
    file_suffix : `str`
        Suffix of the files to parse.

    Returns
    -------
    `list`\[`str`\]
        List of files in the directory.
    """
    return [str(path) for path in pathlib.Path(dir_path).glob(f"**/*{file_suffix}") if path.is_file()]
