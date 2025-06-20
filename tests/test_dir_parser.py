"""Unit tests for the directory-parsing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from muscopy.dir_parser import (
    numpy_parser,
    png_parser,
    recursive_file_parser,
    recursive_numpy_parser,
    tiff_parser,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# ----------------------------------------------------------------------------- #
# Helper utilities for the test suite
# ----------------------------------------------------------------------------- #


def _touch(path: Path) -> None:
    """Create an empty file (and any missing parent dirs) for the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


# ----------------------------------------------------------------------------- #
# Tests for the non-recursive parsers
# ----------------------------------------------------------------------------- #


def test_numpy_parser_returns_only_npy_files(tmp_path: Path) -> None:
    """numpy_parser must return every *.npy file in the *top-level* directory."""
    # Arrange - create files
    good_files = {tmp_path / "a.npy", tmp_path / "b.npy"}
    other_files = {
        tmp_path / "a.png",
        tmp_path / "b.tiff",
        tmp_path / "nested" / "c.npy",  # should *not* be picked up (subdir)
    }
    for f in good_files | other_files:
        _touch(f)

    # Act
    result = set(map(Path, numpy_parser(str(tmp_path))))

    # Assert
    assert result == good_files
    # Guarantee no unintended paths sneaked in
    assert all(p.suffix == ".npy" for p in result)


@pytest.mark.parametrize(
    ("parser", "suffix"),
    [
        (png_parser, ".png"),
        (tiff_parser, ".tiff"),
    ],
)
def test_image_parsers_only_return_matching_suffixes(
    tmp_path: Path,
    parser: Callable[[str], list[str]],
    suffix: str,
) -> None:
    good = tmp_path / f"file{suffix}"
    bad = tmp_path / f"file_not{suffix}.txt"
    _touch(good)
    _touch(bad)

    # Dynamically pick the parser under test
    result = parser(str(tmp_path))

    assert result == [str(good)]


def test_parsers_return_empty_list_for_no_matches(tmp_path: Path) -> None:
    """Every parser should yield an empty list when no file matches."""
    assert numpy_parser(str(tmp_path)) == []
    assert png_parser(str(tmp_path)) == []
    assert tiff_parser(str(tmp_path)) == []


# ----------------------------------------------------------------------------- #
# Tests for the recursive parsers
# ----------------------------------------------------------------------------- #


def test_recursive_numpy_parser_finds_nested_files(tmp_path: Path) -> None:
    """recursive_numpy_parser must discover *.npy files in *any* sub-folder."""
    top_level = tmp_path / "root.npy"
    nested = tmp_path / "lvl1" / "lvl2" / "deep.npy"
    unrelated = tmp_path / "lvl1" / "file.png"

    for f in (top_level, nested, unrelated):
        _touch(f)

    expected = {str(top_level), str(nested)}
    result = set(recursive_numpy_parser(str(tmp_path)))

    assert result == expected


def test_recursive_file_parser_generic_suffix(tmp_path: Path) -> None:
    """recursive_file_parser should work for arbitrary suffixes."""
    suffix = ".log"
    files_of_interest = {
        tmp_path / "one.log",
        tmp_path / "dir" / "two.log",
    }
    other_file = tmp_path / "dir" / "skip.txt"

    for f in files_of_interest | {other_file}:
        _touch(f)

    result = set(recursive_file_parser(str(tmp_path), suffix))
    assert result == {str(p) for p in files_of_interest}


def test_recursive_file_parser_ignores_directories(tmp_path: Path) -> None:
    suffix = ".data"
    folder_path = tmp_path / f"folder{suffix}"
    folder_path.mkdir()  # directory, not file
    _touch(tmp_path / f"valid{suffix}")

    out = recursive_file_parser(str(tmp_path), suffix)
    assert len(out) == 1
    assert out[0].endswith(suffix)


# ----------------------------------------------------------------------------- #
# Basic negative-path tests
# ----------------------------------------------------------------------------- #


def test_parser_raises_on_non_existent_dir(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        numpy_parser(str(missing))
