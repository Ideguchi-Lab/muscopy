"""Validate release artifact metadata against pyproject.toml."""

# This is a standalone CI utility script, not an import package.
# ruff: noqa: INP001

from __future__ import annotations

import email.message
import email.parser
import os
import tarfile
import tomllib
import zipfile
from pathlib import Path


def _parse_metadata(text: str) -> email.message.Message:
    return email.parser.Parser().parsestr(text)


def _wheel_metadata(path: Path) -> email.message.Message:
    with zipfile.ZipFile(path) as wheel:
        required_files = {"muscopy/py.typed", "muscopy/__init__.py"}
        missing = required_files.difference(wheel.namelist())
        if missing or not any(name.endswith(".dist-info/licenses/LICENSE") for name in wheel.namelist()):
            msg = f"{path}: missing package files or bundled license: {sorted(missing)}"
            raise RuntimeError(msg)
        metadata_name = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
        return _parse_metadata(wheel.read(metadata_name).decode())


def _sdist_metadata(path: Path) -> email.message.Message:
    with tarfile.open(path) as sdist:
        relative_names = {name.partition("/")[2] for name in sdist.getnames()}
        missing = {"muscopy/py.typed", "muscopy/__init__.py", "LICENSE"}.difference(relative_names)
        if missing:
            msg = f"{path}: missing package files: {sorted(missing)}"
            raise RuntimeError(msg)
        pkg_info_name = next(name for name in sdist.getnames() if name.endswith("/PKG-INFO"))
        pkg_info = sdist.extractfile(pkg_info_name)
        if pkg_info is None:
            msg = f"Could not read {pkg_info_name} from {path}"
            raise RuntimeError(msg)
        return _parse_metadata(pkg_info.read().decode())


def _distribution_metadata(path: Path) -> email.message.Message:
    if path.suffix == ".whl":
        return _wheel_metadata(path)
    return _sdist_metadata(path)


def _distribution_paths(dist_dir: Path) -> list[Path]:
    return sorted(path for path in dist_dir.iterdir() if path.suffix == ".whl" or path.name.endswith(".tar.gz"))


def _normalize_specifier_set(specifier: str) -> str:
    return ",".join(sorted(part.strip() for part in specifier.split(",")))


def _check_pypi_metadata(path: Path, metadata: email.message.Message) -> None:
    if "LICENSE" not in metadata.get_all("License-File", []):
        msg = f"{path}: LICENSE is not declared in metadata"
        raise RuntimeError(msg)

    for requirement in metadata.get_all("Requires-Dist", []):
        if "@" in requirement:
            msg = f"{path}: direct URL dependency is not accepted by PyPI: {requirement}"
            raise RuntimeError(msg)


def _main() -> None:
    project_config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project = project_config["project"]
    expected_metadata = {
        "Name": project["name"],
        "Version": project["version"],
        "Requires-Python": project["requires-python"],
        "License-Expression": project["license"],
    }
    expected_extras = set(project.get("optional-dependencies", {}))

    tag_name = os.environ.get("GITHUB_REF_NAME")
    if tag_name is not None:
        expected_tag = f"v{project['version']}"
        if tag_name != expected_tag:
            msg = f"Release tag is {tag_name!r}, expected {expected_tag!r}"
            raise RuntimeError(msg)

    paths = _distribution_paths(Path("dist"))
    if not paths:
        msg = "No wheel or source distribution found in dist/"
        raise RuntimeError(msg)

    for path in paths:
        metadata = _distribution_metadata(path)
        for field, expected in expected_metadata.items():
            actual_value = metadata[field]
            expected_value = expected
            if field == "Requires-Python":
                actual_value = _normalize_specifier_set(actual_value)
                expected_value = _normalize_specifier_set(expected_value)
            if actual_value != expected_value:
                msg = f"{path}: {field} is {actual_value!r}, expected {expected_value!r}"
                raise RuntimeError(msg)

        actual_extras = set(metadata.get_all("Provides-Extra", []))
        if actual_extras != expected_extras:
            msg = f"{path}: Provides-Extra is {sorted(actual_extras)}, expected {sorted(expected_extras)}"
            raise RuntimeError(msg)

        _check_pypi_metadata(path, metadata)


if __name__ == "__main__":
    _main()
