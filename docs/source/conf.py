"""Configuration file for the Sphinx documentation builder."""

import os
import sys
from pathlib import Path

# Resolve paths relative to this file so the build works regardless of the current directory
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "examples"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Muscopy"
copyright = "2024-2026, Ideguchi-Lab"  # noqa: A001
author = "Masato Fukushima, Kohki Horie, Takuro Ideguchi"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []
autosectionlabel_prefix_document = True
default_role = "any"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_class_signature = "separated"
autodoc_member_order = "bysource"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

html_context = {
    "mode": "production",
}

pygments_style = "sphinx"
pygments_dark_style = "monokai"

# Configure Sphinx-Gallery ignore pattern based on environment.
# GitHub Actions sets CI=true and Read the Docs sets READTHEDOCS=True.
ignore_pattern = r"__init__\.py"
if os.getenv("CI", "").lower() == "true" or os.getenv("READTHEDOCS", "") == "True":
    # Ignore MLB simulation examples in hosted builds where muscopy_mlbsim is not available
    ignore_pattern = r"(__init__|odt_with_mlb_simulation|idt_with_mlb_simulation)\.py"

sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "gallery",
    "filename_pattern": r".*\.py",
    "ignore_pattern": ignore_pattern,
    "plot_gallery": True,
    "run_stale_examples": True,
    "first_notebook_cell": ("%matplotlib inline\n"),
    "thumbnail_size": (800, 550),
}

suppress_warnings = ["config.cache"]
