"""Configuration file for the Sphinx documentation builder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path("..", "muscopy").resolve()))
sys.path.insert(0, str(Path("..", "muscopy/examples").resolve()))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Muscopy"
copyright = "2025, Masato Fukushima"  # noqa: A001
author = "Masato Fukushima"

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
exclude_patterns = []
autosectionlabel_prefix_document = True
default_role = "any"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_class_signature = "separated"
autodoc_member_order = "bysource"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = "_static"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

html_context = {
    "mode": "production",
}

pygments_style = "sphinx"
pygments_dark_style = "monokai"

sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "gallery",
    "filename_pattern": r".*\.py",
    "ignore_pattern": r"__init__\.py",
    "plot_gallery": True,
    "run_stale_examples": True,
    "first_notebook_cell": ("%matplotlib inline\n"),
    "thumbnail_size": (800, 550),
}

suppress_warnings = ["config.cache"]
