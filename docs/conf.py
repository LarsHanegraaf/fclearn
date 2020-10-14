"""Sphinx configuration."""

import sphinx_rtd_theme  # noqa: F401

project = "fclearn"
author = "Lars Hanegraaf"
copyright = f"2020, {author}"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_rtd_theme",
    "sphinx_autodoc_typehints",
]
autoclass_content = "both"
html_theme = "sphinx_rtd_theme"
