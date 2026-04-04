# Sphinx configuration — **source directory** is ``docs/`` (``index.rst`` → root ``index.html`` for RTD).
# ``docs/tutorials/`` is a normal doc path (``tutorials/...``) with nothing under ``sphinx/``.
#
# Local build (from repo root):
#   sphinx-build -b html docs docs/_build/html
#
# Read the Docs: ``.readthedocs.yaml`` points ``sphinx.configuration`` here.

from __future__ import annotations

import os
import sys
import tomllib

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

with open(os.path.join(_ROOT, "pyproject.toml"), "rb") as _pf:
    _release = tomllib.load(_pf)["project"]["version"]

project = "bsrn"
copyright = "Dazhi Yang"
author = "Dazhi Yang"
release = _release
version = _release

# Master document at docs root → index.html for Read the Docs root URL.
root_doc = "index"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx_design",
    "nbsphinx",
]

templates_path = [os.path.join("sphinx", "_templates")]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "sphinx/_build",
    "**/.ipynb_checkpoints",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}

extlinks = {
    "issue": ("https://github.com/dazhiyang/bsrn/issues/%s", "GH%s"),
    "pull": ("https://github.com/dazhiyang/bsrn/pull/%s", "GH%s"),
    "ghuser": ("https://github.com/%s", "@%s"),
}

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
pygments_style = "tango"
pygments_dark_style = "monokai"

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/dazhiyang/bsrn",
    "header_links_before_dropdown": 6,
    "navbar_align": "left",
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}
html_static_path = [os.path.join("sphinx", "_static")]
html_logo = os.path.join("sphinx", "_static", "logo.jpg")
html_favicon = os.path.join("sphinx", "_static", "logo.jpg")
