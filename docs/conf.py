# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'pRNN'
copyright = '2025, Dan Levenstein'
author = 'Dan Levenstein'
release = '0.95'

extensions = [
    'sphinx.ext.autodoc',          # core autodoc
    'sphinx.ext.napoleon',         # for Google/NumPy-style docstrings
    'sphinx_autodoc_typehints',    # for type hints support
    'sphinx_autodoc_typehints'
]

autodoc_mock_imports = ["minigrid", "hydra", "miniworld", "seaborn"]
autosummary_generate = True
autodoc_typehints = "description"
autoclass_content = "both"  # include both class docstring and __init__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
