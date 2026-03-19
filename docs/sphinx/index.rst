Welcome to bsrn documentation!
============================

.. raw:: html

   <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 2rem;">
     <img src="_static/logo.jpg" alt="bsrn" style="height: 64px; width: auto;">
     <h1 style="margin: 0;">bsrn</h1>
   </div>

**bsrn** is a Python package for the `Baseline Surface Radiation Network (BSRN) <https://bsrn.awi.de/>`_.
It provides automated quality control (QC), solar geometry, clear-sky modeling, clear-sky detection (CSD),
cloud enhancement event (CEE) detection, irradiance separation, data retrieval, and visualization tools
for BSRN station-to-archive files.

The package implements a six-level QC pipeline based on the BSRN Operations Manual and Forstinger et al. (2021).
It supports multiple clear-sky models and detection methods, with seamless integration of MERRA-2 data.

.. grid:: 1 2 2 2
    :gutter: 4

    .. grid-item-card::  User Guide
        :link: user_guide/index
        :link-type: doc

        Learn how to install and get started with bsrn. Includes a high-level overview of the library's capabilities.

    .. grid-item-card::  Tutorials
        :link: tutorials/index
        :link-type: doc

        Step-by-step tutorials in Jupyter notebooks for core bsrn workflows.

    .. grid-item-card::  API Reference
        :link: api/index
        :link-type: doc

        Detailed documentation of all functions, classes, and modules in the bsrn package.

    .. grid-item-card::  Examples
        :link: examples
        :link-type: doc

        Explore additional examples demonstrating bsrn in action.

    .. grid-item-card::  GitHub
        :link: https://github.com/dazhiyang/bsrn

        View the source code, report issues, and contribute to the project on GitHub.

.. toctree::
   :maxdepth: 2
   :hidden:

   user_guide/index
   tutorials/index
   api/index
   examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
