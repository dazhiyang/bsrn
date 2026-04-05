Package Overview
================

**bsrn** is a Python package designed for the `Baseline Surface Radiation Network (BSRN) <https://bsrn.awi.de/>`_.
It provides a comprehensive suite of tools for researchers and practitioners working with high-quality surface radiation measurements.

Modules
-------

The package is organized into several key modules:

- :py:mod:`bsrn.archive`: Logical-record (LR) specs, validation, and formatting for
  station-to-archive ``.dat`` output; see :doc:`sphinx/api/archive`.
- :py:mod:`bsrn.io`: BSRN station-to-archive reader, FTP retrieval, and remote fetchers
  (SoDa McClear / CAMS CRS, MERRA-2 on Hugging Face); see :doc:`sphinx/api/io`.
- :py:mod:`bsrn.qc`: A multi-level quality control pipeline (see :doc:`tutorials/2.quality_control`).
- :py:mod:`bsrn.modeling`: Clear-sky modeling and detection algorithms.
- :py:mod:`bsrn.physics`: Fundamental solar position and astronomical calculations.
- :py:mod:`bsrn.visualization`: Tools for visualizing solar data and QC results.
- :py:mod:`bsrn.utils`: Utility functions for data manipulation and analysis (including
  :func:`~bsrn.utils.averaging.pretty_average` for LR0100-style averaging; see
  :doc:`tutorials/3.time_averaging`).

Data Pipeline
-------------

Typical workflows in `bsrn` involve:

1. **Reading**: Loading raw BSRN data into pandas DataFrames.
2. **Modeling**: Generating clear-sky expectations using atmospheric data.
3. **Quality Control**: Applying automated tests to identify questionable data points.
4. **Analysis**: Performing high-level research tasks like clear-sky detection or irradiance separation.
5. **Visualization**: Creating publication-quality plots.
