Package Overview
================

**bsrn** is a Python package designed for the `Baseline Surface Radiation Network (BSRN) <https://bsrn.awi.de/>`_.
It provides a comprehensive suite of tools for researchers and practitioners working with high-quality surface radiation measurements.

Modules
-------

The package is organized into several key modules:

- :py:mod:`bsrn.io`: Data readers and writers for BSRN station-to-archive files.
- :py:mod:`bsrn.qc`: A multi-level quality control pipeline.
- :py:mod:`bsrn.modeling`: Clear-sky modeling and detection algorithms.
- :py:mod:`bsrn.physics`: Fundamental solar position and astronomical calculations.
- :py:mod:`bsrn.visualization`: Tools for visualizing solar data and QC results.
- :py:mod:`bsrn.utils`: Utility functions for data manipulation and analysis.

Data Pipeline
-------------

Typical workflows in `bsrn` involve:

1. **Reading**: Loading raw BSRN data into pandas DataFrames.
2. **Modeling**: Generating clear-sky expectations using atmospheric data.
3. **Quality Control**: Applying automated tests to identify questionable data points.
4. **Analysis**: Performing high-level research tasks like clear-sky detection or irradiance separation.
5. **Visualization**: Creating publication-quality plots.
