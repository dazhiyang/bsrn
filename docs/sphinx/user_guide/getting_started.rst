Getting Started
===============

This guide provides a quick introduction to using `bsrn` for quality control and clear-sky modeling.

BSRNDataset (recommended)
-------------------------

The simplest way to work with a BSRN archive file is through
:class:`~bsrn.dataset.BSRNDataset`, which reads the file, resolves
station metadata, and exposes chainable pipeline methods:

.. code-block:: python

   import bsrn

   ds = bsrn.BSRNDataset.from_file("data/QIQ/qiq0125.dat.gz")

   df = ds.solpos()            # solar position + extraterrestrial
   df = ds.clear_sky()         # clear-sky irradiance (Ineichen)
   df = ds.qc()                # 6-level quality control flags

Functional API
--------------

All pipeline steps are also available as standalone functions,
useful when working with non-BSRN data or custom DataFrames:

.. code-block:: python

   from bsrn.physics.geometry import add_solpos_columns
   from bsrn.modeling.clear_sky import add_clearsky_columns
   from bsrn.qc.wrapper import run_qc

   df = add_solpos_columns(df, lat=47.80, lon=124.49, elev=170)
   df = add_clearsky_columns(df, lat=47.80, lon=124.49, elev=170)
   df = run_qc(df, lat=47.80, lon=124.49, elev=170)

Key Features
------------

- **Automated Quality Control**: 6 levels of QC based on BSRN standards.
- **Clear-sky Modeling**: Support for Ineichen, McClear, and REST2.
- **Clear-sky Detection**: Multiple algorithms (Reno, Ineichen, etc.).
- **Cloud Enhancement detection**: Killinger, Yang, Gueymard.
- **Irradiance Separation**: Erbs, BRL, Engerer2, Yang4.
