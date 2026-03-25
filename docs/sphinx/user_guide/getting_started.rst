Getting Started
===============

This guide provides a quick introduction to using `bsrn` for quality control and clear-sky modeling.

Basic Usage
-----------

.. code-block:: python

   from bsrn.io.reader import read_station_to_archive
   from bsrn.modeling.clear_sky import add_clearsky_columns

   # Read a BSRN station-to-archive file
   df = read_station_to_archive("data/QIQ/qiq0124.dat.gz", logical_records="lr0100")

   # Add clear-sky columns (REST2 fetches MERRA-2 from Hugging Face automatically)
   df = add_clearsky_columns(df, station_code="QIQ", model="rest2")

Key Features
------------

- **Automated Quality Control**: 6 levels of QC based on BSRN standards.
- **Clear-sky Modeling**: Support for Ineichen, McClear, and REST2.
- **Clear-sky Detection**: Multiple algorithms (Reno, Ineichen, etc.).
- **Cloud Enhancement detection**: Killinger, Gueymard, Wang.
- **Irradiance Separation**: Erbs, BRL, Engerer2, Yang4.
