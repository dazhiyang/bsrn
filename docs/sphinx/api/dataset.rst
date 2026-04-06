BSRNDataset
===========

The central data object for one monthly BSRN station file.
``BSRNDataset`` wraps station identity, validated Pydantic logical
records (``LR0100``, ``LR0300``, ``LR4000``), and auto-resolved
metadata into a single typed model.

Pipeline methods delegate to the existing standalone functions so
both the object-oriented and functional APIs remain available.

Minute-level columns come from :meth:`~bsrn.dataset.BSRNDataset.data`:
call ``ds.data()`` for LR0100 means only, or pass ``include`` for
optional LR0300 / LR4000 fields.

.. autosummary::
   :toctree: generated/

   bsrn.dataset.BSRNDataset

Construction
------------

.. autosummary::
   :toctree: generated/

   bsrn.dataset.BSRNDataset.from_file

Data access
-----------

.. autosummary::
   :toctree: generated/

   bsrn.dataset.BSRNDataset.data

Pipeline methods
----------------

.. autosummary::
   :toctree: generated/

   bsrn.dataset.BSRNDataset.solpos
   bsrn.dataset.BSRNDataset.clear_sky
   bsrn.dataset.BSRNDataset.qc

Plotting
--------

You can quickly generate plots directly from the dataset.

.. autosummary::
   :toctree: generated/

   bsrn.dataset.BSRNDataset.plot
   bsrn.dataset.BSRNPlot
   bsrn.dataset.BSRNPlot.daily
   bsrn.dataset.BSRNPlot.table
