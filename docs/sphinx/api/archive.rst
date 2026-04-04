Station-to-archive (LR)
=======================

Logical-record specifications, validation, and string formatting for BSRN
station-to-archive ``.dat`` files. For reading parsed LR data into pandas, see
:func:`~bsrn.io.reader.read_station_to_archive` in :doc:`io`.

Core
----
.. autosummary::
   :toctree: generated/

   bsrn.archive.ArchiveRecordBase
   bsrn.archive.get_azimuth_elevation

Logical record classes
----------------------
.. autosummary::
   :toctree: generated/

   bsrn.archive.LR0001
   bsrn.archive.LR0002
   bsrn.archive.LR0003
   bsrn.archive.LR0004
   bsrn.archive.LR0005
   bsrn.archive.LR0006
   bsrn.archive.LR0007
   bsrn.archive.LR0008
   bsrn.archive.LR0100
   bsrn.archive.LR4000
   bsrn.archive.LR4000CONST

Each ``LR*`` model implements ``get_bsrn_format`` for archive text; ``get_azimuth_elevation`` is defined alongside those helpers in ``bsrn.archive.archive_lr_formats``.

Metadata and lookup tables
--------------------------
.. autosummary::
   :toctree: generated/

   bsrn.archive.LR_SPECS
   bsrn.archive.STATION_METADATA
   bsrn.archive.TOPOGRAPHIES
   bsrn.archive.SURFACES
   bsrn.archive.QUANTITIES
   bsrn.archive.PYRGEOMETER_BODY
   bsrn.archive.PYRGEOMETER_DOME

Submodules
----------
.. autosummary::
   :toctree: generated/

   bsrn.archive.archive_lr_formats
   bsrn.archive.records_base
   bsrn.archive.records_models
   bsrn.archive.specs
   bsrn.archive.validation
