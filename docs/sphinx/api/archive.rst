Station-to-archive (LR)
=======================

Logical-record specifications, validation, and string formatting for BSRN
station-to-archive ``.dat`` files. For reading parsed LR data into pandas, see
:func:`~bsrn.io.reader.read_station_to_archive` in :doc:`io`.

The concrete ``LR*`` types are **Pydantic v2** models (:class:`pydantic.BaseModel` subclasses).
Field layout and Fortran-width metadata come from :data:`~bsrn.archive.LR_SPECS`; each field’s
``validate_func`` name resolves to a callable in :mod:`bsrn.archive.validation`.

- **Scalar and header fields** use :func:`~bsrn.archive.records_models.lr_spec`, which attaches
  ``json_schema_extra`` and an :class:`~pydantic.functional_validators.AfterValidator` built by
  :func:`~bsrn.archive.records_base.make_archive_after_validator`.
- **LR0100 / LR4000 minute columns** use :func:`~bsrn.archive.records_models.lr_spec_field` plus
  model :func:`~pydantic.field_validator` methods that pass ``yearMonth`` into
  :func:`~bsrn.archive.validation.LR0100_validateFunction` /
  :func:`~bsrn.archive.validation.LR4000_validateFunction`.

Core
----
.. autosummary::
   :toctree: generated/

   bsrn.archive.ArchiveRecordBase
   bsrn.archive.get_azimuth_elevation

Field helpers (``records_models`` / ``records_base``)
-----------------------------------------------------
.. autosummary::
   :toctree: generated/

   bsrn.archive.records_models.lr_spec
   bsrn.archive.records_models.lr_spec_field
   bsrn.archive.records_base.make_archive_after_validator

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

Each ``LR*`` model implements ``get_bsrn_format`` for archive text; ``get_azimuth_elevation`` is defined alongside those helpers in :mod:`bsrn.archive.archive_lr_formats`.

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
