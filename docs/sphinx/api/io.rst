I/O Operations
==============

Readers
-------
.. autosummary::
   :toctree: generated/

   bsrn.io.reader.read_lr0100
   bsrn.io.reader.read_lr0300
   bsrn.io.reader.read_lr4000
   bsrn.io.reader.read_station_to_archive

Data Fetching
-------------
.. autosummary::
   :toctree: generated/

   bsrn.io.retrieval.get_bsrn_file_inventory
   bsrn.io.retrieval.download_bsrn_single
   bsrn.io.retrieval.download_bsrn_stn
   bsrn.io.retrieval.download_bsrn_mon
   bsrn.io.retrieval.download_bsrn_files
   bsrn.io.mcclear.fetch_mcclear
   bsrn.io.merra2.fetch_rest2

CAMS Radiation Service (CRS)
----------------------------
.. autosummary::
   :toctree: generated/

   bsrn.io.crs.check_crs_availability
   bsrn.io.crs.download_crs
   bsrn.io.crs.fetch_crs_hf
   bsrn.io.crs.add_crs_columns

NSRDB (National Solar Radiation Database)
-----------------------------------------
.. autosummary::
   :toctree: generated/

   bsrn.io.nsrdb.check_nsrdb_availability
   bsrn.io.nsrdb.download_nsrdb
   bsrn.io.nsrdb.fetch_nsrdb_hf
   bsrn.io.nsrdb.add_nsrdb_columns

BSRN Inventory Helpers
----------------------
.. autosummary::
   :toctree: generated/

   bsrn.io.retrieval.parse_bsrn_filename
   bsrn.io.retrieval.months_from_ftp_filenames
