Tutorials
=========

This section provides step-by-step tutorials for using `bsrn` for various tasks.
These tutorials are provided as interactive Jupyter notebooks.

**1.data_downloading** — BSRN FTP credentials, browse station file inventory with
:func:`~bsrn.io.retrieval.get_bsrn_file_inventory`, and download station-to-archive
``.dat.gz`` files with :func:`~bsrn.io.retrieval.download_bsrn_stn`.

**2.quality_control** — QIQ December LR0100 data; **Part A** runs each ``*_test`` tier
(PPL, ERL, closure, diffuse *k*, *k*-indices, tracker-off); **Part B** uses
:func:`~bsrn.qc.wrapper.run_qc` and compares tier counts; **§8** plots one UTC day with
``flagTracker`` failures (visual inspection emphasized).

**3.time_averaging** — explicit time aggregation with
:func:`~bsrn.utils.averaging.pretty_average` (**floor** / **ceiling** / **center**,
``match_ceiling_labels``, samples per interval).

**4.clear_sky_detection** — McClear-based clear-sky columns, QC, Reno clear-sky
detection via :func:`~bsrn.utils.clear_sky_detection.detect_clearsky`, and CSD-point
visualization (QIQ September example).

**5.cloud_enhancement_event** — REST2 clear-sky (MERRA-2 via Hugging Face), QC
(closure, diffuse ratio, tracker-off), and Wang-style cloud enhancement detection with
:func:`~bsrn.utils.cee_detection.detect_cee`.

.. toctree::
   :maxdepth: 1
   :caption: Core Workflows

   1.data_downloading
   2.quality_control
   3.time_averaging
   4.clear_sky_detection
   5.cloud_enhancement_event
