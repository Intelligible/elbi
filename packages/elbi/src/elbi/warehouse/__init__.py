"""The data warehouse: sync external sources into a Delta Lake lakehouse and query it.

A registry of connectors, incremental extraction, and open table storage, built to run
**anywhere** from one config knob. Extraction is [dlt](https://dlthub.com); tables are
Delta Lake written via delta-rs (`deltalake`); the query engine is DuckDB (already the
platform's compute backend). Delta needs no external catalog (a table is a directory of
Parquet plus a transaction log), so a laptop runs the same code path with nothing extra
to install.

Portability comes from a single seam (`storage.py`): `STORAGE_URI` selects a local
directory on a laptop or the user's own S3/GCS/Azure bucket in the cloud, with no other
code change.
"""

from __future__ import annotations
