# External data

Official Watermark V1 source: https://www.ffi.no/en/research/watermark

For installation, place the official `WatermarkV1.zip` in this directory and
run `python scripts/install_watermark.py` from the repository root.
Keep the complete official ZIP after extraction. It also contains channel
sets beyond NCS1 and is a long-term source resource for future experiments.
Do not remove Watermark resources as disposable installation caches.

Keep the ZIP, the 60 extracted NCS1 records, the official replay engine, and
`source_manifest.json`. The current installer extracts only NCS1 and the
engine; the other channels remain available in the complete ZIP. The source
manifest retains the archive SHA256 and each installed file's provenance.
The archive and extracted data are excluded from Git.
