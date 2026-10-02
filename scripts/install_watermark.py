"""Extract NCS1 records and the unmodified replayfilter used by the generator."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
archive = ROOT / "data/external/WatermarkV1.zip"
entries = []
with zipfile.ZipFile(archive) as z:
    for item in z.infolist():
        source = item.filename
        if source.startswith("Watermark/input/channels/NCS1/mat/") and source.endswith(".mat"):
            dest = ROOT / "data/external/ncs1" / Path(source).name
        elif source == "Watermark/matlab/replayfilter.m":
            dest = ROOT / "third_party/watermark" / Path(source).name
        elif source == "Watermark/doc/readme.txt":
            dest = ROOT / "third_party/watermark/official_readme.txt"
        else:
            continue
        raw = z.read(item)  # also verifies the ZIP CRC
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        entries.append({"path": dest.relative_to(ROOT).as_posix(),
                        "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
with archive.open("rb") as f:
    digest = hashlib.file_digest(f, "sha256").hexdigest()
manifest = {"source": "https://www.ffi.no/en/research/watermark", "version": "Watermark V1, 2016-11-30",
            "archive_sha256": digest, "files": entries}
(ROOT / "data/external/source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Installed {len(entries)} verified files; archive SHA256 {digest}")
