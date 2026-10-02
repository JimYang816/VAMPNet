"""Record hashes of simulation generation code, configuration, and shards."""
import hashlib
import json
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    files = sorted([*ROOT.glob("matlab/**/*.m"), ROOT / "configs/simulation.json",
                    ROOT / "scripts/check_environment.m", *ROOT.glob("scripts/*.py"),
                    ROOT / "third_party/watermark/replayfilter.m"])
    code = {p.relative_to(ROOT).as_posix(): digest(p) for p in files}
    shards = {p.name: digest(p) for p in sorted(args.dataset.glob("*/*.h5"))}
    report = {"pipeline_sha256": code, "shard_sha256": shards,
              "source_manifest_sha256": digest(ROOT / "data/external/source_manifest.json")}
    (args.dataset / "provenance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Recorded {len(code)} code/config and {len(shards)} shard hashes")
