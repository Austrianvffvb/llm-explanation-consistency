"""Fetch the two pinned source files; no API requests or model inference."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "data/source_manifest.json").read_text())
for name, expected in manifest["sha256"].items():
    path = ROOT / name
    if path.exists():
        content = path.read_bytes()
    else:
        url = f"https://raw.githubusercontent.com/milesaturpin/cot-unfaithfulness/{manifest['revision']}/{name}"
        with urlopen(url, timeout=60) as response:
            content = response.read()
    if hashlib.sha256(content).hexdigest() != expected:
        raise SystemExit(f"Source checksum mismatch: {name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(content)
    print(f"Verified: {name}")
