"""Download official UCI data; preserve missing values for fold-local imputation."""
import hashlib
import io
import json
import urllib.request
import zipfile

import pandas as pd

from heart_risk.settings import DATA_PATH, FRIENDLY_COLUMNS, RAW_COLUMNS, ROOT, SOURCE_URL


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    if list(raw.columns) != RAW_COLUMNS:
        raise ValueError("Expected the 14 original Cleveland columns in source order")
    clean = raw.replace("?", float("nan")).apply(pd.to_numeric, errors="raise")
    if clean["diagnosis"].isna().any() or not clean["diagnosis"].isin(range(5)).all():
        raise ValueError("Diagnosis must contain UCI codes 0 through 4")
    clean["diagnosis"] = (clean["diagnosis"] > 0).astype(int)
    clean.columns = FRIENDLY_COLUMNS
    return clean.drop_duplicates().reset_index(drop=True)


def acquire_data(refresh: bool = False) -> pd.DataFrame:
    raw_path = ROOT / "data/raw/processed.cleveland.data"
    manifest_path = ROOT / "data/source_manifest.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not raw_path.exists():
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        member = next(n for n in archive.namelist() if n.endswith("processed.cleveland.data"))
        raw_path.write_bytes(archive.read(member))
    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if manifest_path.exists() and not refresh:
        expected = json.loads(manifest_path.read_text())["source_sha256"]
        if expected != digest:
            raise ValueError("Raw source checksum mismatch; restore data or deliberately --refresh")
    raw = pd.read_csv(raw_path, names=RAW_COLUMNS, na_values="?")
    clean = clean_data(raw)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(DATA_PATH, index=False)
    manifest = {"source_url": SOURCE_URL, "source_file": raw_path.name,
                "source_sha256": digest, "cleaned_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
                "source_rows": len(raw), "cleaned_rows": len(clean),
                "missing_values": clean.isna().sum().to_dict(),
                "target_rule": "diagnosis > 0 maps to 1; diagnosis == 0 maps to 0",
                "license": "CC BY 4.0", "citation": "https://doi.org/10.24432/C52P4X"}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return clean
