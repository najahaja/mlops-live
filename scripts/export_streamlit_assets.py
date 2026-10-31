"""Export a sanitized, read-only asset bundle for the public Streamlit demo (webapp/assets/).

The public Streamlit Community Cloud app cannot reach the local MLflow registry (mlflow.db,
mlruns/) or the operational prediction store (predictions.db). This script copies the current
champion fallback artifact (models/model.joblib) and the verified evaluation reports into
webapp/assets/, removing local filesystem paths and process details before publication.

Source files are only read; they are never modified. Existing asset files are not overwritten
unless --overwrite is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "webapp" / "assets"
ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

# Report files (relative to repo root) published in sanitized form.
REPORT_SOURCES: dict[str, str] = {
    "metrics.json": "models/metrics.json",
    "holdout_test_metrics.json": "reports/generated/holdout_test_metrics.json",
    "feb_2024_out_of_time_evaluation.json": (
        "reports/generated/feb_2024_out_of_time_evaluation.json"
    ),
    "controlled_drift_retraining_report.json": (
        "reports/generated/controlled_drift_retraining_report.json"
    ),
    "http_load_test_report.json": "reports/generated/http_load_test_report.json",
    "load_test_report.json": "reports/generated/load_test_report.json",
}

# Keys removed entirely because they describe local machine state (paths, PIDs, log files).
SENSITIVE_KEYS = frozenset(
    {
        "sandbox_directory",
        "sandbox_cleaned",
        "server_process",
        "output_json",
        "tracking_uri",
        "test_file",
        "log_file",
        "isolated_predictions_db",
        "run_id",
    }
)

_ABSOLUTE_PATH_PATTERN = re.compile(
    r"((?<![A-Za-z])[A-Za-z]:[\\/])|(^\\\\)|(^/(home|Users|tmp|var|mnt)/)|(sqlite:///)|(file://)",
)
REDACTED = "[redacted-local-path]"


def sanitize_payload(payload: Any) -> Any:
    """Recursively drop sensitive keys and redact absolute local paths from a JSON payload."""
    if isinstance(payload, dict):
        return {
            key: sanitize_payload(value)
            for key, value in payload.items()
            if key not in SENSITIVE_KEYS
        }
    if isinstance(payload, list):
        return [sanitize_payload(item) for item in payload]
    if isinstance(payload, str):
        if _ABSOLUTE_PATH_PATTERN.search(payload):
            return REDACTED
        # Normalize Windows-style relative paths (data\\processed\\x.parquet) for display.
        return payload.replace("\\", "/")
    return payload


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new(path: Path, content: bytes, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing asset: {path.name}")
    path.write_bytes(content)


def verify_model_on_holdout(model_path: Path, test_file: Path) -> dict[str, Any] | None:
    """Re-evaluate the exported model on the isolated holdout split (read-only)."""
    if not test_file.exists():
        logger.warning("Holdout file %s not found; skipping verification.", test_file)
        return None

    import joblib
    import pandas as pd

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from mlops_live.data.clean import FEATURE_COLUMNS, TARGET_COLUMN
    from mlops_live.models.evaluate import evaluate_predictions

    pipeline = joblib.load(model_path)
    df = pd.read_parquet(test_file)
    metrics = evaluate_predictions(df[TARGET_COLUMN], pipeline.predict(df[FEATURE_COLUMNS]))
    return {"split": "jan_2024_isolated_holdout_test", **metrics.to_dict()}


def download_zone_lookup(timeout: float = 20.0) -> bytes | None:
    """Download the public NYC TLC taxi zone lookup table (zone names for UI dropdowns)."""
    try:
        import requests

        response = requests.get(ZONE_LOOKUP_URL, timeout=timeout)
        response.raise_for_status()
        content = response.content
        if not content.startswith(b'"LocationID"') and not content.startswith(b"LocationID"):
            logger.warning("Unexpected zone lookup format; skipping.")
            return None
        return content
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Could not download taxi zone lookup (%s); skipping.", exc)
        return None


def export_assets(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    repo_root: Path = REPO_ROOT,
    overwrite: bool = False,
    include_zone_lookup: bool = True,
    verify_holdout: bool = True,
) -> dict[str, Any]:
    """Export the sanitized model and report bundle. Returns the written manifest."""
    model_src = repo_root / "models" / "model.joblib"
    if not model_src.exists():
        raise FileNotFoundError("models/model.joblib not found; train the model first.")

    output_dir.mkdir(parents=True, exist_ok=True)

    model_dst = output_dir / "model.joblib"
    if model_dst.exists() and not overwrite:
        raise FileExistsError("Refusing to overwrite existing asset: model.joblib")
    shutil.copyfile(model_src, model_dst)

    exported_reports: list[str] = []
    for asset_name, relative_src in REPORT_SOURCES.items():
        src = repo_root / relative_src
        if not src.exists():
            logger.warning("Report %s not found; skipping.", relative_src)
            continue
        payload = json.loads(src.read_text(encoding="utf-8"))
        content = json.dumps(sanitize_payload(payload), indent=2).encode("utf-8")
        _write_new(output_dir / asset_name, content, overwrite)
        exported_reports.append(asset_name)

    zone_lookup_exported = False
    if include_zone_lookup:
        zone_bytes = download_zone_lookup()
        if zone_bytes is not None:
            _write_new(output_dir / "taxi_zone_lookup.csv", zone_bytes, overwrite)
            zone_lookup_exported = True

    metrics_payload: dict[str, Any] = {}
    metrics_src = repo_root / "models" / "metrics.json"
    if metrics_src.exists():
        metrics_payload = json.loads(metrics_src.read_text(encoding="utf-8"))
    champion = metrics_payload.get("mlflow", {}).get("champion", {})

    import sklearn

    holdout_check = (
        verify_model_on_holdout(model_dst, repo_root / "data" / "processed" / "test.parquet")
        if verify_holdout
        else None
    )

    manifest = {
        "exported_at_utc": datetime.now(UTC).isoformat(),
        "model_file": "model.joblib",
        "model_sha256": sha256_of(model_dst),
        "source_model_sha256": sha256_of(model_src),
        "model_name": metrics_payload.get("selected_model", "unknown"),
        "registry_version": str(champion.get("version", "unknown")),
        "registry_alias": champion.get("alias", "champion"),
        "registered_model_name": metrics_payload.get("mlflow", {}).get(
            "registered_model_name", "unknown"
        ),
        "feature_columns": metrics_payload.get("feature_columns", []),
        "target_column": metrics_payload.get("target_column", "duration_minutes"),
        "scikit_learn_version": sklearn.__version__,
        "reports": exported_reports,
        "zone_lookup_exported": zone_lookup_exported,
        "holdout_verification": holdout_check,
    }
    _write_new(
        output_dir / "manifest.json", json.dumps(manifest, indent=2).encode("utf-8"), overwrite
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing assets.")
    parser.add_argument(
        "--skip-zone-lookup", action="store_true", help="Do not download taxi zone names."
    )
    parser.add_argument(
        "--skip-holdout-check", action="store_true", help="Skip holdout re-evaluation."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    manifest = export_assets(
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        include_zone_lookup=not args.skip_zone_lookup,
        verify_holdout=not args.skip_holdout_check,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
