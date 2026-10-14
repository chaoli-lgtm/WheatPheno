"""Integrity records for source releases and installed distributions."""

from pathlib import Path, PurePosixPath
import importlib.metadata
import json
import re

from .paths import project_root
from .models import model_manifest, load_model
from .pipeline import sha256, save_json, _report, _output


def verify():
    records = {}
    for target in ("AGB", "Yield"):
        model = load_model(target)
        records[target] = {"integrity": "passed", "trees": len(model.named_steps["model"].estimators_)}
    return {"status": "passed", "models": records, "PH": "CHM PCT95 x 100; no learned weights"}


def _relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("Unsafe path in release manifest.")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError("Unsafe path in release manifest.")
    return Path(*path.parts)


def verify_manifest(root, package_root=None):
    """Verify all files belonging to the detected installation layout.

    Source releases include source, tests and documentation. Wheel installs
    verify package code and distributed assets using separate roots, without
    silently skipping files that should have been installed.
    """
    root = Path(root).resolve()
    package_root = Path(package_root or Path(__file__).parent).resolve()
    manifest_path = root / "RELEASE_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("RELEASE_MANIFEST.json is missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 2:
        raise ValueError("Release manifest format 2 is required; rebuild the source release.")
    source = (root / "pyproject.toml").is_file() and (root / "src/wheatpheno").is_dir()
    mode = "source" if source else "installed"
    records = manifest.get("files" if source else "installed_files")
    if not isinstance(records, list) or not records:
        raise ValueError(f"The release manifest has no {mode} file records.")
    roots = {"source": root, "assets": root, "package": package_root}
    expected = set()
    mismatches = []
    for record in records:
        role = "source" if source else record.get("root")
        if role not in roots or (not source and role == "source"):
            raise ValueError("Invalid installed root in release manifest.")
        relative = _relative_path(record.get("path"))
        key = (role, relative.as_posix())
        if key in expected:
            raise ValueError("Duplicate file entry in release manifest.")
        expected.add(key)
        digest = record.get("sha256", "")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid SHA256 in release manifest.")
        path = roots[role] / relative
        label = relative.as_posix() if source else f"{role}/{relative.as_posix()}"
        if not path.is_file() or not path.resolve().is_relative_to(roots[role]) or sha256(path) != digest:
            mismatches.append(label)
    if not source:
        if not any(role == "package" for role, _ in expected) or not any(role == "assets" for role, _ in expected):
            raise ValueError("Installed manifest must cover both executable code and assets.")
        # Detect extra installed code or datasets instead of accepting a partial
        # manifest or silently allowing an unreviewed asset in the distribution.
        actual = {("package", path.relative_to(package_root).as_posix())
                  for path in package_root.rglob("*.py") if "__pycache__" not in path.parts}
        for folder in ("models", "examples", "benchmarks"):
            actual.update(("assets", path.relative_to(root).as_posix())
                          for path in (root / folder).rglob("*") if path.is_file())
        mismatches.extend(f"unlisted:{role}/{path}" for role, path in sorted(actual - expected))
    return {"layout": mode, "files_checked": len(records), "mismatches": mismatches}


def review(output_dir):
    root = project_root()
    report = _report("release_review")
    report["models"] = verify()
    report["manifest"] = verify_manifest(root)
    report["dependencies"] = {name: importlib.metadata.version(name)
                              for name in ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "rasterio"]}
    report["privacy"] = {
        "public_examples": json.loads((root / "examples/manifest.json").read_text(encoding="utf-8")),
        "complete_raw_tables_published": False,
        "model_privacy": "Fixed tree weights encode learned thresholds and leaf values; this is not a differential privacy guarantee.",
    }
    report["limitations"] = model_manifest().get("known_limitations", [])
    report["external_review"] = "Not performed. Local automated checks do not constitute peer review."
    report["status"] = "passed" if not report["manifest"]["mismatches"] else "failed"
    out = _output(output_dir)
    save_json(out / "review.json", report)
    return report
