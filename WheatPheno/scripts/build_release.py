"""Build the reviewed public source archive using the standard library only.

Usage: python scripts/build_release.py --output dist/WheatPheno-0.4.0.zip
No private training directory, output directory or environment is traversed.
The archive contains WheatPheno/RELEASE_MANIFEST.json; a matching manifest is
also written beside pyproject.toml after the archive has been built.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "WheatPheno"
TOP_LEVEL = {
    ".gitignore", ".gitattributes", "app.py", "README.md", "LICENSE", "DATA_LICENSE.md",
    "CONTRIBUTING.md", "SECURITY.md", "CITATION.cff", "CHANGELOG.md",
    "pyproject.toml", "start_ui.ps1", "start_ui.bat", "start_ui.sh",
    "requirements-lock.txt",
}
PATTERNS = (
    "src/wheatpheno/*.py", "docs/*.md", "tests/*.py", "configs/*.json",
    "scripts/*.py", ".github/workflows/*.yml", ".github/workflows/*.yaml",
)
MODEL_FILES = {
    "models/manifest.json", "models/schema.json",
    "models/AGB_ExtraTrees_full_refit.joblib",
    "models/Yield_ExtraTrees_full_refit.joblib",
}
BENCHMARK_FILES = {
    "benchmarks/README.md", "benchmarks/model_comparison.csv",
    "benchmarks/repetition_metrics.csv", "benchmarks/subgroup_metrics.csv",
}
# Data names are deliberately frozen, so dropping another CSV/NPZ into the
# examples directory does not silently make it part of a public release.
EXAMPLE_FILES = {
    "examples/manifest.json",
    "examples/tables/agb_demo.csv", "examples/tables/agb_expected.csv",
    "examples/tables/yield_demo.csv", "examples/tables/yield_expected.csv",
}
# Add synthetic files by exact name after review; a directory wildcard must
# never make a new dataset public. Historical scene archives are excluded.
SYNTHETIC_FILES = {
    "examples/synthetic/training_demo.csv",
    "examples/synthetic/feature_input.npz",
    "examples/synthetic/README.md",
}
OPTIONAL_FILES = {".streamlit/config.toml"}
FORBIDDEN_PARTS = {
    "private", "raw", "raw_data", "rawdata", "data_raw", "output", "outputs",
    "results", ".venv", "venv", "env", ".git", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "cache", "caches", ".cache", "node_modules",
    "dist", "build",
}
BINARY_SUFFIXES = {".joblib", ".npz", ".png"}
PRIVATE_PATH = re.compile(r"(?i)(?:[a-z]:[\\/]+Users[\\/]|E:[\\/])")
MANIFEST_NAME = "RELEASE_MANIFEST.json"


def _reject_links(path: Path, root: Path) -> None:
    """Reject symbolic links and Windows junctions on every relative component."""
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        info = current.lstat()
        reparse = getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if stat.S_ISLNK(info.st_mode) or reparse:
            raise ValueError(f"Links and reparse points are not publishable: {relative.as_posix()}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes the project: {relative.as_posix()}")


def collect_files(root: Path = ROOT) -> list[tuple[str, bytes]]:
    """Return an immutable byte snapshot of allowlisted, inspected files."""
    root = Path(root).resolve()
    selected = {root / name for name in TOP_LEVEL if (root / name).exists()}
    for pattern in PATTERNS:
        selected.update(root.glob(pattern))
    required = MODEL_FILES | EXAMPLE_FILES | BENCHMARK_FILES | SYNTHETIC_FILES | {
        "README.md", "app.py", "pyproject.toml", "LICENSE", "DATA_LICENSE.md",
        ".gitattributes",
    }
    for name in required:
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"Required public release file is missing: {name}")
        selected.add(path)
    selected.update(root / name for name in OPTIONAL_FILES if (root / name).exists())
    snapshot = []
    for path in sorted(selected, key=lambda p: p.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        name = relative.as_posix()
        if any(part.casefold() in FORBIDDEN_PARTS for part in relative.parts):
            raise ValueError(f"Forbidden release path: {name}")
        _reject_links(path, root)
        if not path.is_file():
            raise ValueError(f"Release entry is not a regular file: {name}")
        data = path.read_bytes()
        if path.suffix.casefold() not in BINARY_SUFFIXES:
            if b"\r" in data:
                raise ValueError(f"Text files must use LF line endings for portable integrity checks: {name}")
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise ValueError(f"Text file must use UTF-8: {name}") from exc
            match = PRIVATE_PATH.search(text)
            if match:
                line = text.count("\n", 0, match.start()) + 1
                raise ValueError(f"Local workstation path detected in {name}:{line}; remove it before release.")
            if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text):
                raise ValueError(f"Non-English CJK text detected in {name}; translate it before release.")
        # The archive supports examples, not full plot tables or complete fields.
        if name.startswith("examples/") and path.suffix == ".csv":
            if len(data) > 512_000 or len(data.splitlines()) > 201:
                raise ValueError(f"Public example table exceeds the reviewed small-example limit: {name}")
        if name.startswith("examples/") and path.suffix == ".npz":
            if len(data) > 16_000_000:
                raise ValueError(f"Public example archive is too large: {name}")
            # Inspect archive metadata only; binary arrays are not text scanned.
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                if sum(entry.file_size for entry in archive.infolist()) > 64_000_000:
                    raise ValueError(f"Expanded example archive exceeds the small-example limit: {name}")
        snapshot.append((name, data))
    return snapshot


def installed_records(snapshot: list[tuple[str, bytes]]) -> list[dict]:
    """Record exactly the code and assets installed by a wheel.

    Source integrity includes repository tests and documentation. Installed
    integrity uses separate package and asset roots, because wheel layouts do
    not retain the source repository directory structure.
    """
    contents = dict(snapshot)
    project = tomllib.loads(contents["pyproject.toml"].decode("utf-8"))
    mappings = {}
    for name in contents:
        if name.startswith("src/wheatpheno/") and name.endswith(".py"):
            mappings[("package", name.removeprefix("src/wheatpheno/"))] = name
    for destination, names in project["tool"]["setuptools"]["data-files"].items():
        if destination != "share/wheatpheno" and not destination.startswith("share/wheatpheno/"):
            raise ValueError("Wheel data destinations must stay within share/wheatpheno.")
        folder = Path(destination).relative_to("share/wheatpheno")
        for name in names:
            if any(character in name for character in "*?[]"):
                raise ValueError("Wheel assets require exact filenames, not wildcard data inclusion.")
            if name == MANIFEST_NAME:
                continue  # A manifest cannot hash itself.
            if name not in contents:
                raise ValueError(f"Wheel asset is outside the reviewed source allowlist: {name}")
            key = ("assets", (folder / Path(name).name).as_posix())
            if key in mappings:
                raise ValueError(f"Duplicate installed artifact path: {key}")
            mappings[key] = name
    return [
        {"root": role, "path": path, "source_path": source,
         "size": len(contents[source]), "sha256": hashlib.sha256(contents[source]).hexdigest()}
        for (role, path), source in sorted(mappings.items())
    ]


def release_manifest(snapshot: list[tuple[str, bytes]]) -> dict:
    return {
        "format_version": 2,
        "package": PREFIX,
        "scope": "Reviewed code, fixed model artifacts, aggregate benchmarks and limited public examples.",
        "manifest_self_hash": False,
        "file_count": len(snapshot),
        "total_bytes": sum(len(data) for _, data in snapshot),
        "files": [
            {"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in snapshot
        ],
        "installed_files": installed_records(snapshot),
    }


def build_release(output: Path, root: Path = ROOT) -> dict:
    root = Path(root).resolve()
    output = Path(output).absolute()
    if output.suffix.casefold() != ".zip":
        raise ValueError("The output filename must end in .zip.")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Refusing to replace an existing release archive; choose a new output path.")
    snapshot = collect_files(root)
    manifest = release_manifest(snapshot)
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Validate output ancestry as well, without following a link to a different
    # destination than the caller can see.
    for parent in (output.parent, *output.parent.parents):
        info = parent.lstat()
        reparse = getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if parent.is_symlink() or reparse:
            raise ValueError("The release destination must not pass through a link or junction.")
    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_symlink():
        raise ValueError("The release manifest destination must not be a link.")
    handle, temporary = tempfile.mkstemp(prefix=".wheatpheno_release_", suffix=".zip", dir=output.parent)
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, data in [*snapshot, (MANIFEST_NAME, manifest_bytes)]:
                item = zipfile.ZipInfo(f"{PREFIX}/{name}", date_time=(1980, 1, 1, 0, 0, 0))
                item.compress_type = zipfile.ZIP_DEFLATED
                item.external_attr = 0o100644 << 16
                archive.writestr(item, data)
        # Check once more immediately before publication; no unrelated file is
        # intentionally overwritten by the builder.
        if output.exists():
            raise FileExistsError("The requested release path was created during packaging.")
        temporary_path.rename(output)
        manifest_path.write_bytes(manifest_bytes)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Package only the reviewed WheatPheno public allowlist.")
    parser.add_argument("--output", required=True, type=Path, help="New ZIP path; existing files are not overwritten.")
    args = parser.parse_args()
    try:
        manifest = build_release(args.output)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Release blocked: {exc}\n")
    print(f"Created {args.output}: {manifest['file_count']} reviewed files plus {MANIFEST_NAME}.")


if __name__ == "__main__":
    main()
