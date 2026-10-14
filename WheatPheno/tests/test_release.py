"""Public allowlists and integrity must survive Git and wheel installation."""

import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import numpy as np
import pytest

from wheatpheno.audit import verify_manifest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_builder", ROOT / "scripts/build_release.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


@pytest.fixture
def release_source(tmp_path):
    root = tmp_path / "source"
    names = (builder.MODEL_FILES | builder.EXAMPLE_FILES | builder.SYNTHETIC_FILES |
             builder.BENCHMARK_FILES | {"README.md", "app.py", "pyproject.toml",
                                        "LICENSE", "DATA_LICENSE.md", ".gitattributes"})
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name in ("pyproject.toml", ".gitattributes"):
            path.write_bytes((ROOT / name).read_bytes().replace(b"\r\n", b"\n"))
        elif path.suffix == ".npz":
            stream = io.BytesIO()
            np.savez_compressed(stream, synthetic=np.ones((2, 2)))
            path.write_bytes(stream.getvalue())
        else:
            path.write_bytes(b"{}\n" if path.suffix == ".json" else b"example\n")
    package = root / "src/wheatpheno"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b'"""Synthetic release fixture."""\n')
    (package / "predict.py").write_bytes(b"VALUE = 1\n")
    return root


def make_manifest(root):
    manifest = builder.release_manifest(builder.collect_files(root))
    (root / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def install_fixture(root, destination):
    manifest = make_manifest(root)
    assets, package = destination / "share/wheatpheno", destination / "site-packages/wheatpheno"
    for record in manifest["installed_files"]:
        path = (assets if record["root"] == "assets" else package) / record["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / record["source_path"], path)
    shutil.copyfile(root / "RELEASE_MANIFEST.json", assets / "RELEASE_MANIFEST.json")
    return assets, package


def test_source_and_installed_integrity_cover_different_layouts(release_source, tmp_path):
    assets, package = install_fixture(release_source, tmp_path / "installed")
    source_result = verify_manifest(release_source)
    installed_result = verify_manifest(assets, package)
    assert source_result["layout"] == "source"
    assert installed_result["layout"] == "installed"
    assert not source_result["mismatches"]
    assert not installed_result["mismatches"]
    assert not (assets / "src").exists()
    assert not (assets / "README.md").exists()


@pytest.mark.parametrize("damage", ["missing_asset", "changed_code", "extra_code", "extra_data"])
def test_installed_review_rejects_incomplete_or_unreviewed_contents(release_source, tmp_path, damage):
    assets, package = install_fixture(release_source, tmp_path / "installed")
    if damage == "missing_asset":
        (assets / "examples/tables/agb_demo.csv").unlink()
    elif damage == "changed_code":
        (package / "predict.py").write_bytes(b"VALUE = 2\n")
    elif damage == "extra_code":
        (package / "unexpected.py").write_bytes(b"VALUE = 3\n")
    else:
        (assets / "examples/private.csv").write_bytes(b"secret\n")
    assert verify_manifest(assets, package)["mismatches"]


def test_windows_git_checkout_preserves_release_hashes(release_source, tmp_path):
    if shutil.which("git") is None:
        pytest.skip("Git is required for the checkout integrity regression.")
    make_manifest(release_source)
    def run(*args, cwd=release_source):
        subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)
    run("git", "init")
    run("git", "config", "core.autocrlf", "true")
    run("git", "add", ".")
    run("git", "-c", "user.name=Release tests", "-c", "user.email=tests@example.invalid",
        "commit", "-m", "Synthetic integrity fixture")
    clone = tmp_path / "windows-checkout"
    run("git", "-c", "core.autocrlf=true", "clone", str(release_source), str(clone))
    assert verify_manifest(clone)["mismatches"] == []


def test_source_archive_excludes_old_scenes_and_unapproved_data(release_source, tmp_path):
    extras = ["examples/scenes/old_field.npz", "examples/synthetic/unapproved.csv",
              "examples/tables/full_training.csv", "models/unapproved.joblib",
              "models/unapproved.json", "benchmarks/unapproved.csv", "docs/demo_overview.png"]
    for name in extras:
        path = release_source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not approved")
    archive = tmp_path / "source.zip"
    builder.build_release(archive, release_source)
    with zipfile.ZipFile(archive) as zipped:
        names = {name.removeprefix("WheatPheno/") for name in zipped.namelist()}
    assert not names.intersection(extras)
    assert builder.SYNTHETIC_FILES.issubset(names)
    assert verify_manifest(release_source)["mismatches"] == []


@pytest.mark.parametrize("violation", ["crlf", "cjk", "private_path"])
def test_release_guards_reject_nonportable_or_private_text(release_source, violation):
    values = {"crlf": "two\r\nlines\r\n", "cjk": chr(0x4E2D),
              "private_path": "C:" + "/Users/person/private.txt"}
    (release_source / "README.md").write_bytes(values[violation].encode("utf-8"))
    with pytest.raises(ValueError):
        builder.collect_files(release_source)


def test_wheel_asset_globs_are_rejected(release_source):
    snapshot = builder.collect_files(release_source)
    snapshot = [(name, data.replace(b'"models/schema.json"', b'"models/*.json"'))
                if name == "pyproject.toml" else (name, data) for name, data in snapshot]
    with pytest.raises(ValueError, match="exact filenames"):
        builder.release_manifest(snapshot)


@pytest.mark.parametrize("bad", ["../outside", "/absolute", "relative/../outside", "C:" + "/outside"])
def test_manifest_paths_cannot_escape_root(release_source, bad):
    manifest = make_manifest(release_source)
    manifest["files"][0]["path"] = bad
    (release_source / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsafe path"):
        verify_manifest(release_source)
