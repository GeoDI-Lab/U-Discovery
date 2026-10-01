"""Release inventory excludes local runtime files, not unexpected shipped data."""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from udiscovery.validate import (
    _validate_artifact_scope,
    _validate_checksum_files,
    _validate_repository_hygiene,
    release_files,
    sha256,
)

update_manifest = runpy.run_path(str(ROOT / "scripts/update_manifest.py"))["update_manifest"]


def write(root, relative, content="portable content\n"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_runtime_and_installation_files_do_not_change_release_inventory(tmp_path):
    write(tmp_path, "src/package.py")
    write(tmp_path, "documentation/outputs/example.txt")
    # Local files exercise every exclusion and all content scans. In
    # particular a bogus local CHECKSUMS file must never be discovered.
    ignored = [
        ".git/credentials.json", ".venv/lib/unsafe.pkl", "venv/lib/unsafe.pt",
        "src/__pycache__/cache.pyc", ".pytest_cache/cache.txt", "build/result.pth",
        "dist/archive.ckpt", "src/project.egg-info/SOURCES.txt",
        "outputs/figure8/unsafe.pt", "outputs/CHECKSUMS.sha256",
        ".DS_Store", "src/._package.py", "src/cache.pyc", "src/cache.pyo", "run.log",
    ]
    for relative in ignored:
        write(tmp_path, relative, "API_KEY=" + "a" * 24 + "\n")
    files = release_files(tmp_path)
    assert [path.relative_to(tmp_path).as_posix() for path in files] == [
        "documentation/outputs/example.txt", "src/package.py",
    ]
    update_manifest(tmp_path)
    assert _validate_checksum_files(tmp_path) == []
    assert _validate_repository_hygiene(tmp_path) == []
    assert _validate_artifact_scope(tmp_path) == []


def test_manifest_still_detects_new_release_file_and_unsafe_checkpoint(tmp_path):
    write(tmp_path, "README.md")
    update_manifest(tmp_path)
    write(tmp_path, "src/unreviewed.py")
    write(tmp_path, "results/unsafe.pt")
    errors = _validate_checksum_files(tmp_path)
    assert any("does not cover release files" in error and "unreviewed.py" in error for error in errors)
    assert any("does not cover release files" in error and "unsafe.pt" in error for error in errors)
    assert any("unsafe pickle-capable" in error and "results/unsafe.pt" in error
               for error in _validate_artifact_scope(tmp_path))


def test_real_env_remains_in_scope_and_is_flagged(tmp_path):
    write(tmp_path, ".env.example", "API_KEY=replace-me\n")
    write(tmp_path, ".env", "API_KEY=" + "b" * 24 + "\n")
    names = {path.name for path in release_files(tmp_path)}
    assert names == {".env", ".env.example"}
    errors = _validate_repository_hygiene(tmp_path)
    assert any("local environment/credential file found: .env" in error for error in errors)
    assert any("possible plaintext credential in .env" in error for error in errors)
    assert not any(".env.example" in error for error in errors)


def test_manifest_is_deterministic_and_frozen_checksums_are_unchanged(tmp_path):
    data = write(tmp_path, "data/frozen.txt")
    frozen = write(tmp_path, "data/CHECKSUMS.sha256", f"{sha256(data)}  frozen.txt\n")
    original = frozen.read_bytes()
    write(tmp_path, "z.txt")
    write(tmp_path, "a.txt")
    manifest = update_manifest(tmp_path)
    first = manifest.read_bytes()
    update_manifest(tmp_path)
    assert manifest.read_bytes() == first
    names = [line.split("  ", 1)[1] for line in manifest.read_text().splitlines()]
    assert names == sorted(names)
    assert "MANIFEST.sha256" not in names
    assert frozen.read_bytes() == original
    assert _validate_checksum_files(tmp_path) == []
    data.write_text("changed frozen data\n")
    assert any("checksum mismatch: data/frozen.txt" in error
               for error in _validate_checksum_files(tmp_path))


def test_scope_and_size_checks_apply_only_to_release_inventory(tmp_path):
    write(tmp_path, "data/1500m/description.md")
    local = tmp_path / "outputs/large.bin"
    local.parent.mkdir()
    with local.open("wb") as output:
        output.truncate(51 * 1024 * 1024)
    errors = _validate_artifact_scope(tmp_path)
    assert any("out-of-scope scale artifacts" in error for error in errors)
    assert not any("50 MiB" in error for error in errors)
    published = tmp_path / "large.bin"
    local.rename(published)
    assert any("50 MiB" in error and "large.bin" in error
               for error in _validate_artifact_scope(tmp_path))
