"""Fetch and safely extract the pinned MovieLens 1M archive."""

from __future__ import annotations

import hashlib
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
ARCHIVE = RAW_DIR / "ml-1m.zip"
EXTRACT_DIR = RAW_DIR / "extracted"
EXPECTED_SHA256 = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"
SOURCES = (
    "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "https://huggingface.co/datasets/Recommenders/MovieLens/resolve/main/ml-1m.zip",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_archive() -> None:
    if ARCHIVE.exists():
        actual = sha256_file(ARCHIVE)
        if actual != EXPECTED_SHA256:
            raise ValueError(f"Existing archive checksum mismatch: {actual}")
        print(f"Verified existing archive SHA-256: {actual}")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    temporary = ARCHIVE.with_suffix(".zip.part")
    errors: list[str] = []
    for url in SOURCES:
        try:
            print(f"Downloading from {url}")
            request = urllib.request.Request(url, headers={"User-Agent": "portfolio-study/1.0"})
            with (
                urllib.request.urlopen(request, timeout=30) as response,
                temporary.open("wb") as out,
            ):
                while block := response.read(1024 * 1024):
                    out.write(block)
            actual = sha256_file(temporary)
            if actual != EXPECTED_SHA256:
                raise ValueError(f"SHA-256 mismatch for {url}: {actual}")
            temporary.replace(ARCHIVE)
            print(f"Verified archive SHA-256: {actual}")
            return
        except (OSError, urllib.error.URLError, ValueError) as exc:
            errors.append(f"{url}: {exc}")
            temporary.unlink(missing_ok=True)
    raise RuntimeError("Could not fetch the pinned MovieLens archive.\n" + "\n".join(errors))


def extract_archive() -> None:
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    root = EXTRACT_DIR.resolve()
    with zipfile.ZipFile(ARCHIVE) as archive:
        for member in archive.infolist():
            target = (EXTRACT_DIR / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe archive member path: {member.filename}")
        archive.extractall(EXTRACT_DIR)

    source_dir = EXTRACT_DIR / "ml-1m"
    required = ("ratings.dat", "movies.dat", "users.dat", "README")
    missing = [name for name in required if not (source_dir / name).is_file()]
    if missing:
        raise ValueError(f"The archive is missing expected files: {missing}")
    print(f"Extracted the archive into {source_dir}")
    print("The demographics file is present in the source archive but is not loaded by this study.")


if __name__ == "__main__":
    download_archive()
    extract_archive()
