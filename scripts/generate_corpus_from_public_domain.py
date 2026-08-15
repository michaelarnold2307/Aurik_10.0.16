#!/usr/bin/env python3
"""Public-Domain Corpus Generator für Aurik — §15.2.

Lädt gemeinfreie Audio-Aufnahmen von Internet Archive, Musopen und Freesound (CC0)
herunter und erstellt manifest.yaml-Einträge nach corpus/MANIFEST_SCHEMA.yaml.

Nutzung:
  python scripts/generate_corpus_from_public_domain.py --material vinyl --count 5
  python scripts/generate_corpus_from_public_domain.py --all --dry-run
  python scripts/generate_corpus_from_public_domain.py --from-urls urls.txt

Abhängigkeiten: requests, yt-dlp (optional für Internet Archive), librosa
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = REPO_ROOT / "corpus"

# ── Public-Domain-Quellen ──────────────────────────────────────────────────
# Jeder Eintrag: (url, material, era_year, genre, defect_types, attribution, license)
# Nur gemeinfreie / CC0-Quellen. Kein urheberrechtlich geschütztes Material.

PUBLIC_DOMAIN_SOURCES: dict[str, list[dict[str, Any]]] = {
    # Echte, verifizierte Public-Domain-Aufnahmen (US: veröffentlicht ≤ 1926).
    # Alle Einträge: Internet Archive 78rpm Collection (Public Domain Mark 1.0).
    # Verifiziert am 2026-08-15: Lizenz-Metadaten, Jahr ≤ 1926, Audio-Checksummen
    # im corpus/shellac/manifest.yaml (test_corpus_integrity grün).
    "shellac": [
        {
            "url": "https://archive.org/download/enrico-caruso-giuseppe-vesrdi-rigoletto-la-donna-emobile-gramophone-da-1303/EnricoCaruso%2CGiuseppeVesrdi%2CRigoletto%2CLaDonnaEMobile%2CGramophone%2CDA1303.mp3",
            "era_year": 1908,
            "genre": "opera",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Enrico Caruso — Rigoletto: La donna è mobile (Internet Archive, 78rpm Collection)",
            "license": "Public Domain",
            "vocal": True,
        },
        {
            "url": "https://archive.org/download/78_ada-celeste-ada-heavenly-aida_enrico-caruso-giueseppi-verdi_gbia0290541a/A%C3%AFda%20-%20Celeste%20A%C3%AFda%20%28Heavenly%20Aida%21%29%20-%20Enrico%20Caruso.mp3",
            "era_year": 1911,
            "genre": "opera",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Enrico Caruso — Aïda: Celeste Aïda (Internet Archive, 78rpm Collection)",
            "license": "Public Domain",
            "vocal": True,
        },
        {
            "url": "https://archive.org/download/enrico-caruso-eduardo-di-capuai-osole-mio-gramophone-da-1303/EnricoCaruso%2CEduardoDiCapuai%2COSoleMio%2CGramophoneDA1303.mp3",
            "era_year": 1916,
            "genre": "opera",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Enrico Caruso — O Sole Mio (Internet Archive, 78rpm Collection)",
            "license": "Public Domain",
            "vocal": True,
        },
        {
            "url": "https://archive.org/download/1919-USA-Archives-1919-07-00-Joseph-Samuels-Dance-Orch-Taxi/1919%20%28USA%29%20Archives%201919%2007%2000%20Joseph%20Samuels%27%20Dance%20Orch%20-%20Taxi.mp3",
            "era_year": 1919,
            "genre": "jazz",
            "defect_types": ["surface_noise", "wow"],
            "attribution": "Joseph Samuels' Dance Orchestra — Taxi (Internet Archive, 78rpm Collection)",
            "license": "Public Domain",
            "vocal": False,
        },
        {
            "url": "https://archive.org/download/78_eddie-leonard-blues_irving-kaufman-v-l-stanton_gbia0286489b/Eddie%20Leonard%20Blues%20-%20Irving%20Kaufman%20-%20V.%20L.%20Stanton.mp3",
            "era_year": 1922,
            "genre": "jazz",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Irving Kaufman — Eddie Leonard Blues (Internet Archive, 78rpm Collection)",
            "license": "Public Domain",
            "vocal": True,
        },
        {
            "url": "https://archive.org/download/farawaybluesraw/Far%20away%20Blues%20raw.mp3",
            "era_year": 1923,
            "genre": "blues",
            "defect_types": ["surface_noise", "wow"],
            "attribution": "Bessie Smith & Clara Smith — Far Away Blues (Internet Archive, 78rpm Collection)",
            "license": "Public Domain",
            "vocal": True,
        },
        {
            "url": "https://archive.org/download/78_dirty-hands-dirty-face_bob-thomas-jolson-clark-leslie_gbia0086661a/Dirty%20Hands%21%20Dirty%20Face%21%20-%20Bob%20Thomas%20-%20Jolson.mp3",
            "era_year": 1923,
            "genre": "jazz",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Bob Thomas (Jolson) — Dirty Hands! Dirty Face! (Internet Archive, 78rpm Collection)",
            "license": "Public Domain",
            "vocal": True,
        },
        {
            "url": "https://archive.org/download/78_slik-pinden-danse-orkester-efim-schachmeister-edward/Slik-pinden%20-%20Efim%20Schachmeister.mp3",
            "era_year": 1924,
            "genre": "jazz",
            "defect_types": ["surface_noise", "wow"],
            "attribution": "Efim Schachmeister Danse Orkester — Slik-pinden (Internet Archive, 78rpm Collection)",
            "license": "Public Domain",
            "vocal": False,
        },
        {
            "url": "https://archive.org/download/78_rock-a-bye-my-baby-blues_frank-wright-and-frank-bessinger-the-two-franks-hill-yoe_gbia0236696b/Rock-A-Bye%20My%20Baby%20Blues%20-%20Frank%20Wright%20and%20Frank%20Bessinger%20%28The%20Two%20Franks%29.mp3",
            "era_year": 1924,
            "genre": "jazz",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "The Two Franks — Rock-A-Bye My Baby Blues (Internet Archive, 78rpm Collection)",
            "license": "Public Domain",
            "vocal": True,
        },
        {
            "url": "https://archive.org/download/78_quartet-emin-op59-no2-3rd-mvt-allegretto_virtuoso-string-quartet/D_955_Cc_4690-3.flac",
            "era_year": 1924,
            "genre": "classical",
            "defect_types": ["surface_noise", "wow"],
            "attribution": "Virtuoso String Quartet — Beethoven op. 59 Nr. 2, 3. Satz (Internet Archive, 78rpm Collection)",
            "license": "Public Domain",
            "vocal": False,
        },
        {
            "url": "https://archive.org/download/78_vesti-la-giubba_sig-caruso-leoncavallo_gbia7002950a/VESTI%20LA%20GIUBBA%20-%20SIG.%20CARUSO%20-%20Leoncavallo.mp3",
            "era_year": 1925,
            "genre": "opera",
            "defect_types": ["surface_noise", "wow", "hiss"],
            "attribution": "Enrico Caruso — Pagliacci: Vesti la giubba (Internet Archive, 78rpm Collection)",
            "license": "Public Domain",
            "vocal": True,
        },
        {
            "url": "https://archive.org/download/78_scherazade-op35-pt3_royal-opera-house-orchestra-covent-garden/C_1288_CR_563-4.flac",
            "era_year": 1925,
            "genre": "classical",
            "defect_types": ["surface_noise", "wow"],
            "attribution": "Royal Opera House Orchestra — Rimski-Korsakow: Scheherazade op. 35, Teil 3 (Internet Archive, 78rpm Collection)",
            "license": "Public Domain",
            "vocal": False,
        },
    ],
    # Hinweis: Für Vinyl, Tape, Kassette und Digital existieren faktisch keine
    # US-gemeinfreien Aufnahmen (Formate nach 1930). Diese Kategorien werden mit
    # selbst generierten CC0-Referenzen abgedeckt (scripts/generate_corpus.py).
    "vinyl": [],
    "tape": [],
}

# ── Minimum required by corpus quality gate ─────────────────────────────────
MIN_FILES_PER_MATERIAL = 5
MIN_TOTAL_FILES = 20
MIN_MATERIALS = 4


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _duration_s(filepath: Path) -> float:
    """Get audio duration in seconds using ffprobe or librosa."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(filepath),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(result.stdout.strip())
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
    try:
        import librosa

        return float(librosa.get_duration(path=str(filepath)))
    except Exception:
        logger.warning("Cannot determine duration for %s, defaulting to 0.0", filepath)
        return 0.0


def _sample_rate(filepath: Path) -> int:
    """Get audio sample rate."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "stream=sample_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(filepath),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return int(result.stdout.strip().split("\n")[0])
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
        logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
    try:
        import librosa

        return int(librosa.get_samplerate(path=str(filepath)))
    except Exception:
        return 0


def download_file(url: str, dest: Path, timeout: int = 120) -> bool:
    """Download a file from url to dest. Returns True on success."""
    if dest.exists():
        logger.info("Already exists: %s", dest)
        return True

    _ensure_dir(dest.parent)

    # Try yt-dlp first (handles Internet Archive, YouTube, etc.)
    try:
        subprocess.run(
            ["yt-dlp", "--no-playlist", "-o", str(dest), url],
            check=True,
            timeout=timeout,
            capture_output=True,
        )
        if dest.exists() and dest.stat().st_size > 0:
            return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

    # Fallback: requests
    try:
        import requests

        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        return dest.stat().st_size > 0
    except Exception as e:
        logger.error("Download fehlgeschlagen for %s: %s", url, e)
        return False


def generate_from_sources(
    material: str | None = None,
    count: int = 5,
    dry_run: bool = False,
) -> dict[str, list[Path]]:
    """Download public-domain files and return paths grouped by material."""
    downloaded: dict[str, list[Path]] = {}

    materials = [material] if material else list(PUBLIC_DOMAIN_SOURCES)
    for mat in materials:
        sources = PUBLIC_DOMAIN_SOURCES.get(mat, [])
        dest_dir = CORPUS_ROOT / mat / "damaged"
        mat_downloaded: list[Path] = []

        for src in sources[:count]:
            url = src["url"]
            filename = Path(urlparse(url).path).name or f"{mat}_{src['era_year']}.wav"
            dest = dest_dir / filename

            if dry_run:
                logger.info("DRY-Ausfuehrung would download: %s → %s", url, dest)
                mat_downloaded.append(dest)
                continue

            if download_file(url, dest):
                mat_downloaded.append(dest)
                logger.info("Downloaded: %s", dest)

        downloaded[mat] = mat_downloaded

    return downloaded


def build_manifest(
    material: str,
    audio_files: list[Path],
    source_meta: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a manifest.yaml dict for a material category."""
    entries: list[dict[str, Any]] = []
    meta_by_name: dict[str, dict[str, Any]] = {}
    if source_meta:
        for sm in source_meta:
            filename = Path(urlparse(sm["url"]).path).name
            meta_by_name[filename] = sm

    for af in sorted(audio_files):
        if not af.exists():
            continue
        meta = meta_by_name.get(af.name, {})
        entry: dict[str, Any] = {
            "file": str(af.relative_to(CORPUS_ROOT / material)),
            "duration_s": round(_duration_s(af), 3),
            "sample_rate": _sample_rate(af),
            "material": meta.get("material", material),
            "era_year": meta.get("era_year", 0),
            "genre": meta.get("genre", "unknown"),
            "condition": "damaged",
            "channels": 2,
            "bit_depth": 16,
        }
        if meta.get("defect_types"):
            entry["defect_types"] = meta["defect_types"]
        if meta.get("attribution"):
            entry["source_attribution"] = meta["attribution"]
        if meta.get("license"):
            entry["license"] = meta["license"]
        if meta.get("url"):
            entry["source_url"] = meta["url"]
        if meta.get("vocal") is not None:
            entry["vocal"] = meta["vocal"]
        if af.exists():
            entry["checksum_sha256"] = _sha256(af)

        entries.append(entry)

    return {
        "corpus_version": "1.0.0",
        "material": material,
        "description": f"Public-Domain {material} recordings for Aurik real-audio validation",
        "entries": entries,
    }


def write_manifests(downloaded: dict[str, list[Path]], dry_run: bool = False) -> list[Path]:
    """Write manifest.yaml for each material category. Returns list of written manifests."""
    written: list[Path] = []
    for material, files in downloaded.items():
        manifest_path = CORPUS_ROOT / material / "manifest.yaml"
        manifest = build_manifest(material, files)

        if dry_run:
            logger.info("DRY-Ausfuehrung would write: %s", manifest_path)
            written.append(manifest_path)
            continue

        _ensure_dir(manifest_path.parent)
        manifest_path.write_text(
            yaml.dump(manifest, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        written.append(manifest_path)
        logger.info("Wrote manifest: %s (%d entries)", manifest_path, len(manifest["entries"]))

    return written


def import_from_urls_file(urls_file: Path, material: str, dry_run: bool = False) -> list[Path]:
    """Import audio from a text file containing one URL per line."""
    if not urls_file.exists():
        logger.error("URLs file not found: %s", urls_file)
        return []

    urls = [line.strip() for line in urls_file.read_text().splitlines() if line.strip() and not line.startswith("#")]
    dest_dir = CORPUS_ROOT / material / "damaged"
    downloaded: list[Path] = []

    for url in urls:
        filename = Path(urlparse(url).path).name or f"import_{len(downloaded)}.wav"
        dest = dest_dir / filename
        if dry_run:
            logger.info("DRY-Ausfuehrung would download: %s → %s", url, dest)
            downloaded.append(dest)
        elif download_file(url, dest):
            downloaded.append(dest)

    return downloaded


def check_coverage() -> dict[str, Any]:
    """Check current corpus coverage against minimum requirements."""
    materials_found: list[str] = []
    total_files = 0
    per_material: dict[str, int] = {}

    for mat_dir in sorted(CORPUS_ROOT.iterdir()):
        if not mat_dir.is_dir() or mat_dir.name.startswith("."):
            continue
        manifest = mat_dir / "manifest.yaml"
        if not manifest.exists():
            continue
        try:
            data = yaml.safe_load(manifest.read_text())
            count = len(data.get("entries", []))
            per_material[mat_dir.name] = count
            total_files += count
            materials_found.append(mat_dir.name)
        except Exception:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
            continue

    return {
        "materials_found": materials_found,
        "materials_required": MIN_MATERIALS,
        "total_files": total_files,
        "total_required": MIN_TOTAL_FILES,
        "per_material": per_material,
        "meets_minimum": (len(materials_found) >= MIN_MATERIALS and total_files >= MIN_TOTAL_FILES),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", choices=list(PUBLIC_DOMAIN_SOURCES), help="Nur ein Material herunterladen")
    parser.add_argument("--all", action="store_true", help="Alle Materialien")
    parser.add_argument("--count", type=int, default=5, help="Max Dateien pro Material")
    parser.add_argument("--dry-run", action="store_true", help="Nur anzeigen, nichts herunterladen")
    parser.add_argument("--from-urls", type=Path, help="URLs aus Textdatei importieren")
    parser.add_argument("--check", action="store_true", help="Nur Coverage prüfen")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.check:
        coverage = check_coverage()
        print(json.dumps(coverage, indent=2, ensure_ascii=False))
        return 0 if coverage["meets_minimum"] else 1

    if args.from_urls:
        if not args.material:
            print("ERROR: --material required with --from-urls", file=sys.stderr)
            return 1
        downloaded = import_from_urls_file(args.from_urls, args.material, args.dry_run)
        grouped = {args.material: downloaded}
    elif args.material or args.all:
        grouped = generate_from_sources(
            material=args.material if not args.all else None,
            count=args.count,
            dry_run=args.dry_run,
        )
    else:
        print("ERROR: use --all, --material X, --from-urls, or --check", file=sys.stderr)
        return 1

    manifests = write_manifests(grouped, args.dry_run)

    if args.dry_run:
        print(f"DRY-RUN: would write {len(manifests)} manifests for {sum(len(v) for v in grouped.values())} files")
    else:
        coverage = check_coverage()
        print(json.dumps(coverage, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
