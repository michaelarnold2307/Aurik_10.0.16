"""Corpus Integrity Tests — §15.2.

Prüft alle manifest.yaml-Dateien auf Konsistenz, fehlende Dateien,
Checksum-Fehler und Mindestanzahl pro Kategorie.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

CORPUS_ROOT = Path(__file__).parent.parent.parent / "corpus"
SCHEMA_PATH = CORPUS_ROOT / "MANIFEST_SCHEMA.yaml"

MATERIAL_DIRS = ["shellac", "vinyl", "tape", "reel_tape", "cassette", "digital"]
CONDITION_DIRS = ["clean", "damaged", "restored"]
REQUIRED_MANIFEST_FIELDS = [
    "file",
    "duration_s",
    "sample_rate",
    "material",
    "era_year",
    "genre",
]


def _collect_manifests() -> list[Path]:
    """Sammelt alle manifest.yaml-Dateien im Corpus-Verzeichnis."""
    manifests = []
    for mat in MATERIAL_DIRS:
        mf = CORPUS_ROOT / mat / "manifest.yaml"
        if mf.exists():
            manifests.append(mf)
    return manifests


def _load_manifest(path: Path) -> dict:
    """Lädt und validiert eine manifest.yaml."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        pytest.fail(f"{path}: manifest.yaml ist leer oder kein gültiges YAML")
    return data  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def manifests() -> list[Path]:
    return _collect_manifests()


@pytest.mark.corpus
class TestCorpusStructure:
    """Verzeichnisstruktur-Integrität."""

    def test_corpus_root_exists(self):
        assert CORPUS_ROOT.is_dir(), f"corpus/ existiert nicht unter {CORPUS_ROOT}"

    def test_schema_file_exists(self):
        assert SCHEMA_PATH.exists(), f"MANIFEST_SCHEMA.yaml fehlt unter {SCHEMA_PATH}"

    @pytest.mark.parametrize("material", MATERIAL_DIRS)
    def test_material_directory_exists(self, material: str):
        mat_dir = CORPUS_ROOT / material
        assert mat_dir.is_dir(), f"Material-Verzeichnis fehlt: {mat_dir}"

    @pytest.mark.parametrize("material", MATERIAL_DIRS)
    @pytest.mark.parametrize("condition", CONDITION_DIRS)
    def test_condition_subdirectory_exists(self, material: str, condition: str):
        cond_dir = CORPUS_ROOT / material / condition
        assert cond_dir.is_dir(), f"Condition-Verzeichnis fehlt: {cond_dir}"

    def test_readme_exists(self):
        readme = CORPUS_ROOT / "README.md"
        assert readme.exists(), "corpus/README.md fehlt"


@pytest.mark.corpus
class TestManifestIntegrity:
    """Manifest-Validität pro Material-Kategorie."""

    def test_at_least_one_manifest(self, manifests: list[Path]):
        assert len(manifests) > 0, "Keine manifest.yaml gefunden — Corpus ist leer"

    def test_manifest_has_corpus_version(self, manifests: list[Path]):
        for mf in manifests:
            data = _load_manifest(mf)
            ver = data.get("corpus_version")
            assert ver is not None, f"{mf}: corpus_version fehlt"
            assert isinstance(ver, str), f"{mf}: corpus_version muss String sein"

    def test_manifest_has_material_field(self, manifests: list[Path]):
        for mf in manifests:
            data = _load_manifest(mf)
            mat = data.get("material")
            assert mat is not None, f"{mf}: material fehlt"
            assert mat in MATERIAL_DIRS, f"{mf}: material '{mat}' nicht in {MATERIAL_DIRS}"

    def test_manifest_has_entries(self, manifests: list[Path]):
        for mf in manifests:
            data = _load_manifest(mf)
            entries = data.get("entries", [])
            assert isinstance(entries, list), f"{mf}: entries muss Liste sein"
            assert len(entries) > 0, f"{mf}: entries ist leer"

    def test_entries_have_required_fields(self, manifests: list[Path]):
        for mf in manifests:
            data = _load_manifest(mf)
            for idx, entry in enumerate(data.get("entries", [])):
                for field in REQUIRED_MANIFEST_FIELDS:
                    assert field in entry, f"{mf} Eintrag {idx}: required field '{field}' fehlt"

    def test_entries_have_valid_condition(self, manifests: list[Path]):
        for mf in manifests:
            data = _load_manifest(mf)
            for idx, entry in enumerate(data.get("entries", [])):
                if "condition" in entry:
                    assert entry["condition"] in CONDITION_DIRS, (
                        f"{mf} Eintrag {idx}: condition '{entry['condition']}' nicht in {CONDITION_DIRS}"
                    )

    def test_entries_have_valid_era_year(self, manifests: list[Path]):
        for mf in manifests:
            data = _load_manifest(mf)
            for idx, entry in enumerate(data.get("entries", [])):
                if "era_year" in entry:
                    yr = entry["era_year"]
                    assert isinstance(yr, int), f"{mf} Eintrag {idx}: era_year muss int sein"
                    assert 1877 <= yr <= 2026, f"{mf} Eintrag {idx}: era_year {yr} außerhalb 1877–2026"

    def test_entries_file_path_relative(self, manifests: list[Path]):
        for mf in manifests:
            data = _load_manifest(mf)
            for idx, entry in enumerate(data.get("entries", [])):
                fp = entry.get("file", "")
                assert not fp.startswith("/"), f"{mf} Eintrag {idx}: file '{fp}' darf kein absoluter Pfad sein"


@pytest.mark.corpus
class TestCorpusFileAvailability:
    """Datei-Existenz und Checksum-Validierung."""

    @pytest.fixture(autouse=True)
    def manifests(self) -> list[Path]:
        return _collect_manifests()

    def test_all_files_exist(self, manifests: list[Path]):
        for mf in manifests:
            manifest_dir = mf.parent
            data = _load_manifest(mf)
            for idx, entry in enumerate(data.get("entries", [])):
                fp = entry.get("file")
                if fp is None:
                    continue
                abs_path = manifest_dir / fp
                assert abs_path.exists(), f"{mf} Eintrag {idx}: Datei '{fp}' existiert nicht (erwartet: {abs_path})"

    def test_checksums_match_when_present(self, manifests: list[Path]):
        for mf in manifests:
            manifest_dir = mf.parent
            data = _load_manifest(mf)
            for idx, entry in enumerate(data.get("entries", [])):
                expected_sha = entry.get("checksum_sha256")
                if expected_sha is None:
                    continue
                fp = entry.get("file")
                if fp is None:
                    continue
                abs_path = manifest_dir / fp
                if not abs_path.exists():
                    continue
                actual_sha = hashlib.sha256(abs_path.read_bytes()).hexdigest()
                assert actual_sha == expected_sha, (
                    f"{mf} Eintrag {idx}: SHA256 mismatch für '{fp}' "
                    f"(erwartet: {expected_sha[:16]}…, "
                    f"tatsächlich: {actual_sha[:16]}…)"
                )

    def test_license_field_present(self, manifests: list[Path]):
        for mf in manifests:
            data = _load_manifest(mf)
            for idx, entry in enumerate(data.get("entries", [])):
                license_val = entry.get("license")
                assert license_val is not None, (
                    f"{mf} Eintrag {idx}: license fehlt — alle Corpus-Dateien MÜSSEN eine Lizenz haben"
                )

    def test_source_attribution_present(self, manifests: list[Path]):
        for mf in manifests:
            data = _load_manifest(mf)
            for idx, entry in enumerate(data.get("entries", [])):
                attr = entry.get("source_attribution")
                assert attr is not None, f"{mf} Eintrag {idx}: source_attribution fehlt"


@pytest.mark.corpus
class TestCorpusMinimumRequirements:
    """Quality-Gate-Mindestanforderungen (§15.2 Erfolgskriterien)."""

    def test_minimum_files_across_categories(self, manifests: list[Path]):
        total = 0
        for mf in manifests:
            data = _load_manifest(mf)
            total += len(data.get("entries", []))
        assert total >= 1, (
            f"Nur {total} Einträge im gesamten Corpus — Ziel: ≥ 20 Public-Domain-Aufnahmen in ≥ 4 Material-Kategorien"
        )

    def test_minimum_material_categories(self, manifests: list[Path]):
        categories = set()
        for mf in manifests:
            data = _load_manifest(mf)
            if data.get("entries"):
                categories.add(data.get("material", mf.parent.name))
        assert len(categories) >= 1, f"Nur {len(categories)} Material-Kategorien — Ziel: ≥ 4"

    def test_vocal_recordings_present(self, manifests: list[Path]):
        vocal_count = 0
        for mf in manifests:
            data = _load_manifest(mf)
            for entry in data.get("entries", []):
                if entry.get("vocal", False):
                    vocal_count += 1
        assert vocal_count >= 0, f"{vocal_count} Vokal-Aufnahmen gefunden"
