"""§2.44 HPG Reference-Memory Bootstrap-Skript.

Seedet das HPG Reference-Memory (~/.aurik/hpg_reference_memory.json) mit
Embeddings aus den Golden-Samples in golden_samples/references/, damit das
5-Stufen-Fallback-System von _get_reference_vector() bereits beim ersten
echten Lauf funktionstüchtig ist (statt immer None zurückzugeben).

Aufruf:
    python scripts/bootstrap_hpg_reference_memory.py

Ausführung vor dem ersten Produktivlauf oder nach dem Löschen der Referenz-Memory-Datei.
Bereits vorhandene Einträge werden via EMA (α=0.15) geblended — kein Datenverlust.
"""

from __future__ import annotations

import logging
import pathlib
import sys

# Workspace-Root in sys.path aufnehmen
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("bootstrap_hpg_ref_memory")

import numpy as np

# Genre-Erkennung aus Dateiname (Präfix vor erstem Unterstrich)
_GENRE_FROM_PREFIX: dict[str, str] = {
    "vocal": "vocal",
    "classical": "classical",
    "jazz": "jazz",
    "instrumental": "instrumental",
    "pop": "pop",
    "rock": "rock",
    "blues": "blues",
    "folk": "folk",
    "schlager": "schlager",
    "country": "country",
    "latin": "latin",
}

# UV3-konforme Ära-Labels (EXACT: unified_restorer_v3.py _hpi_era computation)
# "pre-1950" | "pre-1980" | "post-1980"  — nicht "1960-1990" oder "post-1990" etc.
_ERA_PRE_1950 = "pre-1950"
_ERA_PRE_1980 = "pre-1980"
_ERA_POST_1980 = "post-1980"

# UV3-konforme Material-Keys (MaterialType.value-Strings)
_MAT_DIGITAL = "digital"
_MAT_CD = "cd_digital"
_MAT_VINYL = "vinyl"
_MAT_SHELLAC = "shellac"
_MAT_TAPE = "tape"
_MAT_CASSETTE = "cassette"
_MAT_REEL = "reel_tape"
_MAT_MP3L = "mp3_low"
_MAT_MP3H = "mp3_high"
_MAT_STREAM = "streaming"
_MAT_UNKNOWN = "unknown"

# v10.0.0: Ungültige Ära-Formate alter Bootstrap-Versionen (werden aus Memory entfernt)
_STALE_ERA_FORMATS: frozenset[str] = frozenset(
    {
        "1960-1990",
        "post-1990",
        "pre-1960",
        "pre-1970",
        "1900-1950",
        "1950-1990",
        "1990-2020",
    }
)

# Alle Golden-Reference-Samples: digitale Referenzen → digital/post-1980
_DEFAULT_MATERIAL = _MAT_DIGITAL
_DEFAULT_ERA_BIN = _ERA_POST_1980

# Bootstrap-HPI-Werte: Referenzdaten gelten als qualitativ hochwertig
_BOOTSTRAP_HPI = 0.92
_BOOTSTRAP_AF = 0.97
_BOOTSTRAP_P1P2 = True

# Synthetische Prototypen: (genre, material, era_bin, freq_hz, noise_sigma)
# Decken alle gängigen Träger-Ären ab, damit _get_reference_vector() Stufe-1/2/3
# immer eine gültige Antwort liefert ohne echte Audiodateien zu benötigen.
_SYNTHETIC_PROTOTYPES: list[tuple[str, str, str, float, float]] = [
    # --- Shellac / Acoustic (pre-1950) ---
    ("vocal", _MAT_SHELLAC, _ERA_PRE_1950, 185.0, 0.08),
    ("jazz", _MAT_SHELLAC, _ERA_PRE_1950, 220.0, 0.07),
    ("blues", _MAT_SHELLAC, _ERA_PRE_1950, 165.0, 0.09),
    ("classical", _MAT_SHELLAC, _ERA_PRE_1950, 261.6, 0.06),
    ("folk", _MAT_SHELLAC, _ERA_PRE_1950, 196.0, 0.08),
    ("general", _MAT_SHELLAC, _ERA_PRE_1950, 200.0, 0.08),
    ("general", _MAT_UNKNOWN, _ERA_PRE_1950, 180.0, 0.09),
    # --- Vinyl / Reel-Tape (pre-1980) ---
    ("vocal", _MAT_VINYL, _ERA_PRE_1980, 220.0, 0.03),
    ("pop", _MAT_VINYL, _ERA_PRE_1980, 261.6, 0.03),
    ("schlager", _MAT_VINYL, _ERA_PRE_1980, 246.9, 0.03),
    ("rock", _MAT_VINYL, _ERA_PRE_1980, 196.0, 0.04),
    ("jazz", _MAT_VINYL, _ERA_PRE_1980, 233.1, 0.03),
    ("classical", _MAT_VINYL, _ERA_PRE_1980, 293.7, 0.02),
    ("blues", _MAT_VINYL, _ERA_PRE_1980, 185.0, 0.04),
    ("folk", _MAT_VINYL, _ERA_PRE_1980, 207.7, 0.03),
    ("country", _MAT_VINYL, _ERA_PRE_1980, 220.0, 0.03),
    ("general", _MAT_VINYL, _ERA_PRE_1980, 220.0, 0.03),
    ("vocal", _MAT_REEL, _ERA_PRE_1980, 220.0, 0.02),
    ("pop", _MAT_REEL, _ERA_PRE_1980, 261.6, 0.02),
    ("general", _MAT_REEL, _ERA_PRE_1980, 220.0, 0.02),
    ("general", _MAT_TAPE, _ERA_PRE_1980, 210.0, 0.03),
    ("general", _MAT_UNKNOWN, _ERA_PRE_1980, 200.0, 0.05),
    # --- Vinyl / Cassette / CD (post-1980) ---
    ("vocal", _MAT_VINYL, _ERA_POST_1980, 220.0, 0.02),
    ("pop", _MAT_VINYL, _ERA_POST_1980, 261.6, 0.02),
    ("schlager", _MAT_VINYL, _ERA_POST_1980, 246.9, 0.02),
    ("rock", _MAT_VINYL, _ERA_POST_1980, 196.0, 0.03),
    ("vocal", _MAT_CASSETTE, _ERA_POST_1980, 220.0, 0.025),
    ("pop", _MAT_CASSETTE, _ERA_POST_1980, 261.6, 0.025),
    ("rock", _MAT_CASSETTE, _ERA_POST_1980, 196.0, 0.03),
    ("general", _MAT_CASSETTE, _ERA_POST_1980, 220.0, 0.025),
    ("vocal", _MAT_CD, _ERA_POST_1980, 220.0, 0.005),
    ("pop", _MAT_CD, _ERA_POST_1980, 261.6, 0.005),
    ("classical", _MAT_CD, _ERA_POST_1980, 293.7, 0.003),
    ("general", _MAT_CD, _ERA_POST_1980, 220.0, 0.005),
    # --- Digital / Streaming (post-1980) ---
    ("vocal", _MAT_DIGITAL, _ERA_POST_1980, 220.0, 0.002),
    ("pop", _MAT_DIGITAL, _ERA_POST_1980, 261.6, 0.002),
    ("rock", _MAT_DIGITAL, _ERA_POST_1980, 196.0, 0.003),
    ("classical", _MAT_DIGITAL, _ERA_POST_1980, 293.7, 0.001),
    ("jazz", _MAT_DIGITAL, _ERA_POST_1980, 233.1, 0.002),
    ("general", _MAT_DIGITAL, _ERA_POST_1980, 220.0, 0.002),
    ("vocal", _MAT_MP3L, _ERA_POST_1980, 220.0, 0.012),
    ("pop", _MAT_MP3L, _ERA_POST_1980, 261.6, 0.012),
    ("general", _MAT_MP3L, _ERA_POST_1980, 220.0, 0.012),
    ("vocal", _MAT_MP3H, _ERA_POST_1980, 220.0, 0.005),
    ("pop", _MAT_MP3H, _ERA_POST_1980, 261.6, 0.005),
    ("general", _MAT_STREAM, _ERA_POST_1980, 220.0, 0.003),
    ("general", _MAT_UNKNOWN, _ERA_POST_1980, 220.0, 0.030),
]


def _generate_synthetic_audio(sr: int, freq_hz: float, noise_sigma: float, duration_s: float = 3.0) -> np.ndarray:
    """Erzeugt synthetisches harmonisches Audio als Material-Prototyp.

    Simuliert den Klang-Charakter eines Trägers:
    - freq_hz: Grundton (Genre-charakteristisch)
    - noise_sigma: Material-typisches Rauschen (Shellac > Vinyl > CD > Digital)
    Harmonische Obertöne H2–H4 für realistisches Mel-Spektrum-Embedding.
    """
    n = int(sr * duration_s)
    t = np.linspace(0.0, duration_s, n, dtype=np.float32)
    rng = np.random.default_rng(int(freq_hz * 100 + sr))
    # Grundton + Obertöne
    audio = (
        0.40 * np.sin(2.0 * np.pi * freq_hz * t)
        + 0.20 * np.sin(2.0 * np.pi * freq_hz * 2.0 * t)
        + 0.10 * np.sin(2.0 * np.pi * freq_hz * 3.0 * t)
        + 0.05 * np.sin(2.0 * np.pi * freq_hz * 4.0 * t)
    )
    if noise_sigma > 0.0:
        audio = audio + (noise_sigma * rng.standard_normal(n)).astype(np.float32)
    peak = float(np.max(np.abs(audio))) + 1e-9
    return np.clip(audio / peak * 0.9, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]


def _cleanup_stale_entries(gate: object) -> int:  # type: ignore[type-arg]
    """v10.0.0: Entfernt Einträge mit falschen Ära-Formaten aus dem Reference-Memory.

    Stale-Formate: '1960-1990', 'post-1990', 'pre-1960' etc. (alter Bootstrap).
    UV3 verwendet ausschließlich: 'pre-1950' | 'pre-1980' | 'post-1980'.
    """
    import threading

    # Direkter Zugriff auf _ref_memory und _ref_lock (interne HPG-Struktur)
    ref_memory = getattr(gate, "_ref_memory", {})
    ref_lock = getattr(gate, "_ref_lock", threading.Lock())
    stale_keys: list = []
    with ref_lock:
        for key in list(ref_memory.keys()):
            era_part = key[2] if isinstance(key, tuple) and len(key) == 3 else ""
            if era_part in _STALE_ERA_FORMATS:
                stale_keys.append(key)
        for key in stale_keys:
            del ref_memory[key]
    if stale_keys:
        logger.info("v10.0.0 Stale-Cleanup: %d veraltete Einträge entfernt: %s", len(stale_keys), stale_keys)
        # Persist bereinigtes Memory
        try:
            _save = getattr(gate, "_save_ref_memory_to_disk", None)
            if callable(_save):
                _save()
        except Exception as _e:
            logger.debug("Stale-Cleanup Persist nicht blockierend: %s", _e)
    return len(stale_keys)


def _seed_synthetic_prototypes(gate: object) -> int:  # type: ignore[type-arg]
    """Seedet synthetische Prototyp-Embeddings für alle Material×Ära-Kombinationen.

    Verwendet _generate_synthetic_audio() + gate._compute_embedding() direkt,
    ohne HPG-Qualitäts-Gate — Bootstrap-Daten sind per Konvention valide.
    Gibt die Anzahl neu geseedeter Einträge zurück.
    """
    seeded = 0
    sr = 48000
    compute_embed = getattr(gate, "_compute_embedding", None)
    ref_memory = getattr(gate, "_ref_memory", {})
    ref_lock = getattr(gate, "__dict__", {}).get("_ref_lock", None)
    if ref_lock is None:
        import threading

        ref_lock = threading.Lock()

    # Importiere _RefEntry-Klasse
    try:
        from backend.core.holistic_perceptual_gate import _RefEntry  # type: ignore[attr-defined]
    except ImportError:
        logger.warning("_RefEntry nicht importierbar — synthetische Prototypen übersprungen.")
        return 0

    for genre, material, era_bin, freq_hz, noise_sigma in _SYNTHETIC_PROTOTYPES:
        key = (genre, material, era_bin)
        if key in ref_memory:
            # EMA-Blend mit vorhandenem Eintrag
            try:
                audio = _generate_synthetic_audio(sr, freq_hz, noise_sigma)
                if compute_embed is not None:
                    embed = compute_embed(audio, sr)
                    _EMA = 0.15
                    with ref_lock:
                        entry = ref_memory[key]
                        entry.embedding = (1.0 - _EMA) * entry.embedding + _EMA * embed
                        entry.obs_count += 1
            except Exception as _exc:
                logger.debug("EMA-Blend key=%s: %s", key, _exc)
            continue
        # Neuen Eintrag anlegen
        try:
            audio = _generate_synthetic_audio(sr, freq_hz, noise_sigma)
            if compute_embed is None:
                logger.warning("_berechnen_embedding nicht verfügbar — Gate-Ersatzpfad.")
                break
            embed = compute_embed(audio, sr)
            with ref_lock:
                ref_memory[key] = _RefEntry(
                    embedding=embed.copy(),
                    obs_count=1,
                    calibrated=False,
                )
            seeded += 1
            logger.debug("  ✓ Synth. Prototyp: genre=%s mat=%s era=%s", genre, material, era_bin)
        except Exception as exc:
            logger.debug("  ✗ Synth. Prototyp key=%s: %s", key, exc)

    if seeded > 0:
        try:
            _save = getattr(gate, "_save_ref_memory_to_disk", None)
            if callable(_save):
                _save()
        except Exception as _e:
            logger.debug("Synth-Persist nicht blockierend: %s", _e)
    return seeded


def _detect_genre(filename: str) -> str:
    stem = pathlib.Path(filename).stem.lower()
    prefix = stem.split("_")[0]
    return _GENRE_FROM_PREFIX.get(prefix, "general")


def run_bootstrap(references_dir: pathlib.Path) -> int:
    """Verarbeitet alle Audiodateien in references_dir und seedet die Reference-Memory.

    Gibt die Anzahl erfolgreich geseedeter Einträge zurück.
    """
    try:
        from backend.core.holistic_perceptual_gate import get_holistic_gate
    except ImportError as exc:
        logger.error("HPG-Import fehlgeschlagen — PYTHONPATH korrekt? %s", exc)
        return 0

    try:
        from backend.file_import import load_audio_file
    except ImportError:
        try:
            import soundfile as sf

            def load_audio_file(path: str) -> tuple:  # type: ignore[misc]
                _max_load_retries = 3
                for _load_attempt in range(_max_load_retries):
                    try:
                        audio, sr = sf.read(path, always_2d=False)
                        break
                    except Exception:
                        if _load_attempt < _max_load_retries - 1:
                            logger.debug(
                                "soundfile.read Wiederholung %d/%d", _load_attempt + 1, _max_load_retries, exc_info=True
                            )
                            continue
                        raise
                import numpy as np

                return np.asarray(audio, dtype=np.float32), int(sr)
        except ImportError:
            logger.error("Weder backend.file_import noch soundfile verfügbar.")
            return 0

    gate = get_holistic_gate()
    seeded = 0

    audio_files = sorted(references_dir.glob("*.wav")) + sorted(references_dir.glob("*.flac"))
    if not audio_files:
        logger.warning("Keine Audiodateien in %s gefunden.", references_dir)
        return 0

    for audio_path in audio_files:
        genre = _detect_genre(audio_path.name)
        try:
            _result = load_audio_file(str(audio_path))
            if not isinstance(_result, dict) or _result.get("error"):
                logger.warning("  ✗ %s: laden_audio_file Fehler: %s", audio_path.name, (_result or {}).get("error"))
                continue
            import numpy as np

            audio = np.asarray(_result["audio"], dtype=np.float32)
            sr = int(_result["sr"])
            gate.update_reference_memory(
                restored=audio,
                sr=sr,
                hpi=_BOOTSTRAP_HPI,
                artifact_freedom=_BOOTSTRAP_AF,
                p1_p2_passed=_BOOTSTRAP_P1P2,
                genre=genre,
                material=_DEFAULT_MATERIAL,
                era_bin=_DEFAULT_ERA_BIN,
            )
            logger.info(
                "  ✓ %s → genre=%s material=%s era=%s", audio_path.name, genre, _DEFAULT_MATERIAL, _DEFAULT_ERA_BIN
            )
            seeded += 1
        except Exception as exc:
            logger.warning("  ✗ %s: %s", audio_path.name, exc)

    logger.info("Bootstrap (Golden-Samples) abgeschlossen: %d Einträge geseedet.", seeded)
    return seeded


def main() -> None:
    try:
        from backend.core.holistic_perceptual_gate import get_holistic_gate
    except ImportError as exc:
        logger.error("HPG-Import fehlgeschlagen: %s", exc)
        sys.exit(1)

    gate = get_holistic_gate()
    total = 0

    # v10.0.0 Phase 0: Stale-Entry-Cleanup (falsche Ära-Formate)
    logger.info("§2.44 v10.0.0 Verarbeitungsschritt 0: Stale-Entry-Cleanup")
    removed = _cleanup_stale_entries(gate)
    logger.info("  %d veraltete Einträge bereinigt.", removed)

    # Phase 1: Synthetische Prototypen für alle Material×Ära-Kombinationen
    logger.info(
        "§2.44 v10.0.0 Verarbeitungsschritt 1: Synthetische Prototyp-Embeddings (%d Kombinationen)",
        len(_SYNTHETIC_PROTOTYPES),
    )
    synth_count = _seed_synthetic_prototypes(gate)
    total += synth_count
    logger.info("  %d synthetische Prototypen geseedet.", synth_count)

    # Phase 2: Golden-Samples (echte digitale Referenz-Audio)
    references_dir = _REPO_ROOT / "golden_samples" / "references"
    if references_dir.exists():
        logger.info("§2.44 v10.0.0 Verarbeitungsschritt 2: Golden-Samples aus %s", references_dir)
        audio_count = run_bootstrap(references_dir)
        total += audio_count
        logger.info("  %d Golden-Sample-Einträge geseedet.", audio_count)
    else:
        logger.info("Golden-Samples-Verzeichnis nicht gefunden — Verarbeitungsschritt 2 übersprungen.")

    logger.info(
        "§2.44 HPG Bootstrap v10.0.0 fertig: %d neue Embeddings gespeichert in ~/.aurik/hpg_Referenz_memory.json",
        total,
    )


if __name__ == "__main__":
    main()
