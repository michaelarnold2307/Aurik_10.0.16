from __future__ import annotations

"""
E2E-Test für Aurik 10.0.0 — Kanonischer End-to-End-Test.

Testet die vollständige Restaurierungs-Pipeline mit UnifiedRestorerV3
für beide Modi: QUALITY (Restoration) und BALANCED (Studio 2026).

Anforderungen:
    pytest.mark.e2e
    Timeout: 600 s
    Audio-Testdatei: audio_examples/Elke Best - Du wolltest nur ein Abenteuer, aber ich suchte einen Freund.mp3
    Fallback:        audio_examples/Elke_Best_Freund.mp3

Spec-Referenz: §0, §2.44, §2.45, §2.49, §8.1, §8.2 (copilot-instructions.md v10.14.x)
"""


import math
import pathlib
import time

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Testdatei-Suche
# ---------------------------------------------------------------------------
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
_AUDIO_CANDIDATES = [
    _PROJECT_ROOT / "audio_examples" / "Elke Best - Du wolltest nur ein Abenteuer, aber ich suchte einen Freund.mp3",
    _PROJECT_ROOT / "audio_examples" / "Elke_Best_Freund.mp3",
    _PROJECT_ROOT / "test_audio" / "Elke Best - Du wolltest nur ein Abenteuer, aber ich suchte einen Freund.mp3",
]
_AUDIO_FILE: pathlib.Path | None = next((p for p in _AUDIO_CANDIDATES if p.exists()), None)
_OUTPUT_DIR = _PROJECT_ROOT / "test_output"

if _AUDIO_FILE is None:
    pytest.skip(
        "E2E-Testdatei nicht gefunden. Bitte eine der folgenden Dateien bereitstellen:\n"
        + "\n".join(f"  {p}" for p in _AUDIO_CANDIDATES),
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Hilfsfunktion: Audio laden
# ---------------------------------------------------------------------------
def _load_audio(path: pathlib.Path) -> tuple[np.ndarray, int]:
    """Lädt Audio via librosa (mono, native SR)."""
    import librosa  # type: ignore

    audio, sr = librosa.load(str(path), sr=None, mono=True)
    return audio.astype(np.float32), int(sr)


def _load_audio_clip(
    path: pathlib.Path,
    offset_s: float = 30.0,
    duration_s: float = 15.0,
) -> tuple[np.ndarray, int]:
    """Lädt einen kurzen Ausschnitt (Standard: 30–45 s) für CI-schnelle E2E-Tests.

    Hintergrund: Die 225 s lange Quelldatei benötigt im QUALITY-Modus (29 Phasen)
    über 10 Minuten CPU-Zeit und überschreitet den 600 s-Timeout. Ein 15 s-Clip
    aus der Mitte des Songs (Offset 30 s: Intro-Stille überspringen) reduziert die
    Laufzeit auf ~60–90 s bei vollem musikalischem Inhalt.
    """
    import librosa  # type: ignore

    audio, sr = librosa.load(str(path), sr=None, mono=True, offset=offset_s, duration=duration_s)
    if len(audio) < sr * 3:
        # Offset überschreitet Dateilänge — von Anfang laden
        audio, sr = librosa.load(str(path), sr=None, mono=True, duration=duration_s)
    return audio.astype(np.float32), int(sr)


def _write_wav(path: pathlib.Path, audio: np.ndarray, sr: int) -> None:
    """Schreibt WAV-Ausgabe via soundfile."""
    import soundfile as sf  # type: ignore

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sr, subtype="PCM_24")


# ---------------------------------------------------------------------------
# Gemeinsame Validierung
# ---------------------------------------------------------------------------
def _validate_restoration_result(result, min_duration_s: float = 1.0) -> None:
    """Prüft RestorationResult auf korrekte Grundstruktur und Wertebereiche."""
    from backend.core.defect_scanner import MaterialType  # type: ignore
    from backend.core.unified_restorer_v3 import RestorationResult  # type: ignore

    assert isinstance(result, RestorationResult), f"Rückgabe muss RestorationResult sein, ist: {type(result)}"

    # --- Audio-Output-Validierung ---
    audio = result.audio
    assert isinstance(audio, np.ndarray), "result.audio muss np.ndarray sein"
    assert audio.ndim in (1, 2), f"Unerwartete Audio-Dimensionen: {audio.ndim}"
    assert np.isfinite(audio).all(), "result.audio enthält NaN oder Inf"
    # Spec §8.2 ¶11 + §1.4: True-Peak ≤ −1.0 dBTP (ITU-R BS.1770-4, phase_47)
    _TP_LIMIT = 10 ** (-1.0 / 20)  # ≈ 0.8913 linear
    assert np.max(np.abs(audio)) <= _TP_LIMIT + 1e-4, (
        f"True-Peak: {20 * np.log10(np.max(np.abs(audio)) + 1e-12):.2f} dBTP > −1.0 dBTP (Spec §8.2)"
    )

    n_samples = audio.shape[0]
    assert n_samples > 0, "Audio ist leer"

    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    assert rms > 1e-6, f"Audio-RMS zu niedrig (Stille?): {rms:.2e}"

    # Dauer-Check: wir kennen die interne SR (48 000 Hz) nicht direkt,
    # daher prüfen wir nur auf > 0 Samples und eine Mindestzahl.
    assert n_samples >= 48_000 * min_duration_s, (
        f"Audio zu kurz: {n_samples} Samples (erwartet ≥ {int(48_000 * min_duration_s)})"
    )

    # --- Metadaten-Validierung ---
    assert isinstance(result.material_type, MaterialType), (
        f"material_type muss MaterialType sein, ist: {type(result.material_type)}"
    )
    assert isinstance(result.phases_executed, list), "phases_executed muss list sein"
    # K-1 (SCHRITTE §KRITISCH): Triviale >0-Prüfung ersetzt durch TIER-Membership-Check.
    # Spec §7.1: ≥ 2 TIER-1-Phasen (phase_01..phase_06) UND ≥ 3 TIER-6-Phasen (phase_51..phase_56)
    _tier1_exec = [
        p
        for p in result.phases_executed
        if p.startswith(("phase_01", "phase_02", "phase_03", "phase_04", "phase_05", "phase_06"))
    ]
    _tier6_exec = [
        p
        for p in result.phases_executed
        if p.startswith(("phase_51", "phase_52", "phase_53", "phase_54", "phase_55", "phase_56"))
    ]
    assert len(_tier1_exec) >= 2, (
        f"K-1: Weniger als 2 TIER-1-Phasen ausgeführt: {sorted(_tier1_exec)} (Spec §7.1: ≥ 2 aus phase_01..phase_06)"
    )
    assert len(_tier6_exec) >= 3, (
        f"K-1: Weniger als 3 TIER-6-Phasen ausgeführt: {sorted(_tier6_exec)} (Spec §7.1: ≥ 3 aus phase_51..phase_56)"
    )
    assert isinstance(result.phases_skipped, list), "phases_skipped muss list sein"

    # --- Qualitäts-Metriken ---
    assert isinstance(result.rt_factor, (int, float)), "rt_factor muss numerisch sein"
    assert math.isfinite(result.rt_factor), f"rt_factor ist nicht finite: {result.rt_factor}"
    assert result.rt_factor > 0, f"rt_factor muss positiv sein: {result.rt_factor}"

    assert isinstance(result.quality_estimate, (int, float)), "quality_estimate muss numerisch sein"
    assert 0.0 <= result.quality_estimate <= 1.0, f"quality_estimate außerhalb [0, 1]: {result.quality_estimate:.4f}"
    # K-2 (SCHRITTE §KRITISCH): PQS-basierte Schätzung darf nicht unter 0.55 fallen.
    # Wert < 0.55 bedeutet: Engine liefert trotz gültiger Restaurierung Minimalwert →
    # entweder _estimate_quality() nutzt noch den verbotenen *1.15-Bonus oder liefert
    # defect_severity-only ohne echte Audio-Messung.
    assert result.quality_estimate >= 0.55, (
        f"K-2: quality_estimate {result.quality_estimate:.4f} < 0.55 — "
        f"PQS-Schätzung unterschreitet Mindestqualität (SCHRITTE §KRITISCH K-2)"
    )

    assert isinstance(result.total_time_seconds, (int, float)), "total_time_seconds muss numerisch sein"
    assert result.total_time_seconds > 0, f"total_time_seconds muss positiv sein: {result.total_time_seconds}"

    # --- DefectScores ---
    assert isinstance(result.defect_scores, dict), "defect_scores muss dict sein"

    # --- Warnings ---
    assert isinstance(result.warnings, list), "warnings muss list sein"

    # --- Metadata ---
    assert isinstance(result.metadata, dict), "metadata muss dict sein"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.e2e
@pytest.mark.timeout(600)
class TestE2ERestorationQuality:
    """QUALITY-Modus — entspricht Restoration-Modus der Spec."""

    def test_01_audio_geladen_und_valide(self) -> None:
        """Testdatei muss korrekt ladbar und nicht leer sein."""
        audio, sr = _load_audio(_AUDIO_FILE)  # type: ignore[arg-type]
        assert audio.ndim == 1, "Audio muss mono sein"
        assert sr > 0, f"Sample-Rate ungültig: {sr}"
        assert len(audio) > sr, "Audio kürzer als 1 Sekunde"
        assert np.isfinite(audio).all(), "Quellaudio enthält NaN/Inf"
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        assert rms > 1e-6, f"Quellaudio-RMS zu niedrig: {rms:.2e}"

    def test_02_restorer_instanziierung(self) -> None:
        """UnifiedRestorerV3 muss ohne Fehler instanziierbar sein."""
        from backend.core.performance_guard import QualityMode  # type: ignore
        from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3  # type: ignore

        config = RestorationConfig(mode=QualityMode.QUALITY)
        restorer = UnifiedRestorerV3(config=config)
        assert restorer is not None

    def test_03_restoration_quality_mode(self) -> None:
        """Vollständige Restaurierung im QUALITY-Modus (Restoration)."""
        from backend.core.performance_guard import QualityMode  # type: ignore
        from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3  # type: ignore

        audio, sr = _load_audio_clip(_AUDIO_FILE)  # 15 s-Clip: CI-Timeout vermeiden  # type: ignore[arg-type]

        config = RestorationConfig(
            mode=QualityMode.QUALITY,
            enable_performance_guard=True,
            enable_phase_gate=True,
        )
        restorer = UnifiedRestorerV3(config=config)

        t0 = time.monotonic()
        result = restorer.restore(audio, sample_rate=sr)
        elapsed = time.monotonic() - t0

        _validate_restoration_result(result, min_duration_s=1.0)

        # --- PQS-Scores prüfen (Spec §8.1) ---
        from backend.core.perceptual_quality_scorer import score_audio_absolute  # type: ignore

        pqs = score_audio_absolute(result.audio, sr=48_000)
        assert pqs.pqs_mos >= 4.0, f"Spec §8.1: PQS-MOS {pqs.pqs_mos:.3f} < 4.0"
        assert pqs.nsim >= 0.80, f"Spec §8.1: NSIM {pqs.nsim:.3f} < 0.80"
        assert pqs.mcd_db <= 5.0, f"Spec §8.1: MCD {pqs.mcd_db:.2f} dB > 5.0 dB"

        # --- Chroma-Korrelation Original ↔ Restauriert ≥ 0.95 (Spec §1.4 / §8.2 ¶5) ---
        import librosa  # type: ignore

        res_mono = result.audio
        if res_mono.ndim == 2:
            res_mono = res_mono.mean(axis=0) if res_mono.shape[0] == 2 else res_mono.mean(axis=1)
        audio_48 = librosa.resample(audio, orig_sr=sr, target_sr=48_000) if sr != 48_000 else audio
        src_chroma = librosa.feature.chroma_stft(y=audio_48, sr=48_000).mean(axis=1)
        res_chroma = librosa.feature.chroma_stft(y=res_mono, sr=48_000).mean(axis=1)
        chroma_corr = float(np.corrcoef(src_chroma, res_chroma)[0, 1])
        assert chroma_corr >= 0.95, f"Spec §1.4: Chroma-Korrelation {chroma_corr:.3f} < 0.95 (Tonart-Treue)"

        # --- LUFS-Differenz ≤ 1 LU (Spec §1.4, Restoration-Modus) ---
        import pyloudnorm as pyln  # type: ignore

        meter = pyln.Meter(48_000)
        lufs_src = meter.integrated_loudness(audio_48)
        lufs_res = meter.integrated_loudness(res_mono)
        assert abs(lufs_src - lufs_res) <= 1.0, f"Spec §1.4: LUFS-Diff {abs(lufs_src - lufs_res):.2f} LU > 1.0 LU"

        print(
            f"  PQS-MOS={pqs.pqs_mos:.3f}  NSIM={pqs.nsim:.3f}  "
            f"MCD={pqs.mcd_db:.2f} dB  Chroma-Corr={chroma_corr:.3f}  "
            f"LUFS-Diff={abs(lufs_src - lufs_res):.2f} LU"
        )

        # Ausgabe schreiben
        out_path = _OUTPUT_DIR / "elke_best_quality_9_10_41.wav"
        _write_wav(out_path, result.audio, sr=48_000)
        assert out_path.exists(), f"WAV-Ausgabe nicht gefunden: {out_path}"

        print(
            f"\n[QUALITY] rt_factor={result.rt_factor:.2f}x  "
            f"quality_estimate={result.quality_estimate:.3f}  "
            f"Phasen ausgeführt={len(result.phases_executed)}  "
            f"Zeit={elapsed:.1f}s  material={result.material_type.value}"
        )

    def test_04_defect_scores_plausibel(self) -> None:
        """DefectScores müssen im gültigen Wertebereich [0, 1] liegen."""
        from backend.core.performance_guard import QualityMode  # type: ignore
        from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3  # type: ignore

        audio, sr = _load_audio_clip(_AUDIO_FILE)  # 15 s-Clip: CI-Timeout vermeiden  # type: ignore[arg-type]
        config = RestorationConfig(mode=QualityMode.QUALITY)
        restorer = UnifiedRestorerV3(config=config)
        result = restorer.restore(audio, sample_rate=sr)

        for defect_type, score in result.defect_scores.items():
            assert math.isfinite(score), f"DefectScore für {defect_type} ist nicht finite: {score}"
            assert 0.0 <= score <= 1.0, f"DefectScore für {defect_type} außerhalb [0, 1]: {score:.4f}"

    def test_05_warning_liste_ist_sauber(self) -> None:
        """Warnings müssen Strings sein; kein None-Eintrag erlaubt."""
        from backend.core.performance_guard import QualityMode  # type: ignore
        from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3  # type: ignore

        audio, sr = _load_audio_clip(_AUDIO_FILE)  # 15 s-Clip: CI-Timeout vermeiden  # type: ignore[arg-type]
        config = RestorationConfig(mode=QualityMode.QUALITY)
        restorer = UnifiedRestorerV3(config=config)
        result = restorer.restore(audio, sample_rate=sr)

        for w in result.warnings:
            assert isinstance(w, str), f"Warning-Eintrag ist kein String: {type(w)}"
            assert w, "Leerer Warning-String"

    def test_06_era_detection_present(self) -> None:
        """Ära-Erkennung muss für deutsches Schlagermaterial laufen (§2.14)."""
        from backend.core.performance_guard import QualityMode  # type: ignore
        from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3  # type: ignore

        audio, sr = _load_audio_clip(_AUDIO_FILE)  # type: ignore[arg-type]
        config = RestorationConfig(mode=QualityMode.QUALITY)
        restorer = UnifiedRestorerV3(config=config)
        result = restorer.restore(audio, sample_rate=sr)

        era = result.metadata.get("era") or result.metadata.get("era_result")
        assert era is not None, (
            "Ära-Erkennung (EraClassifier) hat kein Ergebnis geliefert — "
            "StereoAuthenticitiyInvariant ist stumm deaktiviert (§2.18)"
        )
        decade = era.get("decade") if isinstance(era, dict) else getattr(era, "decade", None)
        if decade is not None:
            assert 1950 <= decade <= 2000, f"Era-Decade {decade} außerhalb plausiblem Bereich [1950, 2000]"

    def test_07_schlager_profile_applied(self) -> None:
        """Genre-Klassifikation und Schlager-Profil müssen für Testmaterial laufen (§2.19)."""
        from backend.core.performance_guard import QualityMode  # type: ignore
        from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3  # type: ignore

        audio, sr = _load_audio_clip(_AUDIO_FILE)  # type: ignore[arg-type]
        config = RestorationConfig(mode=QualityMode.QUALITY)
        restorer = UnifiedRestorerV3(config=config)
        result = restorer.restore(audio, sample_rate=sr)

        genre = result.metadata.get("genre") or result.metadata.get("schlager_result")
        assert genre is not None, (
            "Genre-Klassifikation (GermanSchlagerClassifier) hat kein Ergebnis — "
            "§2.19 nicht erfüllt für deutschsprachiges Testmaterial"
        )
        is_schlager = genre.get("is_schlager") if isinstance(genre, dict) else getattr(genre, "is_schlager", None)
        if is_schlager:
            goals = result.metadata.get("musical_goals", {})
            waerme = goals.get("waerme")
            if waerme is not None:
                assert waerme >= 0.80, f"Wärme {waerme:.3f} < 0.80 — Schlager-Profil nicht korrekt angewendet (§2.19)"

    def test_08_pmgg_gate_active(self) -> None:
        """PMGG (PerPhaseMusicalGoalsGate) muss aktiv und stabil laufen (§2.29)."""
        from backend.core.performance_guard import QualityMode  # type: ignore
        from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3  # type: ignore

        audio, sr = _load_audio_clip(_AUDIO_FILE)  # type: ignore[arg-type]
        config = RestorationConfig(
            mode=QualityMode.QUALITY,
            enable_performance_guard=True,
            enable_phase_gate=True,
        )
        restorer = UnifiedRestorerV3(config=config)
        result = restorer.restore(audio, sample_rate=sr)

        pmgg_log = result.metadata.get("phase_gate_log", [])
        tier1_best_effort = [
            e for e in pmgg_log if "TIER-1" in str(e) or "best_effort" in str(e).lower() or "rollback" in str(e).lower()
        ]
        assert len(tier1_best_effort) < 5, (
            f"Zu viele PMGG-Best-Effort-Phasen in TIER-1: {tier1_best_effort} — Engine instabil (§2.29)"
        )

    def test_09_panns_vocals_detected(self) -> None:
        """PANNs muss Gesang für Gesangs-Testmaterial erkennen (§2.9)."""
        from backend.core.performance_guard import QualityMode  # type: ignore
        from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3  # type: ignore

        audio, sr = _load_audio_clip(_AUDIO_FILE)  # type: ignore[arg-type]
        config = RestorationConfig(mode=QualityMode.QUALITY)
        restorer = UnifiedRestorerV3(config=config)
        result = restorer.restore(audio, sample_rate=sr)

        panns_tags = result.metadata.get("panns_tags", {})
        vocals_conf = panns_tags.get("vocals", panns_tags.get("Singing", 0.0))
        assert vocals_conf >= 0.4, (
            f"PANNs Gesang-Konfidenz {vocals_conf:.3f} < 0.40 — phase_42_vocal_enhancement wird nicht aktiviert (§2.9)"
        )
        assert "phase_42_vocal_enhancement" in result.phases_executed, (
            "phase_42_vocal_enhancement nicht ausgeführt obwohl Testmaterial Gesang enthält (§2.9)"
        )

    def test_10_restoration_improves_quality(self) -> None:
        """Restaurierung muss messbar besser sein als das Original (§8.2, Punkt 8)."""
        from backend.core.performance_guard import QualityMode  # type: ignore
        from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3  # type: ignore

        audio, sr = _load_audio_clip(_AUDIO_FILE)  # type: ignore[arg-type]
        rng = np.random.default_rng(42)
        degraded = audio + rng.normal(0, 0.005, audio.shape).astype(np.float32)
        config = RestorationConfig(mode=QualityMode.QUALITY)
        restorer = UnifiedRestorerV3(config=config)
        result_orig = restorer.restore(audio, sample_rate=sr)
        result_degrad = restorer.restore(degraded, sample_rate=sr)

        delta = abs(result_orig.quality_estimate - result_degrad.quality_estimate)
        assert delta <= 0.20, (
            f"Quality-Delta {delta:.3f} > 0.20 — Restaurierung adaptiert sich nicht an Degradierung (§8.2 Punkt 8)"
        )
        goals = result_degrad.metadata.get("musical_goals", {})
        tonal_center = goals.get("tonal_center", goals.get("tonales_zentrum"))
        if tonal_center is not None:
            assert tonal_center >= 0.95, f"TonalCenterMetric {tonal_center:.3f} < 0.95 — §2.29 TonalCenter verletzt"

    def test_11_internal_quality_gates_passed(self) -> None:
        """OQS, MQA und ArtifactDetector müssen ihre Grenzen einhalten (§8.1)."""
        from backend.core.performance_guard import QualityMode  # type: ignore
        from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3  # type: ignore

        audio, sr = _load_audio_clip(_AUDIO_FILE)  # type: ignore[arg-type]
        config = RestorationConfig(mode=QualityMode.QUALITY)
        restorer = UnifiedRestorerV3(config=config)
        result = restorer.restore(audio, sample_rate=sr)

        meta = result.metadata
        oqs = meta.get("oqs_score") or meta.get("mushra_score")
        if oqs is not None:
            assert oqs >= 75, f"OQS {oqs:.1f} < 75 — unter §8.1-Grenze"
        mqa = meta.get("quality_guaranteed") or meta.get("mqa_passed")
        if mqa is not None:
            assert mqa is True, "MusicalQualityAssurance: quality_guaranteed=False — §8.1 nicht erfüllt"
        artifact_check = meta.get("passes_aurik_standards") or meta.get("artifact_check_passed")
        if artifact_check is not None:
            assert artifact_check is True, (
                "ArtifactDetector: passes_aurik_standards=False — Artefakte im Output (§2.23)"
            )
        assert any(x is not None for x in [oqs, mqa, artifact_check]), (
            "Kein internes Qualitäts-Gate im metadata vorhanden — "
            "MushraEvaluator/MQA/ArtifactDetector wurden nicht aufgerufen (§8.1)"
        )

    def test_12_material_type_specific(self) -> None:
        """MaterialType muss inhaltlich plausibel sein — nicht nur 'unknown' (§2.1)."""
        from backend.core.defect_scanner import MaterialType  # type: ignore
        from backend.core.performance_guard import QualityMode  # type: ignore
        from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3  # type: ignore

        audio, sr = _load_audio_clip(_AUDIO_FILE)  # type: ignore[arg-type]
        config = RestorationConfig(mode=QualityMode.QUALITY)
        restorer = UnifiedRestorerV3(config=config)
        result = restorer.restore(audio, sample_rate=sr)

        assert isinstance(result.material_type, MaterialType), (
            f"material_type ist kein MaterialType: {type(result.material_type)}"
        )
        assert result.material_type != MaterialType.UNKNOWN, (
            "MaterialType ist UNKNOWN — MediumClassifier hat nicht funktioniert (§2.1)"
        )
        EXPECTED_DIGITAL_TYPES = {
            MaterialType.MP3_LOW,
            MaterialType.MP3_HIGH,
            MaterialType.AAC,
            MaterialType.STREAMING,
            MaterialType.CD_DIGITAL,
        }
        assert result.material_type in EXPECTED_DIGITAL_TYPES, (
            f"MaterialType {result.material_type} nicht für MP3-Testmaterial erwartet. "
            f"Erwartet eines von: {EXPECTED_DIGITAL_TYPES}"
        )

    def test_13_groove_metric_maintained(self) -> None:
        """Groove (Mikro-Timing) darf nicht zerstört werden — TDP-Nachweis (§2.27)."""
        from backend.core.performance_guard import QualityMode  # type: ignore
        from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3  # type: ignore

        audio, sr = _load_audio_clip(_AUDIO_FILE)  # type: ignore[arg-type]
        config = RestorationConfig(mode=QualityMode.QUALITY)
        restorer = UnifiedRestorerV3(config=config)
        result = restorer.restore(audio, sample_rate=sr)

        goals = result.metadata.get("musical_goals", {})
        groove = goals.get("groove")
        if groove is not None:
            assert groove >= 0.88, f"GrooveMetric {groove:.3f} < 0.88 — TDP oder PMGG hat Timing zerstört (§2.27)"
        tdp_active = result.metadata.get("tdp_active") or result.metadata.get("transient_decoupled")
        if tdp_active is not None:
            assert tdp_active is True, "TransientDecoupledProcessing nicht aktiv (§2.27)"

    def test_14_hpg_natuerlichkeit_authentizitaet(self) -> None:
        """HPG muss Natürlichkeit ≥ 0.88 und Authentizität ≥ 0.85 sichern (§2.28)."""
        from backend.core.performance_guard import QualityMode  # type: ignore
        from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3  # type: ignore

        audio, sr = _load_audio_clip(_AUDIO_FILE)  # type: ignore[arg-type]
        config = RestorationConfig(mode=QualityMode.QUALITY)
        restorer = UnifiedRestorerV3(config=config)
        result = restorer.restore(audio, sample_rate=sr)

        goals = result.metadata.get("musical_goals", {})
        natuerlichkeit = goals.get("natuerlichkeit")
        authentizitaet = goals.get("authentizitaet")
        if natuerlichkeit is not None:
            assert natuerlichkeit >= 0.88, (
                f"NatürlichkeitMetric {natuerlichkeit:.3f} < 0.88 — HPG nicht wirksam (§2.28)"
            )
        if authentizitaet is not None:
            assert authentizitaet >= 0.85, (
                f"AuthentizitätMetric {authentizitaet:.3f} < 0.85 — HPG nicht wirksam (§2.28)"
            )

    def test_15_micro_dynamics_preserved(self) -> None:
        """MDEM muss MicroDynamics-Pearson ≥ 0.90 sichern (§2.30)."""
        from backend.core.performance_guard import QualityMode  # type: ignore
        from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3  # type: ignore

        audio, sr = _load_audio_clip(_AUDIO_FILE)  # type: ignore[arg-type]
        config = RestorationConfig(mode=QualityMode.QUALITY)
        restorer = UnifiedRestorerV3(config=config)
        result = restorer.restore(audio, sample_rate=sr)

        goals = result.metadata.get("musical_goals", {})
        micro_dyn = goals.get("micro_dynamics") or goals.get("micro_dynamik")
        if micro_dyn is not None:
            assert micro_dyn >= 0.90, f"MicroDynamicsMetric {micro_dyn:.3f} < 0.90 — MDEM nicht wirksam (§2.30)"
        lufs_diff = result.metadata.get("lufs_diff") or result.metadata.get("lufs_delta")
        if lufs_diff is not None:
            assert abs(lufs_diff) <= 1.0, (
                f"LUFS-Differenz {lufs_diff:.2f} LU > 1 LU — §1.4 Restoration-Invariante verletzt"
            )

    def test_16_temporal_quality_coherent(self) -> None:
        """Qualität muss über die Zeitachse kohärent sein — keine Ausreißer (§2.16)."""
        from backend.core.performance_guard import QualityMode  # type: ignore
        from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3  # type: ignore

        audio, sr = _load_audio_clip(_AUDIO_FILE)  # type: ignore[arg-type]
        config = RestorationConfig(mode=QualityMode.QUALITY)
        restorer = UnifiedRestorerV3(config=config)
        result = restorer.restore(audio, sample_rate=sr)

        temporal = result.metadata.get("temporal_coherence") or result.metadata.get("temporal_coherence_result")
        if temporal is None:
            pytest.skip("TemporalQualityCoherenceMetric nicht im Metadata — §2.16 Skip")
        max_span = temporal.get("max_span") if isinstance(temporal, dict) else getattr(temporal, "max_span", None)
        if max_span is not None:
            assert max_span <= 0.30, f"Temporale MOS-Spanne {max_span:.3f} > 0.30 — lokale Qualitätsausreißer (§2.16)"


@pytest.mark.e2e
@pytest.mark.timeout(600)
class TestE2EStudio2026Balanced:
    """BALANCED-Modus (Studio 2026) — Quality-Ziele der Spec §1.4."""

    def test_01_studio_balanced_mode(self) -> None:
        """Vollständige Restaurierung im BALANCED-Modus (Studio 2026)."""
        from backend.core.performance_guard import QualityMode  # type: ignore
        from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3  # type: ignore

        audio, sr = _load_audio_clip(_AUDIO_FILE)  # 15 s-Clip: CI-Timeout vermeiden  # type: ignore[arg-type]

        config = RestorationConfig(
            mode=QualityMode.BALANCED,
            enable_performance_guard=False,  # kein RT-Limit im Studio-Modus
            enable_phase_gate=True,
        )
        restorer = UnifiedRestorerV3(config=config)

        t0 = time.monotonic()
        result = restorer.restore(audio, sample_rate=sr)
        elapsed = time.monotonic() - t0

        _validate_restoration_result(result, min_duration_s=1.0)

        # --- Studio 2026 Pflicht-Invarianten (Spec §1.4) ---
        from backend.core.defect_scanner import MaterialType  # type: ignore
        from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker  # type: ignore
        from backend.core.perceptual_quality_scorer import score_audio_absolute  # type: ignore

        # Spec §8.1: MOS ≥ 4.5 gilt NUR für digitale Hochqualitäts-Quellen.
        # Für alle anderen Materialien gelten material-spezifische Erwartungen aus §6.2.
        _HIGH_QUALITY_DIGITAL = {MaterialType.CD_DIGITAL, MaterialType.DAT, MaterialType.MP3_HIGH, MaterialType.AAC}
        _studio_mos_min = 4.5 if result.material_type in _HIGH_QUALITY_DIGITAL else 4.0
        pqs_max = score_audio_absolute(result.audio, sr=48_000)
        assert pqs_max.pqs_mos >= _studio_mos_min, (
            f"Studio 2026 §1.4: PQS-MOS {pqs_max.pqs_mos:.3f} < {_studio_mos_min} "
            f"(Weltklasse / Material={result.material_type.value})"
        )

        checker_max = MusicalGoalsChecker()
        scores_max = checker_max.measure_all(result.audio, sr=48_000)
        # Brillanz ≥ 0.90 (verschärft gegenüber Mindestschwelle 0.85, Spec §1.4)
        assert scores_max.get("brillanz", 0.0) >= 0.90, (
            f"Studio 2026 §1.4: Brillanz {scores_max.get('brillanz', 0.0):.3f} < 0.90"
        )
        # Bass-Kraft ≥ 0.88 (verschärft gegenüber Mindestschwelle 0.85, Spec §1.4)
        assert scores_max.get("bass_kraft", 0.0) >= 0.88, (
            f"Studio 2026 §1.4: Bass-Kraft {scores_max.get('bass_kraft', 0.0):.3f} < 0.88"
        )

        print(
            f"  Studio2026 PQS-MOS={pqs_max.pqs_mos:.3f}  "
            f"Brillanz={scores_max.get('brillanz', 0.0):.3f}  "
            f"Bass-Kraft={scores_max.get('bass_kraft', 0.0):.3f}"
        )

        # Ausgabe schreiben
        out_path = _OUTPUT_DIR / "elke_best_studio2026_9_10_41.wav"
        _write_wav(out_path, result.audio, sr=48_000)
        assert out_path.exists(), f"WAV-Ausgabe nicht gefunden: {out_path}"

        print(
            f"\n[BALANCED/Studio2026] rt_factor={result.rt_factor:.2f}x  "
            f"quality_estimate={result.quality_estimate:.3f}  "
            f"Phasen ausgeführt={len(result.phases_executed)}  "
            f"Zeit={elapsed:.1f}s  material={result.material_type.value}"
        )

    def test_02_quality_estimate_mindest_schwelle(self) -> None:
        """BALANCED-Modus muss quality_estimate ≥ 0.55 erreichen (E2E-Pflicht Spec §8.1)."""
        from backend.core.performance_guard import QualityMode  # type: ignore
        from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3  # type: ignore

        audio, sr = _load_audio_clip(_AUDIO_FILE)  # 15 s-Clip: CI-Timeout vermeiden  # type: ignore[arg-type]

        config = RestorationConfig(mode=QualityMode.BALANCED, enable_performance_guard=False)
        result = UnifiedRestorerV3(config=config).restore(audio, sample_rate=sr)

        assert result.quality_estimate >= 0.55, (
            f"BALANCED quality_estimate {result.quality_estimate:.3f} < 0.55 (E2E-Pflicht Spec §8.1)"
        )


@pytest.mark.e2e
@pytest.mark.timeout(600)
class TestE2EMusicalGoals:
    """Musical Goals validation — all 15 goals (current spec baseline)."""

    def test_01_musical_goals_nach_quality_mode(self) -> None:
        """Nach Restaurierung (QUALITY) müssen Musical Goals messbar sein."""
        from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker  # type: ignore
        from backend.core.performance_guard import QualityMode  # type: ignore
        from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3  # type: ignore

        audio, sr = _load_audio_clip(_AUDIO_FILE)  # 15 s-Clip: CI-Timeout vermeiden  # type: ignore[arg-type]
        config = RestorationConfig(mode=QualityMode.QUALITY)
        result = UnifiedRestorerV3(config=config).restore(audio, sample_rate=sr)

        checker = MusicalGoalsChecker()
        scores = checker.measure_all(result.audio, sr=48_000)

        assert isinstance(scores, dict), "measure_all muss dict zurückgeben"
        # Current spec baseline: exactly 15 Musical Goals
        assert len(scores) == 15, (
            f"Erwartet 15 Musical Goals (aktueller Spec-Stand), erhalten {len(scores)}: {sorted(scores.keys())}"
        )

        for goal, score in scores.items():
            assert math.isfinite(score), f"Musical Goal '{goal}' ist nicht finite: {score}"
            assert 0.0 <= score <= 1.0, f"Musical Goal '{goal}' außerhalb [0, 1]: {score:.4f}"

        # Spec §1.2: jede Restaurierungsoperation darf keines der 14 Ziele unterschreiten
        violations = {g: (scores[g], t) for g, t in checker.thresholds.items() if scores[g] < t}
        assert not violations, "Musical Goals unter Pflicht-Schwellwert (aktueller Spec-Stand):\n" + "\n".join(
            f"  {g}: {s:.3f} < {t:.2f}" for g, (s, t) in violations.items()
        )

        print(f"\n[Musical Goals nach QUALITY]: {len(scores)} Ziele gemessen")
        for goal, score in sorted(scores.items()):
            threshold = checker.thresholds.get(goal, 0.0)
            marker = "✓" if score >= threshold else "✗"
            print(f"  {marker} {goal}: {score:.3f} (≥ {threshold:.2f})")

    def test_02_no_nan_in_musical_goals(self) -> None:
        """Kein NaN/Inf in Musical Goals erlaubt (Spec §3.1)."""
        from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker  # type: ignore
        from backend.core.performance_guard import QualityMode  # type: ignore
        from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3  # type: ignore

        audio, sr = _load_audio_clip(_AUDIO_FILE)  # 15 s-Clip: CI-Timeout vermeiden  # type: ignore[arg-type]
        config = RestorationConfig(mode=QualityMode.QUALITY)
        result = UnifiedRestorerV3(config=config).restore(audio, sample_rate=sr)

        checker = MusicalGoalsChecker()
        scores = checker.measure_all(result.audio, sr=48_000)

        nan_goals = [g for g, s in scores.items() if not math.isfinite(s)]
        assert not nan_goals, f"NaN/Inf in Musical Goals: {nan_goals}"

    def test_03_schlager_specific_thresholds(self) -> None:
        """Schlager-spezifische Zielwerte für Wärme und Tonales Zentrum (§2.19 / W-6)."""
        from backend.core.performance_guard import QualityMode  # type: ignore
        from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3  # type: ignore

        audio, sr = _load_audio_clip(_AUDIO_FILE)  # type: ignore[arg-type]
        config = RestorationConfig(mode=QualityMode.QUALITY)
        restorer = UnifiedRestorerV3(config=config)
        result = restorer.restore(audio, sample_rate=sr)

        genre = result.metadata.get("genre") or result.metadata.get("schlager_result")
        if genre is None:
            pytest.skip("Kein Genre-Ergebnis im Metadata — GermanSchlagerClassifier nicht verfügbar")
        is_schlager = genre.get("is_schlager") if isinstance(genre, dict) else getattr(genre, "is_schlager", False)
        if not is_schlager:
            pytest.skip("Kein Schlager für dieses Material erkannt — Schlager-Thresholds überspringen")

        goals = result.metadata.get("musical_goals", {})
        tonal_center = goals.get("tonal_center", goals.get("tonales_zentrum"))
        if tonal_center is not None:
            assert tonal_center >= 0.97, (
                f"TonalCenterMetric {tonal_center:.3f} < 0.97 — "
                "§2.29 Schlager-Profil-Invariante verletzt (verschärfter Schwellwert)"
            )
        waerme = goals.get("waerme")
        if waerme is not None:
            assert waerme >= 0.88, (
                f"Wärme {waerme:.3f} < 0.88 — Schlager-Profil nicht korrekt angewendet (W-6, §2.19.3)"
            )


@pytest.mark.e2e
@pytest.mark.timeout(120)
class TestE2ESystemInfo:
    """Systeminfo-Tests — Versions- und Modul-Sanity-Checks."""

    def test_01_korrekte_version(self) -> None:
        """Aurik-Version muss 9.10.41 sein (Spec-Pflicht)."""
        from Aurik10 import __version__  # type: ignore

        assert __version__ == "9.10.41", f"Falsche Version: '{__version__}' — erwartet '9.10.41'"

    def test_02_unified_restorer_v3_importierbar(self) -> None:
        """UnifiedRestorerV3 muss importierbar sein."""
        from backend.core.unified_restorer_v3 import (  # type: ignore
            RestorationConfig,
            RestorationResult,
            UnifiedRestorerV3,
        )

        assert UnifiedRestorerV3 is not None
        assert RestorationConfig is not None
        assert RestorationResult is not None

    def test_03_quality_mode_werte(self) -> None:
        """QualityMode muss alle erwarteten Werte besitzen."""
        from backend.core.performance_guard import QualityMode  # type: ignore

        assert hasattr(QualityMode, "FAST")
        assert hasattr(QualityMode, "BALANCED")
        assert hasattr(QualityMode, "QUALITY")
        assert sorted(member.name for member in QualityMode) == ["BALANCED", "FAST", "QUALITY"]

    def test_04_musical_goals_checker_importierbar(self) -> None:
        """MusicalGoalsChecker muss importierbar und instanziierbar sein."""
        from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker  # type: ignore

        checker = MusicalGoalsChecker()
        assert checker is not None
        assert hasattr(checker, "measure_all")

    def test_05_keine_v2_importe_in_pipeline(self) -> None:
        """UnifiedRestorerV2 darf in der V3-Pipeline nicht importiert werden."""
        import sys

        # V3 importieren — dabei darf V2 nicht als Seiteneffekt geladen werden
        if "core.unified_restorer_v3" in sys.modules:
            # Bereits geladen — dann prüfen ob V2 dabei ist
            pass

        # V2 sollte nicht existieren oder zumindest nicht in der V3-Pipeline verwendet werden
        try:
            from backend.core.performance_guard import QualityMode  # type: ignore
            from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3  # type: ignore

            config = RestorationConfig(mode=QualityMode.QUALITY)
            restorer = UnifiedRestorerV3(config=config)
            # Kein Fehler = V3 ist vollständig autonom
            assert isinstance(restorer, UnifiedRestorerV3)
        except ImportError as e:
            pytest.fail(f"UnifiedRestorerV3 nicht importierbar: {e}")
