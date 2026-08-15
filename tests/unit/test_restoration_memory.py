"""Unit-Tests für RestorationMemory (§2.70, v10.0.0)."""

import pytest

from backend.core.restoration_memory import RestorationMemory, get_restoration_memory


@pytest.fixture()
def mem(tmp_path):
    """Erzeugt eine frische RestorationMemory-Instanz mit tmp-Pfad (kein globaler Singleton-State)."""
    return RestorationMemory(path=tmp_path / "restoration_memory.json")


@pytest.mark.unit
class TestRestorationMemoryBasic:
    def test_get_prior_returns_none_for_unknown_key(self, mem):
        result = mem.get_prior((1960, "vinyl", "abc12345"))
        assert result is None

    def test_save_and_get_prior_round_trip(self, mem):
        key = (1970, "tape", "deadbeef")
        mem.save_result(key, {"strength": 0.5}, hpi_achieved=0.82)
        result = mem.get_prior(key)
        assert result is not None
        assert result["hpi_achieved"] == pytest.approx(0.82, abs=1e-4)

    def test_save_does_not_overwrite_better_prior(self, mem):
        key = (1970, "tape", "deadbeef")
        mem.save_result(key, {"strength": 0.5}, hpi_achieved=0.90)
        mem.save_result(key, {"strength": 0.3}, hpi_achieved=0.70)  # schlechterer Prior
        result = mem.get_prior(key)
        assert result["hpi_achieved"] == pytest.approx(0.90, abs=1e-4), (
            "Schlechterer Prior soll besseren nicht überschreiben"
        )

    def test_save_overwrites_with_better_prior(self, mem):
        key = (1970, "tape", "deadbeef")
        mem.save_result(key, {"strength": 0.5}, hpi_achieved=0.70)
        mem.save_result(key, {"strength": 0.7}, hpi_achieved=0.92)  # besserer Prior
        result = mem.get_prior(key)
        assert result["hpi_achieved"] == pytest.approx(0.92, abs=1e-4)

    def test_hpi_zero_not_saved(self, mem):
        key = (1970, "tape", "cafe1234")
        mem.save_result(key, {}, hpi_achieved=0.0)
        assert mem.get_prior(key) is None

    def test_hpi_negative_not_saved(self, mem):
        key = (1970, "tape", "babe5678")
        mem.save_result(key, {}, hpi_achieved=-0.5)
        assert mem.get_prior(key) is None

    def test_multiple_keys_independent(self, mem):
        key1 = (1960, "vinyl", "aaaaaaaa")
        key2 = (1980, "cd", "bbbbbbbb")
        mem.save_result(key1, {}, hpi_achieved=0.88)
        mem.save_result(key2, {}, hpi_achieved=0.75)
        assert mem.get_prior(key1)["hpi_achieved"] == pytest.approx(0.88, abs=1e-4)
        assert mem.get_prior(key2)["hpi_achieved"] == pytest.approx(0.75, abs=1e-4)

    def test_persistence_across_instances(self, tmp_path):
        """Gespeicherte Priors überleben eine neue Instanz (Disk-Persistenz)."""
        path = tmp_path / "restoration_memory.json"
        m1 = RestorationMemory(path=path)
        key = (1970, "vinyl", "persist01")
        m1.save_result(key, {"x": 1}, hpi_achieved=0.88)

        m2 = RestorationMemory(path=path)
        result = m2.get_prior(key)
        assert result is not None
        assert result["hpi_achieved"] == pytest.approx(0.88, abs=1e-4)

    def test_stats_track_prior_hits_and_misses(self, mem):
        key = (1970, "vinyl", "stats0101")
        mem.save_result(key, {"strength": 0.5}, hpi_achieved=0.81)

        assert mem.get_prior(key) is not None
        assert mem.get_prior((1970, "vinyl", "stats_missing")) is None

        stats = mem.get_stats()
        assert stats["prior_requests"] >= 2
        assert stats["prior_hits"] >= 1
        assert stats["prior_misses"] >= 1

    def test_stats_track_save_rejections(self, mem):
        key = (1970, "tape", "stats0202")
        mem.save_result(key, {"strength": 0.5}, hpi_achieved=0.9)
        mem.save_result(key, {"strength": 0.2}, hpi_achieved=0.7)  # schlechter als vorhandener Prior
        mem.save_result((1971, "tape", "stats0303"), {}, hpi_achieved=0.0)

        stats = mem.get_stats()
        assert stats["save_attempts"] >= 3
        assert stats["save_success"] >= 1
        assert stats["save_rejected_not_better"] >= 1
        assert stats["save_rejected_hpi"] >= 1


class TestRestorationMemorySingleton:
    def test_singleton_returns_same_instance(self):
        a = get_restoration_memory()
        b = get_restoration_memory()
        assert a is b


class TestRestorationMemoryVocalSaveGuard:
    """§2.70 Guard: RestorationMemory-Save darf Vocal-Material (panns_singing ≥ 0.35)
    NICHT blockieren. BUG-Fix v10.0.0: _panns_singing_for_hpi < 0.35 aus UV3-Save-Bedingung
    entfernt — andernfalls wird JEDER Vokal-Run nie gespeichert → timbral_fidelity bleibt
    dauerhaft auf cold-start-Niveau (≈0.747).
    """

    def test_uv3_vocal_save_guard_panns_condition_removed(self):
        """UV3-Source-Code darf _panns_singing_for_hpi < 0.35 NICHT als Save-Bedingung enthalten.

        Linter-Guard: Bei Reintroduktion des Bugs sofortiger Testfehler.
        """
        import pathlib

        uv3_path = pathlib.Path(__file__).parent.parent.parent / "backend" / "core" / "unified_restorer_v3.py"
        assert uv3_path.exists(), f"UV3 nicht gefunden: {uv3_path}"
        source = uv3_path.read_text(encoding="utf-8")
        # Suche den Save-Block (identifiziert durch _af_save >= 0.95)
        save_block_start = source.find("_af_save >= 0.95")
        assert save_block_start > 0, "§2.70 Save-Bedingung nicht in UV3 gefunden"
        # Prüfe 200 Zeichen nach dem Block-Start: kein panns_singing-Ausschluss
        save_block_snippet = source[save_block_start : save_block_start + 200]
        assert "_panns_singing_for_hpi < 0.35" not in save_block_snippet, (
            "§2.70 BUG REINTRODUCED: _panns_singing_for_hpi < 0.35 in Save-Bedingung! "
            "Vocal-Material (panns_singing ≥ 0.35) wird NIE in RestorationMemory gespeichert. "
            "Fix: Bedingung entfernen (v10.0.0 §2.70)."
        )

    def test_vocal_material_save_works_normally(self, mem):
        """Cassette+Schlager-Keys (typisch für Vocal-Material) werden korrekt gespeichert."""
        key = (1970, "cassette", "cassette_schlager_1970")
        mem.save_result(key, {"strength": 0.6, "nr_wet": 0.8}, hpi_achieved=0.56)
        result = mem.get_prior(key)
        assert result is not None, "Vocal/Cassette-Prior nicht gespeichert"
        assert result["hpi_achieved"] == pytest.approx(0.56, abs=1e-4)

    def test_uv3_emotional_arc_variable_connected_to_hpi(self):
        """§2.44 Linter-Guard: _arc_result muss nach _emotional_arc_result propagiert werden.

        Bug: _emotional_arc_result war nie zugewiesen → HPI verwendete immer 1.0 (arc nie gemessen).
        Fix (v10.0.0): Assignment-Block nach post-MDEM EmotionalArc-Block.
        """
        import pathlib

        uv3_path = pathlib.Path(__file__).parent.parent.parent / "backend" / "core" / "unified_restorer_v3.py"
        assert uv3_path.exists(), f"UV3 nicht gefunden: {uv3_path}"
        source = uv3_path.read_text(encoding="utf-8")
        # Prüfe, dass der Fix-Block vorhanden ist (Linter: verhindert Reintroduktion des Bugs)
        assert "_emotional_arc_result = _arc_result" in source, (
            "§2.44 BUG REINTRODUCED: _arc_result wird nicht nach _emotional_arc_result propagiert! "
            "Emotional-Arc-Score ist für den HPI immer 1.0 (nie gemessen). "
            "Fix: Assignment nach post-MDEM EmotionalArc-Block einfügen (v10.0.0 §2.44)."
        )
        # Prüfe, dass Floor-Guard vorhanden ist (verhindert HPI-Null-Kollaps)
        assert "max(\n            0.25," in source or "max(0.25," in source, (
            "§2.44 FLOOR FEHLT: _emotional_arc_for_hpi ohne 0.25-Floor! "
            "preservation_score=0.0 würde HPI auf 0.0 kollabieren lassen."
        )

    def test_uv3_studio_mode_uses_method_not_attribute(self):
        """§UV3-Crash-Guard: self._is_studio_mode darf in UV3 nicht existieren.

        Bug (v10.0.0): UV3.restore() rief self._is_studio_mode (Attribut) statt
        self.is_studio_mode() (Methode) auf → AttributeError am Ende von restore() →
        gesamte UV3-Restaurierung (42 Phasen, ~96 min) wurde weggeworfen;
        RestaurierDenker-Fallback exportierte Original-Audio.
        Fix: self._is_studio_mode → self.is_studio_mode() in der Metadata-Assemblierung.
        """
        import pathlib

        uv3_path = pathlib.Path(__file__).parent.parent.parent / "backend" / "core" / "unified_restorer_v3.py"
        assert uv3_path.exists(), f"UV3 nicht gefunden: {uv3_path}"
        source = uv3_path.read_text(encoding="utf-8")
        # VERBOTEN: self._is_studio_mode als Attribut-Zugriff (existiert nicht!)
        assert "self._is_studio_mode" not in source, (
            "§UV3-CRASH BUG REINTRODUCED: self._is_studio_mode (Attribut) in UV3! "
            "UV3 hat nur die Methode self.is_studio_mode() — Attribut-Zugriff löst "
            "AttributeError am Ende von restore() aus → gesamte Restaurierung wird "
            "weggeworfen (5791s Verarbeitungszeit verloren). "
            "Fix: Alle Vorkommen durch self.is_studio_mode() ersetzen."
        )

    def test_vqi_formant_stability_uses_bandpass_not_full_spectrum(self):
        """§VQI-Formant-Guard: LPC-Bandpass 200-3400 Hz und Order 14 Pflicht.

        Bug (v10.0.0): _compute_formant_stability() verwendete LPC-Order=50 (sr/1000+2 bei
        48 kHz) auf dem vollen Mix ohne Bandpass-Filter → formant=0.139 als Messartefakt
        bei HF-Boost durch Restaurierung (Centroid-Shift 1664→4101 Hz).
        Fix: Bandpass 200-3400 Hz vor LPC, Order=14.
        """
        import pathlib

        vqi_path = (
            pathlib.Path(__file__).parent.parent.parent
            / "backend"
            / "core"
            / "musical_goals"
            / "vocal_quality_index.py"
        )
        assert vqi_path.exists(), f"VQI nicht gefunden: {vqi_path}"
        source = vqi_path.read_text(encoding="utf-8")
        # Bandpass-Import in _compute_formant_stability (scipy)
        assert "sosfiltfilt" in source, (
            "§VQI-Formant-BUG: Kein Bandpass in _compute_formant_stability! "
            "LPC auf vollem Mix ohne Filter → formant≈0.0 bei HF-Boost (Centroid-Shift)."
        )
        # Fester LPC-Order 14 (nicht sr/1000+2=50 bei 48kHz)
        assert "lpc_order = 14" in source, (
            "§VQI-Formant-BUG: lpc_order != 14 in _compute_formant_stability! "
            "sr/1000+2=50 bei sr=48000 erfasst alle Mix-Spektralspitzen als 'Formanten'. "
            "Standard für 200-3400 Hz Vokal-Formant-Analyse: Order 14."
        )


class TestFixAbcCassetteHissNoveltyFlutter:
    """Linter-Guard-Tests für Fix A (Phase 29 Kassette+MP3), Fix B (SFT NOVELTY_CRIT
    Restoration-Rollback) und Fix C (Phase 12 Kassette confidence-Threshold).

    Root Causes (Elke Best Restoration, v10.0.0):
    - Fix A: Phase 29 Bypass mit falschem 22.0 dB Threshold und 8kHz+ Hiss-Band.
      Kassetten-Hiss liegt in 4-8 kHz (MP3 schneidet 8kHz+ weg) → immer bypass.
    - Fix B: SFT ArtifactRescue setzte wet=0.30 auch bei NOVELTY_CRIT=0.551
      (55% Halluzination) → Musical Noise in allen Downstream-Phasen.
    - Fix C: phase_12 hatte CASSETTE nicht im Tape-Threshold-Block (0.25).
    """

    def test_phase29_cassette_mp3_hiss_band_is_4_to_8khz(self):
        """Fix A: Phase 29 muss für Kassette+MP3 Hiss-Band 4-8 kHz verwenden.

        Bug: Hiss-Band war fest 8kHz+ → Kassetten-Hiss in 4-8 kHz nicht erkannt
        → HF-SNR=28.3 dB > 22 dB → Bypass → OMLSA nie ausgeführt.
        Fix: Hiss-Band 4-8 kHz + Threshold 36 dB für cassette+mp3.
        """
        import pathlib

        p29_path = (
            pathlib.Path(__file__).parent.parent.parent
            / "backend"
            / "core"
            / "phases"
            / "phase_29_tape_hiss_reduction.py"
        )
        assert p29_path.exists(), f"phase_29 nicht gefunden: {p29_path}"
        source = p29_path.read_text(encoding="utf-8")
        # Materialadaptiver Hiss-Band-Zweig für cassette+mp3
        assert "_p29_chain_has_mp3" in source, (
            "Fix A fehlt: _p29_chain_has_mp3 nicht in phase_29! "
            "MP3-Erkennung für Kassette+MP3-Hiss-Band-Selektion fehlt."
        )
        assert "_p29_is_tape_mat" in source, (
            "Fix A fehlt: _p29_is_tape_mat nicht in phase_29! Tape-Material-Erkennung für Hiss-Band fehlt."
        )
        # Kassetten-Hiss-Band 4-8 kHz (nicht 8kHz+)
        assert "4000.0" in source and "8000.0" in source, (
            "Fix A fehlt: Hiss-Band 4000-8000 Hz nicht in phase_29! Kassetten-Hiss bei MP3 liegt in 4-8 kHz."
        )
        # Threshold 36.0 dB für cassette+mp3
        assert "36.0" in source, (
            "Fix A fehlt: Bypass-Threshold 36.0 dB nicht in phase_29! "
            "Alter Threshold 22.0 dB verursachte immer bypass bei Kassette+MP3."
        )
        # Alter universeller 22.0-Threshold darf nicht mehr als einzige Threshold-Logik stehen
        # (er darf irgendwo in Kommentaren stehen, aber nicht als einziger Threshold)
        assert "36.0" in source and "35.0" in source, (
            "Fix A: Phase 29 soll materialadaptive Thresholds haben (36.0 für cassette+mp3, 35.0 für sonstige)."
        )

    def test_uv3_sft_novelty_crit_high_triggers_full_rollback_in_restoration(self):
        """Fix B / §G71 (GEBOTE.md): SFT NOVELTY_CRIT verwendet dynamische Wet-Ceilings aus CalibrationContext.

        §v10.35: NIEMALS wet=0.0 für NOVELTY_CRIT — selbst kritische Novelty behält min. 20% Phase-Output.
        LEVEL_COLLAPSE löst wet=0.0 aus (Prioritätskette §G70).
        NOVELTY_CRIT verwendet depth-adaptive Ceilings aus calibrate_sft_thresholds() (§G71 (GEBOTE.md)).
        min_strength skaliert mit restorability_score (§G71 (GEBOTE.md)).
        """
        import pathlib

        uv3_path = pathlib.Path(__file__).parent.parent.parent / "backend" / "core" / "unified_restorer_v3.py"
        assert uv3_path.exists(), f"UV3 nicht gefunden: {uv3_path}"
        source = uv3_path.read_text(encoding="utf-8")
        # LEVEL_COLLAPSE löst vollständigen Rollback aus (§v10.38)
        assert "LEVEL_COLLAPSE" in source and "_sft_wet = 0.0" in source, (
            "§v10.38 fehlt: LEVEL_COLLAPSE muss _sft_wet = 0.0 triggern (Audio zerstört)."
        )
        # Novelty-Wert wird aus Flag-String geparst
        assert "_sft_novelty_val" in source, (
            "Fix B fehlt: _sft_novelty_val nicht in UV3! Novelty-Wert muss aus NOVELTY_CRIT(x.xxx)-Flag geparst werden."
        )
        # is_studio_mode()-Check (kein Rollback in Studio 2026)
        assert "not self.is_studio_mode()" in source, (
            "Fix B fehlt: is_studio_mode()-Check in SFT ArtifactRescue nicht vorhanden! "
            "Restoration-only Rollback muss via not self.is_studio_mode() gesichert sein."
        )
        # §G71 (GEBOTE.md) Dynamische Wet-Ceilings aus CalibrationContext (nicht mehr hartcodiert 0.80/0.70)
        assert "get_sft_wet_ceilings" in source, (
            "§G71 (GEBOTE.md) fehlt: get_sft_wet_ceilings() nicht in UV3! "
            "Wet-Ceilings müssen aus calibrate_sft_thresholds() kommen, nicht hartcodiert 0.70/0.80."
        )
        # §G71 (GEBOTE.md) min_strength aus Restorability (joint_calibrator)
        jcal_path = pathlib.Path(__file__).parent.parent.parent / "backend" / "core" / "joint_calibrator.py"
        jcal_source = jcal_path.read_text(encoding="utf-8") if jcal_path.exists() else ""
        assert "restorability_score" in jcal_source, (
            "§G71 (GEBOTE.md) fehlt: joint_calibrator muss restorability_score für adaptive min_strength akzeptieren."
        )

    def test_phase12_cassette_has_same_confidence_threshold_as_tape(self):
        """Fix C: MaterialType.CASSETTE muss denselben confidence-Threshold 0.25 wie TAPE haben.

        Bug: Phase 12 hatte _MIN_CONFIDENCE_FOR_CORRECTION = 0.25 nur für MaterialType.TAPE.
        MaterialType.CASSETTE blieb bei 0.40 → Log: 'Konfidenz 0.000 < 0.40'
        → konservativer Fallback → keine Flutter-Vollkorrektur auf Kassettenmaterial.
        Fix: if material in (MaterialType.TAPE, MaterialType.CASSETTE): threshold = 0.25.
        """
        import pathlib

        p12_path = (
            pathlib.Path(__file__).parent.parent.parent / "backend" / "core" / "phases" / "phase_12_wow_flutter_fix.py"
        )
        assert p12_path.exists(), f"phase_12 nicht gefunden: {p12_path}"
        source = p12_path.read_text(encoding="utf-8")
        # Prüfe exakt das kombinierte Pattern — nicht erste CASSETTE-Erwähnung
        assert "in (MaterialType.TAPE, MaterialType.CASSETTE)" in source, (
            "Fix C fehlt: 'in (MaterialType.TAPE, MaterialType.CASSETTE)' nicht in phase_12! "
            "if material in (MaterialType.TAPE, MaterialType.CASSETTE): threshold = 0.25"
        )
        # Der 0.25-Threshold muss im Block nach dem kombinierten Pattern stehen
        block_idx = source.find("in (MaterialType.TAPE, MaterialType.CASSETTE)")
        threshold_idx = source.find("_MIN_CONFIDENCE_FOR_CORRECTION = 0.25", block_idx)
        assert threshold_idx >= 0, (
            "Fix C fehlt: _MIN_CONFIDENCE_FOR_CORRECTION = 0.25 nicht direkt nach "
            "'in (MaterialType.TAPE, MaterialType.CASSETTE)'-Block! Fix C nicht angewendet."
        )


class TestSSC1VQIFix:
    """Guard-Tests für SSC-1 VQI=0.000 Bug-Fix (None-safe Konvertierung in aurik_denker.py).

    Root Cause: dict.get("vqi", 0.0) gibt None zurück wenn Key vorhanden aber Value=None.
    float(None) → TypeError → gesamter try-Block bricht ab → SSC-1 speichert VQI=0.000.
    Fix: None-safe Konvertierung mit isinstance-Prüfung auf (int, float).
    """

    def test_ssc1_vqi_none_safe_pattern_in_aurik_denker(self) -> None:
        """aurik_denker.py muss None-safe VQI-Konvertierung für SSC-1 Store enthalten."""
        import pathlib

        denker_path = pathlib.Path("denker/aurik_denker.py")
        assert denker_path.exists(), "denker/aurik_denker.py nicht gefunden"
        source = denker_path.read_text(encoding="utf-8")
        assert '_ssc_vqi_raw = _rest_metadata.get("vqi")' in source, (
            "SSC-1 VQI-Fix fehlt: '_ssc_vqi_raw = _rest_metadata.get(\"vqi\")' nicht in aurik_denker.py! "
            "float(None) → TypeError wenn metadata['vqi'] = None → SSC-1 speichert VQI=0.000."
        )
        assert "isinstance(_ssc_vqi_raw, (int, float)) and _ssc_vqi_raw > 0.0" in source, (
            "SSC-1 VQI-Fix fehlt: isinstance-Prüfung auf (int, float) nicht gefunden! "
            "Fix muss sein: float(_ssc_vqi_raw) if isinstance(_ssc_vqi_raw, (int, float)) and _ssc_vqi_raw > 0.0"
        )
        # Sicherstellen dass der fehlerhaft Pattern entfernt wurde
        assert 'float(_rest_metadata.get("vqi", 0.0))' not in source, (
            "Fehlerhaftes Pattern noch vorhanden: float(_rest_metadata.get('vqi', 0.0)) gibt None zurück "
            "wenn Key vorhanden aber Value=None → TypeError!"
        )

    def test_apr_adaptive_post_repair_block_in_aurik_denker(self) -> None:
        """aurik_denker.py: §APR muss Studio-2026-only sein (§0a Crossfire-Modus-Invariante)."""
        import pathlib

        denker_path = pathlib.Path("denker/aurik_denker.py")
        assert denker_path.exists(), "denker/aurik_denker.py nicht gefunden"
        source = denker_path.read_text(encoding="utf-8")
        assert "§APR" in source, "§APR Adaptive Post-Repair Block fehlt in aurik_denker.py!"
        # §0a Crossfire-Modus-Invariante: §APR darf NUR in Studio 2026 laufen
        assert "_apr_is_studio" in source, (
            "§APR Mode-Gate fehlt: '_apr_is_studio' nicht in aurik_denker.py! "
            "§APR muss auf Studio 2026 beschränkt sein (§0a: kein Bandpass-EQ-Boost in Restoration)."
        )
        assert "_apr_is_studio and not _rest_rollback" in source, (
            "§APR Mode-Gate nicht korrekt verknüpft: '_apr_is_studio and not _rest_rollback' fehlt! "
            "§0a Crossfire-Modus-Invariante verletzt."
        )
        assert "_apr_mos_before" in source and "_apr_mos_after" in source, (
            "§APR VERSA Gate fehlt: _apr_mos_before/_apr_mos_after nicht gefunden!"
        )
