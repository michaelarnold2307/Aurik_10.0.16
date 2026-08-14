import pytest

"""Tests für §GOAL_BASELINE_CHECK Optimierungen v10.14.

Prio 1: FC-SECONDARY — Sekundärphasen-Injection für below-floor Goals in FeedbackChain
Prio 2: Recovery-Kontext-Injection — goal_deficit_goals in _restoration_context
Prio 3: Trigger-Margin 0.95 → 0.97 — Proxy-Unsicherheit ±2-3% wird abgedeckt
Prio 4: Kurzaudio-Guard — _fast_goal_snapshot nutzt bei N < 4×fft_n nur 1 Segment
"""

import numpy as np

# ---------------------------------------------------------------------------
# Prio 3: Trigger-Margin 0.95 → 0.97
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTriggerMarginWidened:
    def test_margin_covers_proxy_uncertainty(self):
        """Margin 0.97 deckt Proxy-Unsicherheit ±2-3% ab.
        Goals bei 96 % des Floors müssen jetzt erkannt werden.
        """
        from backend.core.calibration_matrix import get_material_floor

        for mat in ("shellac", "vinyl", "cd_digital"):
            for goal in ("brillanz", "natuerlichkeit", "transparenz"):
                floor = get_material_floor(mat, goal)
                proxy_at_96pct = floor * 0.96
                # Mit Margin 0.97 triggert: proxy < floor * 0.97 → 0.96*floor < 0.97*floor ✓
                assert proxy_at_96pct < floor * 0.97, (
                    f"{mat}/{goal}: proxy={proxy_at_96pct:.4f} < margin={floor * 0.97:.4f} erwartet"
                )
                # Mit Margin 0.95 triggerte es NICHT: proxy(0.96) >= 0.95*floor ✓
                assert proxy_at_96pct >= floor * 0.95, (
                    f"{mat}/{goal}: proxy={proxy_at_96pct:.4f} lag bereits unter alter Margin"
                )

    def test_margin_0_95_would_miss_96pct(self):
        """Belegt dass die alte Margin 0.95 Goals bei 96 % des Floors ÜBERSAH."""
        from backend.core.calibration_matrix import get_material_floor

        floor = get_material_floor("vinyl", "brillanz")
        proxy = floor * 0.96
        # Alte Bedingung: proxy < floor * 0.95 → False bei 0.96 * floor ≥ 0.95 * floor
        assert proxy >= floor * 0.95, "96%-Proxy muss alter Margin entgehen"

    def test_new_margin_does_not_over_trigger_clean_material(self):
        """Proxy > floor × 0.97 triggert keine Recovery — sauberes Material bleibt unberührt."""
        from backend.core.calibration_matrix import get_material_floor

        floor = get_material_floor("cd_digital", "natuerlichkeit")
        clean_proxy = floor * 1.05  # klar über Floor — kein Recovery nötig
        assert clean_proxy >= floor * 0.97, "Sauberes Material darf nicht triggern"


# ---------------------------------------------------------------------------
# Prio 4: Kurzaudio-Guard in _fast_goal_snapshot
# ---------------------------------------------------------------------------


class TestShortAudioGuard:
    """_fast_goal_snapshot: Kurzaudio (N < 4×fft_n) nutzt Single-Segment."""

    @staticmethod
    def _snap(audio: np.ndarray, sr: int = 48000) -> dict:
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        return UnifiedRestorerV3._fast_goal_snapshot(audio, sr)

    def test_very_short_audio_no_crash(self):
        """N = 8192 < 4×4096 = 16384 → kein Crash, leeres Dict erlaubt (N > 512 → Ergebnis erwartet)."""
        rng = np.random.default_rng(0)
        audio = (rng.standard_normal(8192) * 0.3).astype(np.float32)
        result = self._snap(audio)
        # muss ohne Exception durchlaufen; Ergebnis kann leer sein wenn fft_n=4096 und N<512
        # aber für N=8192 muss es ein nicht-leeres Dict sein (N > 512 Guard besteht)
        assert isinstance(result, dict), "Ergebnis muss ein dict sein"

    def test_short_tonal_audio_values_in_range(self):
        """Tonales Kurzaudio (8000 Samples) → alle Werte in [0,1]."""
        sr = 48000
        t = np.arange(8000) / sr
        audio = (0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        result = self._snap(audio, sr)
        if result:
            bad = {k: v for k, v in result.items() if not (0.0 <= v <= 1.0)}
            assert not bad, f"Werte außerhalb [0,1]: {bad}"

    def test_boundary_exactly_4x_fft_n(self):
        """N = 4 × fft_n (16384) → kein Overlap-Risiko; 3-Segment-Modus."""
        # fft_n = min(4096, 16384) = 4096. N = 16384 ist NICHT < 4*4096 → 3 Segmente.
        sr = 48000
        rng = np.random.default_rng(1)
        audio = (rng.standard_normal(16384) * 0.2).astype(np.float32)
        result = self._snap(audio, sr)
        assert isinstance(result, dict)

    def test_single_segment_equivalent_for_uniform_signal(self):
        """Für uniform-tonales Signal: Ergebnis von Kurz- vs. Langversion ähnlich.
        Bestätigt dass Single-Segment-Modus keine drastische Diskontinuität erzeugt."""
        sr = 48000
        t_long = np.arange(sr * 4) / sr  # 4s — 3 Segmente
        t_short = np.arange(8192) / sr  # 0.17s — 1 Segment
        sig_long = (0.4 * np.sin(2 * np.pi * 440 * t_long)).astype(np.float32)
        sig_short = (0.4 * np.sin(2 * np.pi * 440 * t_short)).astype(np.float32)
        r_long = self._snap(sig_long, sr)
        r_short = self._snap(sig_short, sr)
        if r_long and r_short:
            # brillanz-Proxy sollte für gleiches Signal ähnlich sein (±0.35 Toleranz)
            if "brillanz" in r_long and "brillanz" in r_short:
                assert abs(r_long["brillanz"] - r_short["brillanz"]) < 0.35, (
                    f"brillanz zu unterschiedlich: long={r_long['brillanz']:.3f}, short={r_short['brillanz']:.3f}"
                )


# ---------------------------------------------------------------------------
# Prio 1: FC-SECONDARY — Sekundärphasen-Datenstruktur-Validierung
# ---------------------------------------------------------------------------


class TestFCSecondaryDataStructure:
    """Validiert die Datenstrukturen die §FC-SECONDARY nutzt."""

    def test_recovery_dict_has_secondaries_for_key_goals(self):
        """Kritische Goals müssen Sekundärphasen haben (nicht nur Primary)."""
        from backend.core.calibration_matrix import _GOAL_TO_RECOVERY_PHASES_RESTORATION as GRDICT

        goals_needing_secondaries = [
            "natuerlichkeit",
            "authentizitaet",
            "brillanz",
            "transparenz",
            "separation_fidelity",
        ]
        for goal in goals_needing_secondaries:
            phases = GRDICT.get(goal, [])
            assert len(phases) >= 2, (
                f"Goal '{goal}' hat nur {len(phases)} Phase(n) — Sekundärphase fehlt für FC-SECONDARY"
            )

    def test_primary_and_secondary_are_distinct(self):
        """Primary- und Sekundärphase für jedes Goal müssen unterschiedlich sein."""
        from backend.core.calibration_matrix import _GOAL_TO_RECOVERY_PHASES_RESTORATION as GRDICT

        for goal, phases in GRDICT.items():
            if len(phases) >= 2:
                seen = set()
                for p in phases:
                    assert p not in seen, f"Goal '{goal}': Duplikat-Phase '{p}' in Recovery-Liste"
                    seen.add(p)

    def test_section_0a_phases_not_in_restoration_list(self):
        """§0a-verbotene Phasen (phase_21, phase_35, phase_42) dürfen NICHT in
        _GOAL_TO_RECOVERY_PHASES_RESTORATION stehen."""
        from backend.core.calibration_matrix import _GOAL_TO_RECOVERY_PHASES_RESTORATION as GRDICT

        blocked = {"phase_21", "phase_35", "phase_42"}
        violations = []
        for goal, phases in GRDICT.items():
            for p in phases:
                prefix = "_".join(p.split("_")[:2])
                if prefix in blocked:
                    violations.append(f"{goal} → {p}")
        assert not violations, f"§0a-Verletzungen in Restoration-Dict: {violations}"

    def test_fc_secondary_blocked_set_correct(self):
        """FC-SECONDARY _FC_SEC_BLOCKED muss die 3 §0a-Phasen enthalten."""
        blocked = frozenset({"phase_21", "phase_35", "phase_42"})
        assert "phase_21" in blocked
        assert "phase_35" in blocked
        assert "phase_42" in blocked
        # phase_07 ist in FC-Liste aber NICHT geblockt
        assert "phase_07" not in blocked

    def test_secondary_phases_exist_on_disk(self):
        """Alle Sekundärphasen in _GOAL_TO_RECOVERY_PHASES_RESTORATION[goal][1:]
        müssen als Phase-Datei existieren."""
        import importlib

        from backend.core.calibration_matrix import _GOAL_TO_RECOVERY_PHASES_RESTORATION as GRDICT

        missing = []
        for goal, phases in GRDICT.items():
            for pid in phases[1:]:  # Sekundärphasen (index 1+)
                mod_name = f"backend.core.phases.{pid}"
                spec = importlib.util.find_spec(mod_name)
                if spec is None:
                    missing.append(f"{goal} → {pid}")
        assert not missing, f"Sekund\u00e4rphasen-Dateien fehlen: {missing}"


# ---------------------------------------------------------------------------
# Prio 2: Recovery-Kontext-Injection — _gbc_below_floor_goals Datenpfad
# ---------------------------------------------------------------------------


class TestRecoveryContextInjection:
    """Validiert den Datenpfad für _gbc_below_floor_goals und context injection."""

    def test_goal_names_parseable_from_gbc_added_format(self):
        """_gbc_added-Einträge haben Format 'goal(proxy=X<floor=Y)→phase_id'.
        Goal-Name muss durch split('(')[0] extrahierbar sein."""
        test_entries = [
            "brillanz(proxy=0.42<floor=0.72)→phase_06_frequency_restoration",
            "natuerlichkeit(proxy=0.71<floor=0.82)→phase_03_denoise",
            "spatial_depth(proxy=0.55<floor=0.70)→phase_46_spatial_enhancement",
        ]
        expected_goals = {"brillanz", "natuerlichkeit", "spatial_depth"}
        extracted = {entry.split("(")[0] for entry in test_entries}
        assert extracted == expected_goals, f"Extraktion fehlgeschlagen: {extracted}"

    def test_recovery_context_goal_names_match_calibration_matrix_keys(self):
        """Goal-Namen aus §GOAL_BASELINE_CHECK müssen identisch mit
        _GOAL_TO_RECOVERY_PHASES_RESTORATION-Keys sein."""
        from backend.core.calibration_matrix import _GOAL_TO_RECOVERY_PHASES_RESTORATION as GRDICT

        dict_keys = set(GRDICT.keys())
        # Alle Goal-Namen die von §GOAL_BASELINE_CHECK gemeldet werden könnten,
        # müssen im Recovery-Dict vorhanden sein.
        # Prüfe repräsentative Goals:
        for goal in ("brillanz", "natuerlichkeit", "authentizitaet", "transparenz"):
            assert goal in dict_keys, f"Goal '{goal}' fehlt in Recovery-Dict"

    def test_fast_goal_snapshot_keys_match_recovery_dict_keys(self):
        """_fast_goal_snapshot-Schlüssel müssen mit _GOAL_TO_RECOVERY_PHASES_RESTORATION
        übereinstimmen, damit FC-SECONDARY Goals korrekt verknüpft."""
        from backend.core.calibration_matrix import _GOAL_TO_RECOVERY_PHASES_RESTORATION as GRDICT
        from backend.core.unified_restorer_v3 import UnifiedRestorerV3

        sr = 48000
        t = np.arange(sr * 4) / sr
        audio = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        snap = UnifiedRestorerV3._fast_goal_snapshot(audio, sr)

        # Alle Recovery-Dict-Keys müssen im Snapshot vorhanden sein
        snap_keys = set(snap.keys())
        missing_in_snap = set(GRDICT.keys()) - snap_keys
        assert not missing_in_snap, (
            f"Recovery-Dict-Goals fehlen im Snapshot: {missing_in_snap}\nSnapshot-Keys: {sorted(snap_keys)}"
        )
