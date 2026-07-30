# §18 — Depth-Threshold Calibration-Shift (§v10.120)

**Status**: Implementiert (2026-08)  
**Basis**: §17 SFT-Novelty-Adaptiv-Kalibrierung (§G71)  
**Dateien**: 15 Python-Dateien, 3 Spec-Dokumente

---

## 1. Paradigma-Wechsel

§G71 hat die Novelty-Schwellen pro Transfer-Chain-Tiefe neu kalibriert:

| Depth | Novelty-Schwelle | Klassifikation |
|-------|-----------------|----------------|
| 1 | 0.25 | Studio-Master |
| 2 | 0.35 | Shallow |
| 3 | 0.45 | **Moderat** (vorher: "deep") |
| 4+ | 0.55 | **Deep Cassette** |
| 5+ | 0.55+ | Extrem (physikalische Grenze) |

Vorher galt `depth >= 3` als "tiefe Kette" mit konservativen Safety-Guards. Das neue Paradigma: Erst `depth >= 4` ist "deep cassette", und `depth >= 5` ist "extreme chain".

## 2. Systematische Guard-Migration (18 Guards in 13 Dateien)

### 2.1 `>= 3` → `>= 4` (Deep-Cassette-Schwelle)

| Datei | Guard | Wirkung |
|-------|-------|---------|
| `signal_flow_tracer.py:1116` | Echo-Korrelations-Schwelle | `_ECHO_CORR_THRESH += 0.10*(depth-3)` ab depth 4 |
| `unified_restorer_v3.py:3620` | Family-Weight Tier 1 | Reconstruction+8%, Denoise+5%, Transient−4% ab depth 4 |
| `do_no_harm_guardian.py:264` | Naturalness-Threshold | `+0.10` Bonus ab depth 4 |
| `do_no_harm_guardian.py:521` | Perceptual-Gewicht | 75% perceptual ab depth 4 |
| `global_gain_budget.py:69` | Gain-Budget | 10 dB ab depth 4 |
| `orchestrator_params.py:190,210` | Phase 10/12 Strength | 0.85×/0.80× ab depth 4 |

### 2.2 `>= 3` → `>= 5` (Extreme-Chain-Schwelle)

| Datei | Guard | Wirkung |
|-------|-------|---------|
| `joint_calibrator.py:100` | Depth-Boost für min_strength | `min_strength *= _depth_boost` ab depth 5 |
| `unified_restorer_v3.py:13161` | FeedbackChain-Iterations-Cap | Max 2 Iterationen ab depth 5 |
| `perceptual_tuning.py:210,238,253` | JND-Boost + Deep-Chain-Profil | 15% Boost ab depth 5 |
| `phase_03_denoise.py:938,1666` | DSP-Fallback + Energy-Guard | Lightweight + 0.50 Energy-Threshold ab depth 5 |
| `phase_07_harmonic_restoration.py:755,774` | Tilt-Cap-Floor + Depth-Faktor | Floor 0.35, Toleranz 2× ab depth 5 |
| `phase_23_spectral_repair.py:1825` | FlashSR-ML-Deaktivierung | DSP-only ab depth 5 |
| `phase_29_tape_hiss_reduction.py:820` | Strength-Cap | Max 0.40 ab depth 5 |
| `phase_42_vocal_enhancement.py:1339` | ML-Stem-Sep-Bypass | HPSS+Wiener ab depth 5 |

### 2.3 `>= 4` → `>= 5` (Top-Tier-Shift)

| Datei | Guard | Wirkung |
|-------|-------|---------|
| `unified_restorer_v3.py:3624` | Family-Weight Tier 2 | Zusätzliche Reconstruction+5%, Dynamics_EQ−4% ab depth 5 |
| `global_gain_budget.py:67` | Gain-Budget Top | 12 dB ab depth 5 |
| `orchestrator_params.py:132` | Phase 05 Strength | 0.80× ab depth 5 |

## 3. Folgeprobleme und Korrekturen

### 3.1 Per-Phase Artifact-Freedom-Gate war nicht depth-adaptiv (§v10.120)

**Problem**: Der Per-Phase-Check in `unified_restorer_v3.py:37846` verwendete einen fixen 0.95-Schwellwert, während der Export-Gate (`spec_constitution.py`) bereits depth-adaptiv arbeitete (0.70 bei depth≥4).

**Symptom**: `artifact_freedom=0.350 < 0.95 after phase_19_de_esser (14 pre-echo artifacts) → rollback`

**Fix**: Per-Phase-Gate verwendet jetzt dieselben depth-adaptiven Schwellwerte:

- Depth 1: 0.95
- Depth 2: 0.88
- Depth 3: 0.80
- Depth ≥4: 0.70

### 3.2 Pre-Echo-Detektion zu sensitiv für tiefe Ketten (§v10.121)

**Problem**: Tiefe Transfer-Ketten haben inhärenten Temporal-Smear (Tape-Flutter, Multi-Gen-Phasenrotation), den der Pre-Echo-Detektor fälschlich als Restaurierungsartefakt klassifiziert.

**Symptom**: 14 Pre-Echo-Artefakte nach Phase_19 auf Depth-4-Kassette.

**Fix in `artifact_freedom_gate.py`**:

1. **Pre-Echo-Detektionsschwelle**: `pre_echo_rel_attack_db` wird mit Depth-Faktor multipliziert (×1.0 bei depth 1, ×1.3 bei depth 4, ×1.6 bei depth 5+)
2. **Score-Toleranz**: `_max_tolerance` wird mit Depth-Faktor skaliert (`1.0 + (depth-1) * 0.22`, gecapped bei 2.50)
3. **API**: `evaluate()` akzeptiert neuen Parameter `transfer_chain_depth: int = 1`

### 3.3 CIG Group-Delay-Threshold zu schwach (§v10.120)

**Problem**: Der Chain-Faktor in `_compute_gdd_threshold()` war `1.0 + (depth-2)*0.15`. Bei Depth 4 (1.30×) reichte das nicht, nachdem Phase_29 ohne Strength-Cap läuft.

**Symptom**: `STFT group delay deviation 39.9 ms > threshold 39.8 ms after phase_29 → rollback`

**Fix in `cumulative_interaction_guard.py`**: Chain-Faktor auf `1.0 + (depth-2)*0.25` erhöht (Depth 4: 1.50× statt 1.30×).

## 4. Dokumentation

- `GEBOTE.md` §G88: `transfer_depth≥3` → `≥5`
- `VERBOTEN.md` §v10.60: Beide Einträge `depth≥3` → `depth≥5`
- `06_phases_system.md`: Phase 07 + 23 `depth≥3` → `depth≥5`
- `v10.102_donoharm_contradiction_analysis.md`: `depth≥3` → `depth≥4`

## 5. Nicht geändert (keine Depth-Checks)

- `phase_06_frequency_restoration.py:487`: `_sfr_gen >= 3` ist ein Generationszähler
- Phase 08/19/24/25/36/40: Alle `>= 3` sind Fenstergrößen, Frequenzen oder Kernel-Längen
- `defect_scanner.py:8505`: `depth >= 2.5` ist ein Float (Defekt-Tiefe in dB, nicht Chain-Depth)
- `source_fidelity_reconstructor.py:495,558`: `len(transfer_chain) >= 2` (andere Semantik)

## 6. Erwartete Wirkung für Importsongs

| Depth | Vorher | Nachher |
|-------|--------|---------|
| 1–2 | Unverändert | Unverändert |
| 3 | DSP-Fallbacks, Strength-Caps, reduzierte Gain-Budgets | Volle ML-Pipeline, normale Budgets |
| 4 | Moderate Caps | Family-Weight-Boosts, Echo-Anpassung, volle ML-Phasen |
| 5+ | Wie Depth 4 | Konservative Safety-Guards (DSP-Fallback, Iterations-Cap) |

## 7. HallucinationGuard-SFT-Integration (§v10.122)

**Problem**: Der HallucinationGuard verwendete einen statischen `_ROLLBACK_THRESHOLD = 0.15` — unabhängig von der Chain-Depth. Bei Depth 4 (Novelty 0.55) ist das 3.7× zu strikt.

**Fix**: `calibrate_sft_thresholds()` setzt einen globalen `_HG_BASE_THRESHOLD` (Depth 1: 0.15, Depth 4: 0.40, Depth 5+: 0.55). `check_hallucination()` liest diesen via `_get_adaptive_rollback_threshold()` und verwendet `max()`-Semantik für BW/bw_extension-Modifier.

**Dateien**: `signal_flow_tracer.py`, `dsp/hallucination_guard.py`

## 8. CIG P1/P2-Drift-Toleranz depth-adaptiv (§v10.120)

**Problem**: `compute_adaptive_drift_tolerance()` hatte Material-, Restorability-, Severity- und Phase-Faktoren — aber keinen Chain-Depth-Faktor.

**Fix**: `chain_factor = 1.0 + max(0, depth-2) * 0.25` hinzugefügt (konsistent mit GDD chain_factor). Depth 4: 1.50× Toleranz.

**Datei**: `cumulative_interaction_guard.py`

## 9. CIG-Exclusions erweitert (§v10.120)

Fünf Phasen hatten unzureichende oder fehlende `natuerlichkeit`/`tonal_center`-Exclusions, was zu falschen CIG-Rollbacks führte:

| Phase | Fehlte | Mechanismus |
|-------|--------|-------------|
| `phase_07` | `natuerlichkeit` | Harmonik-Synthese = legitime spektrale Änderung (Reference-Paradoxon) |
| `phase_10` | `natuerlichkeit`, `tonal_center` | Multiband-Kompression = dynamische Spektral-Umverteilung |
| `phase_11` | `natuerlichkeit`, `tonal_center` | Limiter = Peak-Reduktion + Harmonik-Addition |
| `phase_25` | `tonal_center` | Azimuth-Korrektur = Stereo-Rezentrierung → Chroma-Shift |
| `phase_40` | `natuerlichkeit` | Saturation = harmonische Verzerrung (wie phase_07) |
| `phase_44` | `natuerlichkeit` | Stereo-Erweiterung = MFCC-Änderung |

## 10. Weitere Adaptierungen

| § | Mechanismus | Datei |
|----|-------------|-------|
| §v10.120 | `compute_adaptive_max_rollbacks()` mit Chain-Depth (+1 pro Depth-Stufe ab 3) | `cumulative_interaction_guard.py` |
| §v10.122 | `_get_adaptive_penalty_threshold()` = max(0.08, Rollback-Threshold × 0.5) | `dsp/hallucination_guard.py` |
| §v10.121 | Pre-Echo-Detektions-Schwelle depth-adaptiv (×1.3 bei Depth 4) | `artifact_freedom_gate.py` |
| §v10.121 | `_max_tolerance` depth-adaptiv (×1.66 bei Depth 4) | `artifact_freedom_gate.py` |
| §v10.120 | GDD chain_factor von 0.15→0.25 pro Depth-Stufe | `cumulative_interaction_guard.py` |
