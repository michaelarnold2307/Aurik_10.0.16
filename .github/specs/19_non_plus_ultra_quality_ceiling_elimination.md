# §19 — Non-Plus-Ultra: Strukturelle Qualitäts-Deckel beseitigt (§v10.90–§v10.94)

> **Status:** Spec | **Version:** 10.0.94 | **Datum:** 2026-08-10
>
> Die vier unabhängigen Root-Causes für „43→43" (keine messbare Qualitätsverbesserung)
> sind identifiziert und behoben. Diese Spec dokumentiert die architektonischen Fixes
> und die neuen GEBOTE (§G90–§G99), die verhindern, dass Aurik jemals wieder:
>
> - gegen den defekten Input vergleicht
> - Exception-Schlucker ohne Logging verwendet
> - Phasen ohne Cross-Phase-Koordination laufen lässt
> - numerische NaN/Inf-Pfade ungeschützt lässt

---

## §19.1 — Root Cause 1: SFT wet=0.05 verwarf 95% des Outputs (§G71, behoben in v10.40)

### Problem

SFT NOVELTY_CRIT-Schwelle war statisch 0.15 für ALLE Songs. JEDE Phase triggerte NOVELTY_CRIT (Werte 0.40–0.50), und nicht-Repair-Phasen bekamen wet=0.05 — 95% des Outputs wurde verworfen.

### Lösung: Adaptive SFT-Kalibrierung

- **Transfer-Chain-Tiefe adaptiv**: Depth 1→0.25, Depth 2→0.35, Depth 3→0.45, Depth 4+→0.55
- **Restorability-Modulation**: excellent -0.10, good -0.05, fair 0.00, poor +0.10
- **Dynamische Wet-Ceilings**: Non-repair 0.72+0.05×(depth-1), Repair 0.82+0.05×(depth-1), clamped [0.65,0.90]/[0.75,0.95]
- **Adaptive min_strength**: rs≥90→0.20, rs≥60→0.35, rs≥30→0.40, else→0.45

**Dateien**: `signal_flow_tracer.py`, `joint_calibrator.py`, `unified_restorer_v3.py`
**Spec**: `specs/17_sft_novelty_adaptive_calibration.md`
**GEBOTE**: §G68–§G75

---

## §19.2 — Root Cause 2: Kumulative Phasenverzerrung durch lfilter (§v10.65)

### Problem

Sechs DSP-Dateien verwendeten `signal.lfilter` (kausal, minimum-phase) für EQ-Filter.
Nach 15 EQ-Phasen summierte sich die Phasenverzerrung auf > 15 ms Gruppenlaufzeit —
hörbar als Zeitversatz auf Vokaleinsätzen, Pre-Echo und Stereobild-Verschiebung.

### Lösung: Zero-Phase für Magnituden-EQ

- Alle Biquad-EQ-Filter: `signal.lfilter(b, a, audio)` → `signal.sosfiltfilt(sos, audio)`
- Perzeptuelle Bänder (Phase_38, 2–8 kHz): `lfilter` beibehalten (Minimum-Phase verhindert Pre-Ringing)
- Alle 6 DSP-Dateien geprüft, 5 bereits korrekt (filtfilt primary, lfilter nur Fallback < 9 Samples)

**Dateien**: `phase_16_final_eq.py`, `phase_39_air_band_enhancement.py`, `phase_17_mastering_polish.py`, `phase_38_presence_boost.py`, `phase_42_vocal_enhancement.py`

---

## §19.3 — Root Cause 3: HPI verglich gegen defekten Input (§v10.0, §v10.91)

### Problem

`evaluate_restoration()` fiel bei `reference_audio=None` still auf `original` (degraded_input) zurück.
Jede erfolgreiche Restaurierung (Rauschen entfernt, BW erweitert) wurde als „Abweichung vom Original"
gemessen → Score blieb unverändert (43→43).

### Lösung: VERSA + Blinder Referenz-Vektor

**Primärpfad (95%+)**: VERSA MOS — referenzfrei, bewertet nur das restaurierte Audio.
**Fallback 1**: GP-Memory-Referenzvektoren (Genre×Material×Ära).
**Fallback 2** (§v10.91): Blinder Referenz-Vektor — Embedding des saubersten 5s-Fensters via BlindInternalReference.
**Fallback 3**: Direktionale Qualität — misst Original→Restored Verbesserungs-Delta.

```python
# holistic_perceptual_gate.py:_compute_blind_reference_vector()
# Findet das sauberste 5s-Fenster im restaurierten Audio
# Berechnet Mel-Embedding → verwendet als Referenz-Vektor für timbral_ref
# KEIN Audio-Vergleich (Shape-Mismatch vermieden)
```

**Dateien**: `holistic_perceptual_gate.py`, `unified_restorer_v3.py`, `blind_internal_reference.py`
**GEBOTE**: §G90, §G91

---

## §19.4 — Root Cause 4: Numerischer NaN-Kollaps des HPI (§v10.93)

### Problem

1. `max(float('nan'), 0.5) == nan` in Python. Wenn VERSA NaN zurückgab, kollabierte der HPI.
2. `np.log10(np.percentile(frames_rms, 10))` unguarded → `-inf` bei n_frames==0.
3. `noise_ratio` Epsilon-Kollision: 1e-10 in Zähler und Nenner → Stille als "noisy" klassifiziert.

### Lösung: NaN/Inf-Guards auf allen Pfaden

```python
# holistic_perceptual_gate.py:166
mert_sim = float(np.nan_to_num(mert_sim, nan=0.5))  # VOR max()
mert_sim = max(mert_sim, 0.5)

# holistic_perceptual_gate.py:385-394
hpi = mert_sim * timbral * artifact_freedom * emotional_arc_score
if not np.isfinite(hpi):
    logger.warning("HPI product NaN: ... → floor 0.5")
    hpi = 0.5

# excellence_optimizer.py:379-380
noise_floor_db = float(20 * np.log10(max(_p10, 1e-10)))

# difficulty_estimator.py:49-51
noise_ratio = log_mean / max(arith_mean, 1e-8)  # 1e-8 vs 1e-10 im Zähler
```

**Dateien**: `holistic_perceptual_gate.py`, `excellence_optimizer.py`, `difficulty_estimator.py`
**GEBOTE**: §G96, §G97

---

## §19.5 — Cross-Phase-Koordination (§v10.94)

### Problem

Phasen operierten isoliert auf denselben Frequenzbändern:

- P02 (Hum-Removal) notched 50/60 Hz → P37 (Bass-Enhancement) synthetisierte Bass in denselben Bändern
- P10 (Compression) komprimierte Bässe → P26 (Expansion) expandierte dieselben Bänder → Gain-Pumping
- P02 konnte NACH P03 (ML-Denoising) laufen → ML lernte Brumm als „Signal"

### Lösung: DAG-Constraint + Metadata-Handshakes

**C1 — DAG-Constraint (§G95)**:

```python
# phase_dag.py
PhaseConstraint("phase_02_hum_removal", "phase_03_denoise",
    "Hum-Notch-Filter vor ML-NR (§v10.94: verhindert Brumm-Lernen als Signal)")
```

**C2 — P02→P37 Metadata-Handshake (§G94a)**:

- P02 schreibt `modifications["fundamentals"]` = `[50, 60]`
- `_normalize_phase_result` extrahiert → `_restoration_context["hum_notch_freqs"]`
- `_canonical_phase_context_kwargs` injiziert in ALLE Phasen
- P37 liest `hum_notch_freqs` → reduziert `sub_harmonic_gain` um −20% pro überlappendem Fundamental

**C3 — P10→P26 Metadata-Handshake (§G94b)**:

- P10 schreibt `modifications["per_band_gain_db"]` = `{bass: 3.2, low_mid: 2.1}`
- `_normalize_phase_result` extrahiert → `_restoration_context["p10_per_band_gain_db"]`
- P26 liest `p10_per_band_gain_db` → reduziert `max_expansion_db` proportional

**Dateien**: `phase_dag.py`, `unified_restorer_v3.py`, `phase_10_compression.py`, `phase_37_bass_enhancement.py`, `phase_26_dynamic_range_expansion.py`
**GEBOTE**: §G94, §G95

---

## §19.6 — Material-Kalibrierung vervollständigt (§v10.92)

### Problem

- 8 Materialien fehlten in `AUTHENTIC_CHARACTER` → `return 1.0` (keine Preservation)
- 5 Materialien fehlten in `_MATERIAL_THRESHOLD_BONUS` → 0.003-Default
- `predict_quality_score()` war toter Code (nie importiert)
- `confidence` hatte harten 0.95-Deckel unabhängig vom Material
- 14× `return 0.5` Exception-Schlucker ohne Logging

### Lösung

| Fix | Datei | Effekt |
|-----|-------|--------|
| 8 AUTHENTIC_CHARACTER-Einträge | `intentional_artifact_classifier.py` | cassette, kassette, wire_recording, minidisc, dat, aac, streaming, lp |
| 5 THRESHOLD_BONUS-Einträge | `per_phase_musical_goals_gate.py` | lacquer_disc, lp, kassette, aac, streaming |
| `predict_quality_score` aktiviert | `feasibility_controller.py` | Confidence = Material-Ceiling × Restorability |
| 14 Exception-Proxies | 5 Dateien | Zeitdomain-Proxies + `exc_info=True` Logging |

**GEBOTE**: §G92, §G93, §G98, §G99

---

## §19.7 — Datenfluss: Vom Phase-Result zum Cross-Phase-Context

```
Phase-Result.modifications
         │
         ▼
_normalize_phase_result()          ← unified_restorer_v3.py:32535
  Extrahiert fundamentals, per_band_gain_db
         │
         ▼
_restoration_context               ← dict auf self
         │
         ▼
_canonical_phase_context_kwargs()  ← unified_restorer_v3.py:26554
  Injiziert hum_notch_freqs, p10_per_band_gain_db
         │
         ▼
Phase.kwargs                       ← ALLE Phasen empfangen
  kwargs.get("hum_notch_freqs", [])
  kwargs.get("p10_per_band_gain_db", {})
```

---

## §19.8 — HPI-Evaluations-Architektur (final)

```
HPI = mert_sim × timbral_fidelity × artifact_freedom × emotional_arc

mert_sim:
  ├─ VERSA MOS (primary, reference-free)           ✅ §v10.0
  ├─ Spectral Proxy Fallback (rare)                 ✅ BW-Ceiling-Guard
  └─ NaN-Guard: nan_to_num vor max()               ✅ §v10.93

timbral_fidelity:
  ├─ GP-Memory Ref-Vector (genre×material×era)     ✅ §2.44
  ├─ Blinder Ref-Vektor (best 5s embedding)        ✅ §v10.91
  ├─ Direktionale Qualität (original→restored Δ)   ✅ §2.44
  └─ Material-adaptive Floors (Codec, Analog)      ✅ §2.44

Produkt-Guard:
  └─ np.isfinite(hpi) → floor 0.5 + Warning-Log   ✅ §v10.93
```

---

## §19.9 — GEBOTE-Integration

| ID | Regel | Version |
|----|-------|---------|
| §G90 | Blinder-Referenz-Vektor-Pflicht | §v10.91 |
| §G91 | Embedding-basierte-Referenz-Pflicht | §v10.91 |
| §G92 | Material-adaptive-Confidence-Pflicht | §v10.92 |
| §G93 | Exception-Proxy-Pflicht | §v10.92, §v10.93 |
| §G94 | Cross-Phase-Metadata-Pflicht | §v10.94 |
| §G95 | Phase-02-vor-Phase-03-Pflicht | §v10.94 |
| §G96 | HPI-NaN-Guard-Pflicht | §v10.93 |
| §G97 | log10-Null-Guard-Pflicht | §v10.93 |
| §G98 | AUTHENTIC_CHARACTER-Vollständigkeit | §v10.92 |
| §G99 | Equality-of-Materials-Pflicht | §v10.92 |

---

## §19.10 — Verbleibende architektonische Notizen

**Bias-System vs. Material-Ceiling (§09.2)**: Ein uniformes Material-Ceiling (Shellac=0.70, CD=0.95)
ist für Goal-Targets UNGEEIGNET, weil verschiedene Qualitäts-Dimensionen unterschiedlich vom
Trägermaterial beeinflusst werden. Shellac hat z.B. PHYSIKALISCH mehr Wärme (Röhrenmikrofone)
als CD. Das Bias-System bildet diese per-Goal-Physik korrekt ab — kein Fix nötig.

**Gain Budget**: Nur 2/44 Phasen (P12, P28) registrieren beim GlobalGainBudget. Das HF-kumulative
Tracking deckt 3 Phasen (P06, P17, P38) via `hf_cumulative_gain_db` ab. Die Bass-Region hat
kein kumulatives Tracking. Niedriges Risiko, da P37 jetzt via §G94a mit P02 koordiniert.

**HF-Tracking-Ordnungslücke**: P17 (Mastering Polish) kann nach P39 (Air Band) laufen —
P39's HF-Guard sieht P17's Beitrag nicht. Niedriges Risiko, da beide Enhancement-Phasen sind
und P17 nur +1.5 dB beiträgt (subtraktive Phasen dominieren).

---

## Änderungshistorie

| Version | Datum | Änderung |
|---------|-------|----------|
| 10.0.94 | 2026-08-10 | §19.5–§19.7: Cross-Phase-Koordination (§v10.94). DAG-Constraint P02→P03, Metadata-Handshakes P02→P37, P10→P26. |
| 10.0.93 | 2026-08-10 | §19.4: Numerische Stabilität (§v10.93). NaN-Guards, log10-Guards, Epsilon-Differenzierung. |
| 10.0.92 | 2026-08-10 | §19.6: Material-Kalibrierung (§v10.92). AUTHENTIC_CHARACTER, THRESHOLD_BONUS, predict_quality_score, Exception-Proxies. |
| 10.0.91 | 2026-08-10 | §19.3: Blinder Referenz-Vektor (§v10.91). Embedding-basierte Referenz statt Audio-Vergleich. |
| 10.0.90 | 2026-08-10 | Initial: §19.1–§19.2 dokumentieren Root-Causes 1–2 (SFT, lfilter). |
