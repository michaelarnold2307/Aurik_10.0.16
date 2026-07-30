# §19 — Architektur-Roadmap für nächste Major-Version (§v10.124)

**Status**: Teilimplementiert (2026-08)
**Basis**: §18 Depth-Threshold Calibration-Shift, Batch-Log-Tiefenanalyse

---

## 1. MERT-Referenzspeicher depth-adaptiv ✅

**Problem**: `update_reference_memory()` verwendet AF≥0.95 für alle Tiefen. Depth-4-Kassetten erreichen das nie → Referenzspeicher bleibt leer → HPI fällt auf Default-Referenz zurück → Qualitätsbewertung unterschätzt gute Restaurationen.

**Lösung**: `_af_ref_min = min(0.95, _af_min + 0.05)` mit `_af_min` aus `_get_depth_adaptive_af_min(transfer_chain)`.

| Depth | AF-Ref-Min | Wirkung |
|-------|-----------|---------|
| 1 | 0.95 | Unverändert |
| 2 | 0.93 | Leicht gelockert |
| 3 | 0.85 | Moderate Kassetten können speichern |
| 4+ | 0.75 | Deep Cassette bevölkert Referenzspeicher |

**Dateien**: `holistic_perceptual_gate.py`, `unified_restorer_v3.py`

## 2. NaturalnessOptimizer Attack-Detektion für Rauschquellen ✅

**Problem**: `np.percentile(diff, 90)` findet bei rauschenden Kassetten Noise-Peaks statt Musik-Transienten → falsche Attack-Erkennung → blendet Rauschen in restaurierte Transienten.

**Lösung**: Perzentil dynamisch: `90 + max(0, hg_threshold - 0.15) * 20` → 95. Perzentil bei Depth 4.

| Depth | Perzentil | Wirkung |
|-------|----------|---------|
| 1 | 90 | Unverändert |
| 2 | 91 | Minimal |
| 3 | 92.6 | Moderate |
| 4 | 95 | Fokussiert auf echte Transienten |
| 5 | 98 | Nur extreme Peaks |

**Datei**: `naturalness_optimizer.py`

## 3. Kalibrierungsmatrix Chain-Depth-Skalierung (Zukunft)

**Problem**: Goal-Targets in `calibration_matrix.py` sind material- und era-spezifisch, aber nicht depth-adaptiv. Kassette (Depth 1) hat gleiches brillanz-Ziel wie Kassette→MP3→Stream (Depth 3).

**Ansatz**: Post-Processing-Skalierung in `pipeline_calibration.py`:

```python
depth_factor = 1.0 - max(0, depth - 1) * 0.05  # −5% pro Depth-Stufe
for goal in ('brillanz', 'transparenz', 'separation_fidelity'):
    targets[goal] *= depth_factor
```

⚠️ Nicht implementiert — erfordert separates A/B-Testing.

## 4. Phasen-Reihenfolge depth-adaptiv (Zukunft)

**Problem**: `fahrplan.py` ordnet Phasen statisch. Für tiefe Ketten wäre Denoise→De-Esser optimaler als De-Esser→Denoise.

⚠️ Nicht implementiert — erfordert Refactor des gesamten Fahrplan-Systems.

## 5. bw_extension_context flächendeckend ✅

Alle 17 ADDITIVE-Phasen haben jetzt `bw_extension_context=True` im `check_hallucination()`-Aufruf.

**Dateien**: 12 Phasen-Dateien (siehe §18)

---

## Major-Version Delta

| System | v10.0.17 | v10.0.18 (diese Session) |
|--------|----------|--------------------------|
| Depth-Paradigma | ≥3="deep" | ≥4="deep cassette", ≥5="extrem" |
| Quality-Gates | Teilweise depth-blind | Alle 6 Systeme depth-adaptiv |
| MERT-Referenz | Nur Depth 1 | Depth 1–4 bevölkern Speicher |
| Attack-Detektion | 90. Perzentil fix | Depth-adaptiv 90→98 |
| HallucinationGuard | 0.15 fix, nur 4 bw_extension | 0.15–0.55 adaptiv, 17 bw_extension |
| CIG-Exclusions | 2 Phasen | 8 Phasen |
| Rollbacks (Depth 4) | ~20 | ~0 |

## 6. System-Architektur Depth-Adaptivität

```
                    ┌─────────────────────────┐
                    │  calibrate_sft_thresholds │
                    │  (signal_flow_tracer.py)  │
                    │  setzt _HG_BASE_THRESHOLD │
                    └──────────┬──────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│HallucinationGuard│  │NaturalnessOptimizer│ │Phase_19 De-Esser│
│check_hallucination│  │Attack-Perzentil  │   │Strength-Faktor │
│0.15 → 0.55     │    │90% → 98%        │    │1.0 → 0.70     │
└───────┬───────┘    └───────────────┘    └───────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│              Cumulative Interaction Guard                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │GDD       │  │P1/P2 Drift│  │Critical  │  │Max      │ │
│  │chain_fact│  │chain_fact│  │Pair      │  │Rollbacks│ │
│  │1.30→1.50│  │1.30→1.50│  │chain_fact│  │5→7      │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
└─────────────────────────────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ArtifactFreedom │    │   HPI Gate     │    │     PMGG      │
│Per-Phase 0.70 │    │AF-Min 0.70    │    │Regression +0.016│
│Pre-Echo ×1.66│     │                │    │               │
│Tolerance ×1.66│   │                │    │               │
└───────────────┘    └───────┬───────┘    └───────────────┘
                             │
                             ▼
                    ┌───────────────┐
                    │ MERT Reference │
                    │ AF-Ref 0.75    │
                    │ (war: 0.95)   │
                    └───────────────┘
```

## 7. Depth-Referenztabelle (alle Systeme)

| Depth | AF-Min | AF-Ref | HG-Basis | HG-bw_ext | CIG-Drift | CIG-GDD | PMGG | Max-RB | Attack-% | De-Esser |
|-------|--------|--------|----------|-----------|-----------|---------|------|--------|----------|----------|
| 1 | 0.95 | 0.95 | 0.15 | 0.23 | 1.00× | 1.00× | +0.000 | 5 | 90 | 1.00 |
| 2 | 0.88 | 0.93 | 0.20 | 0.30 | 1.00× | 1.00× | +0.000 | 5 | 91 | 1.00 |
| 3 | 0.80 | 0.85 | 0.28 | 0.42 | 1.25× | 1.25× | +0.008 | 6 | 92.6 | 1.00 |
| 4 | 0.70 | 0.75 | 0.40 | 0.60 | 1.50× | 1.50× | +0.016 | 7 | 95 | 0.85 |
| 5 | 0.70 | 0.75 | 0.55 | 0.83 | 1.75× | 1.75× | +0.024 | 8 | 98 | 0.70 |
