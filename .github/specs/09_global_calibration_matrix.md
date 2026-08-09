# Spec §09 — Globale Kalibrierungs-Matrix für reproduzierbare Optimalwerte | §v10 Pleasantness-First

**Aurik 10.11.14+ | Gültig ab: 18. April 2026 | Stand: 19. April 2026 (v10.0.0+) | Normativ übergeordnet über einzelne Phase-Konfigurationen**

---

## Implementierungsstatus

| Symbol | Bedeutung |
| --- | --- |
| ✅ | Implementiert, getestet, produktionsstabil |
| 🔲 | Geplant / Roadmap |

| Komponente | Pfad | Status |
| --- | --- | --- |
| `CANONICAL_THRESHOLDS_RESTORATION/STUDIO2026` | `backend/core/calibration_matrix.py` | ✅ |
| `estimate_song_goal_targets` (Pipeline, SongGoalTargets) | `backend/core/studio_goal_targets.py` | ✅ |
| `estimate_song_goal_targets` (Convenience, dict) | `backend/core/calibration_matrix.py` | ✅ |
| `predict_quality_score` | `backend/core/calibration_matrix.py` | ✅ |
| `compute_adaptive_drift_tolerance` | `backend/core/calibration_matrix.py` | ✅ |
| `compute_tcci`, `compute_ibs` | `backend/core/calibration_matrix.py` | ✅ |
| `get_phase_strength_range` | `backend/core/calibration_matrix.py` | ✅ |
| `get_material_floor` | `backend/core/calibration_matrix.py` | ✅ |
| `get_effective_material_floor` | `backend/core/calibration_matrix.py` | ✅ |
| Effektiver Goal-Target-Resolver (`resolve_effective_goal_targets`) | `backend/core/calibration_matrix.py` / UV3-Integration | ✅ |
| UV3-Integration (SongGoalTargets → PMGG) | `backend/core/unified_restorer_v3.py` | ✅ |

---

## Übersicht

Dieses Dokument ist die **Single Source of Truth** für alle Aurik-globalen Parameter, die eine reproduzierbare, evidenzbasierte Annäherung an den rekonstruierten Studio-Zielklang steuern. Es definiert interne Kalibrierung und Zielableitung, nicht den externen Beweis eines historisch exakt bekannten Referenzsignals.

**Eingaben** (determiniert die Kalibrierung):

1. `material_type` (primäres Trägermedium)
2. `era_decade` (Aufnahmezeit)
3. `genre_label` (Musikstil)
4. `restorability_score` (0–100: wie stark degradiert?)
5. `transfer_chain` (Multipel-Kopie-Stufen)
6. `is_studio_2026` (Mode: Restoration oder Studio 2026)

**Ausgaben** (die Pipeline wird damit kalibriert):

- Goal-Schwellwerte pro Mode (15 Goals)
- Per-Song Goal-Targets (statt nur Floors)
- Effektive, physikalisch gedeckelte Zielwerte pro Importstück
- Phase-Strength-Ranges material-adaptiv
- Gate-Toleranzen (PMGG, CIG, AFG)
- ML-Budget-Grenzen
- Expected Quality Score (vorhersagbar)
- **§v10.40 SFT-Novelty-Schwelle** (adaptiv pro Transfer-Chain-Tiefe + Tier → siehe Spec §17)

---

## §09.0a [RELEASE_MUST] Maximal-Ausbaustufe fuer Defekt-Erkennen, Differenzieren, Dosieren

Diese Invariante gilt **gleichzeitig fuer `Restoration` und `Studio 2026`**.

**Leistungsziel (Weltspitze):**
Defektintelligenz wird in beiden Modi auf die **hoechste jemals veroeffentlichte Ausbaustufe**
festgelegt. Bewertungsziel ist ein Niveau ohne Konkurrenz in drei Dimensionen:
Erkennungsabdeckung, kausale Differenzierungspraezision, lokale Dosiergenauigkeit.

1. **Maximale Erkennungstiefe**: DefectScanner und CausalDefectReasoner muessen in beiden Modi
    mit voller Defektabdeckung laufen (keine Modus-Reduktion der Analyse).
2. **Maximale Differenzierungstiefe**: Defekte sind kausal getrennt zu behandeln,
    sobald getrennte Ursachen und Reparaturpfade definiert sind; keine Sammelklassen,
    wenn eine physikalisch spezifische Ursache vorliegt.
3. **Maximale Dosierpraezision**: Strength-Parameter duerfen nicht global pro Defektklasse
    fixiert werden; event-basierte lokale Orakel sind verpflichtend (Severity, Kontext,
    Schutzmasken, Transfer-Chain, Material).
4. **Modusunterschiede begrenzen nur das Zielbild**: `Restoration` bleibt subtraktiv/tragertreu,
    `Studio 2026` darf erweiternde Phasen nutzen. Erkennungs- und Differenzierungs-Qualitaet
    ist modusinvariant.
5. **Unsicherheitsbehandlung**: Unsicherheit erhoeht zuerst die Evidenzfusion und lokale
    Kausalpruefung, nicht die globale Aggressivitaet der Eingriffe.
6. **Abdeckung aller bekannten Defekte**: Alle in Aurik bekannten Defektklassen,
    DetectionTypes, Ursachen und Event-Unterformen sind in beiden Modi verpflichtend
    vollstaendig zu erkennen, differenzieren und event-lokal zu dosieren.

Referenz auf Umsetzungspfade:

- `backend/core/defect_scanner.py`
- `backend/core/causal_defect_reasoner.py`
- `backend/core/defect_phase_mapper.py`
- Per-Event-Strength-Oracles in `backend/core/phases/phase_*.py`

---

## §09.1 Kanonische Musical-Goal-Schwellwerte

### §09.1a RESTORATION-Mode Schwellwerte (Spec §01 §1.2)

Ziel: Klang in Richtung des rekonstruierten Studio-Zielankers zurückführen.

```python
CANONICAL_THRESHOLDS_RESTORATION = {
    # P1: Primär (Hartregeln, Minimal-Intervention §2.45)
    "natuerlichkeit":              0.90,    # keine unnaturalen Artefakte
    "authentizitaet":              0.88,    # historischer/künstlerischer Charakter bewahrt

    # P2: Kern-Klangtreue
    "tonal_center":                0.95,    # Tonalität präzise
    "timbre_authentizitaet":       0.87,    # Klangfarbe treu zum rekonstruierten Zielanker
    "artikulation":                0.88,    # Transient-Klarheit
    "transient_energie":           0.80,    # Onset-Energie-Erhalt

    # P3: Musikalische Kohärenz
    "emotionalitaet":              0.84,    # Gefühlsausdruck erhalten
    "micro_dynamics":              0.88,    # Feindynamik-Struktur
    "groove":                      0.83,    # Rhythmische Intention

    # P4: Transparenz-Komponenten
    "transparenz":                 0.82,    # Spektral-Klarheit
    "waerme":                      0.75,    # HF-Wärme (Material-Charakter)
    "bass_kraft":                  0.78,    # Subharmonische Kraft
    "separation_fidelity":         0.80,    # L/R Koherenz

    # P5: Räumlichkeit (optional bei Mono-Quelle)
    "brillanz":                    0.78,    # HF-Präsenz
    "spatial_depth":               0.70,    # Spatial-Tiefe
}
```

### §09.1b STUDIO 2026-Mode Schwellwerte (Spec §01 §1.2)

Ziel: Bestmöglicher modernstuido-Klang (Enhancement statt nur Restoration).

```python
CANONICAL_THRESHOLDS_STUDIO2026 = {
    # Erhöhte Ansprüche wegen aktiver Enhancement
    "natuerlichkeit":              0.92,    # Muss natürlich bleiben trotz Enhancement
    "authentizitaet":              0.90,    # Künstler-Intention bewahrt

    "tonal_center":                0.96,
    "timbre_authentizitaet":       0.89,
    "artikulation":                0.90,
    "transient_energie":           0.83,

    "emotionalitaet":              0.87,
    "micro_dynamics":              0.90,    # Enhanced, aber nicht künstlich
    "groove":                      0.85,

    "transparenz":                 0.85,    # Moderne Klarheit
    "waerme":                      0.78,
    "bass_kraft":                  0.80,
    "separation_fidelity":         0.83,

    "brillanz":                    0.82,    # Modern glänzend, aber nicht hart
    "spatial_depth":               0.74,    # Enhanced Spatial
}
```

---

## §09.2 Material-adaptive Goal-Targets (§2.56) ✅

Nicht alle Songs sollten zu denselben Schwellwerten führen. Ein 1920er-Shellac-Aufnahme kann unmöglich 0.78 Brillanz haben (technische Grenze 8 kHz, Rolloff). Ein 1990er CD-Pop sollte 0.87+ Brillanz haben.

### §09.2a PMGG-Blend-Invariante (normativ, v10.0.0)

**Problem**: Fixer 60/40-Blend (60 % canonical, 40 % SGT) erzeugt PMGG-Schwellwerte **über** der physikalischen Ceiling — z.B. `brillanz` Shellac: `0.60 × 0.78 + 0.40 × 0.51 = 0.71` bei physikalischer Grenze 0.51. Resultat: 5 Retries → 15 % Stärke → degradierte Restaurierung.

**Normative Regel — delta-adaptiver Blend (Pflicht in PMGG + Pipeline-Ende)**:

```python
delta = canonical[goal] - sgt[goal]   # positiv = SGT ist niedriger (Material-Constraint)
if delta > 0.10:
    blended = sgt[goal]                # Direkt — Ceiling-Fall, kein Blend sinnvoll
elif delta > 0.04:
    blended = 0.40 * canonical[goal] + 0.60 * sgt[goal]   # Moderate Constraint
else:
    blended = 0.60 * canonical[goal] + 0.40 * sgt[goal]   # Kleine/aufwärts Differenz
blended = float(np.clip(blended, 0.30, 0.99))
```

**Pre-Pipeline Physical Ceiling (Pflicht)**:

Vor dem Phase-Loop muss UV3 `PhysicalCeilingEstimator` auf dem Input-Audio ausführen und `_pmgg_ceiling_capped_targets` bilden:

```python
_pmgg_ceiling_capped_targets = {
    g: float(min(sgt[g], physical_ceiling[g], chain_end_ceiling.get(g, 0.99)))
    for g in sgt
}
```

Diese werden als `adaptive_goal_thresholds` an **jeden** `wrap_phase()`-Aufruf, an `§GOAL_BASELINE_CHECK`, an FeedbackChain und an die Pipeline-Ende-Schwellwertberechnung übergeben.

**Invariante**: PMGG-Blend und Pipeline-Ende-Blend verwenden identische delta-adaptive Logik — kein Split-Behavior zwischen per-Phase-Steuerung und Final-Gate.

### §09.2b [RELEASE_MUST] Effektiver Goal-Target-Resolver (v10.0.0)

Für jedes Importstück MUSS ein vollständiger Target-Vektor für alle **15** anwendbaren Musical Goals berechnet werden. Dieser Vektor ist die einzige normative Vergleichsgrundlage für `§GOAL_BASELINE_CHECK`, PMGG, FeedbackChain, End-Gate, UI und Analyse-Reports.

Die songweiten Zielgewichte, die in diese Berechnung einfließen, stammen normativ aus
`restoration_policy_profile`. `song_goal_weights` ist nur ein Kompatibilitäts-Fallback und darf
nicht als konkurrierende Quelle gepflegt werden.

```python
def resolve_effective_goal_targets(
    *,
    mode: str,
    material_type: str,
    transfer_chain: list[str],
    era_decade: int | None,
    genre_label: str | None,
    restorability_score: float,
    song_goal_weights: dict[str, float],
    physical_ceiling: dict[str, float],
    applicable_goals: set[str],
) -> dict[str, float]:
    canonical = CANONICAL_THRESHOLDS_STUDIO2026 if is_studio(mode) else CANONICAL_THRESHOLDS_RESTORATION
    sgt = estimate_song_goal_targets(
        is_studio_2026=is_studio(mode),
        goal_weights=song_goal_weights,
        restorability_score=restorability_score,
        era_decade=era_decade,
        genre_label=genre_label,
        material_type=material_type,
        transfer_chain=transfer_chain,
    )
    chain_ceiling = estimate_chain_end_goal_ceiling(transfer_chain)

    effective: dict[str, float] = {}
    for goal in applicable_goals:
        floor_eff = get_effective_material_floor(material_type, goal, restorability_score, is_studio(mode))
        target = max(floor_eff, sgt.get(goal, canonical.get(goal, 0.70)))
        target = min(target, physical_ceiling.get(goal, 0.99), chain_ceiling.get(goal, 0.99))
        effective[goal] = float(np.clip(target, 0.20, 0.99))
    return effective
```

**Invarianten:**

- Kein Gate darf direkt gegen rohe `CANONICAL_THRESHOLDS` prüfen, sobald ein `effective_targets`-Vektor verfügbar ist.
- `estimate_song_goal_targets()` darf Ziele anheben oder absenken, aber `PhysicalCeilingEstimator` und das Transferketten-Ende dürfen sie immer nach oben begrenzen.
- `restorability_score < 30` senkt den Zielwert nur über `get_effective_material_floor()` und MUSS in `metadata["degraded_restorability"]` sichtbar sein.
- Inapplicable Goals werden nicht als Fail gewertet, aber mit `applicability_reason` dokumentiert.
- P0-Vokal-Ziele (`vocal_quality`, `formant_fidelity`, VQI) bleiben zusätzliche Vocal-Gates und dürfen die 15 Musical Goals nicht ersetzen.
- Formant-Rollback-Toleranzen sind global kalibriert über `EraVocalProfile` und `resolve_formant_tolerance_db()`. Feste `2.0 dB`-Werte sind nur der moderne Fallback, nicht die Pipeline-Referenz.

### §09.2 Zwei-Ebenen-API (normativ)

| Ebene | Modul | Rückgabe | Einsatz |
| --- | --- | --- | --- |
| **Pipeline** | `backend/core/studio_goal_targets.py` | `SongGoalTargets` (frozen dataclass mit `.targets`, `.confidence`, `.derived`) | UV3 → PMGG `wrap_phase(..., goal_targets=result.targets, goal_targets_confidence=result.confidence)` |
| **Convenience** | `backend/core/calibration_matrix.py` | `dict[str, float]` | Pre-Analysis-Preview, Tests, Bridge/UI, einfache Scores |

**Invariante**: `studio_goal_targets.estimate_song_goal_targets` ist die Pipeline-Referenz. Die Convenience-API darf nicht für PMGG-Integration oder Phase-Steering eingesetzt werden — dort fehlen Confidence, IBS und Chain-Depth-Pullback.

### §09.2b Restoration-Prior-Payload (v10.0.0)

Wenn `restoration_prior` an `estimate_song_goal_targets(...)` oder GP-Proposal-Pfade übergeben
wird, ist die kanonische Struktur:

- `phase_params`: numerische Best-Parameter aus dem erfolgreichen Vorlauf
- `hpi_achieved`: HPI des Vorlaufs
- optionale coalition-aware Felder in `phase_params`:
    `_coalition_learning_factor`, `_dominant_phase_coalition_ratio`, `_phase_coalition_count`

**Invarianten**:

- Coalition-Felder sind advisory-only und dürfen die Zielwertberechnung nie außerhalb der
    Material-/Era-/Restorability-Ceilings pushen.
- Fehlende Coalition-Felder müssen auf neutrales Verhalten zurückfallen (`factor=1.0`).
- PMGG/CIG/AFG/HPI bleiben normative Endinstanzen; restoration_prior darf nur initiale
    Parametervorschläge und milde Gewichtung beeinflussen.

### §09.2c Runtime-Metric-Reliability-Layer (v10.0.0)

Der Closed-Loop-Stack darf Goal-Konfidenzen nicht nur aus statischen Proxys ableiten.
Zusätzlich MUSS ein laufzeitnaher Reliability-Layer aus realen Phase-Deltas und PMGG-
Metadaten gespeist werden.

**Kanonische API**:

```python
from backend.core.metric_reliability_graph import get_metric_reliability_graph

graph = get_metric_reliability_graph()
graph.update_from_phase_delta(...)
runtime_conf = graph.get_goal_reliability(...)
base_w, runtime_w = graph.get_blend_weights(...)
goal_conf = clip(base_w * base_conf + runtime_w * runtime_conf, 0.20, 0.98)
```

**Normative Quellen** (mindestens):

- `phase_deltas[phase_id]["delta"]`
- `pmgg_team_net_delta`
- `pmgg_reconstruction_localized`
- `pmgg_reconstruction_epistemic_confidence`

**Sicherheitsgrenzen**:

- Reliability-Layer ist advisory-only (kein Gate-Override).
- Persistenz nur lokal/offline, atomar, non-blocking.
- Kontext-Key umfasst `mode`, `material`, `era_bucket`, `tcci_bucket`.
- Blend-Gewichte sind cross-run-adaptiv, normiert auf 1.0 und innerhalb
    `base ∈ [0.40, 0.75]`, `runtime ∈ [0.25, 0.60]`.

**Algorithmus**: `target[goal] = floor + kappa × bias + weight_shift`

Wobei:

- `floor` = CANONICAL_THRESHOLDS[goal]
- `kappa` = 0.45 (Restoration) / 0.65 (Studio 2026), skaliert mit Restorability
- `bias` = Summe Era-Bias + Material-Bias + Genre-Bias (siehe §09.2a–c)
- `weight_shift` = `(goal_weight − 1.0) × 0.06` aus §2.56

### §09.2a Era-basierte Goal-Biases (subtrahiert/addiert zum Canonical Floor)

```python
# 1920–1949 (Shellac, WaxCylinder, earlywire)
ERA_BIAS_1920S = {
    "brillanz":     -0.28,   # 8 kHz Rolloff ist Material-Realität
    "transparenz":  -0.18,
    "spatial_depth": -0.14,   # Mono oder Pseudo-Stereo
    "waerme":       +0.14,   # Warm-ätzender Sound ist Charakter
    "authentizitaet": +0.10,
    "natuerlichkeit": +0.08,
}

# 1950–1969 (Vinyl, earlytape, Radiobroadcast)
ERA_BIAS_1950S = {
    "brillanz":     -0.14,   # Vinyl-Rolloff ~10 kHz
    "transparenz":  -0.08,
    "waerme":       +0.10,
    "authentizitaet": +0.08,
}

# 1970–1989 (Reel-Tape, Cassette, Early-Digital)
ERA_BIAS_1970S = {
    "brillanz":     +0.04,
    "transparenz":  +0.04,
    "waerme":       +0.02,
}

# 1990+ (CD, DAT, Digital)
ERA_BIAS_1990S = {
    "brillanz":     +0.10,   # Digital hat volle BW
    "transparenz":  +0.10,
    "artikulation": +0.06,
    "waerme":       -0.04,   # Digitale Kälte
}
```

### §09.2b Material-basierte Goal-Biases

```python
# Shellac, Wax, Wire (ultra-degradiert)
MATERIAL_BIAS_ULTRA_ANALOG = {
    "brillanz":     -0.24,
    "transparenz":  -0.12,
    "waerme":       +0.10,
    "authentizitaet": +0.10,
}

# Vinyl, Tape, Cassette (normal-analog)
MATERIAL_BIAS_ANALOG = {
    "waerme":       +0.10,
    "brillanz":     -0.06,
    "authentizitaet": +0.08,
}

# CD, Digital, Streaming (clean)
MATERIAL_BIAS_DIGITAL = {
    "transparenz":  +0.08,
    "artikulation": +0.06,
    "brillanz":     +0.06,
}
```

### §09.2c Genre-basierte Goal-Biases

```python
GENRE_BIAS_KLASSIK = {
    "spatial_depth": +0.18,   # Saalakustik zentral
    "natuerlichkeit": +0.12,
    "micro_dynamics": +0.10,
    "brillanz":     -0.08,   # Wärmere Interpretation
}

GENRE_BIAS_JAZZ = {
    "waerme":       +0.12,
    "natuerlichkeit": +0.10,
    "authentizitaet": +0.10,
    "transparenz":  -0.04,
}

GENRE_BIAS_POP = {
    "transparenz":  +0.08,
    "artikulation": +0.08,
    "brillanz":     +0.08,    # Pop modern/glänzend
}
```

---

## §09.3 Material-adaptive Strength-Ranges pro Phase (§2.47)

Die Auswahl der Phase-Stärke hängt ab von:

1. Material
2. Restorability (0–100)
3. Defekt-Severity an der Phase
4. GlobalPlan-Vorschlag (§GP)

### §09.3a Beispiel: Phase 03 (Denoise)

```python
# Phase 03 — Denoise (Broadband subtraktiv)
PHASE_03_STRENGTH_RANGES = {
    "vinyl": {
        "restorability": {
            "high": (75, 100),      # Rest ≥ 75: strength ∈ [0.25, 0.45]
            "fair": (50, 74),       # Rest 50–74: strength ∈ [0.35, 0.65]
            "poor": (20, 49),       # Rest < 50: strength ∈ [0.50, 0.85]
        }
    },
    "shellac": {
        "high": (0.20, 0.40),       # Shellac-Restorability fast nie >50
        "fair": (0.40, 0.70),
        "poor": (0.60, 0.90),
    },
    "cd_digital": {
        "high": (0.0, 0.20),        # Nur bei Defekten
        "fair": (0.10, 0.35),
        "poor": (0.30, 0.60),
    },
}

# Auswahl der Range:
# strength_min, strength_max = lookup(phase, material, restorability)
# strength = GlobalPlan.recommend() within [strength_min, strength_max]
```

### §09.3b GlobalPlan Recommendation (§2.47.5)

GlobalPlan nutzt Pareto-Optimierung auf den 15 Goals und empfiehlt eine Stärke, die einen lokalen Optimum-Punkt trifft (musikalischer Sweetspot, nicht Extremum).

```python
# Pseudocode in GP.recommend():
target = estimate_song_goal_targets(...)  # Per-Song-Targets aus §09.2
goal_scores_before = measure_goals(audio_before)
best_strength = None
best_score = -inf

for s in linspace(strength_min, strength_max, n_steps=15):
    audio_test = phase(audio_before, strength=s)
    goal_scores_test = measure_goals(audio_test)

    # Weighted distance zur per-song Target
    distance = sum(
        goal_weight[g] * (goal_scores_test[g] - target[g])^2
        for g in goals
    )
    if distance < best_score:
        best_score = distance
        best_strength = s

return best_strength
```

### §09.3c [RELEASE_MUST] Phase-Strength-Oracles fuer alle profitierenden Phasen (v10.0.0)

`GlobalPlan.recommend()` liefert nur den globalen Song-Kontext. Die finale lokale
Interventionsstaerke jeder Phase wird durch ein **Phase-Strength-Oracle** bestimmt.

**Pflicht gilt fuer alle profitierenden Phasen**: Jede Phase, deren Wirkung ueber einen
kontinuierlichen oder semi-kontinuierlichen Steuerparameter laeuft, MUSS ein eigenes
Strength-Oracle besitzen. Dazu zaehlen insbesondere `strength`, `wet_mix`, `threshold_db`,
`ratio`, `drive`, `mix`, `boost_db`, `cut_db`, `repair_strength`, `context_mix`,
`generation_steps`, `temperature`, `gate_dbfs`, `width_amount`.

**Pflicht-Eingaben pro Oracle:**

1. aktueller 15-Goal-Gap-Vektor gegen `effective_goal_targets`
2. `goal_weights` nach Teamwork-Prinzip (Spec 01 §1.2c)
3. Defekt-Schwere + Lokalitaet der Phase
4. Material, Aera, Genre, Carrier-Chain, Restorability
5. Vocal-/Phonem-/Frisson-/Passaggio-Kontext wenn vokalrelevant
6. PMGG/CIG/AFG-Historie der vorangegangenen Phasen
7. Wall-Time-/ML-Budget-Headroom

**Pflicht-Optimierungsziel:** Weighted-Gap-Closure statt Single-Goal-Maximierung.

```python
def score_candidate(candidate_audio, goal_snapshot_pre, effective_targets, goal_weights):
        goal_snapshot_post = fast_goal_snapshot(candidate_audio)
        weighted_gap_closure = 0.0
        weighted_penalty = 0.0
        for goal in applicable_goals:
                gap_pre = max(0.0, effective_targets[goal] - goal_snapshot_pre[goal])
                gap_post = max(0.0, effective_targets[goal] - goal_snapshot_post[goal])
                closure = max(0.0, gap_pre - gap_post)
                weighted_gap_closure += goal_weights[goal] * closure
                regression = max(0.0, goal_snapshot_pre[goal] - goal_snapshot_post[goal])
                weighted_penalty += goal_weights[goal] * regression
        return weighted_gap_closure - 2.5 * weighted_penalty
```

**Oracle-Klassen (kanonisch):**

| Oracle-Klasse | Einsatz | Kanonischer Suchmodus |
| --- | --- | --- |
| `O1_impulse` | lokale Impuls-/Klick-/Splice-Reparaturen | Window-/Threshold-/Mix-Scan auf Defektinseln |
| `O2_subtractive` | NR/Dereverb/Gate/subtraktive Reparatur | konservativer Control-Floor + Wet/Dry + Banded Caps |
| `O3_spectral_balance` | EQ/BW/Harmonik/Praesenz/Air/Bass | bandbegrenzter Pareto-Scan gegen Zielvektor |
| `O4_time_pitch` | Wow/Flutter/Speed/Pitch/Azimuth/Phase | timing-konsistenter bounded search, kein Wet-only |
| `O5_stereo_field` | Stereo/MS/Raum/Breite/Crosstalk | M/S-koharenter Width-/Mix-Scan mit Mono-Guard |
| `O6_dynamics` | Kompression/Expansion/Limiter/Dynamics | envelope-aware Threshold-/Ratio-/Time-Constant-Scan |
| `O7_vocal_articulation` | DeEsser/Vocal/Lyrics/phonem-/konsonantenbezogen | phonem-/formant-/hnr-aware Steuerprofil |
| `O8_generative_repair` | Spectral repair / Inpainting / Band-gap | Schrittzahl-/Guidance-/Blend-Oracle mit SSIP |
| `O9_periodic_cancellation` | Hum/Print-through/Groove-Echo/ModNoise/IMD | periodic-template fit + residue-minimizing gain |
| `O10_output` | Loudness/Format/TruePeak/Output-Optimierung | exportkonformer Zielwertsolver |

**Transfer-Chain-Faktor [RELEASE_MUST]:**

Die lokale Interventionsstaerke wird zusaetzlich durch einen deterministischen
`chain_factor` aus `material_key`, `transfer_chain` und `chain_confidence` konditioniert. `[SRC:S03,S04]`

```python
def compute_chain_factor(material_key: str, transfer_chain: list[str], confidence: float) -> float:
    mats = unique([material_key] + transfer_chain)
    strict_factor = min(CHAIN_STAGE_FACTOR.get(m, 0.88) for m in mats)
    generation_penalty = clip(0.02 * max(0, len(mats) - 1), 0.0, 0.12)
    strict_factor = clip(strict_factor - generation_penalty, 0.55, 1.0)

    conf_weight = clip((confidence - 0.40) / 0.60, 0.0, 1.0)
    return clip((1.0 - conf_weight) * 0.95 + conf_weight * strict_factor, 0.55, 1.0)
```

**Normative Wirkung:**

1. `driver` wird mit `chain_factor` skaliert.
2. `hard_caps["max_strength"]` wird mit `(0.75 + 0.25 * chain_factor)` skaliert.
3. `hard_caps` MUSS `chain_factor`, `chain_depth`, `chain_confidence` ausgeben.
4. Bei `vinyl -> cassette -> mp3_low` MUSS `chain_factor` niedriger sein als bei reinem `vinyl`. `[SRC:S03,S04]`

**Garantieklausel fuer die 15 Ziele:**

- Kein Oracle darf auf ein Ziel optimieren und dabei die uebrigen 14 ignorieren.
- Wenn ein Goal bereits ueber Ziel liegt, wird Ueberschuss nur genutzt, um andere Goals
    sicher zu schliessen; Ueberschuss allein rechtfertigt keinen aggressiveren Eingriff.
- Wenn alle 15 anwendbaren Goals ueber Ziel liegen, muss das Oracle auf Minimalintervention
    zurueckfallen (`control_strength -> min_safe`).

**Kreuzreferenz:** Die konkrete 64-Phasen-Bindung der Oracle-Klasse steht in Spec 06 §7.1d.

### §09.3d Evidenzklassen fuer P1-Kernschwellen (v10.0.0+)

Zur Vermeidung scheinbarer Praezision ohne Quellenpflicht werden Kern-Schwellen in Evidenzklassen gefuehrt:

| Schwellenfamilie | Evidenzklasse | Mindestanforderung fuer Aenderungen |
| --- | --- | --- |
| Loudness/TruePeak (`BS.1770`/`R128`) | A | Nur normkonforme Anpassungen mit Quellenupdate (`[SRC:S06,S07]`) |
| Artifact-Freedom-Detektion (`musical_noise`, `pre_echo`) | B | Peer-reviewte Belege + Gate-Regression + AFG-Audit (`[SRC:S14,S15]`) |
| Vocal-Guardrails (Formant/Vibrato) | B | Peer-reviewte Belege + Vocal-Regression + Hoertest-Delta (`[SRC:S16,S17,S18]`) |
| Materialadaptive VQI-/MERT-Proxy-Floors | C | Reale Musikfall-Revalidierung + Changelog-Nachweis pro Release |

**Release-Regel:** Klasse-C-Werte sind kalibrierte Betriebsparameter, keine naturgesetzlichen Konstanten; sie duerfen nur unter dokumentierter Revalidierung angepasst werden.

---

## §09.4 PMGG Gate-Toleranzen (§2.29)

PMGG prüft nach jeder Phase, ob die 15 Goals den effektiven Zielwert halten.

### §09.4c Rekonstruktive Konfidenz und Threshold-Multiplikator [RELEASE_MUST v10.0.0]

Rekonstruktive Phasen (`phase_23`, `phase_24`, `phase_50`, `phase_55`) dürfen nicht nur
gegen den defektzentrierten Zielbereich bewertet werden. Für `passed_reconstruction_localized`
gilt zusätzlich eine lokal gestützte Entscheidungskonfidenz aus Zielabdeckung, Kontrollfenster-
Leckage und Threshold-Sicherheitsmarge.

**Normative Metadaten**: `PhaseGateLogEntry.metadata` MUSS für rekonstruktive Entscheidungen
die Felder `pmgg_reconstruction_localized`, `pmgg_reconstruction_confidence`,
`pmgg_reconstruction_reason`, `pmgg_reconstruction_retry_budget_bias`,
`pmgg_reconstruction_retry_budget_bias_reason`, `pmgg_reconstruction_transfer_chain_tcci`,
`pmgg_reconstruction_material_family`, `pmgg_reconstruction_threshold_multiplier`,
`pmgg_reconstruction_threshold_multiplier_reason`, `pmgg_reconstruction_threshold_multiplier_tcci`,
`pmgg_reconstruction_target_coverage`, `pmgg_reconstruction_control_coverage` und
`pmgg_reconstruction_control_regression` enthalten.

**Invariante**: Der Threshold-Multiplikator ist kein freischwebender UI-Wert, sondern Teil der
kanonischen Rekonstruktions-Telemetrie. Ein lokales `threshold_multiplier <= 1.0` darf die
Entscheidung nicht fälschlich neutralisieren; bei kontextuell erfolgreicher Rekonstruktion bleibt
die Entscheidung als lokal gestützt markiert.

### §09.4a Restorability-adaptive Regression-Thresholds

```python
REGRESSION_THRESHOLDS = {
    "restorability_high": {     # Rest ≥ 70: präzise Messung möglich
        "threshold": 0.020,
        "max_retries": 4,
        "retry_strengths": [0.65, 0.50, 0.35, 0.20],
    },
    "restorability_fair": {     # Rest 40–69: moderate Toleranz
        "threshold": 0.035,
        "max_retries": 3,
        "retry_strengths": [0.60, 0.40, 0.20],
    },
    "restorability_poor": {     # Rest < 40: maximal tolerant
        "threshold": 0.040,
        "max_retries": 2,
        "retry_strengths": [0.50, 0.25],
    },
}

# Material-spezifischer Bonus (Carrier-Chain-Inversion §2.44)
MATERIAL_THRESHOLD_BONUS = {
    "wax_cylinder": 0.022,    # extreme Trägerketten-Inversion
    "shellac":      0.018,
    "vinyl":        0.009,
    "tape":         0.007,
    "cd_digital":   0.000,    # clean → kein Bonus
}

# Angewendeter Threshold:
threshold = base_threshold + material_bonus
```

### §09.4b Priority-aware Retry-Budgets (§2.29 v10.0.0)

```python
PRIORITY_MAX_RETRIES = {
    1: 4,   # P1 (Natürlichkeit, Authentizität) — volle Retry-Kaskade
    2: 4,   # P2 (TonalCenter, Timbre, Artikulation) — volle Kaskade
    3: 2,   # P3 (Emotionalität, Groove, MicroDyn) — reduziert
    4: 1,   # P4 (Transparenz, Wärme, Bass, Sep) — Recovery-Lite
    5: 1,   # P5 (Brillanz, Raumtiefe) — Recovery-Lite
}

# Nur ein Ziel mit Priority P ≥ X triggert Retry-Pfad für diese Priority.
# Sub-Threshold Deltas (JND-unterschwellig) werden akzeptiert ohne Retry.
```

### §09.4d Goal-Koalitionen bei Recovery-Prioritäten [RELEASE_MUST v10.0.0]

Wenn mehrere offene Goals dieselbe Primär-Recovery-Phase teilen, darf die
Reschedule-Priorisierung nicht nur den numerischen Gap betrachten. Recovery-Phasen,
die mehrere offene Goals zugleich bedienen können, erhalten einen kleinen Koalitionsbonus
gegenüber isolierten Einzelfall-Phasen.

**Invariante**: Der Koalitionsbonus ist advisory für die Reihenfolge, nicht für die
Injektionsberechtigung. §2.45 Minimal-Intervention, §0a-Guards und Session-Limits bleiben
unverändert.

```

---

## §09.5 Cumulative-Interaction-Guard Drifttoleranzen (§2.48)

Nach jeder Phase wird der kumulative Drift in P1/P2-Goals geprüft.

### §09.5a Material-adaptive Drift-Toleranzen

```python
def compute_adaptive_drift_tolerance(
    restorability: float,       # 0–100
    material_type: str,         # "vinyl", "shellac", etc.
    defect_severity_mean: float, # 0–1
    n_active_phases: int,       # Wie viele Phasen laufen?
) -> float:
    """Adaptive Schwelle für kumulativen P1/P2-Drift.

    Begründung: Stark degradiertes Material mit vielen Phasen braucht
    mehr Spielraum für Zwischendrift (aber muss am Ende unter Threshold sein).
    """

    # Basis-Toleranz aus Restorability
    if restorability >= 70:
        base_tol = -0.05
    elif restorability >= 50:
        base_tol = -0.08
    else:
        base_tol = -0.12

    # Material-Faktor (Carrier-Chain-Inversion §2.44)
    material_factor = {
        "wax_cylinder": 0.22,
        "shellac":      0.18,
        "vinyl":        0.09,
        "tape":         0.07,
        "cd_digital":   0.00,
    }.get(material_type, 0.03)

    # Phasenzahl-Faktor (mehr Phasen = mehr Zwischendrift erwartet)
    phase_factor = min(0.10, n_active_phases * 0.005)

    return base_tol - material_factor - phase_factor
```

---

## §09.6 Artifact Freedom Gate (AFG) Schwellwerte (§2.49)

AFG blockiert Phasen, die Artefakte einführen würden.

```python
ARTIFACT_FREEDOM_MATERIAL_THRESHOLDS = {
    # Menschliche Wahrnehmung (psychoakustische Salienz-Gewichtung)
    "musical_noise": {
        "vinyl":        0.95,      # Vinyl-Artefakte sind hörbar
        "shellac":      0.90,      # Ultra-niedrig (Verlust-Toleranz)
        "cd_digital":   0.98,      # Digital-rein → strik mit Artefakten
    },
    "phase_cancellation": {
        "vinyl_stereo": 0.92,
        "mono_source":  1.00,      # Mono → kein Stereo-Artefakt möglich
    },
    "spectral_holes": {
        "all_materials": 0.95,     # Höchst wahrnehmbares Artefakt
    },
}

# AFG Verdict:
# artifact_freedom_score = 1.0 - (percep_weight × detections / max_detections)
# if artifact_freedom_score < threshold: rollback to best_checkpoint
```

---

### v10-Amendment (2026-07-12)

**IAD severity-graded response** (v10, 2026-07-12):

Siehe §2.49 Amendment in `02_pipeline_architecture.md`. Zusammenfassung der Log-Level:

| Fraktion | Log-Level | Aktion |
| --- | --- | --- |
| `< 1%` | INFO | Keine — spektraler Rest |
| `1–5%` | WARNING | `artifact_freedom_penalty` prüfen, ggf. VETO |
| `≥ 5%` | WARNING + Rescue | §2.23 Dry/Wet-Blend zurück zum Original |

Die AFG-Schwellwerte selbst bleiben unverändert. Nur die _Reaktion_ (Log-Level + Rescue-Trigger) ist jetzt nach Schweregrad gestaffelt.

## §09.7 Expected Quality Scores (Baseline-Vorhersage) ✅

Nach Pre-Analysis ist ein Expected Quality Score vorhersagbar, damit das UI eine Erwartung setzen kann.

```python
def predict_quality_score(
    material_type: str,
    restorability: float,
    defect_severity_mean: float,
    is_studio_2026: bool,
) -> float:
    """Erwarteter OQS (Overall Quality Score) am Ende der Pipeline.

    Basis: Restorability, Material-Limit, Defekt-Schwere.
    """

    # Restorability als Basis (0–100 → 0–1)
    rest_contrib = restorability / 100.0

    # Material-spezifisches Ober-Limit (physikalische Grenzen)
    material_ceiling = {
        "wax_cylinder": 0.55,      # Ultra-degradiert
        "shellac":      0.70,
        "wire_recording": 0.65,
        "vinyl":        0.88,      # Gut restaurierbar
        "tape":         0.85,
        "cassette":     0.80,
        "cd_digital":   0.95,      # Saubere Quellen
    }.get(material_type, 0.75)

    # Defekt-Abzug
    defect_penalty = defect_severity_mean * 0.15  # Max. -15% für schwere Defekte

    # Studio 2026 boost (Enhancement statt nur Restoration)
    studio_boost = 0.08 if is_studio_2026 else 0.0

    base_score = (rest_contrib * material_ceiling) - defect_penalty + studio_boost
    return np.clip(base_score, 0.0, 0.99)
```

---

## §09.8 ML-Budget und Plugin-Lifecycle (§4.6)

### §09.8a Plugin Memory Budgets

```python
ML_PLUGIN_BUDGETS_GB = {
    "SGMSE+":           3.5,    # Diffusion Denoising (heaviest)
    "DeepFilterNet":    2.2,
    "FlashSR":          4.0,    # Spectral Upsampling
    "MDX23C":           3.2,    # Stem Separation
    "CREPE":            1.2,    # Pitch Tracking
    "PANNs":            0.7,    # Genre/Tagging
    "LAION-CLAP":       2.2,    # Semantic Analysis
    "WPE":              0.8,    # Dereverberation DSP
    "OMLSA":            0.5,    # Gate DSP
    # ... weitere Plugins
}

TOTAL_ML_BUDGET_GB = 10.4  # System default für Desktop (adjustable)
```

### §09.8b Eviction-Schwellen

```python
PLM_EVICTION_THRESHOLDS = {
    "ram_percent": 75,          # Wenn RAM > 75%, starte Plugin-Eviction
    "swap_percent": 80,         # Wenn Swap > 80%, blockiere neue ML-Phase (DSP-Fallback)
    "priority_keep": [          # Diese Plugins werden NICHT evicted
        "current_running_plugin",
        "next_queued_plugin",
        "WPE",                  # DSP-Fallback wird immer erhalten
    ],
}
```

---

## §09.9 Implementierungs-Integration (Kopieren als Anhang zur copilot-instructions)

**In `denker/aurik_denker.py` Zeile N:**

```python
# §09 Global Calibration Parameters
from backend.core.calibration_matrix import (
    CANONICAL_THRESHOLDS_RESTORATION,
    CANONICAL_THRESHOLDS_STUDIO2026,
    estimate_song_goal_targets,
    compute_adaptive_drift_tolerance,
    MATERIAL_PRIORITY_PHASES,
)

def restauriere(self, audio, sr, mode="restoration", **kwargs):
    # Pre-Analysis gibt material, era, restorability
    material = medium_result.primary_material
    era = era_result.decade
    restorability = restorability_result.restorability_score

    # Lookup Canonical Thresholds
    thresholds = (
        CANONICAL_THRESHOLDS_STUDIO2026
        if is_studio_2026
        else CANONICAL_THRESHOLDS_RESTORATION
    )

    # Estimate Per-Song Goal Targets
    song_targets = estimate_song_goal_targets(
        is_studio_2026=is_studio_2026,
        goal_weights=goal_weights,
        restorability_score=restorability,
        era_decade=era,
        material_type=material,
        transfer_chain=transfer_chain,
    )

    # Alle Gates nutzen diese Werte
    pmgg.canonical_thresholds = thresholds
    pmgg.song_targets = song_targets

    cig.drift_tolerance = compute_adaptive_drift_tolerance(
        restorability, material, defect_severity_mean, n_active_phases
    )

    # GlobalPlan nutzt diese Targets zur Strength-Empfehlung
    gp.goal_targets = song_targets
    gp.canonical_thresholds = thresholds
```

---

## §09.10 Abgeleitete Meta-Parameter (universell, keine neuen Inputs)

Die folgenden Parameter sind **nicht zusätzliche Benutzereingaben**. Sie werden aus bereits vorhandenen Signalen
(`material_type`, `restorability_score`, `transfer_chain`, Goal-Weights, Gate-Metriken) berechnet und erhöhen die
Robustheit für den gesamten Import-Raum.

### §09.10a Transfer-Chain-Complexity-Index (TCCI)

Zweck: Quantifiziert Mehrgenerationen-Komplexität der Trägerkette als einheitlicher Steuerwert für Gates und Recovery.

```python
def compute_tcci(transfer_chain: list[str]) -> float:
    n = max(1, len(transfer_chain))
    lossy = sum(1 for m in transfer_chain if m in {"mp3_low", "aac", "streaming"})
    analog = sum(1 for m in transfer_chain if m in {"wax_cylinder", "shellac", "vinyl", "tape", "cassette", "wire_recording", "reel_tape"})
    score = 0.18 * (n - 1) + 0.22 * lossy + 0.10 * max(0, analog - 1)
    return float(np.clip(score, 0.0, 1.0))
```

Integration:

- Erhoeht CIG-Drift-Toleranz innerhalb der bereits materialadaptiven Grenzen.
- Erhoeht Recovery-Budget bei hoher Kettenkomplexitaet (statt fruehem Hardstop).

### §09.10b Intervention-Budget-Scalar (IBS)

Zweck: Ein globaler Interventions-Regler pro Song, der Minimal-Intervention und notwendige Defekt-Reparatur balanciert.

```python
def compute_ibs(restorability: float, defect_severity_mean: float, tcci: float) -> float:
    r = 1.0 - np.clip(restorability / 100.0, 0.0, 1.0)
    d = np.clip(defect_severity_mean, 0.0, 1.0)
    budget = 0.55 * r + 0.30 * d + 0.15 * tcci
    return float(np.clip(budget, 0.15, 0.95))
```

Integration:

- Skaliert obere Grenzen von Strength-Ranges (nicht die unteren Sicherheitsgrenzen).
- Wirkt nur advisory; explizite PMGG-Hardcaps bleiben fuehrend.

### §09.10c Target-Confidence-Blend (TCB)

Zweck: Verhindert Uebersteuerung durch unsichere Klassifikationen, indem Per-Song-Targets mit Canonical-Floors gemischt werden.

```python
def blend_targets_with_confidence(
    canonical: dict[str, float],
    song_targets: dict[str, float],
    medium_conf: float,
    era_conf: float,
    genre_conf: float,
) -> dict[str, float]:
    conf = float(np.clip(0.45 * medium_conf + 0.30 * era_conf + 0.25 * genre_conf, 0.0, 1.0))
    return {
        g: (1.0 - conf) * canonical[g] + conf * song_targets[g]
        for g in canonical.keys()
    }
```

Integration:

- PMGG und GoalPriorityProtocol nutzen `blended_targets` statt rohe Song-Targets.
- Niedrige Analyse-Konfidenz zieht automatisch Richtung konservativer Canonical-Werte.

### §09.10d Ceiling-Proximity-Budget (CPB)

Zweck: Erzwingt Abstand zu physikalischen Ceilings (DR/BW), damit Enhancement nicht in Artefakte kippt.

```python
def compute_cpb(material_ceiling: float, current_value: float, mode: str) -> float:
    margin = max(0.0, material_ceiling - current_value)
    safety = 0.70 if mode == "restoration" else 0.50
    return float(np.clip(safety * margin, 0.0, material_ceiling))
```

Integration:

- Additive Phasen duerfen pro Iteration nur einen Anteil von `cpb` verbrauchen.
- Bei `cpb -> 0` muss die Phase auf konservatives Wet/Dry zurueckfallen.

### §09.10d-2 Headroom-Scalar für Additive Phasen (v10.0.0)

Ergänzt §09.10d. Enhancement-Phasen der Familien `harmonic_reconstruction`, `harmonic_enhancement`, `tonal_enhancement`, `source_enhancement`, `stereo_enhancement`, `stereo_generation` erhalten einen Strength-Scalar proportional zum **absoluten Headroom** bis zur physikalischen Decke:

```python
HR_WINDOW = 0.25   # Fenster: innerhalb 0.25 zur Decke → Dämpfung beginnt
HR_GOALS  = ("brillanz", "waerme", "spatial_depth", "bass_kraft", "sep_fidelity")

min_hr = 1.0
for goal in HR_GOALS:
    headroom = max(0.0, ceiling[goal] - current_score[goal])  # absolut
    hr_ratio  = min(1.0, headroom / HR_WINDOW)
    min_hr    = min(min_hr, hr_ratio)

hr_strength = float(np.clip(min_hr, 0.40, 1.0))
# strength_used = combined_strength * hr_strength   (nur wenn hr_strength < 0.95)
```

**Rationale**: Psychoakustisches Sättigungsgesetz — Grenznutzen sinkt asymptotisch nahe der physikalischen Decke; Artefaktrisiko steigt gleichzeitig. Die lineare Skalierung über ein 0.25-Fenster ist konservativ und vermeidet Over-Processing bei Shellac/Wachszylinder. Restorative und Pflicht-Phasen sind ausgenommen (Repair bleibt immer auf voller Stärke).

**VERBOTEN**: `hr_range = ceil - 0.30; hr_ratio = 1 - (curr - 0.30) / hr_range` — relative Normierung mit 0.30-Baseline dämpft CD-Phasen mit 0.39 Headroom (CD `brillanz=0.60, ceil=0.99` → scalar=0.57 obwohl kein Ceiling-Problem).

### §09.10e Retry-Temperature (RT)

Zweck: Vereinheitlicht Retry-Aggressivitaet in PMGG/CIG/AFG fuer schwere vs. leichte Songs.

```python
def compute_retry_temperature(restorability: float, tcci: float, artifact_freedom_score: float) -> float:
    hard_song = 1.0 - np.clip(restorability / 100.0, 0.0, 1.0)
    artifact_risk = 1.0 - np.clip(artifact_freedom_score, 0.0, 1.0)
    t = 0.50 * hard_song + 0.30 * tcci + 0.20 * artifact_risk
    return float(np.clip(t, 0.0, 1.0))
```

Integration:

- Hohe Temperatur: mehr Strength-Reduktionsstufen statt harter Rollback.
- Niedrige Temperatur: schnellere Akzeptanz oder frueheres Skip bei unkritischen Deltas.

### §09.10f Export-Reliability-Score (ERS)

Zweck: Einheitlicher Vertrauensindikator fuer Final-Export, ohne neue Metriken einzufuehren.

```python
def compute_export_reliability(
    hpi: float,
    artifact_freedom: float,
    passed_goals: int,
    total_goals: int,
    reference_confidence: float,
) -> float:
    goal_ratio = 0.0 if total_goals <= 0 else passed_goals / total_goals
    score = (
        0.35 * np.clip(hpi, 0.0, 1.0)
        + 0.30 * np.clip(artifact_freedom, 0.0, 1.0)
        + 0.20 * np.clip(goal_ratio, 0.0, 1.0)
        + 0.15 * np.clip(reference_confidence, 0.0, 1.0)
    )
    return float(np.clip(score, 0.0, 1.0))
```

Integration:

- `ERS < 0.55`: verpflichtende Recovery-Kaskade fortsetzen.
- `ERS >= 0.55`: Export zulaessig, Status weiterhin transparent (`recovered`/`degraded`).

### §09.10g Invarianten

1. Keine neuen Nutzer-Inputs: alle Parameter sind reine Funktionen vorhandener Signale.
2. Advisory-only gegenueber Hard-Gates: PMGG/CIG/AFG/HPI bleiben normative Endinstanzen.
3. Material-Ceilings bleiben unantastbar: DR/BW-Grenzen werden niemals durch Meta-Parameter geloest.
4. Bei Ausnahme: neutraler Fallback (`1.0` oder konservativer Canonical-Wert), Pipeline darf nicht blockieren.

### §09.10h Goal-Coverage-Index (GCI)

Zweck: Gewichtete Zielabdeckung statt roher Pass-Quote. P1/P2-Fehlschlaege zaehlen staerker als P5.

```python
def compute_goal_coverage_index(musical_goals_passed: dict[str, bool]) -> float:
    # P1: 1.4, P2: 1.2, P3: 1.0, P4: 0.8, P5: 0.6
    return weighted_pass_ratio
```

Integration:

- `export_reliability.goal_coverage_index` als zusaetzlicher Qualitaetsanker.
- `goal_deficit_ratio = 1 - GCI` speist Recovery-Pressure.

### §09.10i Calibrated-Reference-Confidence (CRC)

Zweck: Referenzvertrauen aus Target-Confidence plus Kontextstabilitaet (Transfer-Chain + Carrier-Recovery).

```python
def compute_reference_confidence(target_confidence: float, tcci: float, carrier_chain_recovery_ratio: float) -> float:
    # hohe TCCI und hohe Carrier-Inversion senken Referenzvertrauen
    return clipped_confidence_0_1
```

Integration:

- Ersetzt rohe `targets_confidence` als ERS-Input (`reference_confidence`).
- Schafft robuste Bewertung bei starker Carrier-Chain-Inversion (§0d).

### §09.10j Recovery-Pressure-Index (RPI)

Zweck: Ein numerischer Druckindikator, ob Recovery-Pfade noch weiterlaufen sollten.

```python
def compute_recovery_pressure_index(
    fallback_attempts: int,
    rollback_count: int,
    goal_deficit_ratio: float,
) -> float:
    # 0..1 aus Fallback-Versuchen, Rollbacks und Zieldefizit
    return clipped_pressure_0_1
```

Integration:

- `export_reliability.recovery_pressure_index` fuer UI/Diagnostik.
- Advisory-only; harte Export-Gates unveraendert.

---

## §09.11 [RELEASE_MUST] Maximum-Achievable-Score (MAS) — Formale Definition (v10.0.0)

MAS ist der höchste physikalisch erreichbare Goal-Score für einen konkreten Song — gegeben Material, Ära, Genre und Restorability. MAS ist **das primäre Optimierungsziel der Pipeline** — kein Bodengrenzwert, kein Stopp-Signal, sondern aktives Konvergenzziel.

### MAS-Quelle und Ceiling-Clamp

```python
# Schritt 1: SongGoalTargets berechnen (§09.2 Zwei-Ebenen-API)
mas_targets = estimate_song_goal_targets(
    era_decade=era_decade,
    genre_label=genre_label,
    material_chain=material_chain,
    restorability=restorability_score,
).targets  # dict[str, float] — ein Wert pro der 15 Musical Goals

# Schritt 2: Ceiling-Clamp — kein Goal darf physikalisches Material-Maximum überschreiten
for goal, value in mas_targets.items():
    ceiling = PHYSICAL_CEILING.get(material_type, {}).get(goal, 1.0)
    mas_targets[goal] = min(value, ceiling)
```

**Invariante**: Wenn `PHYSICAL_CEILING[material][goal] < CANONICAL_THRESHOLDS[goal]`, ist MAS = PHYSICAL_CEILING — das physikalische Material-Limit hat Vorrang vor dem globalen Canonical-Floor. Dies ist kein Kalibrierungsfehler, sondern physikalische Realität (Shellac/Artikulation: Ceiling < Floor=0.88, WaxCyl/Brillanz: Ceiling=0.55 < Floor=0.78). Die Pipeline zielt auf das Erreichbare, nicht auf das physikalisch Unmögliche.

### PHYSICAL_CEILING — Physikalische Obergrenzen pro Material

```python
PHYSICAL_CEILING: dict[str, dict[str, float]] = {
    "shellac": {
        "brillanz":           0.72,
        "transparenz":        0.72,
        "spatial_depth":      0.55,
        "artikulation":       0.78,
        "separation_fidelity":0.60,
    },
    "wax_cylinder": {
        "brillanz":           0.55,
        "transparenz":        0.60,
        "spatial_depth":      0.45,
        "artikulation":       0.70,
    },
    "vinyl": {
        "brillanz":           0.86,
        "transparenz":        0.84,
        "spatial_depth":      0.80,
    },
    "tape": {
        "brillanz":           0.88,
        "transparenz":        0.86,
    },
    "reel_tape": {
        "brillanz":           0.90,
        "transparenz":        0.88,
    },
    "mp3_low": {
        "brillanz":           0.80,
        "transparenz":        0.78,
        "artikulation":       0.82,
    },
    # cd_digital, dat, mp3_high, flac: kein physikalisches Ceiling (leeres Dict)
}
```

### MAS-Konvergenz-Metrik

```python
MAS_TOLERANCE       = 0.02   # P1/P2 "erreicht" wenn gap ≤ 0.02
MAS_FULL_TOLERANCE  = 0.05   # P3–P5 "erreicht" wenn gap ≤ 0.05
MAS_OVERSHOOT_TOL   = 0.03   # Strength-Clamp wenn post > MAS + 0.03

def compute_mas_convergence(
    current_scores: dict[str, float],
    mas_targets: dict[str, float],
) -> dict:
    gaps = {g: mas_targets[g] - current_scores.get(g, 0.0) for g in mas_targets}
    p1p2_achieved = all(gaps[g] <= MAS_TOLERANCE for g in P1P2_GOALS)
    p3p5_achieved = all(gaps[g] <= MAS_FULL_TOLERANCE for g in P3P5_GOALS)
    overshooting = [
        g for g in mas_targets
        if current_scores.get(g, 0.0) > mas_targets[g] + MAS_OVERSHOOT_TOL
    ]
    return {
        "gaps": gaps,
        "p1p2_achieved": p1p2_achieved,
        "p3p5_achieved": p3p5_achieved,
        "fully_achieved": p1p2_achieved and p3p5_achieved,
        "overshooting_goals": overshooting,
        "max_p1p2_gap": max(gaps[g] for g in P1P2_GOALS),
        "mean_p1p2_gap": sum(gaps[g] for g in P1P2_GOALS) / len(P1P2_GOALS),
    }
```

### Neue Module (zu implementieren)

| Modul | Pfad | Zweck |
| --- | --- | --- |
| `FastGoalProxy` | `backend/core/dsp/fast_goal_proxy.py` [ROADMAP] | DSP-Proxy-Messung aller 15 Goals in ≤ 200 ms, kein ML |
| `_fast_goal_snapshot()` | `backend/core/unified_restorer_v3.py` | Pro-Phase-Aufruf von `FastGoalProxy.measure_fast()` |
| `_check_mas_convergence()` | `backend/core/unified_restorer_v3.py` | Pipeline-Stop-Signal wenn MAS erreicht |
| `PHYSICAL_CEILING` | `backend/core/calibration_matrix.py` | Material-spezifische Goal-Obergrenzen |

### CI-Test-Invarianten

```python
# tests/normative/test_mas_convergence.py
def test_mas_ceiling_clamped_to_physical_material():
    """Shellac-MAS darf physikalisches Ceiling nicht überschreiten."""
    mas = estimate_song_goal_targets("1930s", "jazz", "shellac", 40).targets
    for goal, ceil in PHYSICAL_CEILING["shellac"].items():
        assert mas[goal] <= ceil

def test_mas_target_at_or_above_canonical_floor():
    """MAS muss immer ≥ CANONICAL_THRESHOLDS sein."""
    mas = estimate_song_goal_targets("1970s", "schlager", "vinyl", 70).targets
    for goal in P1P2_GOALS:
        assert mas[goal] >= CANONICAL_THRESHOLDS_RESTORATION[goal]

def test_compute_mas_convergence_early_stop():
    """Wenn alle P1/P2 ≤ MAS_TOLERANCE: fully_achieved muss True sein."""
    scores = {g: mas_targets[g] - 0.01 for g in GOAL_NAMES}  # alle knapp unter MAS
    result = compute_mas_convergence(scores, mas_targets)
    assert result["p1p2_achieved"] is True
```

> Integration: UV3 `_profiled_phase_call_with_delta()` (Spec 02 §2.64); Pipeline-Stop (Spec 02 §2.65); normatives Prinzip (Copilot §0k)

---

## §09.10 [RELEASE_MUST] §GOAL_BASELINE_CHECK — Pre-Pipeline-Absicherung (v10.0.0)

> **Normative Spec-Grundlage** für das in Copilot Instructions §0k beschriebene Prinzip.
> Schließt die CAUSE_TO_PHASES-Lücke: wenn DefectScanner einen Goal-Defizit nicht
> als Defect-Cause erkennt, wird die Pipeline ohne Korrektur-Phase ausgeführt und
> FeedbackChain kann das Goal über das Original-Niveau nicht heben.

### §09.10a Algorithmus (in UV3._execute_pipeline(), vor Phasen)

```python
# Zeitbudget: ≤ 200 ms DSP-Proxy (FastGoalProxy, kein ML)
_goal_snapshot = _fast_goal_snapshot(audio_input, sr)
_effective_targets = resolve_effective_goal_targets(
    mode=mode,
    material_type=material_type,
    transfer_chain=transfer_chain,
    era_decade=era_decade,
    genre_label=genre_label,
    restorability_score=restorability_score,
    song_goal_weights=song_goal_weights,
    physical_ceiling=physical_ceiling.ceiling,
    applicable_goals=applicable_goals,
)

for goal_name, goal_score in _goal_snapshot.items():
    if goal_name not in applicable_goals:
        continue
    _effective_target = _effective_targets[goal_name]
    _threshold = _effective_target * 0.97  # 3% Proxy-Toleranz

    if goal_score < _threshold:
        # Goal liegt unter effektivem Zielwert → Recovery-Phase einfügen
        _recovery_phases = get_goal_recovery_phases(goal_name, is_studio_2026)

        for phase_id in _recovery_phases:
            if phase_id not in selected_phases:
                # §0a-Guard: Studio-Only-Phasen nicht in Restoration einfügen
                if not is_studio_2026 and phase_id in _STUDIO_ONLY_PHASES:
                    continue
                selected_phases.insert(
                    _get_primary_insertion_index(selected_phases, phase_id),
                    phase_id
                )
                logger.info(
                    "goal_baseline_check: goal=%s score=%.3f < target=%.3f "
                    "→ inserting recovery phase=%s",
                    goal_name, goal_score, _threshold, phase_id
                )

        metadata["goal_baseline_gaps"][goal_name] = {
            "score_before": goal_score,
            "effective_target": _effective_target,
            "proxy_threshold": _threshold,
            "recovery_phases_added": _recovery_phases,
        }
```

### §09.10b Invarianten

- **Non-blocking**: Exception in `_fast_goal_snapshot()` → kein Pipeline-Abbruch; normale Ausführung.
- **§0a-Guard** ist aktiv: `_STUDIO_ONLY_PHASES = {"phase_21_exciter", "phase_35_multiband_compression", "phase_42_vocal_enhancement"}` — dürfen niemals in Restoration-`selected_phases` eingefügt werden.
- **§2.45 Minimal-Intervention**: Recovery-Phase wird nur eingefügt wenn `goal_score < effective_target × 0.97` — kein Eingriff bei gesunden Goals.
- **Target-Konsistenz**: `_fast_goal_snapshot()` MUSS für jedes Goal denselben semantischen Messansatz wie `MusicalGoalsChecker`/PMGG verwenden. Abweichende Proxy-Algorithmen sind nur erlaubt, wenn ein Test ihre monotone Äquivalenz gegen die finale Metrik belegt.
- **Disk-Guard**: Alle Phase-IDs in `get_goal_recovery_phases()` MÜSSEN gegen `backend/core/phases/phase_*.py` validiert sein (Test: `test_get_goal_recovery_phases_all_phase_ids_exist_on_disk()`).
- **Keine Duplikate**: Vor dem Einfügen `if phase_id not in selected_phases`.
- **Garantiepfad**: Jedes in `metadata["goal_baseline_gaps"]` registrierte Goal MUSS am Pipeline-Ende entweder `final_score ≥ effective_target` erreichen oder mit `metadata["goal_target_limitations"][goal]` physikalisch begründet und als `degraded`/`recovered_with_limitations` ausgewiesen werden.

---

## §09.11 [RELEASE_MUST] Goal-Recovery-Phase-Mappings (v10.0.0)

> **Normative Vollständig-Tabelle** für `get_goal_recovery_phases()` in
> `backend/core/calibration_matrix.py`. Bis v10.0.0 nur in Copilot Instructions beschrieben —
> hier erstmalig als Spec-Grundlage spezifiziert.

### [RELEASE_MUST] §09.11a `_GOAL_TO_RECOVERY_PHASES_RESTORATION`

```python
_GOAL_TO_RECOVERY_PHASES_RESTORATION: Dict[str, List[str]] = {
    # P0 — Vokalqualität (nur wenn panns_singing ≥ 0.35)
    "VocalQuality":         ["phase_65_vocal_naturalness_restoration", "phase_03_denoise"],
    # phase_65: DSP-Korrektiv (subtraktiv/korrektiv, §0a-konform für Restoration).
    # Adressiert HNR-Verlust nach NR, Spektral-Tilt-Shift, Formant-Drift.
    # phase_42_vocal_enhancement: VERBOTEN in Restoration (§0a) — nie hier eintragen.
    "FormantFidelity":      ["phase_42_vocal_enhancement"],                        # §0a: VERBOTEN in Restoration; Guard blockt es

    # P1 — Natürlichkeit, Authentizität
    "Natürlichkeit":        ["phase_29_tape_hiss_reduction", "phase_03_denoise"],
    "Authentizität":        ["phase_09_declicker", "phase_11_dehum"],

    # P2 — Tonales Zentrum, Timbre, Artikulation
    "TonalCenter":          ["phase_12_wow_flutter_fix", "phase_04_riaa_eq"],
    "Timbre":               ["phase_20_harmonic_analyzer", "phase_07_harmonic_enhancer"],
    "Artikulation":         ["phase_03_denoise", "phase_09_declicker"],
    "TransientEnergie":     ["phase_26_transient_shaper", "phase_08_transient_shaper_hpss"],
    # phase_26: Dosierte Transient-Energie-Wiederherstellung (Restoration-konform)
    # phase_08: HPSS-gestütztes Transient-Enhancement (subtraktiv/korrektiv)

    # P3 — Emotionalität, Mikrodynamik, Groove
    "Emotionalität":        ["phase_40_loudness_normalizer", "phase_26_transient_shaper"],
    "MikroDynamik":         ["phase_26_transient_shaper", "phase_14_dc_offset"],
    "Groove":               ["phase_12_wow_flutter_fix", "phase_15_phase_corrector"],

    # P4 — Transparenz, Wärme, Basskraft, Trennschärfe
    "Transparenz":          ["phase_29_tape_hiss_reduction", "phase_30_hum_remover"],
    "Wärme":                ["phase_07_harmonic_enhancer", "phase_20_harmonic_analyzer"],
    "BassKraft":            ["phase_05_rumble_filter", "phase_26_transient_shaper"],
    "SepFidelity":          ["phase_20_harmonic_analyzer", "phase_03_denoise"],

    # P5 — Brillanz, Raumtiefe
    "Brillanz":             ["phase_06_bandwidth_extension", "phase_07_harmonic_enhancer"],
    "Raumtiefe":            ["phase_46_spatial_enhancement", "phase_33_stereo_expander"],
}
```

### §09.11b `_GOAL_TO_RECOVERY_PHASES_STUDIO_EXTRAS`

```python
# Studio 2026: Zusätzliche Phasen (ergänzend zu Restoration-Liste)
_GOAL_TO_RECOVERY_PHASES_STUDIO_EXTRAS: Dict[str, List[str]] = {
    "VocalQuality":         ["phase_42_vocal_enhancement"],   # Erlaubt in Studio 2026
    "FormantFidelity":      ["phase_42_vocal_enhancement"],   # Erlaubt in Studio 2026
    "Brillanz":             ["phase_23_flashsr_upsampling"],
    "Wärme":                ["phase_21_exciter"],             # Erlaubt in Studio 2026
    "Transparenz":          ["phase_35_multiband_compression"],  # Erlaubt in Studio 2026
    "MikroDynamik":         ["phase_35_multiband_compression"],
}
```

### [RELEASE_MUST] §09.11c Richtungsregel (Anti-Inversion-Guard)

**VERBOTEN**: Recovery-Phase darf nicht in entgegengesetzter Richtung zum Goal-Defizit wirken.

| Goal-Defizit | Verbotene Recovery-Phase | Grund |
| --- | --- | --- |
| `Raumtiefe` zu niedrig | `phase_49_advanced_dereverb` | Entfernt Raumcues — verstärkt Defizit |
| `Wärme` zu niedrig | `phase_31_digital_artifact_repair` | Bereinigt Obertöne — entfernt Wärme |
| `Brillanz` zu niedrig | `phase_05_rumble_filter` | Tiefpasscharakter — senkt Brillanz |
| `BassKraft` zu niedrig | `phase_06_bandwidth_extension` | HF-Erweiterung ohne Bass-Boost |

### §09.11d Disk-Guard (Pflicht-Test)

```python
# tests/normative/test_calibration_matrix.py
def test_get_goal_recovery_phases_all_phase_ids_exist_on_disk():
    """
    Alle Phase-IDs in _GOAL_TO_RECOVERY_PHASES_RESTORATION und
    _GOAL_TO_RECOVERY_PHASES_STUDIO_EXTRAS MÜSSEN auf Disk existieren.
    """
    import glob
    phase_files = {
        Path(p).stem
        for p in glob.glob("backend/core/phases/phase_*.py")
    }
    for goal, phases in {
        **_GOAL_TO_RECOVERY_PHASES_RESTORATION,
        **_GOAL_TO_RECOVERY_PHASES_STUDIO_EXTRAS
    }.items():
        for phase_id in phases:
            assert phase_id in phase_files, (
                f"Recovery phase '{phase_id}' for goal '{goal}' "
                f"does not exist on disk — fix _GOAL_TO_RECOVERY_PHASES"
            )
```

---

## §09.12 [RELEASE_MUST] Restorability-adaptive Floor-Skalierung (v10.0.0)

> **Problem**: `get_material_floor()` liefert absolute Material-Böden (Shellac: 0.72, Vinyl: 0.82,
> CD: 0.90). Bei extremer Degradierung (`restorability < 30`) sind diese Böden physikalisch
> unerreichbar — nicht weil die Pipeline versagt, sondern weil das Material zerstört ist.
> Resultat: §09.10 §GOAL_BASELINE_CHECK und PMGG-Gate lösen endlose Recovery-Kaskaden aus,
> obwohl das **maximal mögliche** Ergebnis bereits erreicht ist.

### §09.12a Algorithmus

```python
# backend/core/calibration_matrix.py

RESTORABILITY_SCALE_MIN = 0.72  # Minimum-Skalierungsfaktor
                                # Böden kollabieren maximal auf 72 % des Material-Bodens
                                # (verhindert unkontrollierten Qualitätsabfall bei schlecht restaurierbarem Material)

def get_effective_material_floor(
    material_type: str,
    goal_name: str,
    restorability_score: float,  # 0–100
) -> float:
    """
    Skaliert material-adaptiven Boden proportional zur Restorierbarkeit.

    Formulierung:
        floor_base   = get_material_floor(material_type, goal_name)
        scale        = max(RESTORABILITY_SCALE_MIN, restorability_score / 100.0)
        floor_eff    = floor_base × scale

    Beispiele (Vinyl, Brillanz, floor_base = 0.82):
        restorability=100 → floor_eff = 0.82 × 1.00 = 0.82   (voll)
        restorability=72  → floor_eff = 0.82 × 0.72 = 0.59   (skaliert)
        restorability=20  → floor_eff = 0.82 × 0.72 = 0.59   (Minimum-Clip)

    Rationale: Restorability-Beitrag ist nicht linear — bei restorability=20
    ist das Material zu 80 % zerstört, aber der Effektiv-Boden soll nicht
    auf 20 % kollabieren (das wäre ein Freibrief für Totalausfall). Minimum
    0.72 bedeutet: auch extremst degradiertes Material MUSS 72 % des Normal-
    bodens erreichen — oder der Export erhält Status "degraded".
    """
    floor_base = get_material_floor(material_type, goal_name)
    scale = max(RESTORABILITY_SCALE_MIN, float(np.clip(restorability_score / 100.0, 0.0, 1.0)))
    return float(floor_base * scale)
```

### §09.12b Verwendung in UV3

**§GOAL_BASELINE_CHECK (§09.10)** MUSS `get_effective_material_floor()` verwenden:

```python
_material_floor = get_effective_material_floor(material_type, goal_name, restorability_score)
```

**PMGG-Gate** verwendet nicht rohe `CANONICAL_THRESHOLDS`, sondern den `effective_targets`-/`_pmgg_ceiling_capped_targets`-Vektor aus §09.2b. `get_material_floor()` bleibt die normative Basis für UI, Tests und Voranalyse; die Pipeline vergleicht gegen effektiv berechnete, restorability- und ceiling-korrigierte Ziele.

### §09.12c Export-Status bei niedrigen Restorability-Werten

```python
# In UV3._execute_pipeline():
if restorability_score < 30:
    metadata["degraded_restorability"] = True
    metadata["degraded_restorability_score"] = restorability_score
    # Export-Status: "degraded" (nicht "recovered") — ehrliche Kommunikation
    # an die UI, dass das Ergebnis physikalisch limitiert ist
```

### §09.12d Invarianten

- **`get_material_floor()` unverändert**: Normative Böden bleiben kanonisch für Tests und UI
- **UV3-Pipeline, PMGG, FeedbackChain und End-Gate** verwenden `effective_targets` aus §09.2b; Pre-Analysis/UI-Preview darf `get_material_floor()` zusätzlich anzeigen, muss aber den effektiven Zielwert als Export-Gate-Ziel ausweisen.
- **Non-breaking**: Bestehende Tests auf `get_material_floor()` schlagen nicht fehl
- **Test-Pflicht**: `test_get_effective_material_floor_restorability_scale()` — überprüft:
  - `restorability=100` → voller Boden
  - `restorability=50` → 50 % Skalierung (über Min)
  - `restorability=10` → RESTORABILITY_SCALE_MIN (0.72) als Boden
  - `restorability=72` → exakt 72 % (Grenzfall Min-Clip)

---

## §09.13 [RELEASE_MUST] Chain-End-Codec-Floor-Override (v10.0.0)

> **Problem**: `get_material_floor()` wird mit dem primären Träger aufgerufen (z.B. `cassette`).
> Bei einer Transfer-Kette wie `cassette → mp3_low` ist der limitierende Faktor jedoch der
> Codec am Kettenende — nicht der analoge Primärträger.
> Resultat: `spatial_depth`-Boden für Kassette (~0.55) liegt über dem physikalisch erreichbaren
> Wert für `mp3_low`-Joint-Stereo-komprimiertes Material (~0.38) → permanente false violations.

### §09.13a Codec-Ceiling-Tabelle

```python
_CHAIN_END_GOAL_CEILINGS: dict[str, dict[str, float]] = {
    "mp3_low":   {"spatial_depth": 0.38, "separation_fidelity": 0.68, "brillanz": 0.52, "transparenz": 0.75},
    "mp3_high":  {"spatial_depth": 0.44, "separation_fidelity": 0.74, "brillanz": 0.62, "transparenz": 0.82},
    "aac":       {"spatial_depth": 0.44, "separation_fidelity": 0.74, "brillanz": 0.60, "transparenz": 0.82},
    "streaming": {"spatial_depth": 0.38, "separation_fidelity": 0.68, "brillanz": 0.50, "transparenz": 0.74},
    "minidisc":  {"spatial_depth": 0.36, "separation_fidelity": 0.66, "brillanz": 0.48, "transparenz": 0.72},
}
```

### §09.13b Algorithmus

```python
def get_material_floor(
    material_type: str,
    goal_name: str,
    restorability_score: float | None = None,  # reserviert für §09.12-Kompatibilität
    transfer_chain: list[str] | None = None,
) -> float:
    floor = _MATERIAL_GOAL_FLOORS.get(material_type, _DEFAULT_FLOORS).get(goal_name, 0.60)
    # Chain-End-Codec-Override: Codec am Kettenende strenger als Primärträger?
    if transfer_chain:
        chain_last = transfer_chain[-1]
        codec_floors = _CHAIN_END_GOAL_CEILINGS.get(chain_last, {})
        codec_floor = codec_floors.get(goal_name)
        if codec_floor is not None and codec_floor < floor:
            floor = codec_floor
    return float(floor)
```

### §09.13c Invarianten

- **Chain-End-Override NUR wenn Codec strenger** (`codec_floor < primary_floor`): ein höherwertiger Codec (z.B. `mp3_high` auf Shellac-Basis) darf den Shellac-Boden nicht lockern.
- **`transfer_chain=None` = kein Override**: bestehende Tests ohne Transfer-Chain schlagen nicht fehl (nicht-breaking).
- **Cascading**: `get_effective_material_floor()` (§09.12) ruft `get_material_floor()` auf und erbt den Chain-End-Override automatisch.
- **Zusammenspiel mit GAF §2.32a**: Beide Mechanismen sind orthogonal. §2.32a deaktiviert das Goal vollständig bei Near-Mono-Codec (corr ≥ 0.83). §09.13 setzt den Boden korrekt, wenn das Goal aktiv bleibt (corr < 0.83 oder Mono-Erkennung nicht greift).
- **Test-Pflicht**: `test_get_material_floor_chain_end_codec_override()`:
  - `('cassette', 'spatial_depth', None, ['mp3_low'])` → `0.38`
  - `('cassette', 'spatial_depth', None, ['mp3_high'])` → `0.44`
  - `('shellac', 'brillanz', None, ['mp3_low'])` → Shellac-Wert (Shellac bereits strenger)
  - `('vinyl', 'spatial_depth', None, None)` → Vinyl-Standardboden

---

## Referenzen

- Spec §01: Musical Goals (15 Ziele)
- Spec §02: Pipeline (Phasen, Gates)
- Spec §2.56: Song-Goal-Importance (per-Song Gewichtung)
- Spec §2.54: Adaptive Schwellwerte
- Spec §0a: Modus-Differenzierung (Restoration vs. Studio 2026)
- Copilot Instructions §2.56, §2.54, §2.29, §2.48, §2.49

---

**Version**: 1.1 | **Datum**: 25. Juni 2026 | **Status**: normativ
