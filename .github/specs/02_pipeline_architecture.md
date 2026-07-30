# Aurik 10 — Spec 02: Pipeline-Architektur

> Kanonischer Pipeline-Ablauf, RestorationResult-Spec, Restaurierungs-Modi,
> StemRemixBalancer, Studio-2026-Verarbeitungskette.

---

## §v10 Pleasantness-First (2026-07-05)

> **HPE ist oberste Instanz.** PMGG darf Phasen ueberspringen, wenn sie
> den Klang fuer menschliche Ohren verschlechtern (§2.29 v10).
> Siehe backend/core/per_phase_musical_goals_gate.py,
> backend/core/human_pleasantness_estimator.py.
> **Kein Rollback-Verbot mehr.** CausalDefectReasoner kann irren — das Ohr nicht.

---

## §v10.303 Phase-0 ML-Pre-Processor (implementiert 2026-07-28)

> **ML vor DSP:** Apollo (Codec-Decompression) → DeepFilterNet v3 (Noise-Floor) →
> Resemble Enhance (Spektrale Reparatur). Wissenschaftliche Reihenfolge gemäß
> Carrier-Chain-Inversion (§2.46). 12 redundante DSP-Phasen werden automatisch
> übersprungen. Goal-Baseline wird nach Phase 0 neu kalibriert.
>
> **Referenzrahmen-Verschiebung (§v10.303.31):** Nach Phase 0 wird die
> Qualitäts-Baseline (`original_audio_for_goals`) von degradiertem Original auf
> Phase-0-Output umgestellt. Alle Post-Pipeline-Prüfungen (Goals, Goosebumps,
> EmotionalArc, IAD, HPI) messen gegen das realistisch Erreichbare.
>
> Siehe `.github/specs/v10.303.17_phase0_architecture.md` für die vollständige
> Architektur-Dokumentation.

---

## §1.4 Restaurierungs-Modi

| Modus | Ziel | Charakteristik |
| --- | --- | --- |
| **`restoration`** | Originalgetreue Restauration — Tonträgerkette invertieren (§2.46) | Erhalt des historischen Klangs, minimaler Eingriff, LUFS-Diff ≤ 1 LU, kein Harmonic-Exciter, GP `mode="restoration"` konservativ |
| **`studio2026`** | Highend-Studio-Klang — Carrier-Chain-Inversion + Enhancement | Modern, kräftig — PQS MOS ≥ 4.5, Brillanz ≥ 0.90, Bass-Kraft ≥ 0.88, GP `mode="studio2026"` aggressiv |

**Restoration-Modus Pflicht-Invarianten:**

- Chroma-Korrelation Original↔Restauriert ≥ 0.95
- LUFS-Differenz ≤ 1 LU
- Kein hinzugefügtes Harmonic-Exciter-Material
- Rauschboden: CD-ähnlicher Export-Boden für alle analogen Tonträger (`shellac`, `wax_cylinder`, `lacquer_disc`, `wire_recording`, `vinyl`, `tape`, `reel_tape`, `cassette`); analoges Hiss-/Oberflächenrauschen darf nicht als Mindestboden zurückkehren (§0a/V21)
- HPI-Gate: `timbral_fidelity` dominant (§2.44) — akustisch nicht unterscheidbar vom Original

**Studio-2026-Modus Pflicht-Invarianten:**

- PQS MOS ≥ 4.5 (internes Spitzenziel)
- Brillanz-Score ≥ 0.90 (verschärft)
- Bass-Kraft ≥ 0.88 (verschärft)
- Rauschboden ≤ −72 dBFS (§0a)
- HPI-Gate: PQS-Improvement dominant (§2.44)

### §1.4b [RELEASE_MUST] Psychoakustische Anti-Maschinenklang-Invariante

Ziel: Hoerbar natuerliche Musikwiedergabe vor technischer Ueberoptimierung.

Pflichtregeln:

1. Additive/Enhancement-Familien MUESSEN vor der finalen Phasenstaerke durch einen
    psychoakustischen Naturalness-Guard skaliert werden.
2. Der Guard MUSS maschinell klingende Muster explizit bestrafen:
    `musical_noise`, `metallic_ringing`, `roughness_regression`,
    kumulative `psycho_delta_penalty` und `rolling_risk`.
3. Bei Vokal-Material (`panns_singing >= 0.35`) MUSS der Guard zusaetzlich
    `breath_naturalness` und `spectral_color_preservation` auswerten.
4. Bei aktivem Guard ist nur Daempfung erlaubt (`scalar <= 1.0`), kein Boost.

Kanonischer Vertragsrahmen (UV3):

```python
naturalness_guard = _compute_naturalness_guard_scalar(
     phase_family=...,
     studio_mode=...,
     panns_singing=...,
     vocal_quality_check=...,            # inkl. breath_naturalness/spectral_color_preservation
     phase_metadata_accumulator=...,     # inkl. n_musical_noise/n_metallic_ringing/
                                                     # roughness_regression/_psycho_runtime_state
)
uq_scalar *= naturalness_guard["scalar"]
```

VERBOTEN:

- Enhancement-Phasen ohne Naturalness-Guard-Skalierung.
- Ignorieren von `rolling_risk` bei vorhandenen Laufzeit-Psycho-Deltas.
- Akzeptieren von wiederkehrendem Musical-Noise/Metallic-Ringing ohne
  konservative Staerkedaempfung.

### §1.4a [RELEASE_MUST] Fail-Fast-Kontrakt für kritische Qualitätsmodule (v10.0.0)

Kritische Qualitätsmodule dürfen in `restoration` und `studio2026` nicht unbemerkt in
qualitativ schwache Platzhalterpfade fallen.

**Kritische Module:**

- `PerceptualQualityScorer` / PQS
- `HolisticPerceptualGate`
- `ArtifactFreedomGate`
- `MusicalGoalsChecker` (P1/P2)

**Pflichtregeln:**

1. Fällt ein kritisches Modul zur Laufzeit aus, MUSS ein strukturierter
    `fail_reason`-Eintrag erzeugt werden (`severity=failed`, `component`, `error_code`).
2. Für `studio2026` ist bei Ausfall von PQS oder HPI kein stiller Positiv-Proxy erlaubt.
    Der Run MUSS in einen kontrollierten Safe-Mode mit striktem End-Gate wechseln
    oder fail-fast abbrechen.
3. Für `restoration` gilt: Primum non nocere hat Vorrang. Bei unklarer Qualitätslage
    MUSS auf das beste artefaktfreie Checkpoint-Audio oder Input zurückgerollt werden.
4. VERBOTEN: Konstante positive Platzhalter (`pqs_improvement=0.1` o. ä.) als
    dauerhafter Ersatz für echte Qualitätsmessung im finalen Exportpfad.

**Invariante:** Ein Export darf nie allein deshalb passieren, weil ein kritischer
Qualitätsdetektor nicht verfügbar war.

### §1.4c [RELEASE_MUST] Disziplin-Parallelisierung via InnovationSuperiorityOrchestrator (v10.0.0)

Ziel: Alle Kern-Disziplinen der Restaurierung (Defektbehebung, Vokalintegritaet,
timbrale Treue, zeitliche Praezision, raeumliche Kohärenz, Zielkonvergenz)
werden parallel im Closed Loop bewertet und als bounded Innovation-Hinweise
fuer die §2.78-Nachsteuerung bereitgestellt.

Pflichtregeln:

1. Orchestrator laeuft pro Phase parallel zum Runtime-Reliability-Layer.
2. Orchestrator ist advisory-only und darf harte Gates nicht ersetzen.
3. Goal-Confidence-Uplifts muessen bounded bleiben (`<= 0.05` pro Goal).
4. Endgueltige Goal-Konfidenzen bleiben auf `[0.20, 0.98]` begrenzt.
5. Telemetrie muss mindestens enthalten:
   `priority_goals`, `recovery_phase_hints`, `goal_confidence_uplift`,
   `discipline_scores`, `innovation_intensity`.

Kanonischer Integrationsrahmen (UV3):

```python
from backend.core.innovation_superiority_orchestrator import (
    get_innovation_superiority_orchestrator,
)

orch = get_innovation_superiority_orchestrator()
plan = orch.build_realtime_plan(...)

for goal, uplift in plan.goal_confidence_uplift.items():
    goal_confidence[goal] = clip(goal_confidence[goal] + uplift, 0.20, 0.98)
```

VERBOTEN:

- direkte Phase-Injektion ohne bestehenden §2.78-Rescheduler-Vertrag,
- ungebundene Konfidenz-Boosts,
- Gate-Bypass durch Innovationslogik.

### §1.4d [RELEASE_MUST] Kombinierte End-Gate-Weltspitzenmassnahmen (v10.0.0)

Ziel: Geplante und ungeplante Qualitaetsmassnahmen muessen im End-Gate als
ein gemeinsamer, widerspruchsfreier No-Harm-Entscheider laufen.

Pflichtregeln:

1. Candidate-Ranking bleibt primaer multi-goal-basiert (weighted-gap + critical drops).
2. `spatial_depth` hat zusaetzliche Harddrop-Sensitivitaet im Ranking (frueher Schutz vor Raumkollaps).
3. Bei `waerme`-Verletzung MUSS ein konservativer `waerme_focus_rescue`-Kandidat evaluiert werden.
4. Der `waerme_focus_rescue` darf nur low-mid gezielt staerken und MUSS eine Stereo-Schutzkappe haben.
5. Endentscheidung bleibt no-harm: Kandidat nur bei objektiver Rank-Verbesserung.
6. Telemetrie MUSS den gesamten Pfad ausweisen (`candidate_ranking_*`, `waerme_focus_rescue*`).
7. Material-Causal-Reconciliation MUSS in die source-penalties einfliessen:
    niedrige Material-Konfidenz oder Material-Defect-Inkonsistenz reduzieren Original-Prior
    und bevorzugen chain-kompatible Checkpoints.
8. HPI-Referenzwahl darf bei Material-Causal-Reconciliation auf
    `best_carrier_checkpoint` wechseln, auch wenn `carrier_chain_recovery_ratio <= 0.15`,
    sofern eine analoge Stufe in der Transferkette vorliegt und Material-Defect-
    Inkonsistenz aktiv ist.

Kanonischer Ablauf (UV3 End-Gate):

```python
if "waerme" in remaining_violations:
    waerme_candidate, waerme_meta = _build_waerme_focus_rescue_candidate(...)
    if waerme_candidate is not None:
        candidate_sources.append(("waerme_focus_rescue", waerme_candidate, penalty))

rank_current = _rank_goal_recovery_candidate(...)
rank_best = min(candidates, key=final_rank_key)

if rank_best < rank_current:
    apply(rank_best)
```

VERBOTEN:

- Waerme-Rettung ohne Stereo-Schutzkappe.
- Wechsel auf Kandidaten trotz verschlechtertem End-Gate-Rank.
- Telemetrie-loser Kandidatwechsel.

---

## §1.5 Studio-2026-Verarbeitungskette (kanonische Reihenfolge nach Defektkorrektur)

```text
1.  Stem-Separation (MDX23C lokal, Kim_Vocal_2/Kim_Inst)
2.  Vocals: VocalAIEnhancement (stimmtyp-adaptiv) + ConsonantEnhancement (Frikative adaptiv)
    + Vocal-Intimitäts-Gate (Pre/Post-Check; Rescue bei Delta < -0.04)
3.  Sub-Mix-Instrumente: genre-adaptiv (guitar/brass/piano/drums nach PANNs)
4.  Reference Mastering (optional): OT-Spektral-Matching, Chroma-Korrelation ≥ 0.92
5.  Multiband-Dynamik: phase_35_multiband_compression
6.  Präsenz & Air: phase_38 + phase_39 (> 12 kHz)
7.  Stereo-Imaging: phase_48 + phase_46
8.  EraAuthenticPerceptualCompletion (wenn Quell-BW < 10 kHz)
9.  Re-Stem-Mix: StemRemixBalancer.balance_remix() — KEIN nacktes vocals + instruments
    Invariante: |LUFS(mix) − L_orig| ≤ 0.3 LU guaranteed
10. Lautheit: phase_40 (−14 LUFS EBU R128)
11. True-Peak-Begrenzung: phase_47 (−1.0 dBTP)
12. Musical Goals: alle 15 Ziele gegen effektive, physikalisch gedeckelte Zielwerte prüfen (verschärfte Studio-Schwellen)
13. Vocos-Synthese (konditionell): wenn PQS-MOS < 4.3
    → Vocos 48 kHz nativ (vocos_48khz.onnx) → BigVGAN-v2 → HiFi-GAN → PGHI-ISTFT
    Fallback-Kaskade gem. Spec 04 §4.5: Vocos primär; BigVGAN-v2 bei Vocos-OOM/Fail;
    HiFi-GAN als tertiärer Fallback; PGHI-ISTFT als deterministischer Letzter-Ausweg.
    VERBOTEN: vocos_mel_spec_24khz.onnx als primäres Modell (§4.4 SOTA-Matrix)
```

### StemRemixBalancer (Pflicht nach getrennter Stem-Verarbeitung)

```python
class StemRemixBalancer:
    """Gain-korrigierter Re-Mix nach getrennter Stem-Verarbeitung.

    Algorithmus:
        1. Vor Separation: L_orig gesamt messen
        2. Vor Separation: vocal_weight via PANNs auf Original (max. 10-s-Excerpt)
           → vocal_weight MUSS vollständig feststehen BEVOR MDX23C startet
        3. Nach Verarbeitung: LUFS pro Stem messen (L_voc', L_inst')
        4. Gain-Korrektur:
           g_voc  = 10 ** ((L_orig_voc  − L_voc')  / 20)
           g_inst = 10 ** ((L_orig_inst − L_inst') / 20)
        5. Re-Mix: mix = g_voc · vocals + g_inst · instruments
        6. Final-Check: |LUFS(mix) − L_orig| ≤ 0.3 LU

    Invarianten:
        - Vocals/Instruments-Verhältnis: ΔdB ≤ ±0.3 dB vs. Original
        - Kein Clipping im Re-Mix (np.clip nach Summation)
        - TonalCenterMetric nach Re-Mix ≥ 98 % des Pre-Remix-Werts
        - Laufzeit: ≤ 0.5 s / Minute Audio
    """
    def balance_remix(self, vocals, instruments, original, sr, vocal_weight=0.5): ...
```

**Pflicht**: Kein nacktes `vocals + instruments` in `UnifiedRestorerV3`.
**Pflicht-Test**: `tests/unit/test_stem_remix_balancer.py` (≥ 20 Tests).

---

### v10-Amendment (2026-07-12)

**MERT-v1-330M weight_norm/Parametrizations-Kompatibilität** (v10, 2026-07-12):

Der MERT-v1-330M-Checkpoint wurde mit einer älteren `transformers`-Version gespeichert, die `torch.nn.utils.weight_norm` verwendete (State-Dict-Keys: `encoder.pos_conv_embed.conv.weight_g`, `weight_v`). Neuere `torch>=2.0`-Versionen verwenden `torch.nn.utils.parametrizations` (Keys: `parametrizations.weight.original0`, `original1`).

Die weights werden von PyTorch automatisch in die neuen Parametrizations-Namen konvertiert. Das Modell ist voll funktionsfähig. Die Warnung beim Laden wird jetzt durch `ignore_mismatched_sizes=True` in `AutoModel.from_pretrained()` unterdrückt.

**Implementierung**: `plugins/mert_plugin.py` → `AutoModel.from_pretrained(..., ignore_mismatched_sizes=True)`.

## §2.2 Pipeline-Ablauf (kanonisch, Code-genau)

### §2.2.0 Sample-Rate-Vertrag (Dual-SR, [RELEASE_MUST])

- `analysis_sr = import_sr` (native): DefectScanner, RestorabilityEstimator, EraClassifier, MediumDetector, classify_clipping/analyse_clipping.
- `processing_sr = 48000`: alle Verarbeitungsphasen (01–64), PMGG, ML-Plugins, Export-Gates.
- Es müssen zwei getrennte Datenpfade geführt werden: `analysis_audio` (native SR) und `processing_audio` (48 kHz).
- Wenn die Normierung `import_sr -> 48000` fehlschlägt, MUSS die Verarbeitung fail-fast abbrechen; ein Weiterlauf der Phasen auf Nicht-48k ist unzulässig.
- Resampling darf nur `processing_audio` betreffen; `analysis_audio` bleibt unverändert in nativer SR.

### §2.2.1 Parallelisierungs-Invariante

- TIER 0 und TIER 1: IMMER sequenziell

### §2.2.2 SCHLAGER_RESTORATION_PROFILE — Definition (GermanSchlagerClassifier)

Wird aktiviert wenn `GermanSchlagerClassifier.is_schlager == True` (Gesamt-Konfidenz ≥ 0.52, gem. §2.19 Spec 03).
**Invariante**: Aktivierungsschwelle ist **0.52** — kein abweichender Wert darf im Code verwendet werden.
Enthält adjustierte GP-Priors und aktivierte Pflicht-Phasen für das Genre.

```python
SCHLAGER_RESTORATION_PROFILE = {
    # GP-Priors (überschreiben die Era-basierten Defaults aus §2.14)
    "gp_priors": {
        "noise_reduction_strength":  {"mean": 0.60, "std": 0.08},   # moderater als 1940er (0.90)
        "reverb_reduction_strength": {"mean": 0.55, "std": 0.10},   # typisch: Hallplatten-Echo
        "eq_correction_strength":    {"mean": 0.50, "std": 0.08},   # Mid-Boost bewahren
        "harmonic_preservation":     {"mean": 0.90, "std": 0.05},   # hohe Harmoniebewahrungs-Prio
        "transient_strength":        {"mean": 0.45, "std": 0.08},   # Schlagzeug-Transienten sanft
    },
    # Pflicht-Aktivierte Phasen (unabhängig von DefectScanner-Ergebnis)
    # §0a INVARIANTE: phase_42_vocal_enhancement ist im Restoration-Modus VERBOTEN
    # (Stem-Enhancement). UV3 _restoration_forbidden_stem_enhancement filtert sie
    # universal heraus — kein forced_phases-Eintrag nötig oder zulässig.
    "forced_phases": [
        "phase_19_de_esser",             # Vintage-Mikrofon → Sibilanten-Spitzen
        "phase_07_harmonic_restoration", # Harmonie-Authentizität (H2/H4-Bewahren)
        "phase_08_transient_preservation",  # Orchester-Attacken
    ],
    # Family-Scalars für SongCalibrationProfile (überschreiben material-basierte Defaults)
    "family_scalars_override": {
        "denoise":        0.65,   # sanfter als Shellac/pre-war (weniger aggressiv)
        "reverb":         0.60,   # Hallplatten sind Stilmerkmal — nicht vollständig entfernen
        "reconstruction": 0.70,
        "dynamics_eq":    0.55,
        "transient":      0.45,
        "general":        0.60,
    },
    # Vokal-Intimität besonders schützen (§2.36 / §8.3 Tiefen-Immersion)
    "vocal_intimacy_guard": True,
    # TonalCenter-Pflicht: Schlager streng tonal — kein Key-Shift toleriert
    "tonal_center_strict": True,
    # Typisches Erscheinungsbild: Analog-Tape (1950–1980)
    "expected_material_range": ["tape_standard", "tape_studio", "vinyl_standard"],
    "expected_era_range": (1950, 1985),
}
```

**Invariante**: `SCHLAGER_RESTORATION_PROFILE["family_scalars_override"]` überschreibt SongCalibrationProfile-Defaults, wird aber durch denselben `global_scalar`-Bound begrenzt (Anti-Overfitting). `SCHLAGER_RESTORATION_PROFILE` wird in `RestorationResult.metadata["schlager_profile_active"]` als `True` protokolliert.

> **Kreuzreferenz Spec 03 §2.19**: Die obige strukturierte Definition (GP-Priors, forced_phases, family_scalars_override, vocal_intimacy_guard) ist die autoritative Spec-02-Vollform. Spec 03 §2.19 ergänzt flache Zielwerte (`groove_dtw_max_ms`, `deessing_strength_cap`, `waerme_target`, `brillanz_target`) — diese sind additive Qualitätsziele, kein Ersatz für GP-Priors und forced_phases. **Implementierungen MÜSSEN beide Spec-Abschnitte konsultieren.** Konflikte: Spec 02 hat Vorrang bei strukturellen Feldern (forced_phases, family_scalars_override); Spec 03 §2.19 bei metrischen Zielwerten.

- TIER 2–4: dürfen parallelisieren; Merge via `np.mean` NUR wenn gleiche Frequenzzone
- TIER 6: IMMER sequenziell (EQ → Polish → LUFS → TruePeak → Format)

```text
Audio-Eingang (mono/stereo, beliebige SR)
    ↓
[Dual-SR-Split]
    │ analysis_audio @ import_sr (unveraendert)
    │ processing_audio @ 48000 Hz (resampled)
    │ Invariante: Kein Processing auf Nicht-48k
    ↓
[DCOffsetPreRemoval]  ← PFLICHT-VORSTUFE vor jeder FFT-Analyse (kein phase_30!)
    │ Standard (alle Materialien): scipy.signal.lfilter([1, -1], [1, -0.9999])
    │   → Hochpass-IIR 1. Ordnung, Pol bei z=0.9999, fc ≈ 0.76 Hz @ 48 kHz
    │   → Sicher für BassKraftMetric: Cutoff << 20 Hz, kein Energieverlust im Bassband
    │ Material-Sonderfall reel_tape (Lücke-H-Fix v10.0.0):
    │   Tape-Transport erzeugt DYNAMISCHEN DC-Drift (Geschwindigkeitsschwankungen
    │   → Pitch-/Amplitudenmodulation → langsame Basislinienwanderung 0.1–2 Hz).
    │   Für material_type == "reel_tape" MUSS segmentweise DC-Entfernung erfolgen:
    │   scipy.signal.lfilter([1, -1], [1, -0.9995])  — aggressiverer Pol (fc ≈ 3.8 Hz)
    │   ODER: scipy.signal.filtfilt([1, -1], [1, -0.9995]) — zero-phase (bevorzugt)
    │   Begründung: causales lfilter erzeugt Phasendrehung < 10 Hz → verfälscht Onset-
    │   Zeitstempel in WowFlutter-Erkennung; filtfilt vermeidet das.
    │   VERBOTEN bei Tape: globale Mittelwert-Subtraktion (np.mean) — erfasst keinen Drift.
    │ Invariante: np.abs(np.mean(audio)) < 1e-6 nach Entfernung
    │ Begründung: DC-Offset verfälscht STFT Bin 0+1 und damit alle
    │   Spektralanalysen (OMLSA-Profil, DefectScanner, HarmonicPreservationGuard).
    │   phase_30 bleibt für Post-Kettenausgleich erhalten, ist aber KEIN Ersatz.
    ↓
[TransientDecoupledProcessing]  ← ZWEITER Schritt (nach DC-Entfernung)
    │ separate(audio, sr) → (audio_percussive, audio_harmonic)
    │ audio_percussive → NUR phase_01 + phase_27 (kein NR, kein EQ!)
    │ audio_harmonic → volle Pipeline
    ↓
[RestorabilityEstimator]  (< 5 s, optional)
    ↓
[SongCalibrationProfile]  (§2.31a, Pflicht)
    │ Input: material_type, mode, restorability_score, input_snr_db,
    │        max_defect_severity, pipeline_confidence
    │ Output: global_scalar + family_scalars
    │ Familien: denoise | reverb | reconstruction | dynamics_eq | transient | general
    │ Invariante: bounded scalars (anti-overfitting) + deterministische Berechnung
    │
    │ [RELEASE_MUST] Bounds (Lücke-G-Fix v10.0.0):
    │   global_scalar       ∈ [0.50, 1.50]  — kein Wert < 0.50 (neutralisiert alle Phasen)
    │                                          kein Wert > 1.50 (Soft-Saturation-Guard umgangen)
    │   family_scalars[*]   ∈ [0.30, 1.80]  — Untergrenze schützt vor Komplettunterdrückung
    │                                          einer Familie; Obergrenze verhindert Überamplitude
    │   VERBOTEN: np.clip(scalar, 0.0, 2.0) — zu weite Grenzen; nur enge Clipping erlaubt
    │   Pflicht: assert 0.50 <= global_scalar <= 1.50 vor Phasen-Ausführung
    ↓
[EraClassifier]  → EraResult (decade, material_prior, confidence)
    ↓
[GermanSchlagerClassifier]  → SchlagerClassificationResult
    │ → aktiviert SCHLAGER_RESTORATION_PROFILE bei is_schlager=True
    ↓
[MediumDetectorResult]  → transfer_chain, primary_material, confidence (aus PreAnalysis-Handover)

    ⚡ PARALLEL (ThreadPoolExecutor max_workers=3):
        EraClassifier + GermanSchlagerClassifier + RestorabilityEstimator gleichzeitig
    (ONNX gibt GIL frei → echte Parallelität)

    ↓
[MusikalischerGlobalplanDienst]  ← Stufe 4 (Cross-Phase-Reasoning)
    │ erstelle_globalplan(audio, sr, use_ml_classifiers=False)  [DSP-only]
    │ 13 Ära-Profile × 7 Genre-Modifikatoren → 17 Per-Phase-Adjustments
    │ Enrichment nach Stufe 8 mit era_decade (→ RestorationConfig.global_plan)
    ↓
[SongGoalImportance]  (§2.56, Pflicht)
    │ estimate_goal_importance(genre, era, material, vocal, restorability,
    │     snr, bandwidth, dynamic_range, stereo, bpm, defects, tilt,
    │     carrier_chain, psychoacoustic, vocal/harmonic/transient)
    │ → SongGoalImportance (14 Gewichte ∈ [0.3, 2.0])
    │ 5 Stufen: Label → Audio → Psychoakustik → Vokal/Harmonik → Interactions
    │ Soft-Cap: w > 1.5 → rational compression k=3.0 (Asymptote 1.83)
    │ P1/P2-Floor ≥ 0.70; Durchreichung als goal_weights an PMGG/CIG/GPP/FC
    │ + UV3 all-phase Kopplung (§2.56a): `harmonic_adaptation_scalar` in `_profiled_phase_call`
    ↓
[DefectScanner]  → DefectAnalysisResult (46 DefectTypes)
    ↓
[CausalDefectReasoner]  → RestorationPlan (62 Kausal-Ursachen)
    ↓
[UncertaintyQuantifier]  → confidence → GP-Bounds adj.
    ↓
[GPParameterOptimizer]  → propose_pareto() → ParameterProposal (Pareto-Front)
    ↓
[HarmonicPreservationGuard]  ← NACH TDP, VOR phase_03/phase_29
    │ extract_harmonic_mask(audio_harmonic, sr) → protected_bins[t,f]
    │ G_floor = 0.85 an Harmonik-Bins, 0.10 sonst
    │
    │ [RELEASE_MUST] Mask-Gültigkeit (Fix L, v10.0.0):
    │ Die Maske ist gültig für phase_03 (Denoise) und phase_29 (Tape-Hiss).
    │ Für alle übrigen Phasen (EQ, Pitch, Stem-Sep, Dereverb etc.) darf die
    │ initiale Maske NICHT unverändert wiederverwendet werden — das harmonische
    │ Spektrum verschiebt sich nach Pitch-Korrektur, EQ und Stem-Separation.
    │ Regel:
    │   (a) phase_03: initiale Maske (berechnet aus audio_harmonic, prä-Denoise).
    │   (a.1) phase_29 (Tape-Hiss): wenn UV3 nach phase_03 einen SNR-Gewinn
    │         > 12 dB misst (snr_after_03 − snr_before_03 > 12.0 dB), MUSS die
    │         Maske VOR phase_29 neu berechnet werden (rauschverdeckte Transienten
    │         sind nach Denoise freigelegt; alte Maske schützt Rauschartefakte
    │         statt echter Harmonik). Übergabe: `recompute_harmonic_mask=True`.
    │         Bei SNR-Gewinn ≤ 12 dB: initiale Maske weiterverwendbar.
    │   (b) phase_42/43 (Vocal), phase_44–45 (Instrument): Maske NEU aus
    │       dem zum Zeitpunkt der Phase aktuellen audio berechnen
    │       (Übergabe als `recompute_harmonic_mask=True` an HPG).
    │   (c) alle übrigen Phasen: kein HPG-Eingriff (Verarbeitungs-Semantik
    │       der Phase definiert selbst ihren Amplituden-Schutz).
    │ VERBOTEN: Globale Maske ohne Ggültigkeit über alle 64 Phasen propagieren.
    ↓
[UnifiedRestorerV3._select_phases()]
    ↓
[PerceptualEmbedder]  → AudioEmbedding (256-dim L2, Pre-Fingerprint)
    ↓
[Phasen-Ausführung]  ← jede Phase gewrapped durch PerPhaseMusicalGoalsGate
    │ 5-s-Sample → measure_quick(6 Ziele) → Rollback bei Δ > REGRESSION_THRESHOLD
    │ SongCalibrationProfile skaliert phasenfamilien-basiert strength/wet-dry
    │ §2.45a Mid-Pipeline-Loudness-Drift-Guard: Nach jeder breitbandig-subtraktiven Phase
    │   Gated-RMS + Sone (§4.1b) + LUFS messen; bei Drift → Envelope-Aware Makeup-Gain
    │   Dreistufig: Per-Phase → Mid-Pipeline (kumulativ) → End-of-Pipeline (final)
    │ §2.56a Global All-Phase Harmonic Adaptation skaliert zusätzlich
    │ implizite strength/wet-dry mit bounded song-context-Scalar
    │ (psychoakustisch priorisiert: P1/P2-Stabilität, Maskierung, Transienten)
    │ MAX_RETRIES = 5; STRENGTHS = [0.65, 0.50, 0.35, 0.25, 0.15]   # kanonisch gem. §2.29 _RETRY_STRENGTHS
    ↓
[EraAuthenticPerceptualCompletion]  (wenn Quell-BW < 10 kHz)
    ↓
[IntroducedArtifactDetector]  → ML_HALLUCINATION / NMF_RESIDUAL_CLICK / etc.
    ↓
[FeedbackChain.run()]  → iteriert bis PQS-MOS konvergiert || max_iterations
    │ Normative Konvergenz-Werte (§8.5, Spec 07 — Single Source of Truth):
    │   MAX_ITERATIONS = 5
    │   CONVERGENCE_DELTA = 0.02  (|mos_n - mos_n-1| < 0.02 → Exit, kein Rollback)
    │   REGRESSION_DELTA  = 0.05  (|mos_n - mos_n-1| > 0.05 → sofortiger Rollback auf bestes Ergebnis)
    │   FORCED_EXIT bei n >= MAX_ITERATIONS (kein Hänger möglich)
    │ VERBOTEN: abweichende Werte in Implementierung (§8.5 ist normativ übergeordnet)
    ↓
[TemporalQualityCoherenceMetric]  (bei Dateien ≥ 25 s)
    ↓
[PerceptualQualityScorer]  → PQSResult (.mos, .nsim, .mcd_db, .spectral_coherence)
    ↓
[ExcellenceOptimizer]  → ExcellenceResult (GP-Params)
    ↓
[MusicalGoalsChecker]  → Dict[str, float] (alle 15 Ziele)
    ↓
[MicroDynamicsEnvelopeMorphing]  § 2.30 — Mikro-Dynamik (400 ms LUFS-Profil-Morphing)
    ↓
[EmotionalArcPreservationMetric.measure()]  (bei Dateien ≥ 30 s — Messung post-MDEM)
    ↓
[correct_emotional_arc()]  (optional: nur wenn Bogen nicht erhalten — Makro-Korrektur 5 s)
    ← §2.30 Post-Smoothing-Quiet-Zone-Clamp Pflicht (siehe §2.30b)
    ↓
[HolisticPerceptualGate]  → HPI-Score (inkl. artifact_freedom §2.49)
    ↓
[GPParameterOptimizer.update()]  ← persistiert Lernerfolg
    ↓
Audio-Ausgang + RestorationResult
```

---

### v10-Amendment (2026-07-12)

**Hip-Hop-Veto bei deutschsprachigem Material** (v10, 2026-07-12):

Deutscher Schlager aus den 1970ern, der durch eine analoge Kette (Vinyl→Cassette→mp3) degradiert wurde, teilt DSP-Merkmale mit Hip-Hop: einfache Harmonik (HSI 0.55–0.85), moderate Onset-Dichte (2.5–6.0/s), komprimierte Dynamik (DR < 15 dB), spektraler Schwerpunkt (1500–3200 Hz). Diese Überlappung führte zu Fehlklassifikationen (»Hip-Hop« statt »Deutscher Schlager«).

**Fix**: Analog zu den existierenden Latin/Reggae-Vetoes in `GermanSchlagerClassifier.classify()`: Wenn `alt_genre == "Hip-Hop"` UND `lang_de_score >= 0.30` (deutschsprachiges Material) UND `n_active >= 1` (Schlager-Evidenz vorhanden) → Override zu Schlager.

**Implementierung**: `backend/core/genre_classifier.py` → nach Latin/Reggae-Veto:

```python
if not is_schlager and n_active >= 1 and alt_genre == "Hip-Hop" and lang_de_score >= 0.30:
    is_schlager = True
    confidence = float(max(confidence, self.SCHLAGER_CONFIDENCE_THRESHOLD))
```

**Einschränkung**: Greift nur bei `lang_de_score >= 0.30`. Bei extrem degradiertem Material ohne erkennbare Sprachsignatur bleibt die Fehlklassifikation möglich — dann ist manuelles Genre-Override nötig.

## §2.30b [RELEASE_MUST] Post-Smoothing-Quiet-Zone-Clamp-Invariante (v10.0.0)

**Normative Reihenfolge in UV3 (kanonisch):**

1. `MicroDynamicsEnvelopeMorphing.morph()` — 400 ms LUFS-Morphing
2. `measure_emotional_arc()` — Messung post-MDEM (≥ 30 s)
3. `correct_emotional_arc()` — Makro-Korrektur (5 s, **nur wenn Bogen nicht erhalten**)

**Systemisches Anti-Pattern (VERBOTEN in allen Gain-Morphing-Funktionen):**

Jede Funktion, die:

1. Pre-Smoothing Guard stille/Fadeout-Frames auf 0 setzt
2. Savitzky-Golay oder Boxcar-Glättung anwendet
3. `np.interp` auf Sample-Ebene interpoliert

**MUSS** nach Schritt 2 UND nach Schritt 3 den Guard erneut anwenden — sonst
verschleppt der Smoother positiven Gain aus Musiksegmenten in still/denoised
Fadeout-Bereiche → Pegelexplosion.

**Quiet-Zone-Grenze (normativ, modul-adaptiv):**

| Modul | Schwellwert | Logik | Rationale |
| --- | --- | --- | --- |
| `morph()` (MDEM, 400 ms) | **−36 dBFS** | Einzel-Bedingung | Feine Zeitauflösung — Frame-Level reicht |
| `correct_arc()` (EmotionalArc, 5 s) | **−42 dBFS + 6 dB-Diff** | Zwei-Bedingungen | 5-s-Segmente enthalten Mix aus Musik und Stille — einfaches -36 dBFS wäre zu aggressiv; zweite Bedingung (`rms_orig > rms_rest + 6 dB`) erkennt das „Denoised-Fadeout-Muster" zuverlässig |

**VERBOTEN:** Einzel-Bedingung (`< −36 dBFS`) in `correct_arc()` ohne zweite Schutz-Bedingung — das würde normale leise Musikpassagen (Pianissimo, Fade-in) fälschlicherweise sperren.

**Kanonisches Muster (Pflicht für MDEM und correct_emotional_arc):**

```python
# 1. Pre-Smoothing Guard
for i in range(len(gain_db)):
    if gain_db[i] > 0.0 and rms_rest[i] < QUIET_THRESH:
        gain_db[i] = 0.0

# 2. Smoother (SG / Boxcar)
gain_db = savgol_filter(gain_db, ...)

# 3. POST-Smoothing Guard (PFLICHT — Smoother kann Guard aus Schritt 1 aufheben)
for i in range(len(gain_db)):
    if gain_db[i] > 0.0 and rms_rest[i] < QUIET_THRESH:
        gain_db[i] = 0.0

# 4. np.interp auf Sample-Ebene
gain_db_interp = np.interp(sample_idx, centres, gain_db)

# 5. Per-Sample Guard (PFLICHT — interp erzeugt Übergangs-Boost Musik→Stille)
quiet_mask = (frame_rms_rest_per_sample < QUIET_THRESH_LINEAR)
gain_db_interp[quiet_mask & (gain_db_interp > 0.0)] = 0.0
```

**Betroffene Module (Pflicht-Implementierung):**

- `backend/core/micro_dynamics_envelope_morphing.py` — `MicroDynamicsEnvelopeMorphing.morph()`
- `backend/core/emotional_arc_preservation.py` — `EmotionalArcPreservationCorrector.correct_arc()`

**Testpflicht:** Regression-Test mit lauter Intro (0–30 s) + denoised Fadeout (30–42 s):

- Kein positiver Gain im Fadeout (`rms_after ≤ rms_before × 1.06`, d. h. < 0.5 dB)
- Datei: `tests/unit/test_emotional_arc_preservation.py::TestCorrectArc::test_36_no_pegelexplosion_in_denoised_fadeout`

---

## §2.56a [RELEASE_MUST] Global All-Phase Harmonic Adaptation (v10.0.0)

`SongGoalImportance` muss nicht nur in Gates, sondern auch in der laufenden
Phasensteuerung wirksam sein, damit alle 64 Phasen harmonisch zusammenarbeiten.

**Normativer Ort:** `UnifiedRestorerV3._profiled_phase_call`

**Pflicht-Algorithmus:**

```python
harmonic_adaptation_scalar = _compute_harmonic_adaptation_scalar(
    phase_id,
    phase_family,
    goal_weights,
    restorability_score,
    material_key,
)
# harmonic_adaptation_scalar in [0.72, 1.18], mit Boundary-Pullback
```

**Verhalten:**

1. Der Skalar wirkt multiplikativ auf implizite `strength` und `wet/dry`.
2. Explizite `strength` (PMGG/Team-Policy/Hard-Cap) hat Vorrang.
3. Die Anpassung ist advisory-only und darf harte Sicherheits-Gates nicht aufweichen.
4. Fehler im Adaptionspfad führen zu neutralem Verhalten (`scalar=1.0`) und `logger.debug`.

**Invarianten:**

- Keine Per-Phase-Divergenz: ein zentraler Pfad für alle 64 Phasen.
- Keine Randwert-Klebung: Pullback von exakten Bound-Rändern.
- Keine Pipeline-Blockade durch Adaptionsfehler.

**Verboten:**

- Statische, song-unabhängige Universal-Strengths ohne Kontextkopplung.
- Gleichzeitige Überschreibung expliziter PMGG-Strength durch §2.56a.

**Rationale:** Senkt unnötige Rollbacks durch entkoppelte Phasenparameter,
bei unverändert strikten End-Gates (§2.44, §2.48, §2.49).

---

## §2.56b [RELEASE_MUST] Per-Phase Strength Oracle und 15-Goal-Teamsteuerung (v10.0.0)

Die zentrale All-Phase-Kopplung (§2.56a) bleibt Pflicht, reicht aber allein nicht aus.
Alle Phasen mit wirksamen Steuerparametern (`strength`, `wet`, `threshold_db`, `ratio`,
`drive`, `mix`, `gain_db`, `q`, `temperature`, `repair_strength`, `context_mix`,
`band_strength`, `stereo_amount`) MUESSEN vor ihrer Ausfuehrung ein
**phasenspezifisches Strength Oracle** konsultieren.

**Normativer Ort:** `UnifiedRestorerV3._profiled_phase_call` direkt nach SongCal-/PMGG-/Team-Policy-
Vorbelegung, aber vor dem eigentlichen Phasenaufruf.

**Pflichtziel:**

1. Nicht ein Einzelziel maximieren, sondern den gewichtet anwendbaren 15er-Zielvektor.
2. Pro Phase genau den lokalen Eingriff waehlen, der den groessten Team-Beitrag zum
     Schliessen der aktuellen Goal-Gaps liefert.
3. P1/P2-Stabilitaet, Vocal-Supremacy, Carrier-Recovery und AFG/PMGG/CIG-Grenzen
     haben absoluten Vorrang vor P3-P5-Gewinn.

**Kanonischer Ablauf:**

```python
goal_snapshot_pre = _fast_goal_snapshot(audio_pre, sr, context)
goal_gaps = compute_goal_gaps(goal_snapshot_pre, effective_goal_targets, applicable_goals)

oracle_profile = resolve_phase_strength_oracle(
        phase_id=phase_id,
        phase_family=phase_family,
        audio=audio_pre,
        sample_rate=sr,
        goal_gaps=goal_gaps,
        goal_weights=goal_weights,
        defect_scores=defect_scores,
        defect_locations=defect_locations,
        locality_factor=phase_locality_factor,
        restorability_score=restorability_score,
        material_key=material_key,
        transfer_chain=transfer_chain,
        chain_confidence=material_confidence,
        vfa_result=vfa_result,
        song_calibration_profile=song_calibration_profile,
        gp_proposal=gp_parameter_proposal,
        phase_history=metadata.get("phase_deltas", {}),
        wall_time_budget_s=wall_time_remaining_s,
)

phase_kwargs = merge_oracle_profile_into_phase_kwargs(base_kwargs, oracle_profile)
audio_post = phase.execute(audio_pre, sr, **phase_kwargs)
```

**Minimaler Oracle-Output (bindend):**

```python
{
        "control_strength": float,       # 0.0-1.0
        "wet_mix": float,                # 0.0-1.0
        "threshold_db": float | None,
        "ratio": float | None,
        "drive": float | None,
        "eq_gain_db": float | None,
        "band_profile": dict[str, float] | None,
        "team_contribution": dict[str, float],      # erwarteter Beitrag je Goal
        "dominant_goal_guard": bool,                # nie single-goal-dominant
                "hard_caps": {
                        "max_strength": float,
                        "max_wet_mix": float,
                        "chain_factor": float,                  # direkte Tontraegerketten-Konditionierung
                        "chain_depth": float,
                        "chain_confidence": float,
                },
}
```

**Transfer-Chain-Weltklasse-Invariante [RELEASE_MUST]:**

- Das Oracle MUSS den gesamten `transfer_chain`-Kontext direkt in der Staerkeberechnung nutzen;
    eine rein indirekte Beruecksichtigung ueber Material-Defaults ist nicht ausreichend. `[SRC:S03,S04]`
- `chain_factor` MUSS deterministisch aus `material_key`, `transfer_chain` und
    `chain_confidence` berechnet werden und sowohl `driver` als auch `hard_caps` beeinflussen. `[SRC:S03,S04]`
- Bei komplexen Mehrfachketten (z. B. `vinyl -> cassette -> mp3_low`) MUSS `chain_factor`
    strenger sein als bei Einzeltraegern gleicher Defektlast. `[SRC:S03,S04]`

**Invarianten:**

- Das Oracle ist **advisory fuer die lokale Parametrisierung**, nicht fuer Export-Gates.
- PMGG-/Policy-/Safety-Caps bleiben fuehrend.
- Explizit uebergebene Hard-Parameter aus Recovery-/Rollback-Pfaden duerfen nur bounds-geklammert,
    nicht ignoriert werden.
- Phasen ohne kontinuierliche Staerke (`binary gate`, `passthrough`, `format only`) muessen
    mindestens `control_strength in {0.0, 1.0}` und `team_contribution` melden.
- Wenn das Oracle ausfaellt: neutraler Fallback auf bestehende implizite Staerke + `logger.debug`. `[SRC:S03]`

**Verboten:**

- Song-unabhaengige Feststaerken als Primaersteuerung.
- Single-goal-Maximierung ohne Weighted-Gap-Check ueber alle 15 Ziele.
- Pro Phase voneinander isolierte Optimierer ohne Zugriff auf `goal_gaps`, `goal_weights`
    und `effective_goal_targets`.

**Kreuzreferenzen:** Spec 01 §1.2c, Spec 06 §7.1d, Spec 09 §09.3c.

---

## Kanonische RestorationResult-Definition

```python
@dataclass
class RestorationResult:
    # ── Pflichtfelder ────────────────────────────────────────
    audio:                np.ndarray
    config:               "RestorationConfig"
    material_type:        "MaterialType"
    defect_scores:        dict["DefectType", float]
    phases_executed:      list[str]
    phases_skipped:       list[str]
    total_time_seconds:   float
    rt_factor:            float
    quality_estimate:     float   # = 0.40·(1−defect_severity) + 0.60·(pqs_mos−1)/4
    warnings:             list[str]
    metadata:             dict[str, Any]
    # ── Optionale Felder ─────────────────────────────────────
    pqs_result:           Optional[Any] = None    # .mos, .nsim, .mcd_db, .spectral_coherence
    musical_goals:        Optional[dict[str, float]] = None   # 15 Ziele → Score
    excellence:           Optional[Any] = None
    temporal_coherence:   Optional[Any] = None    # MOS-Spanne ≤ 0.30
    emotional_arc:        Optional[Any] = None    # Arousal/Valence Pearson
    restorability:        Optional[Any] = None    # 0–100
    confidence:           float = 1.0
    genealogy:            Optional[Any] = None
    harmonic_fingerprint: Optional[Any] = None    # 256-dim L2 Post-Fingerprint
    phase_gate_log:       Optional[list[str]] = None
    adaptive_thresholds:  dict[str, float] = field(default_factory=dict)
    physical_ceiling:     dict[str, float] = field(default_factory=dict)
    goal_applicability:   dict[str, bool] = field(default_factory=dict)
    goal_priority_log:    list[str] = field(default_factory=list)
    preview_mos:          Optional[float] = None
    era_decade:           Optional[int] = None
    # ── §2.38 KMV-Felder ─────────────────────────────────────
    deferred_phases:      list[str] = field(default_factory=list)   # Phasen die Stufe 2 benötigen
    refinement_complete:  bool = False                               # True nach ML-Veredelung
    stufe2_quality_estimate: Optional[float] = None                  # quality nach vollständigem ML-Pass
```

### §2.2.3 [RELEASE_MUST] Experience-Telemetrie-Vertrag (v10.0.0)

Für die Produktions-Closed-Loop-Steuerung müssen folgende Felder in `RestorationResult.metadata`
normativ vorhanden sein (fehlertolerant, aber schema-stabil):

- `song_calibration.cluster_key: str`
- `song_calibration.cluster_policy: dict`
- `joy_runtime_index: {joy_index: float, fatigue_index: float, components: dict}`
  - `components` MUSS enthalten: `frisson_index` (0..1, Gänsehaut-Propensity, Blood & Zatorre 2001),
    alle Sub-Werte NaN/Inf-frei, clipped [0, 1]
  - Mode-Policy: Restoration = advisory-only (kein Audio-Impact); Studio 2026 = konservative bounded Mikro-Kopplung erlaubt
- `auto_improvement_recommendations: {count: int, recommendations: list[dict]}`

**Invarianten:**

1. Alle numerischen Werte sind finite und auf [0,1] bzw. plausible Bereiche begrenzt.
2. Fehlende Upstream-Teile führen zu leeren Strukturen (`{}`, `[]`), nicht zu Schema-Bruch.
3. Die Telemetrie ist advisory-only: sie darf Pipeline/Export nicht blockieren.

**Rationale:** Ohne explizite Freude-/Ermüdungs- und Root-Cause-Telemetrie bleibt die
geschlossene Nachbesserung intransparent und kann in UI/Orchestrator nicht stabil genutzt werden.

### §2.38a ML-Guard-Fallback-Metadaten (PFLICHT)

Wenn eine heavy ML-Stufe wegen RAM-Headroom-Guard nicht gestartet wird, MUESSEN strukturierte Metadaten geschrieben werden.

```python
metadata.setdefault("ml_guard_events", []).append(
    {
        "phase_id": "phase_20_reverb_reduction",
        "model": "SGMSE+",
        "reason": "insufficient_physical_ram_headroom",
        "required_gb": 9.0,
        "available_gb": 6.8,
        "channels": 2,
        "duration_s": 245.3,
        "fallback": "wpe_dsp",
    }
)
```

**Invarianten:**

- Kein Rollback auf Original-Audio als Guard-Reaktion.
- Phase bleibt ausgefuehrt (DSP/Fallback-Pfad) und wird in `phases_executed` gefuehrt.
- Betroffene Phase MUSS in `deferred_phases` eingetragen werden (Stufe-2-KMV-Nachzug).
- `metadata["ml_guard_events"]` ist JSON-serialisierbar und NaN/Inf-frei.

**quality_estimate-Formel (normativ):**

```python
quality_estimate = 0.40 * (1 - defect_severity) + 0.60 * (pqs_mos - 1) / 4
# VERBOTEN: * 1.15 Bonus; quality_estimate aus defect_severity allein
# Clip: max(0.0, min(1.0, quality_estimate))
```

**Serialisierungsregeln:**

- `audio`-Feld wird NICHT in JSON serialisiert
- NaN/Inf-Werte → `null` (via `clean_nans()`)
- `genealogy` → separates `<sha256_prefix>_genealogy.json`
- Neue Felder: immer mit Default `null`

---

## [RELEASE_MUST] §2.29 PerPhaseMusicalGoalsGate — Adaptive Regression-Schwellen

**[RELEASE_MUST] PMGG darf notwendige Phasen nicht stumm verwerfen.**
CausalDefectReasoner-bestimmte Phasen müssen über eine Recovery-Kaskade geführt werden
(Stärke-Reduktion, Team-Policy, alternative sichere Variante). Blindes Skippen ist verboten.
Wenn nach Recovery kein besseres sicheres Ergebnis gefunden wird, ist der Status als
`degraded` zu kennzeichnen (kein stiller Erfolg).

VERBOTEN: `return audio, scores_before, "rollback", 0.0` — Rückgabe von
unverändertem Original-Audio gleichbedeutend mit Phasen-Skip.

> **§2.54 ist übergeordnet**: Die untenstehenden Schwellwerte sind **Notbremsen-Baselines**,
> nicht die Routine-Steuerung. Die Routine-Steuerung ist der iterative
> Messen→Handeln→Validieren-Zyklus (§2.54), gesteuert durch PhaseConductor (§2.52)
> und SongCalibration (§2.47). Die hier genannten REGRESSION_THRESHOLD-Werte sind
> letzte Sicherheitsnetze für katastrophale Fälle — sie dürfen restorative Phasen
> nicht blockieren, wenn das Material den Eingriff braucht und der Defekt messbar
> reduziert wird (auch wenn ein Proxy-Score dabei sinkt).

```python
# Notbremsen-Baselines (restorability-adaptiv):
# Diese Werte definieren die MAXIMALE Proxy-Regression, ab der der Guard
# die Phase iterativ mit reduzierter Stärke wiederholt. Sie sind NICHT die
# Pipeline-Steuerung — PhaseConductor.recommend() und SongCalibration
# steuern die initiale Stärke materialadaptiv BEVOR der Guard prüft.
REGRESSION_THRESHOLD_GOOD: float = 0.020   # restorability ≥ 70
REGRESSION_THRESHOLD_FAIR: float = 0.035   # restorability 40–69
REGRESSION_THRESHOLD_POOR: float = 0.040   # restorability < 40 (reduced from 0.055 v10.0.0 — prevent best_effort cascades)
SAMPLE_DURATION_S: float = 5.0

# Priority-Aware Retry-Budget (v10.0.0 + §2.31b v10.0.0):
_RETRY_STRENGTHS: list[float] = [0.65, 0.50, 0.35, 0.25, 0.15]   # 5 Stufen, Floor 0.15 (Last-Resort)
# §2.31b: initial_strength < 0.90 (SongCal vorreduziert) → Ankerpunkte [0.80, 0.65, 0.50, 0.35, 0.20]
_PRIORITY_MAX_RETRIES: dict[int, int] = {1: 4, 2: 4, 3: 2, 4: 1, 5: 1}
_PRIORITY_THRESHOLD_FACTOR: dict[int, float] = {1: 1.0, 2: 1.0, 3: 1.5, 4: 2.0, 5: 2.0}
# P1/P2: volle Kaskade (4 Retries + Emergency)
# Catastrophic-Threshold: max(0.08, 4.0 × adaptive_threshold) statt fest 0.20 (§2.31b)
# P3: max 2 Retries, 1.5× Regression-Toleranz
#   §2.31b: restorability_tier="good" → 3 Retries; tier="poor" → 1 Retry
# P4/P5: Recovery-Lite (1 konservativer Retry, 2× Toleranz), kein Emergency
#   Psychoakustische Begründung: P4 (Transparenz, Wärme, Bass) und P5 (Brillanz, Räumlichkeit)
#   sind für den Hörer wahrnehmbar — ein einziger Recovery-Versuch verhindert unnötigen
#   Komfort-Verlust ohne die Primum-non-nocere-Garantie (P1/P2) zu schwächen.
# Stagnation-Abbruch: max(0.002, threshold × 0.15) (§2.31b proportional)

# Schnell-Ziele (≤ 200 ms Gesamtcheck):
FAST_GOALS_SUBSET = [
    "natuerlichkeit", "authentizitaet", "tonal_center",
    "timbre_authentizitaet", "artikulation", "emotionalitaet",
    "micro_dynamics", "groove", "transparenz", "waerme",
    "bass_kraft", "separation_fidelity", "brillanz", "spatial_depth",
]
# Phasen-adaptive Sample-Dauer (§9.7.3):
PHASE_SAMPLE_DURATIONS = {
    "phase_30": 1.5,  "phase_05": 1.5,  "phase_02": 2.0,
    "phase_15": 1.5,  "phase_11": 1.5,  "phase_18": 2.0,
}

# Datenfluss-Invariante: restorability_score MUSS aus RestorabilityEstimator stammen:
re_result = RestorabilityEstimator().estimate(audio, sr, defect_analysis)
gate = PerPhaseMusicalGoalsGate()
for phase in selected_phases:
    audio, scores, _ = gate.wrap_phase(
        phase, audio, sr, scores_before,
        restorability_score=re_result.restorability_score,
        applicable_goals=goal_filter.applicable,
    )
# action ∈ {"passed", "retry1"..., "best_effort", "best_effort_rN", "passed_p4p5_tolerated"}
# Invariante: Jede best_effort-Action ist als transparenter Recovery/Degradation-Pfad
# zu behandeln (nie als stiller Success).
```

### §9.7.7 [RELEASE_MUST] PMGG Stable-Metric-Invariante (v10.0.0)

Metriken mit ML-zustandsabhängigem Gewicht **DÜRFEN NICHT** in `_PRECISE_METRICS` für PMGG-Delta-Checks stehen.

**Root-Cause `NatuerlichkeitMetric`**: CREPE-Load-State verändert die internen Gewichte zwischen
`scores_before` (CREPE nicht geladen → `w_crepe=0.0`) und `scores_after` (CREPE geladen → `w_crepe=0.18`).
Das erzeugt Pseudo-Regression Δ ≈ 0.15–0.28 auf unverändertem Audio, triggert die vollständige
P1-Retry-Kaskade (4 Retries + 2 Emergency) und erzwingt Phase_03 best-effort bei strength=0.056.

**Auswirkung auf Gänsehaut-Erlebnis**: Phase_03 bei 5.6 % Wet-Mix erreicht Noise Floor −55 dBFS
statt −72 dBFS. Der Air-Layer (8–20 kHz) und der Vokal-Intimität-Layer (4–8 kHz) bleiben unter
dem Rauschteppich verdeckt → kein „Ohr-in-die-Musik-Legen", keine Tiefen-Immersion.

**Invarianten**:

- `NatuerlichkeitMetric` läuft ausschließlich in `MusicalGoalsChecker` (Export-Gate), nie im PMGG-Delta.
- Neue Metriken vor `_PRECISE_METRICS`-Aufnahme: Eigenrauschen ≤ 0.02 auf identischen Audio-Paaren Pflicht.
- `_PRECISE_OVERRIDE_WARN_MS = 200.0` (angehoben von 120.0).

### §2.29c [RELEASE_MUST] PMGG Restorative-Phase-Baseline-Capping (v10.0.0)

**Problem**: In restorativen Phasen (Denoise, Dereverb, Declip, etc.) misst `scores_before` auf
defekt-belastetem Audio. Bestimmte Defekte **inflationieren** Metriken künstlich:

- Breitbandrauschen hebt `transparenz` (Spectral Crest) und `brillanz` (HF-Energie)
- Hall-Nachhall hebt `waerme` (LF-Energie-Ratio) und verdeckt `authentizitaet`-Verluste
- Dropout-Lücken verfälschen `groove` (Autokorrelation) und `micro_dynamics` (RMS-Envelope)

Nach der Restaurierung sinken die Werte auf **physikalisch korrekte Levels** → PMGG meldet
Falsch-Regression → Retry-Kaskade → best-effort bei minimaler Wet-Strength → Defekte bleiben.

**Lösung**: `_RESTORATIVE_PHASES` + `_CANONICAL_THRESHOLDS` + `effective_scores_before`:

```python
_RESTORATIVE_PHASES: frozenset[str] = frozenset({
    "phase_01",  # Click removal
    "phase_02",  # Hum removal (Kammfilter)
    "phase_03",  # Broadband denoise (OMLSA + ResembleEnhance)
    "phase_05",  # Rumble filter (subtractive LF cleanup)
    "phase_09",  # Crackle removal (BANQUET ONNX — blind neural denoising; targetiert impulsive Vinyl-Crackle)
    "phase_18",  # Noise gate (Silero VAD)
    "phase_20",  # Reverb reduction (SGMSE+)
    "phase_23",  # Spectral inpainting / gap-fill (AudioSR)
    "phase_24",  # Dropout repair (AudioSR)
    "phase_27",  # Click/pop removal
    "phase_29",  # Tape hiss reduction (DeepFilterNet v3 II)
    "phase_30",  # DC offset / near-DC drift removal
    "phase_49",  # Advanced dereverb
    "phase_50",  # STFT spectral inpainting (bin interpolation)
    "phase_56",  # Spectral band gap repair (HEAD_WEAR)
    "phase_57_print_through_reduction",  # Print-through reduction (bidirectional LMS)
})

_CANONICAL_THRESHOLDS: dict[str, float] = {
    "natuerlichkeit": 0.90, "authentizitaet": 0.88, "tonal_center": 0.95,
    "timbre_authentizitaet": 0.87, "artikulation": 0.88, "transient_energie": 0.80,
    "emotionalitaet": 0.84,
    "micro_dynamics": 0.88, "groove": 0.83, "transparenz": 0.82,
    "waerme": 0.75, "bass_kraft": 0.78, "separation_fidelity": 0.80,
    "brillanz": 0.78, "spatial_depth": 0.70,
}
```

**Algorithmus** in `_run_with_retry()`:

```python
# §2.29c Restorative-Phase-Baseline-Capping
if phase_id in _RESTORATIVE_PHASES:
    effective_scores_before = {}
    for goal, measured in scores_before.items():
        canonical = _CANONICAL_THRESHOLDS.get(goal, 0.80)
        effective_scores_before[goal] = min(measured, canonical + 0.05)
else:
    effective_scores_before = scores_before
# Delta-Check: scores_after[g] - effective_scores_before[g]
```

**Invarianten**:

- `_CANONICAL_THRESHOLDS` = Restoration-Mode-Schwellwerte + 0.05 Headroom
- Capping greift nur in `_RESTORATIVE_PHASES` — Enhancement-Phasen nutzen echte `scores_before`
- Defekt-inflationierte Baselines über Canonical+5% werden gedeckelt → kein false Regression-Trigger
- Deterministisch: kein Zufall, keine ML-Abhängigkeit

**Aktualisierte `PHASE_GOAL_EXCLUSIONS`** (v10.0.0 — kanonische Quelle: `backend/core/per_phase_musical_goals_gate.py`):

### §2.29e [RELEASE_MUST] PMGG Team-Koordination via `prior_phase_context` (v10.0.0, erweitert v10.0.0)

**Problem**: Sequenziell korrekte Reparaturen können durch PMGG-Retry indirekt gegeneinander arbeiten,
wenn Folgephasen die Vorphasen-Interventionen als Regression interpretieren.

**Lösung**: PMGG liest `prior_phase_context` und leitet eine team-policy ab.

```python
def _resolve_team_context_policy(phase_id: str, phase_kwargs: dict[str, Any] | None) -> dict[str, Any]:
        # advisory policy for PMGG only
        return {
                "goal_exclusions": set(),
                "threshold_multiplier": 1.0,
                "strength_cap": 1.0,
                "reason": "",
        }
```

**Normative Regel (alle Module/Phasen via Ontologie)**:

- PMGG muss aus `prior_phase_context.last_phase_type` und `get_phase_type(current_phase)`
  eine zentrale Übergangs-Policy ableiten.
- Übergangs-Policy ist für **alle aktiven Phasen** anzuwenden (nicht nur Einzel-Hotfixes):
  - optionale Goal-Exclusions,
  - moderates `threshold_multiplier` (capped),
  - konservatives `strength_cap`.
- Policy ist **advisory-only** (Retry/Strength), Export-Gates bleiben unverändert.

**Normative Spezialregel (`phase_50_spectral_repair`)**:

- Wenn `prior_phase_context` eines von
    `harmonic_restoration_applied`, `frequency_restoration_applied`,
    `spectral_super_resolution_applied` enthält, gilt
    `reason="phase50_after_hf_restoration"`.
- PMGG erweitert Goal-Exclusions um:
    `{"brillanz", "transparenz", "timbre_authentizitaet"}`.
- PMGG darf den adaptiven Threshold moderat skalieren (`×1.15`, capped).
- PMGG deckelt Initial-Strength konservativ (`≤ 0.80`).

**Emergency-Pfad-Invariante**:

```python
def _allow_emergency_retries(..., team_policy):
        # catastrophic retries are skipped when the regression is a known
        # proxy-artifact of intentional prior HF restoration.
```

- Catastrophic/Emergency-Retries müssen team-policy-bewusst entscheiden.
- Für `phase_50` mit `reason="phase50_after_hf_restoration"` sind Emergency-Retries
    zu unterdrücken (kein sinnloses Low-Strength-Looping auf Proxy-Artefakte).

**Team-Telemetrie (v10.0.0, §2.53 RELEASE_MUST)**:

- `PhaseGateLogEntry.metadata` erhält folgende Felder wenn Team-Policy aktiv:
  `team_policy_reason`, `team_excluded_goals`, `team_threshold_mult`, `team_strength_cap`.
- UV3 extrahiert nach Pipeline alle Entries mit gesetztem `team_policy_reason` →
  `self._team_coordination_events`.
- `RestorationResult.metadata["team_coordination"]` enthält:
  `event_count`, `events` (Liste mit phase_id/action/reason/excluded_goals/threshold_mult/strength_cap),
  `phase_type_summary` (Typ-Häufigkeiten aus `_phase_team_context`).
- `bridge.get_experience_insights()` gibt `team_coordination` als Frontend-sicheres Dict zurück.
- Fehlendes team_coordination darf den Export nie blockieren (non-blocking §2.53).

**CONFLICT_REGISTRY (v10.0.0)**:

Explizite Paare in `backend/core/phase_ontology.py` — Phase B darf Arbeit von Phase A NICHT rückgängig machen:

```python
CONFLICT_REGISTRY: dict[str, frozenset[str]] = {
    "phase_09": frozenset({"phase_50"}),             # Crackle → Spectral-Repair
    "phase_07": frozenset({"phase_50", "phase_03", "phase_29"}),  # Harmonik
    "phase_06": frozenset({"phase_28", "phase_29", "phase_50"}),  # BW-Extension
    "phase_23": frozenset({"phase_03", "phase_29"}), # Spektral-Inpainting
    "phase_55": frozenset({"phase_03", "phase_29"}), # Diffusions-Inpainting
    "phase_24": frozenset({"phase_50"}),             # Dropout-Repair
    "phase_01": frozenset({"phase_50", "phase_27"}), # Click-Removal
    "phase_56": frozenset({"phase_29", "phase_03"}), # Bandlücken-Repair
}
```

UV3 `_profiled_phase_call` injiziert `conflict_with_prior_phases: list[str]` in Phase-kwargs
wenn ein Treffer im CONFLICT_REGISTRY vorliegt (`get_conflict_phases(prior_id)` enthält `current_phase_id`).

**Invariante**: Team-Policy beeinflusst nur PMGG-Retry/Strength und liefert `conflict_with_prior_phases`
als Hint an Phasen. Export-Gates (`HolisticPerceptualGate`, `ArtifactFreedomGate`) bleiben unberührt.

```python
PHASE_GOAL_EXCLUSIONS: dict[str, set[str]] = {
    # Broadband denoise: CREPE-Load-State + transient-shape mismatch +
    # K-S NOT invariant for shaped NR §9.7.11 ext (non-uniform NR reshapes
    # chroma-bin balance → key-label flip) + MFCC-Pearson/Centroid-CV
    # disturbed by spectral-envelope change after NR.
    # §2.31b material-adaptive: cd_digital/dat → reduce to {"natuerlichkeit", "artikulation"}.
    "phase_03": {"natuerlichkeit", "artikulation", "authentizitaet", "tonal_center", "timbre_authentizitaet"},
    # DeepFilterNet tape-hiss: same root-causes as phase_03.
    "phase_29": {"artikulation", "authentizitaet", "natuerlichkeit", "tonal_center", "timbre_authentizitaet"},
    # Comb-filter hum removal: G1/G2/G3 notches cause false regressions:
    #   - groove: §9.7.10 rms_env variance-normalisation artefact (50 Hz modulation)
    #   - timbre_authentizitaet: MFCC-Pearson/centroid disturbed by LF notches → false P2
    "phase_02": {"bass_kraft", "authentizitaet", "natuerlichkeit", "transparenz",
                 "groove", "timbre_authentizitaet"},
    # EQ / tonal shaping: broadband frequency shifts invalidate timbre comparisons.
    "phase_04": {"transparenz", "brillanz", "waerme", "authentizitaet", "natuerlichkeit", "timbre_authentizitaet"},
    # TDP/HPSS: Transient-Shaping.
    "phase_08": {"micro_dynamics", "artikulation"},
    # Wow/Flutter: K-S volatile after pitch-/speed-correction + Centroid-CV disturbed.
    "phase_12": {"tonal_center", "timbre_authentizitaet"},
    # Noise gate: VAD mask applies binary gains → micro-dynamics artifacts.
    # artikulation: Note-Attack unterdrückt durch VAD-Gating → false P2 (bestätigt, VERBOTEN.md).
    "phase_18": {"micro_dynamics", "authentizitaet", "emotionalitaet", "groove", "artikulation"},
    # SGMSE+ reverb reduction: SGMSE+ spectral deconvolution disturbs
    # CREPE pitch confidence → natuerlichkeit false P1.
    "phase_20": {"authentizitaet", "natuerlichkeit"},
    # AudioSR spectral inpainting: synthesised gap-fill has no valid reference —
    # same mechanism as phase_24 (Dropout Repair).
    # timbre_authentizitaet: MFCC-Pearson/Centroid-CV disturbed by synthesis.
    "phase_23": {"natuerlichkeit", "brillanz", "authentizitaet", "artikulation", "timbre_authentizitaet"},
    # Dropout repair: synthesised gap-fill; same root-causes as phase_23.
    "phase_24": {"natuerlichkeit", "brillanz", "authentizitaet", "artikulation", "timbre_authentizitaet"},
    # Dereverb: authentizitaet regression from RT60 removal.
    "phase_49": {"authentizitaet"},
    # Vocal processing: NMF separation shifts NatuerlichkeitMetric sub-weights.
    "phase_19": {"natuerlichkeit", "timbre_authentizitaet", "micro_dynamics"},
    # Dithering / noise-shaping: micro-dynamics by design.
    "phase_17": {"micro_dynamics", "natuerlichkeit"},
    # Diffusion inpainting: synthesised content; artikulation reference absent.
    "phase_55": {"artikulation", "micro_dynamics"},
    # Bandwidth extension (AudioSR): adds HF content → brillanz intentionally rises.
    "phase_06": {"brillanz"}, "phase_07": {"brillanz"},
    # Transient / time-domain: micro-dynamics re-shaping alters onset metric.
    "phase_26": {"micro_dynamics", "artikulation"}, "phase_36": {"micro_dynamics", "artikulation"},
    # Passthrough / analysis-only phases: no musical scoring required.
    "phase_28": set(), "phase_05": set(), "phase_30": set(),
    # Click removal (phase_01, phase_27): impulse transients + spectral interpolation.
    #   - artikulation: clicks appear as transients → removal reduces onset-count correlation.
    #   - natuerlichkeit: spectral interpolation at click locations creates MFCC-smoothness
    #     discontinuities (transition from reconstructed frames to undamaged context). CREPE-
    #     based NatuerlichkeitMetric flags these as unnatural → false P1 regression (0.267
    #     confirmed in real-run, PMGG dithered to strength=0.17). Same mechanism as phase_02.
    "phase_01": {"artikulation", "natuerlichkeit"},  # click impulses + interpolation → false P2/P1
    "phase_27": {"artikulation", "natuerlichkeit"},  # click/pop removal — identical to phase_01
    # Crackle removal via BANQUET ONNX (phase_09): phase_09_crackle_removal.py verwendet BANQUET
    # als ML-Backbone für impulsive Vinyl-Crackle-Entfernung (full-band neural spectral modification).
    #   - natuerlichkeit: MFCC-smoothness proxy disturbed by full-band NR (same as phase_03/29).
    #   - groove: crackle events appear as periodic impulsive onsets. GrooveMetric onset-based
    #     DTW proxy registers the change in LF onset density as rhythmic disruption. Real-run
    #     confirmed: regression=0.291 (P1), stagnation across all retries, strength=0.15.
    #     Same mechanism as phase_02 groove exclusion.
    #   - authentizitaet: crackle fills log-spectrum valleys (roughness low before BANQUET);
    #     after processing valleys reappear → roughness rises → false P1. Identical to phase_03.
    #   - timbre_authentizitaet: MFCC-Pearson/centroid-CV disturbed (same as phase_29).
    "phase_09": {"natuerlichkeit", "groove", "authentizitaet", "timbre_authentizitaet"},
    # LyricsGuidedEnhancement (phase_58): Fricative-Ramp-Gain (4–8 kHz) verändert Spektralenveloppe
    # wie shaped NR → K-S-Key-Label-Flip möglich (tonal_center).
    # Vowel-LPC-Shelving und Plosive-Burst ändern MFCC-Pearson/Centroid-CV (timbre_authentizitaet).
    # HINWEIS: Key muss "phase_58_lyrics_guided_enhancement" lauten — NICHT "phase_57"
    # (würde via startswith-Präfix-Matching phase_57_print_through_reduction treffen).
    "phase_58_lyrics_guided_enhancement": {"tonal_center", "timbre_authentizitaet", "artikulation", "emotionalitaet"},
}
```

**Änderungen v10.0.0 → v10.0.0**:

- phase_03/29: brillanz/transparenz entfernt (§9.7.12/13 SNR-robust); tonal_center + timbre_authentizitaet eingefügt (§9.7.11 ext: K-S NOT invariant to shaped NR; Centroid-CV-Disturbance).
- phase_12: **NEU** — K-S volatile nach Pitch-/Speed-Korrektur + Centroid-CV.
- phase_02: tonal_center entfernt (K-S stabil bei Kammfilter).
- phase_18: brillanz/transparenz/tonal_center entfernt; groove hinzugefügt.
- phase_20: brillanz/waerme/transparenz entfernt (§9.7.12/13/14 reverb-invariant).
- phase_23/24: timbre_authentizitaet hinzugefügt (MFCC-Pearson/Centroid-CV gestört durch Synthese).
- phase_49: brillanz/waerme/transparenz entfernt (§9.7.12/13/14 reverb-invariant).
- phase_08: aus Passthrough-Gruppe in eigenen Eintrag verschoben.

### §9.7.8 [RELEASE_MUST] Precise-Metric Audio-Cap (v10.0.0)

`_apply_precise_metric_overrides` kappt Audio auf **max. 2.5 s** vor dem Metric-Loop.

- Alle 7 verbleibenden präzisen Metriken (Brillanz, Wärme, TonalCenter, MicroDynamics,
  Artikulation, SeparationFidelity, Transparenz) sind spektral-stationär über kurze Fenster.
- Ohne Cap: `ArticulationMetric` (Short-Frame 5 ms Hop) und `SeparationFidelityMetric`
  (NMF) benötigen > 2 s/Call auf 60-s-Material → kumulative PMGG-Latenz 4+ s pro Phase.
- Mit 2.5 s Cap: alle 7 Metriken < 200 ms gesamt.

### §9.7.9 [RELEASE_MUST] Material-adaptive PHASE_GOAL_EXCLUSIONS (v10.0.0)

Für hochwertige digitale Quellen (`cd_digital`, `dat`) entfallen Rausch-bedingte Ausschlüsse
bei `phase_03` (Breitband-Denoise) und `phase_29` (DeepFilterNet Tape-Hiss):

**Root-cause**: Die Ausschlüsse für `brillanz`, `authentizitaet`, `transparenz` und `tonal_center`
entstehen durch HF-Rauschminderung auf analogen Medien — Tape-Hiss und Vinyl-Hiss verschieben
spektrale Flatness, ZCR und Rolloff. Digitale Quellen haben kein Breitbandrauschen → diese
Falsch-Regressions-Ursachen treten nicht auf.

**Stabile Ausschlüsse (bleiben für alle Materialien)**:

- `natuerlichkeit`: CREPE-Load-State ändert interne Gewichte material-unabhängig
- `artikulation`: Transient-shape mismatch bei leichter Filterung bleibt relevant
- `tonal_center`: K-S ist bei shaped/HF-selektiver NR **nicht** invariant (§9.7.11 ext v10.0.0) — nicht-uniformes NR verändert Chroma-Bin-Balance → Key-Label-Flip
- `timbre_authentizitaet`: MFCC-Pearson/Centroid-CV gestört durch Spektral-Hüllkurvenänderung nach NR

**Implementierung** in `wrap_phase()` nach dem `PHASE_GOAL_EXCLUSIONS`-Loop:

```python
# §2.31b Material-adaptive exclusion relaxation (v10.0.0, akt. v10.0.0)
if _excluded_goals:
    _mat_str = ... # aus phase_kwargs["material_type"] oder ["material"]
    if _mat_str in {"cd_digital", "dat"} and (
        phase_id.startswith("phase_03") or phase_id.startswith("phase_29")
    ):
        _excluded_goals &= {"natuerlichkeit", "artikulation"}
```

**Qualitätswirkung**: Für digitale Quellen werden `authentizitaet`, `tonal_center` und
`timbre_authentizitaet` jetzt im PMGG-Delta aktiv gemessen → Regressions-Schutz greift für
digitale Pfade wo bisher Falsch-Ausschlüsse standen. brillanz/transparenz/waerme sind seit
§9.7.12/13/14 bei **allen** Materialtypen SNR-robust und nicht mehr ausgeschlossen.

### §9.7.10 [RELEASE_MUST] Groove-Proxy LF-Robustheit (v10.0.0)

**Problem**: `_measure_quick` berechnet die Groove-Metrik via Autokorrelation einer 10 ms-Hop
RMS-Energiehüllkurve `rms_env`. Die Normierungsbasis `autocorr[0]` ist gleich der Gesamtvarianz
von `rms_env`. 50/100 Hz-Hum erzeugt innerhalb jedes 10 ms-Frames (≈ 0.5–1 Hum-Perioden/Frame)
Frame-zu-Frame-Schwankungen, die `autocorr[0]` erhöhen, ohne die 500 ms-Rhythmusperiodizität
zu verändern. Ergebnis: `autocorr[lag_05]` / `autocorr[0]` hängt von der Hum-Stärke ab →
false groove-Delta bei `phase_02_hum_removal`, obwohl der echte Rhythmus unverändert bleibt.
Stagnation Δ=0.000000 entsteht, weil das Artefakt rein normierungsbedingt ist und sich mit der
Filter-Stärke nicht ändert.

**Fix**: 5-Frame Moving-Average (= 50 ms) auf `rms_env` **vor** `np.correlate()`:

```python
# §9.7.10 LF-Robustheit: 5-Frame-MA filtert 50/100 Hz-Hum-Modulation aus rms_env.
# Hum-Periode 10–20 ms → stark gedämpft; Groove-Periode 120–500 ms → nahezu unverändert.
_sw = min(5, len(rms_env) // 4)
if _sw >= 2:
    rms_env = np.convolve(rms_env, np.ones(_sw) / float(_sw), mode="valid")
autocorr = np.correlate(rms_env, rms_env, mode="full")
autocorr = autocorr[len(rms_env) - 1:]
autocorr /= autocorr[0] + 1e-12
```

**Invarianten**:

- `_sw = min(5, len(rms_env) // 4)` → keine Überglättung bei kurzen Clips (< 0.2 s, ≈ 12 Frames → `_sw=3`)
- `_sw < 2` → kein Smoothing (Edge Case: < 8 Frames = < 80 ms Audio)
- Groove-Score bleibt deterministisch (kein stochastischer Anteil)
- `autocorr[0]` nach MA repräsentiert ausschließlich rhythmische Energievarianz

**Tests**: `TestGrooveProxyLFRobustness` (4 Tests, test_74–test_77) in
`tests/unit/test_per_phase_musical_goals_gate.py`.

---

### §9.7.11 [RELEASE_MUST] Krumhansl-Schmuckler tonal_center Proxy (v10.0.0)

**Problem**: Der bisherige `tonal_center`-Proxy maß **Chroma-Konzentrations-Entropie**
(`1 − entropy/log(12)`). Das ist SNR-abhängig: Rauschen/Nachhall/EQ-Filter verteilen
Energie gleichmäßig über alle 12 Chroma-Bins → hohe Konzentration `scores_before`;
nach Denoise/Dereverb sichtbare Spektralpeaks → niedrigere Konzentration `scores_after`
→ false P2-Regression auf **jedem rauschreduzierenden Phase bei beliebiger Stärke**.
Δ≈0 Stagnation bestätigt globale Stärke-Unabhängigkeit = strukturelle Proxy-Invalidität.
Beobachtete Katastrophen in Produktionslogs (2026-03-30):

| Phase | Regression | Δ-Stagnation | Root-Cause |
| --- | --- | --- | --- |
| phase_49_advanced_dereverb | 0.5312 | 0.000010 | Nachhall füllt Chroma-Bins diffus |
| phase_08_transient_preservation | 0.5612 | 0.000025 | HPSS verschiebt harmonisch/perkussiv-Balance |
| phase_04_eq_correction | 0.0753 | 0.000600 | EQ-Notch/Shelf verschiebt Chroma-Bin-Amplituden |
| phase_18_noise_gate | 0.1721 (groove) | 0.002226 | VAD-Gating → Chroma-Sparsität |

**Lösung**: Krumhansl-Schmuckler (1990) Key Detection — SNR-invariant, weil gleichmäßiges
Rauschen alle 24 KS-Scores gleichmäßig hebt → argmax unverändert.

**Algorithmus**:

1. Chroma-Vektor aus FFT-Magnitude (Hann-Fenster, n=4096) über Frequenz > 27.5 Hz
2. Korrelere gegen 24 KS-Dur/Moll-Profile (alle 12 Root-Transpositionen)
3. `key_before = argmax` im Referenzsignal, `key_after = argmax` im verarbeiteten Signal
4. Zirkuläre Semitondistanz `d = min(|k_a − k_b| mod 12, 12 − ...) ∈ [0, 6]`
5. Moduswechsel (Dur ↔ Moll) = +1 Semiton-Äquivalent, max 6
6. `tonal_center = 1 − d/6` → 0 = Tritonus/maximale Verschiebung, 1 = gleiche Tonart

```python
# §9.7.11 Krumhansl-Schmuckler key detection (SNR-invariant)
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
# Both normalized to zero-mean unit-variance for Pearson equivalence via np.dot

def _ks_key(signal_mono, n_fft=4096, sr=48000) -> int:
    spec = np.abs(np.fft.rfft(signal_mono * np.hanning(len(signal_mono)), n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0/sr)
    chroma = np.zeros(12); bins = np.where((freqs > 27.5) & (freqs < 4186))[0]
    np.add.at(chroma, np.round(12*np.log2(freqs[bins]/440+1e-12)).astype(int)%12, spec[bins])
    chroma -= chroma.mean(); chroma /= chroma.std() + 1e-12
    best_r, best_k = -np.inf, 0
    for root in range(12):
        r_maj = np.dot(chroma, np.roll(_ks_maj_n, root))   # _ks_maj_n = normalised
        r_min = np.dot(chroma, np.roll(_ks_min_n, root))
        if r_maj > best_r: best_r, best_k = r_maj, root
        if r_min > best_r: best_r, best_k = r_min, root+12
    return best_k

# Delta score (reference available):
d = min((k_proc % 12 - k_ref % 12) % 12, 12 - ...)   # circular
mode_penalty = 0 if same_mode else 1
tonal_center = 1.0 - min(6, d + mode_penalty) / 6.0
```

**Invarianten**:

- Fallback bei Stille / sehr kurzem Signal → `0.5`
- KS-Profile: Krumhansl & Schmuckler 1990 Table 1 (kanonisch, unveränderlich)
- Pearson-Äquivalenz: Profile werden zu `zero-mean, unit-variance` normiert → `np.dot = n × pearson`
- Kein `assert sr == 48000` nötig (sr-agnostisch durch `rfftfreq(n, d=1/sr)`)
- Deterministisch: kein Zufall in der Berechnung

**PHASE_GOAL_EXCLUSIONS nach §9.7.11** (tonal_center in folgenden Phasen **nicht** mehr ausgeschlossen):
`phase_02`, `phase_04`, `phase_08`, `phase_18`, `phase_49`

These exclusions were removed because the old entropy proxy was SNR-dependent. K-S is key-label-based
and does not react to spectral energy redistribution that doesn't cause a genuine pitch transposition.

**§9.7.11 Extension (v10.0.0/96)**: K-S ist bei **shaped/HF-selektiver NR** (phase_03 OMLSA+ResembleEnhance,
phase_29 DeepFilterNet) **nicht** invariant. Nicht-uniformes NR verändert Chroma-Bin-Balance selektiv
→ Key-Label-Flip möglich. Daher bleiben `tonal_center`-Ausschlüsse für phase_03 und phase_29 bestehen.
Phase_12 (Wow/Flutter) erhält ebenfalls tonal_center-Ausschluss: Pitch-/Speed-Korrektur verschiebt
fundamentale Frequenzen → K-S volatile.

**Tests**: `TestKrumhanslSchmucklerTonalCenter` (24 Tests, test_78–test_101) in
`tests/unit/test_per_phase_musical_goals_gate.py`. Enthält auch §9.7.12/13/14 Proxy-Tests
(brillanz HF Crest, transparenz Multi-Band Crest, waerme Sub-Band-Ratio).

---

## §2.37 [RELEASE_MUST] Frontend-Backend-PreAnalysis-Handover-Architektur (v10.0.0)

### Kernprinzip

Pre-Analyseergebnisse werden **einmalig** bei Import berechnet (`run_pre_analysis()`) und als **direkte Objektreferenz** (nicht Cache-Keys) weitergereicht. Cache-basierte Rekonstruktion in asynchronen Batch-Threads erzeugt Racebedingungen.

### Datenfluss: Import → Analysis → Queue → Batch → Denker

```
UI: _load_file(path)
  │
  ├─→ [A] Hard Cache Clear: _bridge_clear_cache_for_path(old_path)
  │       └─ Alte Caches (defect, era/genre, medium, restorability) aktiv löschen
  │
  ├─→ [B] _pre_analysis_bg() → run_pre_analysis(audio_native, sr_native, ...)
  │       └─ MediumDetector.detect() aufgerufen GENAU 1x (native SR)
  │       └─ Alle 5 Analysen parallel: Medium, Era, Genre, Defect, Restorability
  │       └─ Ergebnisse in Bridge-Cache speichern (LRU, content-addressed)
  │
  ├─→ [C] Frontend speichert: _latest_pre_analysis_result = PreAnalysisResult(...)
  │       └─ Complete object reference (nicht nur Cache-Keys)
  │
  └─→ [D] Mode-Click (Restoration / Studio 2026)
          │
          ├─→ _add_to_queue_with_mode()
          │   └─ queue_item.settings["pre_analysis_result"] = _latest_pre_analysis_result
          │   └─ falls vorhanden: queue_item.settings["cached_defect_result"] = pre_analysis_result.defects
          │
          └─→ BatchProcessingThread.run()
              │
              ├─→ [E] Check queue_item.settings.get("pre_analysis_result"):
              │       IF present: pre_result = settings["pre_analysis_result"]
              │       ELSE: Rekonstruiere von Bridge-Caches (Fallback)
              │       Zusätzlich: konkret verwendetes Defect-Result immer als
              │       `cached_defect_result` an denke()/UV3 weiterreichen
              │
              └─→ [F] AurikDenker.denke(pre_analysis_result=pre_result, ...)
                  │
                  └─→ UV3.restore(cached_medium_kwarg=..., ...)
                      └─ MediumDetector.detect() NICHT aufgerufen (bereits 1x in pre_analysis)
```

### Invarianten (RELEASE_MUST)

| Invariante | Ort | Status |
| --- | --- | --- |
| Hard Cache Clear bei neuem Import | `Aurik10/ui/modern_window.py` line ~11920 | ✅ |
| PreAnalysisResult Storage | `Aurik10/ui/modern_window.py` line ~12691 | ✅ |
| Queue-Handover | `Aurik10/ui/modern_window.py` line ~13939 | ✅ |
| Batch-Prioritization | `Aurik10/ui/modern_window.py` line ~2117 | ✅ |
| Defect-Handover-Absicherung | `Aurik10/ui/modern_window.py` line ~2107 | ✅ |
| Test: Exactly 1 detect() call | `tests/unit/test_pre_analysis_handover_no_double_detect.py` | ✅ |

**Kritische Invariante**: `MediumDetector.detect()` wird **GENAU 1x** aufgerufen (von `run_pre_analysis()`), nie 2x oder 3x.

**Zusätzliche Invariante**: Das für den Run tatsächlich verwendete `DefectAnalysisResult` MUSS `AurikDenker.denke()` und UV3 immer als `cached_defect_result` erreichen. Ein unvollständiges `PreAnalysisResult` darf keinen zweiten Defect-Scan erzwingen, solange bereits ein konkretes Defect-Result im Queue-Kontext vorliegt.

### Fallback-Hierarchie

Falls `queue_item.settings["pre_analysis_result"]` ist `None` (shouldn't happen):

1. Bridge-Cache Rekonstruktion bei einzelnen Caches
2. Wenn Cache incomplete: UV3 führt fehlende Analysen eigenständig aus
3. Monitoring: `metadata["pre_analysis_handover"]` dokumentiert Fallback-Nutzung

### Rationale: Warum nicht Bridge-Cache?

**Problem**: Zeitfenster zwischen Frontend und Batch erlaubt Racebedingungen

```python
# ❌ RACE CONDITION
# Thread 1 (Frontend):
bridge.cache_medium_result(path, medium)
bridge.cache_defect_result(path, defect)

# Fenster (ms) — Batch-Thread könnte stale Cache lesen
# Old cache von vorrigem File könnte persistent sein

# Thread 2 (Batch):
medium = bridge.get_cached_medium_result(path)  # Original oder degradiert?
defect = bridge.get_cached_defect_result(path)  # Aus alter Datei gelesen?
```

**Lösung**: Direct Object Reference (Frozen nach Frontend-Capture, keine Parallelität)

```python
# ✓ DETERMINISTIC
pre_result = queue_item.settings["pre_analysis_result"]  # Complete object
# Immutable nach Frontend-Capture → keine Racebedingungen
```

---

## §2.38 Kontinuierliche ML-Veredelung (KMV) — [RELEASE_MUST]

> **Kernprinzip**: Der PerformanceGuard verwirft überschrittene Phasen nie endgültig — er _deferriert_ sie.
> RT-Limit-Überschreitung führt zu DSP-Fallback für Sofort-Export **plus** automatischer Hintergrund-Veredelung.
>
> **Quality-First Ergänzung (v10.0.0)**: In den nutzerseitigen Standardpfaden
> (GUI/CLI/Batch) wird `no_rt_limit=True` gesetzt. Dadurch darf der Hauptlauf
> Qualität nicht zugunsten von RT reduzieren; `deferred_phases` entstehen dort
> primär durch Ressourcen-/Stabilitäts-Fallbacks (OOM, Headroom, Inference-Timeout),
> nicht durch RT-Budget-Cuts.

### Zweistufiger Export-Ablauf

```text
Stufe 1 (Sofort-Export, Quality-first im Standardpfad)
    │  Standard: no_rt_limit=True (GUI/CLI/Batch)
    │  Optionaler RT-limitierter Pfad: Deferral bei should_skip_phase
    │  Phasen die RT-Limit überschreiten: DSP-Fallback + in deferred_phases eingetragen
    │  Pipeline finalisiert; Qualitäts-Gate bestanden?
    │   └─ Nein → Stufe 1 abgebrochen (Fail-Reason in metadata)
    │   └─ Ja  → Atomischer Export (immediately listenable)
    │              Wenn len(deferred_phases) > 0:
    ↓
Stufe 2 (Hintergrund-ML-Veredelung, LIMIT_BACKGROUND = ∞)
    │  MLRefinementThread startet automatisch nach Stufe-1-Export
    │  Gecachte Analyse-Ergebnisse aus Stufe 1 (kein Neustart von DefectScanner,
    │    EraClassifier, MediumClassifier, GPParameterOptimizer)
    │  Vollständige UV3-Pipeline ohne RT-Limit (no_rt_limit=True)
    │  QThread.LowPriority + os.nice(10) auf Linux
    │  isInterruptionRequested() zwischen jeder Phase prüfen
    │  Qualitätsinvariante: quality(v2) ≥ quality(v1) → sonst alten Export behalten
    └→ Atomischer Export-Overwrite: result_v2.tmp → os.replace(output_path)
       signal: refinement_complete(output_path, final_RestorationResult)
```

### RAM-Guard (Stufe 2 Startbedingung)

```python
import psutil
avail_gb = psutil.virtual_memory().available / 1024**3
if avail_gb < 4.0:
    logger.warning("KMV Stufe 2 übersprungen: nur %.1f GB RAM verfügbar (< 4 GB)", avail_gb)
    return  # Stufe-1-Export bleibt permanent
```

### DeferredRefinementJob (Pflicht-Dataclass)

```python
@dataclass
class DeferredRefinementJob:
    """Queued job for background ML refinement (§2.38)."""
    output_path:          str                       # Pfad der Stufe-1-Exportdatei
    audio_original:       np.ndarray                # Original-Audio (unkomprimiert, pre-pipeline)
    sr:                   int                       # Sample-Rate (48000)
    mode:                 str                       # "restoration" | "studio_2026"
    deferred_phase_ids:   list[str]                 # Phasen die in Stufe 1 deferriert wurden
    cached_defect_result: Any                       # DefectAnalysisResult aus Stufe 1
    cached_era_result:    Any                       # EraResult aus Stufe 1
    cached_medium_result: Any                       # ClassificationResult aus Stufe 1
    stufe1_quality:       float                     # quality_estimate Stufe 1 (Mindest-Benchmark)
    created_at:           float = field(default_factory=time.time)
```

### MLRefinementThread — Signal-Kontrakt

```python
class MLRefinementThread(QThread):
    refinement_started    = pyqtSignal(str, int)    # output_path, n_deferred_phases
    refinement_phase_done = pyqtSignal(str, float)  # phase_id, quality_improvement_delta
    refinement_progress   = pyqtSignal(int, str)    # pct 0–100, phase_name
    refinement_complete   = pyqtSignal(str, object) # output_path, final_RestorationResult
    refinement_cancelled  = pyqtSignal(str)         # output_path → Stufe-1-Export bleibt
```

### Invarianten

- `LIMIT_BACKGROUND = float("inf")` ist ausschließlich für `MLRefinementThread` — niemals für BatchProcessingThread
- Atomisches Schreiben: `output_path.tmp` → `os.replace(output_path)` nach vollständigem Pass
- Kein Downgrade: `if stufe2_result.quality_estimate < job.stufe1_quality: skip_overwrite()`
- Single active refinement: Pro Prozess höchstens ein aktiver `MLRefinementThread`
- Escape-Abbruch: `requestInterruption()` → Stufe-1-Export bleibt unverändert erhalten
- `DeferredRefinementJob.audio_original` registriert in `ml_memory_budget` (Budget-Guard); freigegeben unmittelbar nach Stufe-2-Export oder Abbruch

## §2.38b [RELEASE_MUST] Deferred-Phases vs. Phase-Skip — Formale Abgrenzung

| Konzept | Definition | Erlaubt | Mechanismus |
| --- | --- | --- | --- |
| **Phase-Skip** | Phase wird **permanent** nicht ausgeführt — Original-Audio wird unverändert weitergereicht | **VERBOTEN** für P1/P2-Phasen (§2.29) | — |
| **Phase-Defer** | Phase wird jetzt mit DSP-Fallback ausgeführt, volle ML-Qualität in Stufe 2 nachgeholt | **ERLAUBT** | `deferred_phases.append(phase_id)` + KMV Stufe 2 |

**Invariante**: RT-Limit-Überschreitung → **immer Defer, nie Skip**. Der PerformanceGuard darf `should_skip_phase()` im Quality-First-Pfad (`no_rt_limit=True`) nie zurückgeben, wenn das die einzige Restaurierungsmethode für eine P1/P2-Ursache ist.

```python
# RICHTIG: Phase deferrieren (Stufe 2 holt nach)
result.deferred_phases.append(phase_id)
phase_result = _run_phase_dsp_fallback(phase_id, audio, kwargs)  # temporärer DSP-Fallback

# VERBOTEN: Phase-Skip auf Original-Audio
# return audio, scores_before, "rollback", 0.0  ← nicht erlaubt gemäß §2.29
```

**Deferred-Phases-Priorisierung in Stufe 2**:

1. Phasen mit P1/P2-Zielbezug (höchste Priorität)
2. Phasen mit P3-Zielbezug
3. Alle übrigen (P4/P5 Recovery-Lite)

Innerhalb jeder Prioritätsgruppe entscheidet die Reihenfolge im ursprünglichen Pipeline-Plan. Bei erneutem Ressourcenmangel: Phase für nächsten Anlauf vormerken, nicht dauerhaft ausführen.

**Endlosschleifen-Prävention**: Nach 3 fehlgeschlagenen Deferred-Aufholversuchen wird die Phase als `"non_recoverable"` markiert. `RestorationResult.metadata["deferred_failed"]` wird befüllt. Weitere automatische Versuche unterbleiben bis zu einem manuellen Neustart.

## §2.39 OOM-Recovery-Checkpoint-System — [RELEASE_MUST]

**Kernprinzip**: `systemd-oomd`-Kill oder `MemoryError` führen nie zu Totalverlust. Pipeline-Zwischenstand wird atomar auf Disk persistiert und beim nächsten Start automatisch zur Wiederaufnahme angeboten.

### Checkpoint-Lifecycle

| Schritt | Komponente | Aktion |
| --- | --- | --- |
| 1 | `_execute_pipeline()` MemoryError-Handler | `save_checkpoint()` → `sessions/<stem>_oom_checkpoint.json` + `_oom_audio.wav` |
| 2 | `ModernMainWindow.__init__` (1,5 s QTimer) | `find_pending_checkpoints()` → Dialog "Restaurierung fortsetzen?" |
| 3 | Nutzer bestätigt | `_resume_from_checkpoint()` → Original laden → normale Restaurierung |
| 4 | Erfolgreicher Abschluss | `delete_checkpoint()` → Cleanup |

### Modul: `backend/core/recovery_checkpoint.py`

```python
@dataclass
class RecoveryCheckpoint:
    input_path: str
    output_path: str
    phases_executed: list[str]
    phases_remaining: list[str]
    mode: str                              # "restoration" | "studio_2026"
    material_type: str                     # MaterialType.value
    era_decade: int | None
    defect_scores: dict[str, float]        # {defect_type: severity}
    defect_scores_full: dict[str, dict]    # Full DefectScore with locations
    restorability_score: float | None
    spectral_fingerprint: dict[str, float]
    quality_estimate_at_failure: float
    musical_goals_at_failure: dict[str, float]
    audio_wav_path: str                    # FLOAT WAV (verlustfrei)
    sample_rate: int
    original_input_path: str
    timestamp: float
    aurik_version: str = "9.10.57"
    failure_phase: str = ""
    failure_reason: str = "MemoryError"
```

### Pfad-Durchleitung

```text
BatchProcessingThread
  → denke(input_path=, output_path=)
    → restauriere()
      → _orchestriere()
        → RestaurierDenker.restauriere()
          → UV3 restore(input_path=, output_path=)
            → self._recovery_ctx
              → _execute_pipeline MemoryError-Handler
                → save_checkpoint()
```

## §2.40 Vollpipeline-Determinismus (PFLICHT)

Die komplette UV3-Kette muss fuer identische Eingaben deterministisch reproduzierbar sein.

```python
# Determinismus-Vertrag (normativ)
assert max_abs_err <= 1e-6
assert rms_err <= 1e-7
assert result_a.phases_executed == result_b.phases_executed
```

Pflichtregeln:

- Alle Seeds zentral setzen und im Result-Metadata dokumentieren.
- Keine unseeded Zufallsfunktionen in Produktionspfaden.
- Vergleichslaeufe mit identischen Prozessparametern (Threads, Mode, Config).

## §2.41 Structured Fail-Reason Taxonomie (PFLICHT)

`RestorationResult.metadata["fail_reasons"]` ist eine Liste strukturierter Eintraege.

Pflichtfelder pro Eintrag:

- `phase_id`
- `reason_code` (z. B. `ml_guard_low_ram`, `goal_regression_p1`, `quality_gate_fail`)
- `severity` (`info|warning|error`)
- `action` (`fallback|retry|best_effort|blocked`)
- `details` (JSON-serialisierbar, NaN/Inf-frei)

**Invariante:** Kein freier String-only Fehlerpfad ohne reason_code in Kernmodulen.

## §2.42 [RELEASE_MUST] Pipeline-Stabilitäts-Kontrakt (v10.0.0)

Zusammenfassung aller Stabilitäts-Invarianten. Jede Verletzung einer dieser Regeln ist ein Release-Blocker.

| ID | Mechanismus | Spezifikation | Schutz gegen |
| --- | --- | --- | --- |
| S-01 | Per-Phase-Inference-Timeout | §3.9.1 spec 08 | BLAS-Deadlock, korruptes Modell |
| S-02 | SIGTERM-Handler + Emergency-Checkpoint | §3.9.2 spec 08 | Graceful OS-Shutdown ohne Datenverlust |
| S-03 | Phase-Output-Guard (`@phase_output_guard`) | §3.9.3 spec 08 | NaN/Inf-Propagation aus ML-Ausgaben |
| S-04 | ThreadPoolExecutor-Lifecycle (shutdown) | §3.9.4 spec 08 | Zombie-Threads, Ressourcen-Leaks |
| S-05 | ml_memory_budget Startup-Reconciliation | §3.9.5 spec 08 | Stale-Allokation nach SIGKILL |
| S-06 | Structured Exception Logging | §3.9.6 spec 08 | Stille Fehler, leer `fail_reasons` |
| S-07 | Audio-Buffer-RAM-Guard | §3.9.7 spec 08 | OOM durch sehr große Audio-Dateien |
| S-08 | Lock-Acquisition-Order (ARM→PLM→MLBudget) | §3.9.8 spec 08 | Deadlock zwischen ARM und PLM |
| S-09 | MLRefinementThread Buffer-Release in finally | §3.9.9 spec 08 | RAM-Leak bei KMV-Abbruch |
| S-10 | watchdog + requestInterruption → terminate() | §11.4 spec 08 | Freeze > 90–300 min (Desktop-Watchdog, §K 64×RT) |
| S-11 | OOM-Recovery-Checkpoint (MemoryError-Pfad) | §2.39 | Python MemoryError → kein Totalverlust |
| S-12 | §2.38 KMV Stufe 2 mit 4 GB RAM-Guard | §2.38 | OOM bei Hintergrund-ML-Veredelung |
| S-13 | §2.38a ML-Headroom-Guard vor ML-Load | §2.38a | OOM während Modell-Laden |
| S-14 | Hybrid-Release-Mode (primary/fallback/blocked) | §13 spec 08 | Crash durch quarantänisierte Modelle |
| S-15 | Atomic File Writes (.tmp → os.replace) | §3.1 spec 08 | Korrupte Ausgabedatei bei Abbruch |

### Stabilitäts-Priorisierung

- **S-01 bis S-09**: Neue Invarianten aus Tiefenanalyse v10.0.0 — RELEASE_MUST.
- **S-10 bis S-15**: Bestehende Invarianten — bereits implementiert, hier zur Referenz.

### Für jedes neue Kernmodul / jede neue Phase gilt zusätzlich (§9.1 Checkliste):

- `try`/`except` mit §2.41-konformem `fail_reasons`-Eintrag (S-06).
- `@phase_output_guard` oder äquivalente manuelle Absicherung (S-03).
- `ml_memory_budget.try_allocate()` vor ML-Load mit `release()` in Fehler-Pfad (S-13).
- Kein `ThreadPoolExecutor` ohne Shutdown in Cleanup (S-04).
- `_check_audio_buffer_size()` bei direktem `soundfile.read()` (S-07).
- **[RELEASE_MUST] Längen-Invariante**: `len(phase_output) == len(phase_input)` — Phasen dürfen die Signallänge nicht verändern. `_execute_pipeline()` korrigiert akkumulierten Längendrift am Ausgang (Trim bei Überlänge, Zero-Pad bei Unterlänge). Dies betrifft insbesondere PGHI-basierte Phasen mit `padded=False` (letztes unvollständiges Fenster wird weggelassen) — Abhilfe: `n_samples=len(audio_in)` immer an `pghi_reconstruct_from_stft()` übergeben.

### [RELEASE_MUST] Invarianten

- Checkpoint-Audio als `FLOAT` WAV — verlustfrei, kein Encoding-Verlust
- Ablauf: 7 Tage (`_MAX_CHECKPOINT_AGE_S`) — danach automatische Bereinigung
- Thread-safe: Alle Writes über `.tmp` + `os.replace` (POSIX-atomar)
- Datenschutz: Lyrics-Text NICHT im Checkpoint (§2.36 Pflicht)
- Wiederaufnahme nutzt das **Original-Audio** (nicht das Checkpoint-Audio) für volle Qualität
- Checkpoint-Audio dient als Fallback wenn Original fehlt
- **VERBOTEN**: Checkpoint-Audio als Primärquelle für Re-Restaurierung (Doppelverarbeitung degradiert Qualität)

---

## §2.44 [RELEASE_MUST] Holistic Perceptual Gate (v10.0.0)

Letztes Gate vor Export. Misst **Gesamt-Hörverbesserung** statt nur Einzel-Goals.

### Referenz-Paradoxon (Restoration)

Das Ziel ist Nähe zum **unbekannten Studio-Original**, aber wir besitzen nur den **degradierten Input**. Je erfolgreicher die Restaurierung, desto unähnlicher wird der Output dem degradierten Input. Deshalb misst `timbral_fidelity` nicht bloße Ähnlichkeit zum Input, sondern **strukturelle akustische Kohärenz**:

- **Spectral-Envelope-Kontinuität**: Keine unnatürlichen Lücken oder Spitzen im Frequenzspektrum
- **Crest-Factor-Konsistenz**: Dynamik-Verhältnis bleibt physikalisch plausibel
- **MFCC-Stabilität**: Klangfarben-Koeffizienten zeigen keine abrupten Sprünge

**Referenz-Anker-Strategie** (Restorability-abhängig):

- **Restorability > 70** (leichte Degradation): Input ist gute Annäherung ans Original → `timbral_fidelity` gegen Input
- **Restorability 50–70** (mittlere Degradation): Gewichtete Mischung aus Input-Referenz (60 %) und MERT-Referenz-Vektor aus GP-Memory (40 %)
- **Restorability ≤ 50** (schwere Degradation): Input zu weit vom Original entfernt → MERT-Referenz-Vektor aus GP-Memory (genre × material × ära) als primärer Anker (70 %), Input nur noch für musikalische Identität (30 %)

### MERT-Referenz-Embedding-Aufbau (v10.0.0)

Die GP-Memory-Referenz-Vektoren werden **automatisch** aus dem Verarbeitungsverlauf aufgebaut — kein manuelles Kuratieren nötig:

**Bootstrap (Cold-Start)**:

- Beim ersten Start: 12 Genre-Prototypen aus vortrainierten MERT-Embeddings (im Modell-Bundle enthalten, ~2 MB)
- Abdeckung: je 1 Prototyp pro Genre-Cluster (Schlager, Oper, Klassik, Jazz, Rock, Pop, Blues, Soul, Electronic, Latin, Folk, Metal)
- Ära-Differenzierung: 3 Ära-Bins (pre-1960, 1960–1990, post-1990) × 12 Genres = 36 Basis-Vektoren

**Inkrementeller Aufbau**:

- Nach jeder **erfolgreichen** Restaurierung (HPI > 0.5 UND artifact_freedom ≥ 0.95 UND alle P1/P2-Goals bestanden):
  - MERT-Embedding des Outputs wird in GP-Memory unter `genre × material × ära_bin` gespeichert
  - Exponential Moving Average (α = 0.15) mit bestehendem Referenz-Vektor → konvergiert ohne Ausreißer
- **Qualitäts-Gate für Referenz-Updates**: Nur Outputs mit HPI > 0.5 fließen ein — verhindert, dass mittelmäßige Restaurierungen die Referenz verschlechtern
- **Mindest-Observationen**: Referenz-Vektor wird erst ab 3 Beobachtungen als "kalibriert" markiert; davor: Bootstrap-Prototyp mit erhöhter Unsicherheit (GP-Lengthscale × 1.5)

**Fallback-Kaskade** (wenn kein passender Referenz-Vektor existiert):

1. Gleiche Genre-Familie + nächstliegende Ära → GP-Memory
2. Gleiche Ära + nächstliegendes Genre → GP-Memory
3. Bootstrap-Prototyp für Genre-Cluster
4. Genre-agnostischer Ära-Median (alle Genres der Ära gemittelt)
5. Kein Referenz-Vektor → `timbral_fidelity` rein gegen Input (Restorability-unabhängig)

### HPI-Formeln

Quelle: `[SRC:S06,S07,S08,S09,S10,S11]`

**Restoration**: `HPI = MERT_similarity(input, output) × timbral_fidelity(input, output) × artifact_freedom × emotional_arc_preservation`

- `timbral_fidelity` dominant: strukturelle Klangkohärenz (nicht bloße Input-Ähnlichkeit)
- `artifact_freedom` (§2.49): Artefakt-Freiheit — Musical Noise, Pre-Echo, Spectral Holes = 0
- MERT_similarity: musikalische Identität bewahren (Melodie, Harmonie, Rhythmus)
- `emotional_arc_preservation`: Arousal/Valence-Bogen + **Makrodynamik** (Vers-/Refrain-/Bridge-Pegelrelationen bleiben erhalten) + Lyrics-Salienz (§2.36: Phonem-Boost-Verhältnisse im Output konsistent mit Enhanced-Zielwerten)
- RestorabilityEstimator > 0.85 → strengeres Gate

**Studio 2026**: `HPI = studio_quality_gain × PQS_improvement × artifact_freedom × emotional_arc_preservation`

- PQS-Improvement dominant (Qualität steigern > Original-Treue)
- `studio_quality_gain`: Abstand zu Referenz-Studioniveau (−14 LUFS, Noise ≤ −72 dBFS)
- `artifact_freedom` (§2.49): auch Enhancement darf keine Artefakte erzeugen
- MERT-Ähnlichkeit fließt mit reduziertem Gewicht ein (musikalische Identität bewahren, nicht Klangfarbe)

**Beide Modi**: `HPI > 0` → Export | `HPI ≤ 0` → Rollback auf weniger aggressive Variante

### Evidenzklassen fuer Kern-Schwellen (Stand 2026-05-20)

| Regel | Evidenzklasse | Quellenbasis | Governance-Regel |
| --- | --- | --- | --- |
| Loudness/TruePeak-Guardrails | A | `[SRC:S06,S07]` | Direkt aus Normen ableitbar, nur normkonforme Updates |
| `artifact_freedom < 0.95` Veto + Musical-Noise/Pre-Echo-Detektion | B | `[SRC:S14,S15]` + materialadaptive Kalibrierung | Aenderungen nur mit Gate-Regression + Audit |
| Formant-/Vibrato-Vokalinvarianten | B | `[SRC:S16,S17,S18]` + `[SRC:S08,S09,S10,S11]` | Aenderungen nur mit Vocal-Regression + Hoertest-Delta |
| MERT-Floor und materialadaptive VQI-Floors | C | Primär kalibriert, theoretisch gestuetzt durch `[SRC:S08,S09,S10,S11]` | Quartalsweise Revalidierung gegen reale Musikfaelle verpflichtend |

**Invariante:** Klasse-C-Schwellen duerfen ohne neue Evidenz nicht als universale Naturkonstanten deklariert werden; sie bleiben kalibrierte Betriebsparameter.

**[BUG-FIX v10.0.0] Material-adaptive `timbral_fidelity` Floor** (Bug 5):

`HPI > 0` allein reicht nicht als Export-Bedingung. Ein `timbral_ref` von 0.318 (Vinyl, Restoration) mit MERT=0.65 ergibt `HPI ≈ 0.32 > 0` → Export trotz massiver Klangverfärbung.

```python
# Material-adaptive timbral floors (nach §0a / Spec 09 / calibration_matrix.py):
_TIMBRAL_FLOORS = {
    "shellac": 0.40, "wax_cylinder": 0.35, "lacquer_disc": 0.38,
    "vinyl": 0.55, "tape": 0.55, "reel_tape": 0.55, "cassette": 0.50,
    "cd_digital": 0.75, "dat": 0.70, "mp3_low": 0.60, "unknown": 0.55,
    "minidisc": 0.60,   # ATRAC-Kompression — gleiche Floor-Klasse wie mp3_low (v10.0.0)
}
_tf_floor = _TIMBRAL_FLOORS.get(material, 0.55)
# Restorability-Skalierung: sehr beschädigtes Material (< 40) hat niedrigeren erreichbaren timbral
_tf_floor_adj = _tf_floor * max(0.60, restorability_score / 100.0)
if timbral < _tf_floor_adj:
    passed = False  # timbral_fidelity unterhalb material-adaptiver Untergrenze
```

**Invariante**: Diese Prüfung erfolgt in `HolisticPerceptualGate._evaluate_restoration()` NACH der `passed = hpi > 0.0 and artifact_freedom >= 0.95` Zeile. Der FailReason ist `TIMBRAL_BELOW_FLOOR`.

> **[RELEASE_MUST] VERSA-Primärpflicht**: In der HPI-Berechnung ist **VERSA** das primäre MOS-Modell (`use_versa_in_loop=True` — immer aktiv, produktionsstabil). MERT fungiert ausschließlich als Proxy-Fallback wenn VERSA fehlschlägt (`metadata["mert_proxy_used"] = True`). **VERBOTEN**: `use_versa_in_loop=False` oder MERT als primäre Qualitätsmetrik bei verfügbarem VERSA. Referenz: Spec 04 SOTA-Matrix, copilot-instructions.md VERBOTEN-Liste.

### §2.44a [RELEASE_MUST] carrier_chain_recovery_ratio — UV3-Pflichtfeld (v10.0.0)

UV3 MUSS nach der letzten Carrier-Phase (§2.46 Stufe 4) folgende Metadata-Felder befüllen:

```python
# In UV3._execute_pipeline(), nach letzter Carrier-Phase:
pre_carrier_audio = metadata["_pre_carrier_audio"]  # gespeichert vor erster Carrier-Phase
post_carrier_audio = current_audio.copy()

# Spektrale Korrelation via normalisierte MFCC-Cross-Correlation
recovery_ratio = 1.0 - spectral_correlation(pre_carrier_audio, post_carrier_audio)

metadata["carrier_chain_recovery_ratio"] = float(np.clip(recovery_ratio, 0.0, 1.0))
metadata["best_carrier_checkpoint"] = post_carrier_audio  # Referenz für §1.2a End-Goals

# Schwellwerte:
# > 0.15 = signifikante Carrier-Inversion → §1.2a Referenz-Shift aktiv
# > 0.35 = massive Inversion (Shellac, Multi-Gen) → HPI MERT-Referenz-Anker verstärkt
# ≤ 0.15 = geringe Inversion (CD, MP3) → Standard-Referenz gegen degradierten Input
```

**Invariante**: `carrier_chain_recovery_ratio` ist ein Pflichtfeld in `RestorationResult.metadata`. Fehlt es, greift der Fallback `0.0` (kein Referenz-Shift).

### HPI-Gewichtungs-Semantik

Die HPI-Multiplikation ist **nicht** gleichgewichtet — die Faktoren operieren auf unterschiedlichen Wertebereichen:

| Faktor | Wertebereich | Rolle |
| --- | --- | --- |
| `timbral_fidelity` | [0.8, 1.0] | Geringe Varianz — dominiert durch **Sensitivität**: kleine Abweichung → großer HPI-Einbruch |
| `artifact_freedom` | [0.0, 1.0] | **Veto-Faktor**: < 0.95 → Gate-Fail (Primum non nocere) |
| `MERT_similarity` | [0.5, 1.0] | Musikalische Identität — verhindert, dass Restaurierung das Stück verändert |
| `emotional_arc` | [0.7, 1.0] | Dynamik-Bogen + Makrodynamik — Narrative Struktur erhalten |

Ein Artefakt (`artifact_freedom` = 0.5) killt den HPI härter als eine leichte Timbre-Abweichung (`timbral_fidelity` = 0.95) — das ist beabsichtigt.

### §0d CCR-Timbral-Floor (v10.0.0) [RELEASE_MUST]

**Problem**: Wenn die Carrier-Chain-Inversion erfolgreich war (`carrier_chain_recovery_ratio > 0.15`),
entfernt sich das restaurierte Signal intentional vom degradierten Input. In diesem Fall misst
`timbral_fidelity` gegen den Input einen niedrigen Wert — obwohl das Ergebnis besser klingt.
Das kann zu einem falschen HPI-Fail führen, obwohl MERT-Similarity hoch ist (musikalische Identität erhalten).

**Lösung** (implementiert in `HolisticPerceptualGate`):

```python
# §0d CCR-Timbral-Floor: Verhindern, dass intentionelle Carrier-Inversion
# einen falschen HPI-Fail erzeugt.
if mert_sim >= 0.74 and carrier_checkpoint_used:
    timbral_floor = float(np.clip(mert_sim * 0.90, 0.65, 0.95))
    timbral_fidelity = max(timbral_fidelity, timbral_floor)
    metadata["ccr_timbral_floor_applied"] = True
    metadata["ccr_timbral_floor_value"] = timbral_floor
```

**Invariante**: Nur aktiv wenn `carrier_checkpoint_used=True` (UV3 setzt dieses Flag nach Carrier-Phasen)
UND `mert_sim ≥ 0.74` (musikalische Identität ist erhalten). Kein Floor bei reinen Enhancement-Phasen.

### §2.44b [RELEASE_MUST] HPG Reference-Memory — UV3-Wiring-Pflicht (v10.0.0)

**Normative Pflicht**: UV3 MUSS `_hg.update_reference_memory()` nach jedem erfolgreichen HPI-Lauf aufrufen.
Dieselbe Bedingung wie RestorationMemory-Save (`HPI > 0.0 AND artifact_freedom ≥ 0.95`).

```python
# In UV3._execute_pipeline(), nach RestorationMemory-Save (§2.70):
# §2.44 [RELEASE_MUST] HPG Reference-Memory: update nach erfolgreichem HPI.
if float(_hpi_result.hpi) > 0.0 and _af_save >= 0.95:
    try:
        _hg.update_reference_memory(
            restored=restored_audio,
            sr=sample_rate,
            hpi=float(_hpi_result.hpi),
            artifact_freedom=_af_save,
            p1_p2_passed=bool(_hpi_result.passed),
            genre=_hpi_genre,
            material=_hpi_material,
            era_bin=_hpi_era,
        )
    except Exception as _ref_mem_exc:
        logger.debug("§2.44 ReferenceMemory.update non-blocking: %s", _ref_mem_exc)
```

**Kaltstart / Bootstrap**: `scripts/bootstrap_hpg_reference_memory.py` seeded
`~/.aurik/hpg_reference_memory.json` aus `golden_samples/references/`. Pflicht nach Neuinstallation.

**VERBOTEN**: `update_reference_memory()` nicht aufrufen → `_ref_memory` bleibt leer →
alle 5 Fallback-Stufen liefern None → `timbral_fidelity` nutzt stets nur den degradierten Input
→ Restaurierungsqualität systematisch unterschätzt → HPG-Lernkurve deaktiviert.

### v10-Amendment (2026-07-12)

**Final-Export-Audio-Gate: Restoration-Mode verwendet `best_carrier_checkpoint`** (v10, 2026-07-12):

Das finale Export-Gate (§2.44/§2.49) in `unified_restorer_v3.py` vergleicht das restaurierte Audio standardmäßig gegen `analysis_audio` (degradiertes Original). Eine gute Restaurierung klingt jedoch _anders_ (sauberer) als das degradierte Original → der Artifact-Freedom-Detektor meldet fälschlich `af=0.000` ("100 % Unterschied") → unnötiger Rollback.

**Fix**: Im Restoration-Mode (`not self.is_studio_mode()`) verwendet das Final-Gate jetzt `self._best_carrier_checkpoint` als Referenz — das Zwischenergebnis, gegen das der Restorer tatsächlich optimiert hat. Falls kein Carrier-Checkpoint existiert, fällt das Gate auf `analysis_audio` zurück. Studio-Mode bleibt unverändert.

**Implementierung**: `backend/core/unified_restorer_v3.py` → Zeile ~18629:

```python
if not self.is_studio_mode():
    _carrier_ref = getattr(self, "_best_carrier_checkpoint", None)
    _final_gate_ref = ...(_carrier_ref if _carrier_ref else analysis_audio)...
else:
    _final_gate_ref = ...analysis_audio...
```

**Erwartetes Ergebnis**: Kein falscher `af=0.000`-Rollback mehr. Das finale Tor vergleicht »finaler Output vs bester Zwischenstand« — genau das, was es in der Restoration intendiert.

## §2.45 [RELEASE_MUST] Minimal-Intervention-Prinzip (v10.0.0, aktualisiert §2.54)

**Restoration**: Phasen ohne hörbare Verbesserung werden NICHT angewendet:

- `perceptual_delta > 0` nachweisen (MERT-Embedding-Distanz oder timbral_fidelity-Delta)
- `perceptual_delta ≤ 0` → Stärke iterativ reduzieren (§2.54 Messen→Handeln→Validieren);
  erst nach 3 Iterationen ohne Verbesserung → Phase-Skip

**Studio 2026**: Volle Enhancement-Kette aktiv, aber jede Phase muss Klanggewinn nachweisen:

- `perceptual_delta > 0` Pflicht — auch Enhancement-Phasen müssen messbaren Nutzen zeigen
- Phasen ohne messbaren Klanggewinn nach 3 Iterationen → Skip

> **§2.54 Kontext**: `perceptual_delta > 0` ist das **Ziel**, nicht die Abbruchbedingung.
> Wenn die erste Stärke keinen positiven Delta bringt, wird die Stärke adaptiv angepasst
> (PhaseConductor-Empfehlung × reduzierte Wetness), nicht sofort geskippt.
> Erst wenn nach dem iterativen Zyklus kein positives Delta erreichbar ist, wird geskippt.

## §2.45a [RELEASE_MUST] Mid-Pipeline-Loudness-Drift-Guard (v10.0.0, erweitert v10.0.0)

Quelle: `[SRC:S06,S07]`

### Problem

Die finale LUFS-Invariante (`LUFS-Differenz ≤ 1 LU`) schützt den Export, aber nicht zwingend frühe, hörbare Pegelkollapse innerhalb der subtraktiven Phasenkette.

### Pflicht-Invarianten

- Für **breitbandig-subtraktive** Phasen (Denoise, Dereverb, Noise-Gate, Surface-Noise) MUSS ein material-adaptiver per-Phase-RMS-Drift-Guard aktiv sein.
- **Ausnahme §2.45a-VI**: Spektralband-Filter (HPF, LPF, Notch, Bandpass) dürfen **keinen** per-Phase-Makeup-Gain-Guard haben — ihr Energieverlust ist beabsichtigt (Carrier-Inversion). Für diese Phasen übernimmt Stufe 2 (Mid-Pipeline) und Stufe 3 (End-of-Pipeline) der Kaskade die Überwachung.
- Ein Guard darf die Phase nicht trivialisieren (`strength=0`/Bypass als Standardreaktion ist unzulässig).
- Bei Überschreitung des material-adaptiven RMS-Drift-Limits gilt: primär Dry/Wet-Rescue (mehr Dry-Anteil), sekundär sichere Makeup-Gain-Kompensation.
- Gain-Limits müssen den DSP-Peak-Guard nutzen: `np.percentile(np.abs(audio), 99.9)`.
- Phase-Metadaten müssen `rms_drop_db` und `loudness_makeup_db` ausweisen.
- Pipeline-Metadaten müssen stärkste Pegelabfälle separat ausweisen (z. B. `phase_regression_top_drops`).

### §2.45a-I [RELEASE_MUST] Gated-RMS-Pflicht (v10.0.0)

Alle RMS-Messungen in Loudness-Drift-Guards MÜSSEN **gated** erfolgen:

- Frame-basiert: Signal in Frames aufteilen (≈ 2048 Samples / ~43 ms bei 48 kHz)
- Gate-Schwellwert: nur Frames mit RMS > −50 dBFS berücksichtigen (Stille-Frames ignorieren)
- Mindest-Gate-Ratio: wenn < 5 % der Frames den Gate passieren → Fallback auf ungated-RMS
- Stereo-Behandlung: vor dem Framing zu Mono downmixen (`(L + R) * 0.5`), nicht `.reshape(-1)` (interleaved Samples mischen L/R-Information)

**Rationale**: Globaler RMS misst Stille mit. Subtraktive Phasen (Denoise) reduzieren Stille-RMS drastisch (−35 → −80 dBFS), während Musik-RMS nahezu unverändert bleibt. Globaler RMS täuscht dadurch einen Pegelkollaps vor, der perzeptuell nicht existiert → unnötige Makeup-Gain-Kompensation → Stille wird re-amplifiziert.

**VERBOTEN**: `np.mean(audio**2)` oder `np.sqrt(np.mean(audio**2))` in Loudness-Guards (misst Stille mit).

> ⚠️ Der Gated-RMS-Schwellwert (−50 dBFS) ist der **Messgate** für die RMS-Referenz — er bestimmt,
> welche Frames in die Pegelreferenz einfließen. Er ist NICHT identisch mit dem **Gain-Gate** von
> `apply_musical_gain_envelope` — das Gain-Gate MUSS −36 dBFS betragen (§2.45a-V).

### §2.45a-II [RELEASE_MUST] Envelope-Aware Gain (v10.0.0)

Makeup-Gain-Kompensation MUSS **musik-selektiv** (envelope-aware) erfolgen:

- Gain-Envelope: Frame-basierte Gate-Entscheidung (identisches Framing wie Gated-RMS)
- Musikalische Frames (RMS > Gate): Gain wird angewendet
- Stille-Frames (RMS ≤ Gate): **kein Gain** (Faktor 1.0 — Signal unverändert)
- Crossfade an Gate-Übergängen: 10 ms Hann-Fenster-Smoothing (keine harten Sprünge)
- Tail-Handling: Samples jenseits des letzten vollständigen Frames werden explizit gemessen und gegated (Default: kein Gain)

**Rationale**: Uniformer Gain (`audio * g`) amplifiziert Stille-Segmente gleichermaßen wie Musik. Nach Denoising enthält Stille typisch −80 dBFS — uniformer Gain von +4 dB hebt sie auf −76 dBFS, was bei niedrigem Rauschboden hörbar sein kann und die Entrauschung teilweise rückgängig macht.

**VERBOTEN**: `audio *= gain_factor` als Makeup-Kompensation in Loudness-Guards.

### §2.45a-III [RELEASE_MUST] Soft-Limiter-Invarianten (v10.0.0)

Wenn Makeup-Gain Peaks über die digitale Grenze treibt, MUSS ein Soft-Limiter eingreifen:

- **Typ**: `tanh`-basiertes Shaping: `0.92 + 0.08 * tanh((|x| - 0.92) / 0.08)`
- **Bedingung**: NUR wenn `peak_after_gain > 0.98` (echtes Clipping-Risiko), NICHT als routinemäßiger Post-Gain-Schritt
- **Finaler Clip**: `np.clip(audio, -1.0, 1.0)` nach Soft-Limiter als Sicherheitsnetz

**Rationale**: Ein Soft-Limiter bei 0.92 als Routine-Schritt nach jedem Gain komprimiert musikalische Peaks um bis zu 3 dB. Bei 3 Stufen (per-Phase + Mid-Pipeline + End-of-Pipeline) akkumuliert sich die Kompression → Dynamikverlust → §0-Verletzung.

**VERBOTEN**: Unbedingter Soft-Limiter nach Makeup-Gain (komprimiert Musikdynamik ohne Clipping-Risiko).

### §2.45a-IV Dreistufige Guard-Kaskade (v10.0.0)

Die Pipeline implementiert 3 Ebenen Loudness-Drift-Protection:

| Stufe | Trigger | Messung | Scope |
| --- | --- | --- | --- |
| **1. Per-Phase** | Nach jeder **breitbandig-subtraktiven** Phase (Denoise/Dereverb/Noise-Gate/Surface-Noise) — **nicht** HPF/LPF/Notch (§2.45a-VI) | Gated-RMS Δ vs. Phase-Eingang | Einzelphase |
| **2. Mid-Pipeline** | Nach jeder Phase im Loop | Gated-RMS vs. Pipeline-Start (`_afg_pre_pipeline_audio`) | Kumulativ bis Checkpoint |
| **3. End-of-Pipeline** | Nach Phase-Loop, vor Export-Gates | Gated-RMS vs. Pipeline-Start | Gesamt-Pipeline |

**Interaktions-Invarianten:**

- Jede Stufe verwendet Gated-RMS (§2.45a-I) und Envelope-Aware Gain (§2.45a-II)
- Soft-Limiter nur bei peak > 0.98 (§2.45a-III) — verhindert kumulative Dynamik-Kompression
- Die Stufen sind redundant-sichernd konzipiert: Stufe 2 fängt kumulative Drift, die Stufe 1 nicht einzeln erkennt
- Stufe 3 ist das finale Sicherheitsnetz (`_MAX_CUMULATIVE_LEVEL_DROP_DB ≈ 0.915 dB` = ~10 % Amplitude)

### §2.45a-V [RELEASE_MUST] Makeup-Gain-Gate −36 dBFS (v10.0.0)

**Das fundamentale Missverständnis**: Das Gated-RMS-Messgate (§2.45a-I, −50 dBFS) und das
Makeup-Gain-Gate von `apply_musical_gain_envelope` (§2.45a-II) sind **zwei verschiedene Schwellwerte**
mit unterschiedlichen Rollen:

| Schwellwert | Rolle | Wo | Richtwert |
| --- | --- | --- | --- |
| **Mess-Gate** | Welche Frames fließen in die RMS-Referenz | `_rms_dbfs_gated()` | −50 dBFS |
| **Gain-Gate** | Welche Frames erhalten Makeup-Gain | `apply_musical_gain_envelope(gate_dbfs=...)` | **−36 dBFS** |

**Warum −36 dBFS für das Gain-Gate (nicht −50 dBFS)?**

Vinyl- und Shellac-Oberflächenrauschen liegt typisch bei **−35 bis −42 dBFS**. Ein Gain-Gate von
−50 dBFS klassifiziert dieses Oberflächenrauschen als "Musik" → bekommt Makeup-Gain →
Pegelexplosion in Intro/Outro/Fadeout. Bestätigt in Produktion:

- `phase_05_rumble_filter`: `gate_dbfs=-50.0` → stille Fadeout-Bereiche mit Vinyl-Rauschen
  (~−40 dBFS) werden als Musikframes eingestuft → Makeup-Gain boosted Rauschboden → hörbare
  Pegelexplosion zu Beginn und am Ende des Songs (bestätigt 2026-04-25)
- `correct_arc()`: gleicher Mechanismus → −42 dBFS per-sample Guard fixiert auf −36 dBFS (§2.30b)

**Normative Regel (bindend für alle Phasen):**

```python
# RICHTIG — Vinyl/Shellac-sicher:
filtered = apply_musical_gain_envelope(
    audio, gain_factor,
    gate_dbfs=-36.0,   # ← MUSS -36.0 sein, NICHT -50.0
    crossfade_ms=10.0,
    sr=sample_rate,
)

# FALSCH — Pegelexplosion auf Vinyl/Shellac:
filtered = apply_musical_gain_envelope(
    audio, gain_factor,
    gate_dbfs=-50.0,   # ← Rauschboden bei -40 dBFS passiert dieses Gate → VERBOTEN
    ...
)
```

**Ausnahme**: `apply_musical_gain_envelope` enthält einen adaptiven Gate-Mechanismus,
der den Schwellwert automatisch anhebt wenn der 5th-Perzentil-RMS über `gate_dbfs + 12 dB`
liegt. Dieser Mechanismus ist NICHT als alleinige Schutzmaßnahme ausreichend — er greift
nur wenn die Noise-Floor-Evidenz eindeutig genug ist. Der Aufrufer MUSS explizit −36 dBFS
übergeben.

**Checkliste für alle Phases mit Makeup-Gain-Logik:**

- [ ] `apply_musical_gain_envelope(..., gate_dbfs=-36.0, ...)` — NICHT −50.0
- [ ] `self._musical_gain_envelope(..., gate_dbfs=-36.0, ...)` (UV3-intern) — NICHT −50.0
- [ ] `_rms_dbfs_gated(audio)` für RMS-Messung verwendet den internen Default (−50 dBFS) — korrekt
- [ ] Per-Sample-Guard nach `np.interp` (§2.30b) nutzt −36 dBFS
- [ ] Bei positivem Loudness-/Export-/Mastering-Gain mit Quiet-Edge-Risiko: Referenzvergleich Intro/Outro gegen Eingangs-Audio und post-gain Quiet-Edge-Clamp (`limit_quiet_edge_boost(reference_audio, candidate_audio, sr)`) anwenden

**Zusatzinvariante für positive Gain-Pfade (v10.0.0):**

Music-gated Gain allein ist nicht hinreichend, wenn ein Signal bewusst leise Intro-/Outro-Zonen enthält.
Sobald ein Pfad Programm-Lautheit aktiv anhebt (z. B. LUFS-Normalisierung, Floor-Boost, Export-Normalisierung,
Mastering-LUFS, späte Dynamics-Polish), MUSS nach dem positiven Gain ein Referenzvergleich gegen das Eingangs-
signal stattfinden. Die kanonische Umsetzung ist `limit_quiet_edge_boost(reference_audio, candidate_audio, sr)`. `[SRC:S06,S07]`

**Zielinvariante:** Der Mittelteil darf angehoben werden, aber intentional leise Intro-/Outro-Zonen dürfen gegenüber
der Referenz nicht neu explodieren.

**UV3-intern betroffene Stellen (alle drei benötigen −36.0):**

- `_active_quality_intervention()` — per-Phase-Rescue bei Loudness-Kollaps
- Mid-Pipeline-Cumulative-Guard — kumulativer Pegel-Drift nach jeder Phase
- End-of-Pipeline-Guard — finales Sicherheitsnetz vor Export-Gates

### Normativer Scope (typische Kandidaten)

- Denoise / Hiss / Surface-Noise Reduction
- Noise-Gate
- Dereverb
- Rumble-Filter
- Jede Phase mit Makeup-Gain-Kompensation

### Rationale

Schützt §0 (Primum non nocere), §2.45 (Minimal-Intervention) und P1/P2-Pipeline-Ende-Regeln (§2.54) gegen frühe Klangausdünnung, ohne die Defektkorrekturwirkung zu verlieren.

### §2.45a-VI [RELEASE_MUST] Kein Makeup-Gain-Guard in subtraktiven Filtertypen (v10.0.0)

**Fundamental-Invariante**: Hochpassfilter, Tiefpassfilter, Notchfilter und Bandpassfilter entfernen
Energie **absichtlich**. Ein per-Phase-Makeup-Gain-Guard, der diesen Energieverlust kompensiert,
kämpft gegen den Filter — das ist ein **logischer Widerspruch** der zu einer Bug-Endlosschleife führt.

**Mechanismus der Endlosschleife (phase_05 — bestätigt 2026-04-25):**

1. Rumble-Filter (HP 20–80 Hz) entfernt sub-Bass-Rumpelenergie (30+ dB unter Musik-Pegel)
2. `_rms_in_db_ref` wird **vor** dem Filter gemessen → enthält Rumpelenergie
3. `_rms_out_db` wird **nach** dem Filter gemessen → Rumpelenergie fehlt
4. Scheinbarer RMS-Drop → Makeup-Gain-Guard feuert
5. `apply_musical_gain_envelope` boosted Fadeout/Intro-Frames → Pegelexplosion
6. Fix-Versuch: `gate_dbfs=-50.0 → -36.0` → Pegelexplosion bleibt (Referenzmessung war falsch)
7. Nächster Fix-Versuch würde wieder fehlschlagen → Endlosschleife

**Normative Regel:**

| Phasentyp | Makeup-Gain-Guard erlaubt? | Begründung |
| --- | --- | --- |
| HPF / LPF / Notch / Bandpass | ❌ **VERBOTEN** | Energieverlust ist Carrier-Inversion — beabsichtigt |
| Denoise / De-Hiss / Surface-Noise | ✅ Erlaubt | Breitbandig-subtraktiv: Musikpegel-Erhalt wichtig |
| Dereverb | ✅ Erlaubt | Breitbandig-subtraktiv: Musikpegel-Erhalt wichtig |
| Noise-Gate | ✅ Erlaubt | Wirkt auf Dynamik, nicht auf Spektralband |

**Wer überwacht den Gesamtpegel dann?**

Der UV3-Cumulative-Guard (`§2.45a-IV`) misst den kumulativen Pegel-Drift über die gesamte Pipeline
und greift ein wenn nötig. Er ist die richtige Stelle für Pipeline-weite Pegel-Kompensation —
nicht einzelne HP-Filter-Phasen.

```python
# RICHTIG — kein Guard in HPF-Phase:
filtered = apply_highpass_filter(audio, cutoff_hz)
filtered = np.clip(np.nan_to_num(filtered), -1.0, 1.0)
return create_phase_result(audio=filtered, ...)

# FALSCH — Guard kämpft gegen den Filter:
filtered = apply_highpass_filter(audio, cutoff_hz)
rms_drop = rms(filtered) - rms(audio)  # enthält sub-Bass-Energie → scheinbarer Drop
if rms_drop < -threshold:
    filtered = apply_musical_gain_envelope(filtered, makeup_gain, ...)  # Pegelexplosion
```

**Betroffene Phasen (kein Makeup-Gain-Guard erlaubt):**

- `phase_05_rumble_filter` (HPF) — Guard entfernt in v10.0.0 (commit 72d993a)
- `phase_02_hum_removal` (Notch) — Guard entfernt in v10.0.0
- Jede zukünftige Phase die primär als Spektralband-Filter arbeitet

**Zweite Fehlerquelle: UV3 `_active_quality_intervention` + Cumulative Guard (bestätigt 2026-04-25)**

Nach dem Entfernen der per-Phase-Guards trat die Pegelexplosion erneut auf. Ursache:

1. `_active_quality_intervention` berechnet `_prof = _phase_intervention_profile(phase_id)` —
   ohne Eintrag in `_phase_overrides` gilt der General-Default `enable_loudness=True`.
   HPF/Notch-Phasen sehen so immer noch einen scheinbaren Loudness-Kollaps → Makeup-Gain.
2. Der Mid-Pipeline-Cumulative-Guard und der End-of-Pipeline-Guard verwendeten
   `_afg_pre_pipeline_audio` (vor der Pipeline) als Referenz — HPF-Energieentfernung akkumulierte
   als kumulativer RMS-Drop → Guard triggerte Makeup-Gain.

**Vollständige Fix-Checkliste für neue HPF/Notch-Phasen:**

```python
# 1. Phase-Datei: keinen per-Phase-Makeup-Gain-Guard (§2.45a-VI)
# 2. UV3 _phase_overrides: enable_loudness=False setzen
_phase_overrides["phase_XX_name"] = {
    "family": "phase_xx_hpf",
    "enable_loudness": False,
    "enable_stereo": False,
    "enable_transient": False,
}
# 3. UV3 _HPF_NOTCH_CUM_RESET_PHASES: Phase eintragen
_HPF_NOTCH_CUM_RESET_PHASES: frozenset = frozenset({
    "phase_02_hum_removal",
    "phase_05_rumble_filter",
    "phase_XX_name",  # NEU
})
# Nach diesem Phase-Reset berechnet der Cumulative Guard den Drift
# relativ zum Audio NACH der HPF — nicht relativ zum Audio davor.
```

Implementierung in `backend/core/unified_restorer_v3.py`:

- `_phase_overrides`: `phase_02` + `phase_05` mit `enable_loudness=False` (commit 032a83f)
- `_cum_rms_reference_audio` + `_HPF_NOTCH_CUM_RESET_PHASES` im Phase-Loop (commit 032a83f)
- `_cum_guard_ref = _cum_rms_reference_audio` im End-of-Pipeline-Guard (commit 032a83f)

## §2.46 [RELEASE_MUST] Carrier-Chain-Inversion (v10.0.0)

**Restoration-Modus**: Ziel = **gesamte Tonträgerkette invertieren**, nicht Einzel-Defekte reparieren.

**Signalkette** (vorwärts): `Studio-Monitor → Mic/Line → Preamp → Mixer → Carrier-Encoding (Tape/Vinyl/Shellac/Digital) → Alterung → Playback → ADC → Digital-File`

**Restaurierung** (invers, Reihenfolge beachten):

1. ADC-Artefakte entfernen (DC-Offset, Quantisierungsrauschen)
2. Playback-Verzerrungen invertieren (RIAA-Inverse, Azimuth-Korrektur, Wow/Flutter)
3. Alterungsschäden reparieren (Knistern, Dropout, Oxidation)
4. Carrier-Encoding invertieren (Bandrauschen, Vinyl-Groove-Distortion, Shellac-Rauschen)
5. Mixer/Preamp-Charakter: **bewahren** (Recording-Chain-Signatur = Teil des Originals)
6. Studio-Raumklang: **bewahren** (nicht über-entrauschen — Rauschboden material-adaptiv §0a)

**Studio 2026**: Carrier-Chain-Inversion + Enhancement-Kette (§1.5). Mixer/Preamp-Charakter darf modernisiert werden.

> Kreuzreferenz: Slim Core §2.46, Spec 01 §8.2 Rauschboden modus-differenziert

### §2.46a.1 Vinyl-Inference (v10.0.0-Phantom)

Wenn `reel_tape` und `cassette` in der erkannten Kette sind und die
Inhalts-Ära 1950–1990 beträgt, inferiert Aurik eine Vinyl-Zwischenstufe.
Dies ist eine logische Inferenz aus der Defektanalyse — kein physikalisches
Risiko für das Audio.

Implementiert in `backend/core/pre_analysis.py` (§Vinyl-Inference).

### v10-Amendment (2026-07-12)

**Transfer-Chain: Era-Material (Original-Aufnahmemedium) an Position 0** (v10, 2026-07-12):

Der EraClassifier liefert das **Original-Aufnahmemedium** (z.B. `reel_tape` für eine Studio-Aufnahme der 1970er). Dieses muss chronologisch am ANFANG der Tonträgerkette stehen, nicht zwischen Zwischenträger und digitaler Stufe.

**Vorher**: `vinyl → cassette → reel_tape → mp3_low` (falsch)
**Nachher**: `reel_tape → vinyl → cassette → mp3_low` (korrekt)

Chronologische Ordnung: Studio-Aufnahme → Veröffentlichung → Überspielung → Digitalisierung.

**Implementierung** (`backend/core/pre_analysis.py`):
`_era_material` wird separat behandelt und via `chain.insert(0, ...)` an den Anfang gesetzt. `_defect_material` und `physical_analog_sources` bleiben bei Insertion vor der digitalen Stufe.

```python
if _era_injected is not None:
    _chain.insert(0, _era_injected)  # Original-Aufnahmemedium am Anfang
if _chain_injected:
    # Zwischenträger vor digitaler Stufe einfuegen
    for _m in reversed(_chain_injected):
        _chain.insert(_dpos, _m)
```

## §2.46a [RELEASE_MUST] Deep-Transfer-Chain-Pflicht (v10.0.0)

Importsongs mit **3+ Tonträgerstufen** müssen vollständig modelliert werden. Die
Transferkette darf nicht auf Primärträger + eine Sekundärstufe verkürzt werden.

### Invarianten

1. `transfer_chain` bildet reale Mehrfachkopien kausal ab, z. B.
    `shellac -> reel_tape -> cassette -> cd_digital -> mp3_low`.
2. Digitale Zwischenstufen (`cd_digital`, `dat`) dürfen bei lossy Endformaten nicht
    ausgelassen werden, wenn Evidenz vorliegt.
3. Keine Rückwärtssprünge in der Kette: Reihenfolge bleibt gemäß `_MEDIUM_ORDER`.
4. Nach Material-Normalisierung werden Duplikate konsolidiert
    (Konfidenzaggregation via `max`), damit `source_fidelity_generation_count`
    nicht künstlich aufgebläht wird.
5. Die erkannte Mehrfachkette muss bis SongCalibration, SourceFidelity und
    Export-Metadaten propagiert werden.

### Testpflicht

- Mindestens ein Unit-Test für eine 4-stufige Kette mit digitaler Zwischenstufe.
- Mindestens ein Unit-Test für `file_ext=.mp3` mit physikalischer Inferenz und
  4-stufigem Ergebnis.

Referenztests: `tests/unit/test_forensics_medium_detector.py`

## §2.46b [RELEASE_MUST] Spectral-Tilt-Preservation-Invariante (v10.0.0.x)

**Psychoakustische Motivation**: Der Spektral-Tilt (Steigung der mittleren Spektralhüllkurve in
dB/Oktave) kodiert den Ära-Charakter eines Recordings: 1920er ≈ −6 dB/oct, 1970er ≈ −4 dB/oct,
2000er ≈ −3 dB/oct. phase_06 (SBR / Bandwidth-Extension) und phase_39 (Air-Enhancement > 12 kHz)
können den Tilt unbemerkt verschieben und dadurch den Ära-Charakter zerstören, ohne dass
ein Musical-Goal-Verstoß detektiert wird (brillanz steigt, Goal scheinbar erfüllt).
Das Ergebnis klingt wie ein falsch mastered Remaster, nicht wie das Original.

**Invariante**: Jede Phase vom Typ `ADDITIVE` (§2.48a), die den Spektral-Tilt verändert
(HF-Extension, SBR, Air-Enhancement), MUSS sicherstellen, dass die Deviation vom
`era_result.spectral_tilt`-Referenzwert ≤ ±material_tolerance bleibt.

**Material-Toleranz** (Träger mit inhärent ungleichmäßigem Tilt erhalten mehr Spielraum):

| Material | Toleranz (dB/oct) | Begründung |
| --- | --- | --- |
| digital, cd_digital, streaming | ±1.5 | Flacher Referenz-Tilt |
| tape, reel_tape | ±1.875 | Bandcharakter natürlich variabel |
| vinyl | ±2.25 | RIAA-Entzerrung variiert zwischen Pressungen |
| shellac, wax_cylinder, wire_recording | ±3.0 | Stark schwankende Träger-Charakteristika |

**Messung**: `_estimate_spectral_tilt_quick(audio, sr)` — Log2-Regression über aktives Spektrum
identisch zu `EraClassifier._estimate_spectral_tilt()` (wiederverwendet, nicht dupliziert)

**Enforcement in `phase_06.process()`**:

```python
era_result = kwargs.get("era_result", None)
if era_result is not None and hasattr(era_result, "spectral_tilt"):
    tilt_post = _estimate_spectral_tilt_quick(audio_after_sbr, sr)
    tilt_deviation = abs(tilt_post - era_result.spectral_tilt)
    mat_tol = _TILT_MATERIAL_TOLERANCE.get(material_type, 1.5)
    if tilt_deviation > mat_tol:
        # Linearer Cap: Boost-Anteil reduzieren, bis Tilt-Deviation ≤ mat_tol
        cap_factor = 1.0 - min(0.50, (tilt_deviation - mat_tol) / (mat_tol * 2.0))
        # hf_boost neu anwenden mit cap_factor auf Extension-Anteil
        metadata["spectral_tilt_capped"] = {
            "post_tilt": tilt_post, "era_tilt": era_result.spectral_tilt,
            "deviation": tilt_deviation, "tolerance": mat_tol, "cap_factor": cap_factor
        }
```

**Invarianten**:

- Gilt nur für `ADDITIVE`-Phasen — subtraktive Phasen (Denoising) invertieren Carrier-Tilt intentional
- Kein Rollback — nur Boost-Cap (Stärke-Modifikation, nicht Phasen-Ablehnung)
- Kein Guard, wenn `era_result` nicht in `kwargs` (graceful skip ohne Log-Spam)
- `era_result.spectral_tilt = -4.0` ist der Default (§4.x EraClassifier), d. h. Guard ist immer aktiv wenn era übergeben
- Telemetrie: `metadata["spectral_tilt_capped"]` nur wenn tatsächlich gecappt wurde

> Messmethode: `backend/core/era_classifier.py` — `_estimate_spectral_tilt()` (bestehende Methode, nicht kopieren!)
> Aufruf: `backend/core/phases/phase_06_frequency_restoration.py` — `process(..., **kwargs)`

## §2.46c [RELEASE_MUST] Zentraler BW-Hard-Cap nach additiven Phasen (v10.0.0)

**Problem**: Einzelne Phasen (phase_06, phase_07, phase_23, phase_39) haben per-Phase-BW-Limits, aber es gibt keine zentrale Absicherung, dass die **kumulative** Wirkung mehrerer additiver Phasen das physikalische BW-Ceiling des Quellmaterials nicht überschreitet.

**Lösung**: UV3 führt nach dem letzten ADDITIVE-Phase-Block einen zentralen BW-Hard-Cap aus:

```python
# In UV3._execute_pipeline(), nach letztem ADDITIVE-Phase-Block:
_MATERIAL_BW_CEILING_HZ = {
    "wax_cylinder":   5000,
    "wire_recording": 6000,
    "shellac":        8000,
    "lacquer_disc":   8000,
    "vinyl":         16000,
    "tape":          15000,
    "reel_tape":     18000,
    "cassette":      14000,   # alias: tape
    "dat":           22000,
    "minidisc":      20000,
    "cd_digital":    22050,
    "mp3_low":       16000,   # 128 kbps → effektive BW
    "mp3_high":      20000,
    "aac":           20000,
    "streaming":     20000,
    "unknown":       20000,
}

def _post_additive_bw_guard(audio, sr, material_type, mode):
    """Zentraler BW-Guard nach allen additiven Phasen."""
    if mode == "studio_2026":
        return audio  # Studio 2026: volle BW-Extension erlaubt
    ceiling_hz = _MATERIAL_BW_CEILING_HZ.get(material_type, 20000)
    # Butterworth 8th-order zero-phase LPF
    if ceiling_hz < sr / 2 - 100:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(8, ceiling_hz, btype="low", fs=sr, output="sos")
        if audio.ndim == 1:
            audio = sosfiltfilt(sos, audio)
        else:
            for ch in range(audio.shape[-1]):
                audio[..., ch] = sosfiltfilt(sos, audio[..., ch])
        metadata["bw_ceiling_applied_hz"] = ceiling_hz
    return audio
```

**Platzierung in Pipeline**: Nach der letzten Phase mit `family in ("additive", "reconstruction")`, VOR dem FeedbackChain und End-Goals-Check.

**Invarianten**:

- NUR im Restoration-Modus aktiv. Studio 2026 darf volle BW-Extension nutzen (erfordert aber MUSHRA ≥ 3.5 per Extension-Band).
- Kein Rollback — reiner Guard-Filter (transparent für Audio unter Ceiling).
- Material-Keys folgen `SUPPORTED_MATERIALS` (§6.1).
- Telemetrie: `metadata["bw_ceiling_applied_hz"]` nur wenn tatsächlich gefiltert wurde.

## §2.46e [RELEASE_MUST] Hallucination-Guard (v10.0.0)

**Keine additive Phase darf Material in das Ausgangssignal einbringen, das im Eingangssignal physikalisch nicht vorhanden war.** Dies gilt absolut für `restoration`-Modus.

### Drei Kategorien halluzinierten Materials (alle verboten in Restoration)

1. **Harmonik-Halluzination**: Obertöne, die über das physikalische BW-Ceiling (§2.46c / §6.2c) hinausgehen oder deren Amplitude das Trägerprofil überschreitet
2. **Raum-Halluzination**: Raumklang, Reverb-Schwänze oder Stereobreite, die im degradierten Signal nicht nachweisbar sind und nicht aus der Recording-Chain stammen
3. **Textur-Halluzination**: Spektrale Texturen (Harmonischer Hiss, Formant-Muster), die durch ML-Modelle generiert wurden und kein physikalisches Gegenstück im Source-Material haben

### Mess-Gate (`hallucination_guard.py`)

```python
# Pre/Post-Additive-Phase:
# BUG-FIX v10.0.0 (Bug 1): Band-relative Delta-Metrik statt Gesamtenergie-Verhältnis.
# Die alte Formel (energy_new_bins / energy_total) war blind für Air-Band-Halluzinationen,
# da Air-Band-Energie < 0.1% der Gesamtenergie → spectral_novelty nie > 0.03%.
# Neue Metrik: Band-relative Energie-Verhältnis IM CEILING-BAND:
ceiling_band_energy_before = energy_above_ceiling(audio_before, material_bw_ceiling_hz, sr)
ceiling_band_energy_after  = energy_above_ceiling(audio_after,  material_bw_ceiling_hz, sr)
ceiling_band_ratio = ceiling_band_energy_after / max(ceiling_band_energy_before, 1e-10)
# Violation: ceiling_band_ratio > 8.0 ≈ +9 dB Anstieg im Ceiling-Band → Hard-Rollback

# spectral_novelty (Breitband, sekundär):
spectral_novelty = energy_new_bins / energy_total

if spectral_novelty > 0.08:
    phase_score_penalty = 0.3   # PMGG-Penalty
if ceiling_band_ratio > 8.0:
    # BW-Ceiling-Verletzung durch Energie-Anstieg im Ceiling-Band → Hard-Rollback
    return pre_phase_audio, {"hallucination_rollback": True, "bw_ceiling_ratio_rollback": True,
                              "ceiling_band_ratio": ceiling_band_ratio}
if spectral_novelty > 0.15:
    if mode == "restoration":
        # Phase-Rollback — Restoration ist absolut
        return pre_phase_audio, {"hallucination_rollback": True, "spectral_novelty": spectral_novelty}
    else:
        # Studio 2026: MUSHRA-Check entscheidet
        if mushra_score < 3.5:
            return pre_phase_audio, {"hallucination_rollback": True}

# Hard-Rollback unabhängig von spectral_novelty:
if harmonic_ceiling_violation:   # rekonstruierte Harmonics > material BW_CEILING
    return pre_phase_audio, {"bw_ceiling_rollback": True}
```

**Invarianten**:

- Gate wird nach jeder additiven Phase (family `"additive"` oder `"reconstruction"`) aufgerufen
- Restoration: `spectral_novelty > 0.15` → Hard-Rollback ohne Ausnahme
- Studio 2026: Flexibel, aber MUSHRA ≥ 3.5 pflicht für Phase-Accept mit `spectral_novelty > 0.15`
- BW-Ceiling-Verletzung → Hard-Rollback in beiden Modi
- Telemetrie: `metadata["hallucination_rollbacks"]` (Liste betroffener Phase-IDs)

## §2.46f [RELEASE_MUST] Natural-Performance-Artifacts-Guard (v10.0.0)

**Performancebedingte Klangereignisse sind keine Defekte und dürfen nicht entfernt werden.**

### Drei geschützte Kategorien

| Kategorie | Erkennungskriterien | Schutzregel |
| --- | --- | --- |
| **Atemgeräusche** | Energie −55 bis −40 dBFS, Dauer 50–500 ms, spectral_flatness > 0.4, Silero-VAD off | NR-Bypass + Gate-Bypass für dieses Segment |
| **Natürliches Vibrato/Portamento** | F0-Varianz 4–7 Hz, Amplitude ≤ ±50 Cent | Pitch-Phase überspringt Segment; keine Quantisierung |
| **Studio-Early-Reflections** | C80-Proxy > 3 dB in Onset-Fenstern (0–50 ms, §4.5c) | Dereverb wet_mix cap = 0.35; Early Reflections werden nicht entfernt |

### Implementierung (`natural_performance_detector.py`)

```python
def classify_segment(audio_segment, sr, onset_context=False):
    """Returns protection flags for a short audio segment."""
    result = {
        "is_breath": False,
        "is_vibrato": False,
        "has_early_reflection": False,
    }
    energy_dbfs = 20 * np.log10(np.max(np.abs(audio_segment)) + 1e-9)
    flatness = spectral_flatness(audio_segment, sr)

    if -55 <= energy_dbfs <= -40 and flatness > 0.4:
        result["is_breath"] = True

    f0_mod_rate = estimate_f0_modulation_rate(audio_segment, sr)
    f0_mod_cents = estimate_f0_modulation_amplitude_cents(audio_segment, sr)
    if 3.5 <= f0_mod_rate <= 8.0 and f0_mod_cents <= 50:
        result["is_vibrato"] = True

    if onset_context:
        c80 = compute_c80_proxy(audio_segment, sr)
        if c80 > 3.0:
            result["has_early_reflection"] = True

    return result
```

**Integration in UV3**: Vor jeder NR/Pitch/Dereverb-Phase wird `classify_segment()` frameweise aufgerufen. Flags aus `classify_segment()` werden als `_protected_segments` in `phase_context` weitergegeben und von den betroffenen Phasen ausgewertet.

> Kreuzreferenz: Spec 04 §4.5c (Early-Reflection-Guard); §2.46 (Carrier-Chain-Inversion)

## §2.47 [RELEASE_MUST] Adaptive-Intelligence-Prinzip (v10.0.0)

Aurik verarbeitet **kein generisches Audio** — jede Eingabe ist ein einzigartiges Musikstück. Das System muss sich **vor Beginn der Verarbeitung** vollständig an das konkrete Material anpassen.

### Adaptions-Kaskade (kanonische Reihenfolge)

```text
1. MediumDetector.detect()      → transfer_chain, primary_material, composite flag
2. EraClassifier.classify()     → decade, era_profile, vintage_aesthetics
3. GenreClassifier              → genre_label, RESTORATION_PROFILE (5 definierte + DEFAULT)
4. RestorabilityEstimator       → 0–100, tier (GOOD/FAIR/POOR/EXTREME), scale_factor
5. DefectScanner.scan()         → 46 defect_types × severity × locations
6. CausalDefectReasoner         → 49 Ursachen → Phase-Selektion (CAUSE_TO_PHASES)
7. SongCalibrationProfile       → family_scalars [0.30–1.80] + global_scalar [0.50–1.50]
8. SongGoalImportance (§2.56)   → 14 Per-Song-Gewichte [0.3–2.0] aus 5 Stufen
                                   (Label/Audio/Psychoakustik/Vokal-Harmonik/Interactions)
9. GPOptimizer.propose()        → Pareto-optimale Hyperparameter (14-D MOO)
```

**Resultat**: Dieselbe Pipeline verarbeitet Schellack 1928 (SNR 15 dB, BW 7 kHz, Mono) fundamental anders als CD 2005 (SNR 60 dB, BW 20 kHz, Stereo) — ohne manuellen Eingriff.

### GP-Wissenstransfer (v10.0.0)

- GPOptimizer persistiert Beobachtungen pro `gp_memory_key` (Genre × Material)
- **Cross-Material-Generalisierung**: Bei < 10 Beobachtungen für ein neues Material werden Hyperparameter-Priors (Kernel-Lengthscale, Signal-Varianz) aus dem nächstverwandten Material initialisiert gemäß Material-Ähnlichkeitsmatrix (siehe unten)
- **Anti-Overfitting**: `global_scalar ∈ [0.30, 1.80]` begrenzt GP-Vorschläge; Extreme führen zu Conservative-Fallback
- **Batch-Konvergenz**: Bei sequenzieller Verarbeitung mehrerer Dateien gleichen Materials konvergieren GP-Priors → spätere Dateien profitieren von früheren Ergebnissen

### Material-Ähnlichkeitsmatrix (v10.0.0)

Definiert die Transferierbarkeit von GP-Priors zwischen Materialien. Wert = Ähnlichkeit [0, 1]. Bei < 10 GP-Beobachtungen wird der Prior vom Material mit höchstem Ähnlichkeitswert übernommen.

```text
                  shellac  wax_cyl  vinyl_78  vinyl_std  tape_std  tape_stu  cassette  digital  mp3_lossy
shellac             1.00    0.85     0.75      0.40       0.15      0.10     0.10      0.05     0.05
wax_cylinder        0.85    1.00     0.70      0.35       0.10      0.10     0.08      0.05     0.05
vinyl_78rpm         0.75    0.70     1.00      0.65       0.20      0.15     0.15      0.08     0.08
vinyl_standard      0.40    0.35     0.65      1.00       0.45      0.40     0.35      0.15     0.12
tape_standard       0.15    0.10     0.20      0.45       1.00      0.85     0.70      0.25     0.20
tape_studio         0.10    0.10     0.15      0.40       0.85      1.00     0.60      0.35     0.25
cassette            0.10    0.08     0.15      0.35       0.70      0.60     1.00      0.20     0.18
digital_pcm         0.05    0.05     0.08      0.15       0.25      0.35     0.20      1.00     0.55
mp3_lossy           0.05    0.05     0.08      0.12       0.20      0.25     0.18      0.55     1.00
```

**Nutzung bei Cross-Material-Init**:

1. Sortiere Materialien nach Ähnlichkeit absteigend
2. Wähle das ähnlichste Material mit ≥ 10 GP-Beobachtungen
3. Übernimm dessen Kernel-Lengthscale × `(1 / similarity)` (= höhere Unsicherheit bei geringerer Ähnlichkeit)
4. Übernimm Signal-Varianz × `similarity` (= gedämpfter Prior bei geringerer Ähnlichkeit)
5. Bei `similarity < 0.3` → kein Transfer, nur GP-Default-Priors (uninformativ)

### ML-Failure-Degradations-Kaskade (v10.0.0)

Wenn ein ML-Plugin nicht geladen werden kann (OOM, korruptes Modell, ONNX-Fehler), **muss** die Pipeline graceful degradieren statt abzubrechen:

| Failure | Primär-Fallback | Sekundär-Fallback |
| --- | --- | --- |
| DeepFilterNet OOM | OMLSA/IMCRA (§4.5 Spec 04) | Spectral-Gating (Dry-Signal wenn SNR > 35 dB) |
| MDX23C Stem-Sep OOM | NMF-β-Separation (sklearn, β=Itakura-Saito; sdB ≥ 5 Proxy-SDR-Check) | HPSS (librosa.effects.hpss, tertiärer Fallback) |
| AudioSR OOM | Harmonische Oberton-Synthese + PGHI-Phasenrekonstruktion | Spectral-Band-Replication (SBR) |
| MP-SENet OOM (phase_43, ML-De-Esser-Kontext) | OMLSA/IMCRA DSP (Cohen & Berdugo 2002; §4.4) | Bypass (phase_43 Phase-Skip) |
| CREPE Pitch-Track | pYIN (Mauch & Dixon 2014) | YIN (de Cheveigné & Kawahara 2002) |
| MertPlugin OOM | DSP-Analyse: F0+Harmonizität+SpektralFlux-Kohärenz (besser als MFCC) | Bypass (HPI ohne MERT-Anteil) |

**Invariante**: Kein ML-Failure darf die Pipeline vollständig abbrechen. Jede Phase **muss** einen DSP-Fallback haben (§4.4 Spec 04). Der Fallback wird in `RestorationResult.metadata["ml_fallbacks_used"]` protokolliert.

### §2.47 Adaptions-Erweiterungen (v10.0.0)

Vier neue Intelligence-Hebel ergänzen die Adaptions-Kaskade. Sie sind **nach** dem GP-Optimizer aktiv und erhöhen die perceptuelle Präzision ohne neue ML-Modelle.

#### Hebel 1 — Salience-aware PhaseSkipping (`_salience_adjusted_severity`)

`_apply_phase_skipping` in UV3 liest **keine rohe `DefectScore.severity`** mehr direkt, sondern ruft `_salience_adjusted_severity(defect_type)` auf:

```python
def _salience_adjusted_severity(defect_type: str) -> float:
    ds = defect_result.scores.get(defect_type)
    if ds is None:
        return 0.0
    sev = float(ds.severity)               # ERB-adjustiert durch PerceptualSalienceEstimator
    meta = getattr(ds, "metadata", {}) or {}
    n_masked   = int(meta.get("n_masked_events", 0))
    n_salient  = int(meta.get("n_salient_events", 0))
    if n_masked >= 3 and n_salient == 0:   # vollständig ERB-maskiert
        sev *= 0.5                          # zusätzlich -50 % → Phase meist inaktiv
    return sev
```

**Rationale**: Defekte, die durch simultane Maskierung unhörbar sind (ERB-Maskierungskurve), sollen keine Phase einschalten — §0 Minimal-Intervention. Das ERA-Flag `n_masked_events`/`n_salient_events` im `DefectScore.metadata` stammt aus dem `PerceptualSalienceEstimator` (§2.47 Schritt 5).

**Invariante**: Eine Phase, die durch reine Severity-Kalkulation aktiviert würde, aber nur vollständig maskierte Defekte adressiert, wird übersprungen (`_skip_phase()`) — kein Klangschaden durch unnötige Verarbeitung.

#### Hebel 2 — SGMSE+ Tier-0 in `phase_03_denoise` (Richter et al. 2022)

Score-based Diffusion-Denoising als **erster** Processing-Pfad vor dem bisherigen ML-Hybrid-Pfad:

```text
Tier 0  SGMSE+ (diffusion)    — Bedingungen: quality_mode ∈ {quality, maximum}
                                              + (vocal_genre OR panns_singing_confidence ≥ 0.30)
                                              + NOT digital (cd_digital, dat, minidisc)
                                              + NOT use_lightweight
Tier 1  ML-Hybrid (DeepFilterNet/MP-SENet + OMLSA)  — bisheriger Hauptpfad
Tier 2  OMLSA/IMCRA                                 — DSP-Fallback
Tier 3  Spectral-Gating                             — letzter Ausweg
```

**Tier-0-Auslöser**: Vokalmusik profitiert überproportional von Diffusion-Denoising, weil SGMSE+ die Lernverteilung natürlicher Sprachlaute als implizites Prior nutzt — Formanttreue bleibt erhalten. Bei nicht-vokalen Genres oder Digital-Material überwiegt das ML-Hybrid-Verfahren (deterministischer, kein Over-Smoothing).

**Metadata-Markierung**: `phase_result.metadata["sgmse_plus_tier0_applied"] = True` bei Tier-0-Nutzung.

#### Hebel 3 — PhaseConductor (inter-phase adaptive Strength)

Vollständiger Workflow und Invarianten: siehe §2.52 dieses Dokuments.

**Einbettungspunkt**: UV3 `_execute_pipeline`, sequentiell nach §2.31a MidCalibrate-Block.

#### Hebel 4 — Carrier-Formant-Decay-Inversion in `phase_42` (Stage 0.5)

Analoge Tonträger dämpfen charakteristisch den Formantbereich durch mechanische und magnetische Transfer-Verluste:

```python
def _restore_carrier_formant_decay(audio, sr, material_type):
    """Stage 0.5: Invertiert träger-spezifische F1–F4-Unterdrückung via zero-phase Bell-EQ."""
```

Trägertypische Bell-EQ-Profile (Gain in dB, Zentrum-Hz, Q):

| Material | F1-Boost | F2-Boost | F3-Boost | F4-Boost |
| --- | --- | --- | --- | --- |
| vinyl | +0.8 dB @ 800 Hz, Q=2.0 | +1.2 dB @ 1800 Hz, Q=2.5 | +0.6 dB @ 3200 Hz, Q=3.0 | +0.4 dB @ 4500 Hz, Q=3.5 |
| reel_tape | +1.0 dB @ 750 Hz, Q=1.8 | +1.5 dB @ 1700 Hz, Q=2.2 | +0.8 dB @ 3000 Hz, Q=2.8 | — |
| tape | +0.6 dB @ 800 Hz, Q=2.0 | +1.0 dB @ 1800 Hz, Q=2.5 | +0.5 dB @ 3200 Hz, Q=3.0 | — |
| shellac | +2.0 dB @ 600 Hz, Q=1.5 | +3.0 dB @ 1500 Hz, Q=2.0 | +1.5 dB @ 2800 Hz, Q=2.5 | — |
| minidisc | +0.4 dB @ 850 Hz, Q=2.5 | +0.8 dB @ 1900 Hz, Q=3.0 | +0.3 dB @ 3400 Hz, Q=3.5 | — |
| cd_digital | passthrough (kein Formant-Decay) | — | — | — |

**Implementierung**: `scipy.signal.filtfilt` (zero-phase, IIR-Biquad-Peaking) pro Formant. Kein Phasen-Artefakt, kein Pre-Ringing. Stage 0.5 läuft **vor** Stage 1 (Pitch-Korrektur) in `_enhance_channel(audio, sr, material_type=material)`.

> Kreuzreferenz: §2.52 (PhaseConductor), §2.46 (Carrier-Chain-Inversion), Spec 06 §7.4

## §2.47a [RELEASE_MUST] Frontend-Backend-PreAnalysis-Handover-Vertrag (v10.0.0)

Der PreAnalysis-Handover ist als **direkte Objektübergabe** verpflichtend und nicht
als rekonstruierter Cache-Lookup in asynchronen Threads.

### Invarianten

1. `run_pre_analysis()` läuft pro Import genau einmal.
2. `PreAnalysisResult` wird im Frontend als komplettes Objekt gespeichert und über
    Queue-Settings direkt an den Batch-Worker übergeben.
3. Das konkret verwendete `DefectAnalysisResult` wird immer als
    `cached_defect_result` an `AurikDenker.denke()`/UV3 weitergereicht.
4. Bei neuem File-Import wird der vorherige Cache hart gelöscht.
5. `MediumDetector.detect()` wird pro Datei genau einmal ausgeführt.

Detailarchitektur und Ablaufdiagramm: §2.37 dieses Dokuments.

Referenztest: `tests/unit/test_pre_analysis_handover_no_double_detect.py`

## §2.47b [RELEASE_MUST] JND-Effektivitätsschwelle — Sub-Threshold-Phasen-Markierung (v10.0.0.x)

**Psychoakustische Motivation**: Phasen, deren Musical-Goal-Deltas alle unterhalb der
Hörschwelle (JND = Just Noticeable Difference) liegen, bringen keinen perceptuell messbaren
Klanggewinn. Gleichzeitig erhöhen sie das Artefakt-Risiko (§2.49) und verbrauchen CPU-Budget.

Kalibrierungsbasis: **vollständige Musikmischungen mit Gesang** (Pop, Schlager, Jazz, Folk, Oper).
Werte sind normalisierte Score-Äquivalente der perceptuellen JND für komplexe Musikmischungen —
nicht für isolierte Töne. Primärquellen:

- Thoret, Caramiaux, Depalle & McAdams (2021) **JASA** 149:3429 — Timbre-JND in Musikklängen ≈ 1 %
- Caclin, McAdams, Smith & Winsberg (2005) **JASA** 118:2925 — multidimensionale Timbre-JND
- Kreiman & Sidtis (2011) **Foundations of Voice Studies** — Stimmqualitätserkennung
- Krumhansl & Cuddy (2010) **Psychol Learn Motiv** 51:51 — tonale Hierarchie
- Marjieh, Harrison, Lee, Deligiannaki & Jacoby (2023) **Music Percept.** 40:183 — Schlüssel-Salienz
- London (2012) **Hearing in Time** 2. Aufl. — Timing-JND ~8 ms in Musik
- Repp & Su (2013) **Psychon Bull Rev** 20:403 — sensomotorische Synchronisations-JND
- Juslin (2019) **Musical Emotions Explained** Oxford UP — Vokalemotions-Wahrnehmung
- Witek et al. (2017) **PLOS ONE** 12:e0169907 — Groove-Wahrnehmungssensitivität
- Beranek (2016) **J Acoust Soc Am** 139:1548 — Clarity C80-JND ~1 dB (aktualisierte Studie)
- Toole (2018) **Sound Reproduction** 3. Aufl. — Wahrnehmungsschwellen Lautsprecher/Raum
- Glasberg & Moore (2006) **JASA** 119:1705 — revidiertes Loudness-Modell, LF-Zone
- Bregman (1990) **Auditory Scene Analysis** Ch. 2 — Auditory Stream Segregation
- Blauert (1997) **Spatial Hearing** 2. Aufl. — Präzedenz/Hallnachhall-JND
- Choisel & Wickelmaier (2007) **JASA** 121:2718 — räumlicher Eindruck-JND in Mehrkanal

**JND-Schwellwerte pro Musical Goal**:

```python
# backend/core/per_phase_musical_goals_gate.py
JND_MIN_DELTA: dict[str, float] = {
    # P1 — höchste Salienz; Vokalmusik macht diese besonders dominant
    "natuerlichkeit":        0.012,  # Thoret et al. (2021) JASA 149:3429; Caclin et al. (2005) ≈1 %
    "authentizitaet":        0.012,  # Kreiman & Sidtis (2011): Stimmqualität sehr präzise erkannt
    # P2 — strukturelle Musikeigenschaften; tonaler Schwerpunkt am salientesten
    "tonal_center":          0.008,  # Krumhansl & Cuddy (2010); Marjieh et al. (2023): höchste Salienz
    "timbre_authentizitaet": 0.012,  # Caclin et al. (2005); McAdams (2019) Curr Biol 29:R764
    "artikulation":          0.010,  # London (2012) 2. Aufl. ~8 ms; Repp & Su (2013) Psychon
    # P3 — emotionale Hinweise in Stimme sehr präsent (100–300 ms Zeitskala)
    "emotionalitaet":        0.014,  # Juslin (2019) OUP; Zentner et al. (2008) Emotion 8:494
    "micro_dynamics":        0.012,  # Glasberg & Moore (2002) J AES 50:331 zeitvariantes JND
    "groove":                0.010,  # Witek et al. (2017) PLOS ONE; Madison (2006) ≈6 ms
    # P4 — tonale Balance/Raum; längere Integrationszeitfenster, aber sensitiver als vermutet
    "transparenz":           0.012,  # Beranek (2016) JASA 139:1548 C80-JND ~1 dB; Toole (2018)
    "waerme":                0.016,  # Alluri & Toiviainen (2012) Music Percept. 29:459; Howard & Angus (2017)
    "bass_kraft":            0.012,  # Glasberg & Moore (2006) JASA 119:1705; ISO 226:2003
    "separation_fidelity":   0.014,  # Bregman (1990) Auditory Scene Analysis; McDermott (2009) Curr Biol
    # P5 — spektrale Brillanz/Raumtiefe; breiteste Integrationsfenster
    "brillanz":              0.016,  # Siedenburg & McAdams (2017) J New Music Res 46:149
    "spatial_depth":         0.018,  # Blauert (1997) 2. Aufl.; Choisel & Wickelmaier (2007) JASA 121:2718
}
```

**Algorithmus in `_run_with_retry()` (NACH Delta-Berechnung, VOR Retry-Logik)**:

```python
# §2.47b JND Sub-Threshold Check
_applicable = [g for g in applicable_goals if g not in excluded_goals]
_deltas = {g: scores_after.get(g, 0.0) - effective_scores_before.get(g, 0.0) for g in _applicable}
_all_below_jnd = (
    len(_deltas) > 0
    and all(d >= 0.0 for d in _deltas.values())          # nur Verbesserungen
    and all(abs(d) < JND_MIN_DELTA.get(g, 0.015) for g, d in _deltas.items())
)
if _all_below_jnd:
    metadata.setdefault("sub_threshold_phases", []).append(phase_id)
    logger.debug("phase=%s sub_threshold: all deltas < JND, skipping retry", phase_id)
    return audio_out, scores_after, "sub_threshold", wet_ratio
```

**Invarianten**:

- Sub-Threshold → **kein Rollback, kein Retry** — Audiomodifikation wird beibehalten
- Nur auslösbar wenn ALLE Deltas ≥ 0 (keine Regression vorhanden)
- Phasen mit Regression → normale PMGG-Retry-Logik, unabhängig von JND
- `restorative_phases` (§2.29c): Sub-Threshold auch dort anwendbar, JND-Messung auf `effective_scores_before` basieren
- Telemetrie: `RestorationResult.metadata["sub_threshold_phases"]` (liste der Phase-IDs)
- VERBOTEN: Sub-Threshold-Check als Begründung nutzen, um `_MATERIAL_PRIORITY_PHASES` (§6.2a) zu überspringen

> Implementierung: `backend/core/per_phase_musical_goals_gate.py` — `JND_MIN_DELTA` + `_run_with_retry()`
> Referenztest: `tests/unit/test_jnd_sub_threshold.py`

## §2.48 [RELEASE_MUST] Kumulative-Phasen-Interaktions-Guard (v10.0.0, aktualisiert v10.0.0)

Einzelne Phasen können isoliert korrekt arbeiten, aber in Kombination destruktive Effekte erzeugen (z.B. De-Noise + De-Reverb entfernen gemeinsam mehr Raumklang als beabsichtigt).

> **§2.54 ist übergeordnet**: Der Guard ist eine **Notbremse** (letztes Sicherheitsnetz),
> nicht die Routine-Steuerung der Pipeline. Die Routine-Steuerung liegt bei PhaseConductor (§2.52),
> PMGG (§2.29) und SongCalibration (§2.47). Drift-Toleranzen werden **berechnet**, nicht als Konstanten definiert.

### Kumulative P1/P2-Drift-Messung

Nach jeder Phase wird die **kumulative** Gesamt-Regression der P1/P2-Goals (Natürlichkeit, Authentizität, TonalCenter, Timbre, Artikulation) gemessen — nicht nur die Delta-Regression der Einzelphase.

```python
# §2.54 Adaptive Drift-Toleranz (ersetzt feste -0.05-Konstante)
# In _execute_pipeline(), nach jeder Phase:
goals_now = musical_goals_checker.evaluate(current_audio, sr)
cumulative_drift = {g: goals_now[g] - goals_pre_pipeline[g] for g in P1_P2_GOALS}

# §2.48 Carrier-Repair-Exclusions (§2.44 Referenz-Paradoxon):
# Phasen, die Tonträgerschäden invertieren, dürfen authentizitaet/artikulation/
# timbre_authentizitaet vorübergehend senken — das ist intentional, kein Schaden.
effective_drift = apply_phase_specific_exclusions(cumulative_drift, phase_id)

# Drift-Toleranz materialadaptiv berechnen (§2.54):
tolerance = compute_adaptive_drift_tolerance(
    restorability_score, material_type, defect_severity_mean, n_active_phases
)
# Ergebnis: z.B. -0.03 (CD, leicht) bis -0.25 (Shellac-4-Gen, schwer degradiert)

if any(drift < tolerance for drift in effective_drift.values()):
    current_audio = best_perceptual_checkpoint_audio  # Rollback auf BESTES Audio
    logger.warning("phase=%s cumulative_drift=%s tol=%.3f → rollback", phase_id, effective_drift, tolerance)
```

### Carrier-Repair-Phasen-Ausnahmen (§2.44 Referenz-Paradoxon)

Phasen, die Tonträgerschäden invertieren, verändern Chroma/Centroid-Signaturen intentional gegenüber
dem beschädigten Checkpoint. Ein Metrik-Drop gegenüber dem beschädigten Referenzpunkt bedeutet nicht
„Verschlechterung", sondern „das Signal entfernt sich vom Defekt" — genau das ist das Ziel.

| Phase | Ausgeschlossene Goals | Grund |
| --- | --- | --- |
| phase_01, phase_09, phase_27 | authentizitaet, artikulation, timbre_authentizitaet | Click/Crackle-Removal ändert Transient-Profil |
| phase_28, phase_03, phase_29 | authentizitaet, timbre_authentizitaet | Breitband-Rauschentfernung ändert Spektral-Fingerprint |
| phase_12 | authentizitaet, natuerlichkeit, artikulation | Wow/Flutter-Korrektur verschiebt Chromagram |
| phase_24 | authentizitaet, artikulation, natuerlichkeit | Dropout-Repair füllt Lücken mit neuem Content |
| phase_55 | authentizitaet | Diffusion-Inpainting rekonstruiert maskierte Bereiche |

### Kritische Interaktions-Paare (bekannte destruktive Kombinationen)

| Paar | Risiko | Guard |
| --- | --- | --- |
| `phase_03 (De-Hiss) + phase_20/49 (De-Reverb)` | Kumulative Raumklang-Entfernung | Nach De-Reverb: Natürlichkeit ≥ pre_pipeline − 0.03 |
| `phase_29 (NR) + phase_03 (De-Hiss)` | Over-Denoising | Nach zweiter NR-Phase: Rauschboden ≥ Material-Ziel (§0a) |
| `phase_35 (Multiband-Compression) + phase_40 (LUFS-Norm.)` | Dynamik-Verlust | Nach LUFS: MikroDynamik ≥ pre_pipeline − 0.04 |
| `phase_07 (Harmonic-Restoration) + phase_42 (Vocal-AI)` | Frequenz-Doppelung | Nach Vocal-AI: Spectral-Flatness-Check |
| `phase_23/24 (Super-Resolution) + phase_03 (De-Hiss)` | Künstliche Obertöne entrauscht | Super-Res immer VOR De-Hiss (Reihenfolge-Invariante) |

### Kumulative STFT-Phasenkohärenz

Mehrfache STFT→Modifikation→ISTFT erzeugt akkumulierte Phasenfehler (Gruppenlaufzeit-Deviation, Phase-Smearing bei Transienten). Dies ist kein Goal-messbarer Effekt, sondern ein rein technischer Fehler.

**Prüfung**: Nach ≥ 3 STFT-basierten Phasen in Folge:

- `group_delay_deviation = max(|τ_current(f) - τ_original(f)|)` über alle Frequenz-Bins
- Schwellwert: ≤ 5 ms (entspricht ~240 Samples bei 48 kHz)
  - Begründung v10.0.0: 2 ms war unrealistisch. Standard-2048-Punkt-STFT bei 48 kHz hat bereits 42,6 ms Fensterlänge (10,7 ms Hop). Spektralsubtraktions-Filter verschieben pro-Bin-Phase lokal 3–8 ms ohne hörbare Artefakte. Ab 5 ms liegt ein echtes Phase-Distorsions-Problem vor (typisch: unabhängige L/R-IIR-Filter oder falsch kaskadierte STFT-Ketten).
- Überschreitung → letzte STFT-Phase rollback, Alternative ohne STFT versuchen (z.B. PGHI statt GriffinLim, Zero-Phase-Filterung statt STFT-Modifikation)

**Betroffene Phasen** (STFT-basiert): phase_03 (De-Hiss), phase_07 (Harmonic), phase_20/49 (De-Reverb), phase_23/24 (Super-Resolution), phase_29 (NR), phase_35 (Multiband-Comp)

### [RELEASE_MUST] Checkpoint-Verwaltung (§2.54-konform)

- `best_perceptual_checkpoint`: Audio-Snapshot mit dem **höchsten gewichteten P1–P5-Score** über alle bereits akzeptierten Phasen — nicht das **letzte nicht-gerollte**, sondern das perceptuell **beste**
- Bei Rollback: Phase-Skip protokollieren in `RestorationResult.metadata["interaction_rollbacks"]`
- Nach Rollback: nächste Phase erhält `best_perceptual_checkpoint`-Audio
- **Pipeline-Stopp adaptiv**: `max_consecutive_rollbacks = max(5, n_carrier_phases + 2)` — Mehrgenerations-Material (vinyl→tape→mp3) benötigt mehr Carrier-Phasen, die einzeln rollback-anfällig sind. `should_stop` erst wenn NACH materialadaptiver Berechnung die Notbremse-Schwelle gerissen wird UND keine bessere Stärke gefunden wurde.
- **VERBOTEN**: `Max 2 aufeinanderfolgende Rollbacks → Pipeline-Stop` als feste Konstante — das war der Haupt-Bug, der bei Mehrgenerations-Material zu Pipeline-Abbruch nach DC-Offset-Checkpoint führte.

### Phasen-Reihenfolge-Optimierung

CAUSE_TO_PHASES wählt **welche** Phasen aktiv sind. Die **Reihenfolge** der aktiven Phasen folgt der **Carrier-Chain-Inversions-Logik** (§2.46):

1. **ADC-Stufe**: DC-Offset, Quantisierungs-Artefakte (phase_01, phase_31)
2. **Playback-Stufe**: RIAA-Inverse, Azimuth, Wow/Flutter, Speed-Korrektur (phase_06, phase_09, phase_10)
3. **Alterungs-Stufe**: Click/Pop, Dropout, Knistern (phase_02, phase_04, phase_05, phase_11)
4. **Carrier-Encoding-Stufe (subtraktiv)**: NR, De-Hiss, De-Reverb (phase_03, phase_29, phase_20/49)
5. **Carrier-Encoding-Stufe (additiv)**: Super-Resolution, Harmonic-Restoration, Bandwidth-Extension (phase_23, phase_24, phase_07)
6. **Enhancement-Stufe**: Vocal-AI, Stem-Sep, Dynamics, EQ, LUFS (phase_42, phase_35, phase_40)

**Invariante**: Subtraktive Phasen VOR additiven — sonst werden rekonstruierte Obertöne sofort wieder entrauscht.

> Kreuzreferenz: §2.29d (P1/P2 = Pipeline-Ende-Pflicht, §2.54), §2.45 (perceptual_delta), §2.44 (HPI)

## §2.48a [RELEASE_MUST] Phase-Typ-Ontologie — Architektur-Inversion (v10.0.0)

### Prinzip

**Guards dürfen nur feuern, wenn ihre Messvoraussetzung strukturell erfüllt ist** — abgeleitet aus dem intrinsischen Operationstyp der Phase, nicht aus Ausnahmelisten.

Das bisherige Muster (Ausnahmeliste) ist nicht skalierbar: Jede neue Phase braucht manuellen Eintrag in `_RESTORATIVE_PHASE_IDS`, `STFT_PHASES`, `PHASE_GOAL_EXCLUSIONS`. Fehlt ein Eintrag, feuert der Guard falsch → Rollback auf verbessertes Audio.

**Lösung**: `backend/core/phase_ontology.py` definiert `PhaseOperationType` als Enum. Jeder Guard konsultiert den Typ und entscheidet strukturell, ob seine Messung valide ist.

### Phase-Operationstypen (normativ)

| Typ | Beschreibung | Beispiele |
| --- | --- | --- |
| `SUBTRACTIVE` | Entfernt Rauschen/Artefakte | phase_03, phase_09, phase_18, phase_20, phase_27, phase_28, phase_29, phase_49, phase_50 |
| `ADDITIVE` | Fügt neue Signalkomponenten hinzu | phase_06, phase_07, phase_21, phase_22, phase_37, phase_38, phase_39 |
| `CORRECTIVE` | Korrigiert spektrale/zeitliche Eigenschaften | phase_04, phase_12, phase_14, phase_25, phase_30, phase_31, phase_41 |
| `ML_GENERATIVE` | ML-Diffusion/Flow-Matching (kein STFT-kohärenter Ausgang) | phase_42, phase_55, phase_36, phase_64 |
| `DYNAMICS` | Hüllkurven-Verarbeitung | phase_08, phase_10, phase_11, phase_17, phase_19, phase_26, phase_35, phase_40, phase_47 |
| `ANALYSIS_ONLY` | Kein Audio-Output | phase_53 |
| `ENHANCEMENT` | Mix/nicht eindeutig | phase_13, phase_32, phase_46, phase_48, phase_58 |

### Guard-Applicability-Matrix (normativ)

| Guard | Valide für Typen | Invalide für Typen | Wissenschaftliche Grundlage |
| --- | --- | --- | --- |
| **Noise-Texture-Check** (§2.49) | `SUBTRACTIVE` | alle anderen | Schwarz & Grill 2004: BW-Erweiterung verändert Spektral-Tilt intentional |
| **Pre-Echo-Detektor** (§2.49) | `DYNAMICS`, `ENHANCEMENT` | `SUBTRACTIVE`, `ADDITIVE`, `CORRECTIVE`, `ML_GENERATIVE` | Brandenburg & Johnston 1994: Pre-Echo ist ausschließlich Transform-Coding-Artefakt; Residual subtraktiver Phasen ≠ Prä-Transient-Energie |
| **GDD-Check** (§2.48) | `SUBTRACTIVE`, `DYNAMICS`, `ENHANCEMENT` | `ML_GENERATIVE`, `ADDITIVE` | Richter et al. (SGMSE+, TASLP 2022): Diffusionsausgang nicht STFT-phasenkohärent; Synthese erzeugt neue Bins mit eigener Phase |
| **Baseline-Capping** §2.29c | `SUBTRACTIVE` | alle anderen | ITU-R BS.1387 §4.2: Rauschresidual ist kein Artefakt; defekt-inflationierte Baseline ist strukturelles Merkmal subtraktiver Phasen |
| **P1/P2-Drift-Check** (§2.48) | alle außer `ANALYSIS_ONLY` | `ANALYSIS_ONLY` | Audio unverändert → Drift trivial 0.0 |

### Implementierung

```python
# backend/core/phase_ontology.py — normatives Register
from backend.core.phase_ontology import get_phase_type, GDD_VALID_TYPES, NOISE_TEXTURE_VALID_TYPES, PRE_ECHO_VALID_TYPES

# Guard-Entscheidung (Inversion):
phase_type = get_phase_type(phase_id)
if phase_type in NOISE_TEXTURE_VALID_TYPES:      # nur SUBTRACTIVE
    check_noise_texture(...)
if phase_type in PRE_ECHO_VALID_TYPES:           # nur DYNAMICS + ENHANCEMENT
    check_pre_echo(...)
if phase_type in GDD_VALID_TYPES:                # nicht ML_GENERATIVE, nicht ADDITIVE
    check_group_delay(...)
```

**Invariante**: `phase_ontology.py` IST die Wahrheit. Alle Guards leiten ihre Exemptionen ab — keine doppelte Pflege von Ausnahmelisten.

> Implementierung: `backend/core/phase_ontology.py` — `PhaseOperationType`, `get_phase_type()`, Guard-Applicability-Sets.
> Konsumenten: `artifact_freedom_gate.py`, `cumulative_interaction_guard.py`, `per_phase_musical_goals_gate.py`.

## §2.49 [RELEASE_MUST] Artefakt-Freiheits-Gate (v10.0.0)

Quelle: `[SRC:S03,S04,S12,S13]`

Dediziertes Gate für **Artefakt-Erkennung** — unabhängig von den 15 Musical Goals. Eine Phase kann alle Goals bestehen und trotzdem hörbare Artefakte erzeugen.

### Geprüfte Artefakte

| Artefakt | Erkennungsmethode | Schwellwert |
| --- | --- | --- |
| Musical Noise | Spectral-Variance in Stille-Segmenten: isolierte tonale Peaks (> 12 dB über Nachbarn) in Stille/Pausen | 0 Events |
| Pre-Echo | Transient-Onset-Analyse: Energie in 5-ms-Fenster vor Attack ≤ −40 dB relativ zum Attack-Peak | 0 Events |
| Spectral Holes | Bandbreiten-Kontinuitäts-Check: keine Energielücken > 200 Hz im erwarteten Passband (SourceFidelity BW) | 0 Holes |
| Phase-Cancellation | M/S-Korrelation nach Stereo-Processing: `correlation(M, S) ≥ 0.3` (Mono-Kompatibilität) | ≥ 0.3 |
| Metallic Ringing | CQT-Peak-Detection: isolierte resonante Peaks > 6 dB über Nachbar-Bins, Dauer > 50 ms | 0 Events |

### Material-adaptive Schwellwert-Skalierung (v10.0.0)

Feste Schwellwerte führen zu Fehlalarmen bei historischem Material (z.B. Schellack-Oberflächen-Rauschen als "Musical Noise" fehlklassifiziert) oder zu Durchlassfehlern bei Digital-Material. Deshalb werden die Artefakt-Schwellwerte **material-adaptiv** skaliert:

| Artefakt | Digital/CD | Tape | Vinyl | Shellac/Wax |
| --- | --- | --- | --- | --- |
| Musical Noise (Peak-dB) | > 12 dB | > 15 dB | > 18 dB | > 22 dB |
| Pre-Echo (Rel. Attack) | ≤ −40 dB | ≤ −35 dB | ≤ −30 dB | ≤ −25 dB |
| Spectral Holes (Lücke) | > 200 Hz | > 300 Hz | > 400 Hz | > 600 Hz |
| Phase-Cancellation (mono_compat) | ≥ 0.30 | ≥ 0.20 | ≥ 0.20 | ≥ 0.15 |
| Metallic Ringing (Peak-dB) | > 6 dB | > 8 dB | > 10 dB | > 14 dB |

**Logik**: Historische Träger haben inhärent höhere Artefakt-Pegel im Eingangssignal. Was bei einer CD ein klarer Verarbeitungsfehler ist (Musical-Noise-Peak +12 dB), ist bei Shellac Teil des Trägerprofils. Die Erkennung muss nur **neue, durch Verarbeitung eingeführte** Artefakte finden — nicht die vorhandenen des Trägers.

**Direktionalitätspflicht für Musical-Noise-Detektor** (v10.0.0): Subtractive Phasen (Surface-Noise-Profiling, Denoise, Click-Removal) erzeugen ein Residual `restored − orig` dessen Spektrum die **entfernten** Artefakte spiegelt — nicht neu hinzugefügte. Die Spektralpeaks im Residual sind korrekte Entfernungen, keine Artefakte. Implementierungspflicht:

```python
# Nur flaggen wenn restored_spectrum[j] > orig_spectrum[j] × 1.05
# (Energie wurde ADDIERT, nicht subtrahiert)
if rest_spectrum[j] <= orig_spectrum[j] * 1.05:
    continue  # subtractive action — correct removal, not an artefact
```

Ohne diese Prüfung: Surface-Noise-Profiling erzeugt 50 False-Positive-Artefakte → `artifact_freedom=0.000` → Rollback-Loop → Pipeline-Blockade.

**Phase-Cancellation Detektor — Präzisierungen (v10.0.0)**:

Der Phase-Cancellation-Detektor vergleicht im per-phase-Modus die Stereo-Metrik **vor und nach** der Phase (Delta-Check). Folgende Regeln sind **normativ verbindlich**:

1. **Anti-Korrelation-Schwelle**: `lr_corr < −0.20` (nicht `< 0.0`). Werte zwischen 0 und −0.20 entstehen durch STFT-Window-Misalignment, Gate-Transient-Asymmetrie und normale Verarbeitungsunterschiede — sie sind **nicht hörbar** und dürfen nicht als Phase-Cancellation gezählt werden.

2. **Delta-Guard**: Eine Phase wird nur geflaggt, wenn `orig_compat − restored_compat > 0.10`. Kleinere Asymmetrien (< 0.10) durch DSP-Implementierungsdetails (Filter-Rounding, Overlap-Grenzen) sind technische Artefakte, keine perceptuell relevanten Stereo-Probleme.

3. **Near-Mono-Guard**: Wenn das Quellmaterial quasi-mono ist (`orig_compat > 0.65`) UND die verarbeitete Version noch moderat mono-kompatibel ist (`restored_compat > 0.40`), ist die Abweichung durch unabhängige Kanalverarbeitung (Noise-Gate Transient, Dropout-Füllung) **nicht hörbar** — skip. Ausnahme: Echter Stereo-Kollaps (`restored_compat ≤ 0.40`) wird trotzdem geflaggt.

4. **Stereo-Collapse-Guard**: Wenn ein Kanal einen RMS-Abfall > 40 dB gegenüber dem Original-Input verzeichnet (z. B. R-Kanal von −18 dBFS auf −∞), wird **ein Artefakt** erzeugt und der Frame-Loop wird übersprungen (globaler Kollaps überwiegt Frame-Level-Analyse). Voraussetzung: Originales Signal hatte RMS > 1e-4 (kein stiller Quellkanal).

**Implementierung**: `artifact_freedom_gate.py → _detect_phase_cancellation()`

**Implementierung**: `artifact_thresholds = BASE_THRESHOLDS × material_tolerance_factor[material]`. Der `material_tolerance_factor` kommt aus dem MediumDetector-Ergebnis (§2.47 Adaptions-Kaskade Schritt 1).

**Selbstkalibrierung**: Bei den ersten 3 Verarbeitungen eines neuen Material-Typs werden Artefakt-Schwellwerte konservativ (= strenger) angesetzt. Nach 3 erfolgreichen Verarbeitungen (artifact_freedom ≥ 0.98): Schwellwerte auf material-adaptive Normalwerte entspannen.

### Rauschtextur-Kohärenz (Restoration-Modus)

Unabhängig von den 5 Artefakttypen: Die **spektrale Form** des Restrauschens (Noise-Floor-Shape) muss dem originalen Trägerprofil entsprechen. Aggressive Denoising hinterlässt oft ein Restrauschen mit falscher spektraler Färbung.

**Messung**: In Stille-Segmenten (≥ 200 ms, RMS < −50 dBFS):

1. Input-Noise-Profile: Spectral-Tilt (lineare Regression über Log-Magnitude-Spektrum)
2. Output-Noise-Profile: gleiche Berechnung
3. `tilt_deviation = |tilt_output - tilt_input|` in dB/Oktave

**Schwellwerte**:

| Abweichung | Aktion |
| --- | --- |
| ≤ 3 dB/Oktave | OK — Restrauschen hat natürliche Textur |
| 3–6 dB/Oktave | Warnung — `artifact_freedom` −0.05 Penalty |
| > 6 dB/Oktave | Rollback auf letzte NR-Phase — unnatürliche Rauschtextur |

**Typische Fehlerbilder**:

- Vinyl-Denoising → weißes Rauschen (statt rosa-Tilt ≈ −3 dB/Oktave): Over-Denoising der tiefen Frequenzen
- Tape-NR → tonales Rauschen (isolierte NR-Residuen): Musical-Noise-Variante
- Shellac → zu "sauberes" Restrauschen: Ambient-Charakter verloren

### Score-Berechnung

```python
artifact_freedom = 1.0 - (weighted_artifact_count / max_tolerance)
artifact_freedom = np.clip(artifact_freedom, 0.0, 1.0)
```

Gewichtung: Musical Noise = 1.0, Pre-Echo = 0.8, Spectral Holes = 0.6, Phase-Cancellation = 1.0, Metallic Ringing = 0.9

**Perzeptuelle Salienz-Gewichtung**: Die obigen Gewichte werden zusätzlich nach perzeptueller Salienz skaliert:

- **Frequenz**: Artefakte im Bereich 200–5000 Hz (höchste Hörempfindlichkeit, ISO 226) erhalten Faktor 1.0; unter 200 Hz oder über 5000 Hz → Faktor 0.5; über 12 kHz → Faktor 0.2
- **Kontext**: Artefakte in Stille/Pausen-Segmenten (RMS < −40 dBFS) erhalten Faktor 1.5 (stärker hörbar); in Tutti-Passagen (RMS > −20 dBFS) → Faktor 0.5 (maskiert)
- **Dauer**: Artefakte > 100 ms erhalten Faktor 1.5; < 20 ms → Faktor 0.5
- Effektiver Score: `salience_weighted_artifact_count = Σ(type_weight × freq_factor × context_factor × duration_factor)`

### Integration

- **Im HPI**: `artifact_freedom` fließt als Multiplikator in beide HPI-Formeln ein (§2.44)
- **Phase-Level**: Nach jeder Phase prüfen — bei `artifact_freedom < 0.95` → Rollback auf `best_artifact_free_checkpoint`
- **Export-Gate**: `artifact_freedom < 0.95` blockiert regulären Success-Export. Es folgt
    verpflichtend die Recovery-Kaskade; Export nur als `recovered`/`degraded` mit vollständiger Ursache.
- **Protokollierung**: `RestorationResult.metadata["artifact_freedom"]` = Score + Detail-Report (detected_artifacts: list)

### §2.49 Finaler Score — Berechnungsregel (v10.0.0)

**`_artifact_freedom_score` = Minimum aller per-Phase-Scores aller akzeptierten Phasen.**

FALSCH (und verboten): `artifact_gate.evaluate(pre_pipeline_audio, pipeline_output)` — jede echte Restaurierung erzeugt dadurch zwangsläufig `artifact_freedom=0.000`, weil intentionale Signalveränderungen (Rauschen entfernen, Bandbreite erweitern) im Vollvergleich als Artefakte erscheinen.

RICHTIG: Per-Phase-Minimum über alle Phasen, bei denen der Gate-Check durchgeführt wurde (`_min_per_phase_afg_score`). Phasen, die ge-rollt-back wurden, fließen nicht ein.

**[BUG-FIX v10.0.0] IAD → artifact_freedom Kopplung** (Bug 4):

`IntroducedArtifactDetector` (IAD, §2.23) läuft post-Pipeline und erkennt ML-Halluzinationen und neu eingebrachte Artefakte über die gesamte Kette. Sein Ergebnis **MUSS** in `artifact_freedom` einfließen — sonst kann `artifact_freedom=1.0` trotz bekannter IAD-Artefakte einen HPG-VETO-Fail verhindern.

```python
# IAD-Penalty-Formel (UV3, nach IAD-Rescue-Blend):
# fraction=0.025 → af_penalty=0.75 → VETO (< 0.95)
# fraction=0.05  → af_penalty=0.50 → VETO
# fraction=0.10+ → af_penalty=0.0  → vollständiger VETO
_iad_af_penalty = max(0.0, 1.0 - total_contaminated_fraction * 10.0)
self._iad_artifact_fraction_penalty = _iad_af_penalty

# Finale Berechnung (ersetzt einfache Zuweisung):
_iad_af_penalty = getattr(self, "_iad_artifact_fraction_penalty", 1.0)
self._artifact_freedom_score = float(min(_min_per_phase_afg_score, _iad_af_penalty))
```

**Invariante**: IAD-Penalty wird gesetzt für **jedes** `has_artifacts=True` — unabhängig vom Rescue-Threshold. Selbst `fraction=0.010` mit Typen `ml_hallucination`/`musical_noise` = `af_penalty=0.90` → VETO (< 0.95). Der Dry/Wet-Rescue-Blend (`_IAD_RESCUE_THRESHOLD=0.05`) ist ein separater Schritt und darf die Penalty-Logik nicht blockieren.

**Gap-Fix v10.0.0**: Die ursprüngliche Implementierung setzte die Penalty nur innerhalb des `if fraction >= 0.05`-Blocks — damit war `fraction=0.025` (Elke-Best-Fall, Typen `ml_hallucination, musical_noise`) vollständig unsichtbar für den HPG-VETO. **VERBOTEN**: Penalty-Berechnung innerhalb eines Rescue-Threshold-Guards.

### §2.49b [RELEASE_MUST] Post-Pipeline Kumulativer Stereo-Collapse-Guard (v10.0.0)

Per-Phase-δ-Guards fangen nur Single-Phase-Kollapsen (> 40 dB in einer Phase). Kumulativer Stereo-Drift — bei dem 4 Stereo-Phasen jeweils 6–8 dB beitragen — bleibt unsichtbar. Lösung: Post-Pipeline-Vergleich gegen Pre-Pipeline-Baseline.

**Invariante** (direkt nach Phase-Loop, vor `_pmgg_log_entries`-Zuweisung):

```python
if current_audio.ndim == 2 and current_audio.shape[0] == 2:
    cu_imb = abs(L/R_dB(current_audio))      # Imbalance Pipeline-Ausgang
    pp_imb = abs(L/R_dB(afg_pre_pipeline))   # Imbalance Pipeline-Eingang
    if cu_imb > 20.0 and pp_imb < 6.0:       # kumulativer Kollaps
        # Rollback-Kaskade:
        # 1. best_clean_checkpoint — sofern selbst nicht kollabiert (> 20 dB prüfen)
        # 2. afg_pre_pipeline_audio (Primum non nocere)
        current_audio = recovery
```

Schwellwerte: Ausgang-Imbalance > 20 dB; Eingang-Imbalance < 6 dB (Kollaps neu durch Pipeline eingeführt).

### §2.44/§2.49 HPI-Rollback-Checkpoint Stereo-Health-Validation (v10.0.0)

Bevor `_hpi_best_rollback_audio` als Rollback-Ziel verwendet wird: L/R-Imbalance prüfen.

- Checkpoint-Imbalance > 20 dB UND Input war ausgeglichen (< 6 dB) → Checkpoint verwerfen
- Fallback: `original_audio_for_goals` (Primum non nocere)

Ohne diese Prüfung restauriert der HPI-Rollback ein stereo-zerstörtes Signal.

> Kreuzreferenz: §2.44 HPI (artifact_freedom als Multiplikator), §2.48 (Interaktions-Guard), §2.45 (perceptual_delta)

---

### v10-Amendment (2026-07-12)

**IAD-Logging nach Schweregrad gestaffelt** (v10, 2026-07-12):

Die `logger.warning`-Ausgabe des IAD (Introduced Artifact Detector) in `unified_restorer_v3.py` wird jetzt nach Kontaminationsgrad gestaffelt:

| Fraktion | Log-Level | Bedeutung |
| --- | --- | --- |
| `< 1%` | `logger.info` (ℹ️) | Spektraler Rest (z.B. musical_noise auf Cassette). Typisches NR-Residuum, keine hörbare Degradation. |
| `1–5%` | `logger.warning` (⚠️) | Leichter Qualitätsverlust. `artifact_freedom_penalty < 0.95` → VETO-Schwelle wird geprüft. |
| `≥ 5%` | `logger.warning` (⚠️) + Dry/Wet-Rescue | Signifikante Kontamination. §2.23 IAD-Rescue triggert Dry/Wet-Blend zurück zum Original. |

**Begründung**: `fraction=0.003` (0,3 %) `musical_noise` auf Cassette-Material ist normales spektrales Restrauschen jeder NR. `artifact_freedom_penalty=0.997` liegt weit über der 0.95-Veto-Schwelle. Eine WARNING bei <1 % ist ein Fehlalarm, der vom tatsächlichen Qualitätsverlust (≥5 %) ablenkt.

**Implementierung**: `backend/core/unified_restorer_v3.py` → `_iad_is_trivial = _frac_iad < 0.01` → `logger.info` vs `logger.warning`.

## §2.49c [RELEASE_MUST] Psychoakustischer Rauheit/Schärfe-Guard (v10.0.0.x)

**Motivierung**: ArtifactFreedomGate §2.49 prüft strukturelle Artefakte (Spectral Noise,
Holes, Phasenfehler). Multiband-Kompression (phase_35) kann Rauheit erhöhen und
HF-Enhancement (phase_39) Schärfe — beide degradieren das Hörerlebnis (§8.3 Tiefen-Immersion),
passieren aber alle 5 bestehenden Artefakt-Detektoren, weil sie keine strukturellen Fehler
erzeugen, sondern psychoakustische Lästigkeit steigern.

| Metrik | Modell | Schwellwert | Penalty auf `artifact_freedom` |
| --- | --- | --- | --- |
| **Rauheit (roughness)** | Zwicker (1991): AM-Modulationsenergie 15–300 Hz | Δ > 0.15 asper/Phase | −0.05 |
| **Schärfe (sharpness)** | Bismarck (1974): spektraler Schwerpunkt mit g(z)-Gewichtung | Δ > 0.30 acum gesamt | −0.10 |

```python
# backend/core/artifact_freedom_gate.py
_ROUGHNESS_FLAG_ASPER: float = 0.15   # Δrauheit pro Phase in asper
_SHARPNESS_FLAG_ACUM: float  = 0.30   # Δschärfe gesamt in acum
_ROUGHNESS_MATERIAL_TOLERANCE: dict[str, float] = {
    "digital": 1.0, "cd_digital": 1.0, "streaming": 1.0,
    "tape": 1.25, "reel_tape": 1.25,
    "vinyl": 1.5, "minidisc": 1.5,
    "shellac": 2.0, "wax_cylinder": 2.0, "wire_recording": 2.0,
}
```

**Rauheit-Messung (Zwicker-Approximation)**:

1. Hilbert-Transformation → Temporal-Envelope des Signals
2. FFT der Envelope → AM-Modulationsspektrum
3. Rauheit_asper ≈ normierte Energie im 15–300 Hz-Band der Envelope-FFT
4. Referenzwert (1 asper) = 60 dB SPL, 1 kHz, 100 % AM bei 70 Hz

**Schärfe-Messung (Bismarck)**:

1. Bark-Filterbank (24 Bänder, 0–16 kHz)
2. Spezifische Lautheitsdichte N'(z) pro Band (Zwicker)
3. Gewichtungsfunktion: g(z) = 1.0 für z ≤ 16 Bark; g(z) = 0.066 × e^(0.171×z) für z > 16
4. Schärfe_acum = 0.11 × ∫ N'(z) × g(z) × z dz / ∫ N'(z) dz

**Implementierung in `ArtifactFreedomGate.evaluate()`**:

```python
# §2.49c — Guard-Applicability: nur DYNAMICS, ADDITIVE, ENHANCEMENT
if phase_type in (_ROUGHNESS_APPLICABLE_TYPES):
    rough_orig = _compute_roughness_zwicker(orig_mono, sr)
    rough_rest = _compute_roughness_zwicker(rest_mono, sr)
    sharp_orig = _compute_sharpness_bismarck(orig_mono, sr)
    sharp_rest = _compute_sharpness_bismarck(rest_mono, sr)
    mat_tol = _ROUGHNESS_MATERIAL_TOLERANCE.get(_normalize_material(material_type), 1.0)
    roughness_delta = max(0.0, rough_rest - rough_orig)
    sharpness_delta = max(0.0, sharp_rest - sharp_orig)
    rs_penalty = 0.0
    if roughness_delta > _ROUGHNESS_FLAG_ASPER * mat_tol:
        rs_penalty -= 0.05
        detail_report["roughness_flag"] = {"delta_asper": roughness_delta}
    if sharpness_delta > _SHARPNESS_FLAG_ACUM * mat_tol:
        rs_penalty -= 0.10
        detail_report["sharpness_flag"] = {"delta_acum": sharpness_delta}
    artifact_freedom = max(0.0, artifact_freedom + rs_penalty)
```

**Guard-Applicability (§2.48a)**:

- Valide für: `DYNAMICS`, `ADDITIVE`, `ENHANCEMENT`
- Invalide für: `SUBTRACTIVE` (Rauschentfernung reduziert Rauheit intentional), `ML_GENERATIVE`, `CORRECTIVE`

**Invarianten**:

- Δ wird **nur positiv** geprüft (Rauheit/Schärfe dürfen sinken — das ist Verbesserung)
- `Δrauheit = max(0, roughness_output - roughness_input)` — kein Wert < 0 als Penalty
- Material-Toleranz symmetrisch zu §2.49-Schwellwerten
- Felder in `ArtifactFreedomResult`: `roughness_delta_asper`, `sharpness_delta_acum`, `roughness_sharpness_penalty`
- Laufzeit: ≤ 30 ms für 5-s-Sample bei sr=48000 (Bark-Filterbank approximiert mit 24 Butterworth-Bändern)

> Implementierung: `backend/core/artifact_freedom_gate.py` — `_compute_roughness_zwicker()`, `_compute_sharpness_bismarck()`

---

## §2.51 [RELEASE_MUST] Stereo-Kohärenz-Invariante für Phasen (v10.0.0)

### Motivation

Phasen, die L- und R-Kanal **unabhängig** verarbeiten (je Kanal eigener Denoiser, Gate, Kompressor, spektrale Reparatur), können in 2–3 Frames pro Phase `mono_compat < 0.20` erzeugen. Ursache: Minimale Unterschiede in Filterauflösung, Gate-Timing oder Spektralschätzung zwischen den Kanälen. Das §2.49-Gate flaggt diese Frames zu Recht — die Phasen verstoßen gegen §0 (Primum non nocere), weil sie Stereo-Kompatibilität verschlechtern.

Die Lösung ist **nicht** weitere Gate-Relaxation, sondern korrekte Implementierung der betroffenen Phasen.

### Normative Anforderung

Jede Phase, die auf Stereo-Audio operiert und den Signalpegel modifiziert, **MUSS** eine der folgenden zwei Verarbeitungsstrategien verwenden:

**Option A — M/S-Domain (bevorzugt für spektrale Operationen)**:

```
Mid = (L + R) / 2          # Summen-Kanal: Mono-kompatibler Inhalt
Side = (L - R) / 2         # Differenz-Kanal: Stereo-Breite

→  Verarbeite Mid mit voller Algorithmus-Stärke
→  Verarbeite Side mit reduzierter oder keiner Stärke (bewahre Stereo-Breite)
→  Rekonstruiere: L = Mid + Side,  R = Mid - Side
→  Clip: L = np.clip(L, -1.0, 1.0),  R = np.clip(R, -1.0, 1.0)
```

**Wann A**: Harmonische Restaurierung, spektrale Reparatur, Sprach-Enhancement, Dehum, EQ, Sättigungseffekte — immer wenn die Phasen-Verarbeitung tonal auf dem Informations-Inhalt arbeitet.

**Option B — Linked Stereo (für dynamische Verarbeitung)**:

```
signal_level = max(RMS(L), RMS(R))   # oder: np.sqrt(RMS(L)² + RMS(R)²)
gain = compute_gain(signal_level)     # Gain-Kurve einmalig berechnen
L_out = apply_gain(L, gain)           # Gleiches Gain für beide Kanäle
R_out = apply_gain(R, gain)
```

**Wann B**: Noise-Gate (Gate öffnet wenn L ODER R über Threshold), Dropout-Repair (synchrone Erkennung + kohärente Füllung), Multiband-Kompression, Transient-Shaper — immer wenn die Entscheidung (öffnen/schließen, verstärken/dämpfen) von der gemeinsamen Energie-Hüllkurve abhängt.

### Betroffene Phasen (Pflicht-Umsetzung)

| Phase | Problem | Strategie |
| --- | --- | --- |
| `phase_07_harmonic_restoration` | Harmonics separat auf L/R → Anti-Phase-Transients in 2–3 Frames | **Option A** (M/S) — Harmonics auf Mid, Side unverändert |
| `phase_18_noise_gate` | Gate öffnet/schließt für L und R unabhängig → Anti-Phase-Gate-Transients | **Option B** (Linked) — `max(L_rms, R_rms) > threshold → both open` |
| `phase_23_spectral_repair` | Spektrale Lücken auf L/R separat erzeugt minimale Anti-Phasigkeit | **Option A** (M/S) — Reparatur auf Mid, Side minimal bearbeiten |
| `phase_24_dropout_repair` | L/R-Dropouts erkannt und gefüllt unabhängig | **Option B** (Linked) — Dropout-Grenze ist der Eintritt BEIDER Kanäle unter Schwelle; Füllung kohärent |
| `phase_35_multiband_compression` | Kompressor berechnet Gain für L und R separat → L/R-Gain-Differenz in Transienten | **Option B** (Linked) — Gain-Berechnung auf Summen-RMS (`√(L²+R²)/√2`), gleicher Gain auf beide |

### Downstream-Auswirkungen auf Metriken

| Metrik | Auswirkung | Korrekturbedarf |
| --- | --- | --- |
| **Brillanz** | M/S in `phase_07`: Harmonics nur auf Mid → weniger HF-Energie im Side-Kanal. Brillanz-Schwellwert ≥ 0.78 unverändert, aber `BrillanzMetric` muss Stereo-Mid nicht Side-Anteil messen | Kein Schwellwert-Änderungsbedarf; Metrik misst bereits Gesamtspektrum |
| **Raumtiefe** | Linked Stereo in `phase_35`: Einheitlicher Gain erhält Side-kanal besser → Raumtiefe kann leicht steigen | Kein Korrekturbedarf (positive Auswirkung) |
| **SepFidelity** | Kohärente L/R-Füllung in `phase_24`: Dropout-Füllung ist konsistenter mit Stereo-Bild → SepFidelity tendenziell verbessert | Kein Korrekturbedarf |
| **Groove** | Linked Gate in `phase_18`: Transiente Energie wird kohärent erhalten (kein halbes Gate-Öffnen) → Groove-Presenz besser | Kein Korrekturbedarf (positive Auswirkung) |
| **§2.49 Phase-Cancellation** | Nach Implementierung: 5 Phasen passieren Gate ohne Rollback → `_min_per_phase_afg_score` bleibt 1.0 | Kein Korrekturbedarf; Gate-Schwellwerte unverändert |
| **PMGG Wärme §9.7.14** | Wärme nutzt harmonische Oberton-Ratio. M/S ändert Side-Obertöne nicht → Wärme-Proxy stabil | Kein Korrekturbedarf |

### Invariante

Kein Accept-Checkpoint darf `mono_compat < 0.20` in mehr als 5 % der Frames haben (außer das Quellmaterial hatte bereits diese Mono-Inkompatibilität — §2.50 SourceMaterialBaseline).

**Implementierungsprüfung**: `_detect_phase_cancellation()` im §2.49-Gate ist der objective Prüfer. Nach Umsetzung der obigen Phasen dürfen phase_07, phase_18, phase_23, phase_24, phase_35 keine §2.49-Rollbacks mehr auslösen.

> Kreuzreferenz: §2.49 (ArtifactFreedomGate), §2.50 (SourceMaterialBaseline), §7.4 Spec06 (PhaseInterface)

---

## §2.52 [RELEASE_MUST] PhaseConductor — Inter-Phase Adaptive Feedback (v10.0.0)

### Überblick

`PhaseConductor` ist ein **rein DSP-basierter inter-phase Feedback-Controller**. Er misst nach jeder Phase den verbleibenden Signal-Zustand und leitet daraus eine adaptive `strength`-Empfehlung für die **nächste** Phase ab. Kein ML, kein Netzwerkzugriff, kein I/O.

- **Singleton**: `get_phase_conductor()` in `backend/core/phase_conductor.py`; thread-safe (double-checked locking)
- **Session-Scope**: `conductor.reset()` zu Beginn jedes Songs in UV3 `_execute_pipeline`
- **Advisory-only**: PMGG-Strength hat immer Vorrang; alle Empfehlungen sind Hinweise, keine Befehle

### 4D State-Vector

| Dimension | Beschreibung | Normierung | Messzeit (48 kHz, 3 min) |
| --- | --- | --- | --- |
| `noise_floor_db` | 5. Perzentil der Leistungsdichteschätzung (PSD) | dBFS ≤ 0 | < 5 ms |
| `hf_energy_ratio` | Energie 8–24 kHz / Breitband (0–24 kHz) | [0, 1] | < 5 ms |
| `transient_density` | Onset-Rate (librosa.onset.onset_detect) [Events/s] | roh; as_vec() → /20 | < 20 ms |
| `harmonic_coherence` | Autocorrelation-Peak-Ratio auf Mid-Kanal | [0, 1] | < 15 ms |

Gesamt `measure_state()` < **50 ms** pro Aufruf auch für 3 Minuten Audio.

### Referenzgitter und Nearest-Neighbor-Empfehlung

Pro Material gibt es ein vorberechnetes Referenzgitter aus (state_4d → optimal_strength)-Paaren:

```python
# Beispiel-Grid (werden zur Laufzeit nicht trainiert, sind hardcoded DSP-Messungen):
_REFERENCE_GRIDS: dict[str, list[tuple[PhaseState, float]]] = {
    "vinyl":    [...],   # 12 Referenzpunkte × (state_vec, optimal_strength)
    "reel_tape":[...],
    "tape":     [...],
    "shellac":  [...],
    "minidisc": [...],
    "cd_digital":[...],
}
```

**Nearest-Neighbor**: L2-Distanz auf normiertem State-Vektor (noise/−90, hf/1, transient/20, coherence/1). Bei `distance > 0.8` → Fallback auf `_DEFAULT_STRENGTH[phase_id]` (kein Over-Extrapolation).

### Workflow in `_execute_pipeline` (UV3)

```python
# 1. Init vor Phase-Loop
_conductor = get_phase_conductor()
_conductor.reset()
_conductor_strength_hints: dict[str, float] = {}

# 2. Nach jeder erfolgreichen Phase
_conductor.measure_state(current_audio, sr, phase_id=current_phase_id)

# 3. Look-Ahead für nächste Phase
if next_phase_id:
    rec = _conductor.recommend(next_phase_id, state=_conductor.last_state, material_type=material_type)
    if rec.strength_hint is not None:
        _conductor_strength_hints[next_phase_id] = rec.strength_hint

# 4. In _profiled_phase_call: hint injizieren (nur wenn strength nicht explizit gesetzt)
if phase_id in _conductor_strength_hints and "strength" not in explicit_kwargs:
    kwargs["strength"] = _conductor_strength_hints[phase_id]
```

### Invarianten

| Invariante | Beschreibung |
| --- | --- |
| `_NEVER_SKIP` | `frozenset({"phase_01", "phase_09", "phase_12", "phase_14", "phase_15"})` — diese Phasen erhalten nie `skip_recommended=True`, egal wie der State-Vektor aussieht |
| `_MIN_STRENGTH` | `{"phase_03": 0.35, "phase_09": 0.50, ...}` — Untergrenze für kritische Phasen; `recommended_strength = max(rec, _MIN_STRENGTH.get(phase_id, 0.0))` |
| Exception-Sicherheit | Jede Exception in `measure_state` oder `recommend` → `logger.debug(exc)`, Pipeline läuft **unverändert** weiter (kein Abbruch, kein Fehler-Propagation) |
| Kein §0-Verstoß | Wenn `recommended_strength < explicit_strength`: PMGG-Wert gewinnt; wenn `recommended_strength > explicit_strength`: PMGG-Wert gewinnt — ConductorHint beeinflusst nur **nicht explizit gesetzten** Strength |
| Keine ML-Abhängigkeit | Rein DSP, nur numpy + scipy; kein torch, kein ONNX, kein Remote-Call |

### §2.52a PhaseConductor × SongGoalImportance Integration (v10.0.0)

`PhaseConductor.recommend()` erhält optional `goal_weights: dict[str, float]`, die ausschließlich
aus dem zentralen `restoration_policy_profile` abgeleitet werden. `song_goal_weights` ist nur ein
Legacy-Fallback für Kompatibilität, nicht eine konkurrierende Primärquelle.

**Workflow**:

1. UV3 berechnet das `restoration_policy_profile` einmalig in `restore()` (§2.56 Stufe 1–5)
2. UV3 übergibt daraus abgeleitete `goal_weights` an `_conductor.recommend(next_phase_id, state, material_type, goal_weights=goal_weights)`
3. PhaseConductor berücksichtigt Gewichte bei der Strength-Empfehlung:
   - Hohe `transparenz`/`brillanz`-Gewichtung → ADDITIVE-Phasen bekommen leichten Strength-Boost
   - Hohe `natuerlichkeit`/`authentizitaet`-Gewichtung → konservativere Empfehlung (niedrigerer Strength)
4. Modifikation ist **bounded** (±10 % des Basiswerts) und **advisory-only**

**Invarianten**:

- `goal_weights=None` → Fallback auf Uniform-Gewichtung (1.0 für alle Goals)
- Fehler im goal_weights-Pfad → `logger.debug`, neutraler Strength (kein Crash)
- PMGG-Strength hat weiterhin absoluten Vorrang (§2.52 Kein-§0-Verstoß-Invariante)

**Invariante:** Kein Phase-Module oder Subsystem darf eigene songweite Zielgewichte parallel berechnen;
die einzige normative Quelle ist `restoration_policy_profile`.

### Zusammenspiel mit §2.47 PhaseSkipper (Hebel 1 + Hebel 3 Synergie)

```text
DefectScanner → _salience_adjusted_severity() (Hebel 1)
     ↓ severity (ERB-gewichtet)
_apply_phase_skipping → Phase aktiv/inaktiv?
     ↓ wenn aktiv:
_conductor.recommend(next_phase_id, …) (Hebel 3)
     ↓ strength_hint
_profiled_phase_call → Phase läuft mit adaptiver Wetness
```

Hebel 1 entscheidet **ob** eine Phase läuft; Hebel 3 entscheidet **wie stark** sie läuft. Beide zusammen vermeiden Über- und Unter-Processing.

> Implementierung: `backend/core/phase_conductor.py`
> UV3-Integration: `backend/core/unified_restorer_v3.py` — `_execute_pipeline`, `_profiled_phase_call`
> Tests: `tests/unit/test_hebel_intelligence_levers.py` (Hebel 3: Tests 17–26, 32/32 grün)

---

## §2.52b [RELEASE_MUST] SongStructureAnalyzer — Segment-bewusste Pipeline (v10.0.0)

**Motivation**: Ein Vers braucht aggressiveres NR als ein Klimax-Chorus. Ein Instrumentalintro benötigt kein `LyricsGuidedEnhancement`. Eine Stille-Passage benötigt keinen Dereverb. Aurik behandelte bisher alle Segmente identisch — das ist falsches Minimal-Intervention.

**SongStructureAnalyzer** (`backend/core/song_structure_analyzer.py`):

```python
@dataclass
class SongSegment:
    start_s: float
    end_s: float
    label: str          # "intro", "verse", "chorus", "bridge", "outro", "instrumental"
    energy_level: float # [0, 1] — Ø RMS normiert
    has_vocals: bool    # PANNs Singing confidence Ø im Segment
    is_climax: bool     # Frisson-Zone (§2.56/§Frisson) oder Energy-Peak

def analyze_structure(audio: np.ndarray, sr: int) -> list[SongSegment]:
    """
    Librosa Boundary-Erkennung (segment.agglomerative auf MFCC + Chroma).
    Segment-Labels via Beat-Tracking + Energy-Profil heuristisch.
    Laufzeit: ≤ 2 s / Minute Audio.
    """
```

**Segment-adaptive Strength-Modifikation in UV3**:

| Segment-Typ | Phase-Gruppen-Anpassung |
| --- | --- |
| `verse` (leise, Vocals) | NR-Strength × 1.15 (aggressiver), Dereverb × 1.10 |
| `chorus` / `is_climax=True` | NR-Strength × 0.85 (schützend), Kompression × 0.70 — Klimax nicht quetschen |
| `intro` / `outro` | `LyricsGuidedEnhancement` nur wenn `has_vocals=True`; BW-Ceiling strenger (kein Artefakt am Rand) |
| `instrumental` | Vokal-Phasen (phase_19, phase_58) überspringen; Groove × 1.10 |
| Stille (< −50 dBFS) | Alle Enhancement-Phasen überspringen; nur Noise-Gate (phase_18) |

**Integrationsort**: UV3 `_execute_pipeline` — nach `GPOptimizer`, vor Phase-Loop. Segment-Map wird in `_song_segments` gecacht; `_profiled_phase_call` berechnet `_segment_strength_scalar` für das Segment, in dem die aktuelle Zeit liegt.

**Protokollierung**: `metadata["song_structure"]["segments"]` — Liste aller Segmente mit Labels.

**Invariante**: Segment-Stärke-Skalare sind bounded [0.70, 1.30] — kein Phase kann durch Segmentierung auf Null oder über 1.3× getrieben werden. P1/P2-Phase-Schwellwerte bleiben unverändert.

> Implementierung: `backend/core/song_structure_analyzer.py`
> Tests: `tests/unit/test_song_structure_analyzer.py`

---

## §2.52c [RELEASE_MUST] MultiPassScheduler — Iterativer Restaurierungspfad (v10.0.0)

**Motivation**: Phase 03 (NR) senkt den Rauschboden → Phase 09 (Crackle) würde danach bei niedrigerem Grundrauschen präzisere Ergebnisse erzielen. Ein einziger Durchlauf ist suboptimal. Echter Multi-Pass löst diese Sequenz-Abhängigkeit.

**Zweistufiger Multi-Pass** (Pflicht-Kaskade, nicht optional):

```
Pass 1 (Carrier-Chain):  Phasen Stufe 1–4 (§2.46) — ADC/Playback/Alterung/Carrier-Subtraktiv
                         → best_carrier_checkpoint setzen
Pass 2 (Defekt-Refinement, konditionell):
    BEDINGUNG: DefectScanner(restored_after_pass1) zeigt noch ≥ 2 Defekte mit severity > 0.15
    → Nur Top-3-Defekte nochmal (adaptiv: niedrigere Strength = Pass-1-Strength × 0.6)
    → Kein vollständiger Phase-Durchlauf — nur Defekt-spezifische Phasen
    → Maximale 1 Refinement-Pass (kein unbegrenztes Iterieren)
```

**Wall-Time-Budget für MultiPass**:

- Pass 1 darf max. `0.70 × UV3_WALL_TIME_BUDGET` verbrauchen
- Pass 2 darf max. `0.25 × UV3_WALL_TIME_BUDGET` verbrauchen
- Wenn Pass-1 > 0.70 × Budget → kein Pass-2 (normale Pipeline-Fertigstellung)

**Aktivierungsbedingung**:

```python
if (
    material_type in {"shellac", "wax_cylinder", "reel_tape", "lacquer_disc"}
    OR defect_count_pass1 >= 3
) AND wall_time_remaining > UV3_WALL_TIME_BUDGET * 0.25:
    run_pass_2 = True
```

**Nicht aktiviert für**: digitale Quellen (`cd_digital`, `dat`), hohe Restorabilität (> 80), kurze Dateien (< 30 s).

**Protokollierung**: `metadata["multipass"]["passes_run"]`, `metadata["multipass"]["pass2_defects_remaining"]`.

> Implementierung: UV3 `_execute_multipass_pipeline()`
> Tests: `tests/unit/test_multipass_scheduler.py` [ROADMAP]

---

## §2.53 [RELEASE_MUST] Experience-Closed-Loop + Bridge/UI-Propagation (v10.0.0)

### Vertrag

`UnifiedRestorerV3.restore()` MUSS Experience-Telemetrie strukturiert in
`RestorationResult.metadata` bereitstellen und diese MUSS über Bridge/Denker bis
ins Frontend propagiert werden.

### Pflichtfelder

1. `song_calibration.cluster_key`
2. `song_calibration.cluster_policy`
3. `joy_runtime_index` (`joy_index`, `fatigue_index`, `components`)
4. `auto_improvement_recommendations` (`count`, `recommendations[*].focus/action/reason`)
5. `team_coordination` (`event_count`, `events`, `phase_type_summary`)

### Invarianten

- `backend.api.bridge.get_experience_insights()` liefert frontend-sichere Werte
    (NaN/Inf-frei, fehlertolerant, schema-stabil).
- `AurikDenker`/`RestaurierDenker` dürfen Experience-Metadaten nicht verwerfen.
- UI zeigt mindestens Freude-/Ermüdungsindex, Cluster-Policy und Top-Empfehlungen.
- Fehler sind non-blocking: fehlende Experience-Telemetrie blockiert keinen Export,
    wird aber als Degrade-Hinweis protokolliert.

## §2.53a [RELEASE_MUST] Exzellenz-API-Kompatibilitätsvertrag (v10.0.0 / v10.0.0)

### Vertrag

`AurikDenker` MUSS beide Exzellenz-Schnittstellen unterstützen:

1. Primär: `ExzellenzDenker.messe_und_repariere(audio, sr, reference_audio=..., inapplicable_goals=..., mode=..., material=...) -> (audio, goals)`
2. Legacy-Fallback: `ExzellenzDenker.messe_ziele(audio, sr, reference=None)`

### Signatur-Erweiterungen (v10.0.0)

**`reference_audio`** (S6): Originales (prä-Restaurierung) Audio als `np.ndarray | None`. Wird an alle internen `messe_ziele()`-Aufrufe weitergegeben. Ohne `reference_audio` verwendet `TonalCenterMetric` intra-song Chroma-Stabilität (Score 0.50–0.60 für Verse/Chorus-Songs) statt ref-vs-restored Chroma-Korrelation (~0.92+). Bestätigt: tonal_center=0.5501 ohne reference, ~0.95 mit reference (V50). In AurikDenker: `reference_audio=audio` (das `audio`-Argument von `denke()`, vor Restaurierung, 48 kHz, float32).

**`inapplicable_goals`** (S5): `frozenset[str]` mit physikalisch nicht erreichbaren Goal-Namen. Aus `RestorationResult.goal_applicability` des UV3-Runs extrahiert: `frozenset(g for g, ok in rest.goal_applicability.items() if not ok)`. `_count_passed()` und `goals_total` im ExzellenzDenker schließen diese Goals aus Zähler UND Nenner aus (V49). Nur übergeben wenn frozenset nicht leer.

### Invarianten

- Kein harter Bind auf nur eine Methode.
- Bei Legacy-Fallback MUSS ein eindeutiger Stage-Note-Eintrag gesetzt werden:
    `Legacy-Goal-Messpfad`.
- Fehlt die Primärmethode, darf die Pipeline nicht abbrechen, solange Legacy verfügbar ist.
- `messe_ziele(audio, sr, reference=None)` ist die normative interne Mess-Funktion — immer mit `reference=reference_audio` aufrufen, wenn `reference_audio` vorhanden (V50).

## §2.53c [RELEASE_MUST] GAF-Inapplicable-Propagationskette (v10.0.0, V51-Fix S7)

```text
GAF.evaluate_goal_applicability(audio, sr, material, era, ..., transfer_chain)
    → RestorationResult.goal_applicability: dict[str, bool]   ← UV3
    → RestaurierErgebnis.goal_applicability: dict[str, bool]   ← _konvertiere() propagiert (V51)
    → AurikDenker._goal_app_raw: dict[str, bool]  ← initialisiert als {} VOR try-Block (V51)
    → AurikDenker._rest_inapplicable_goals = frozenset(g for g, ok in ... if not ok)
    → ExzellenzDenker.messe_und_repariere(..., inapplicable_goals=_rest_inapplicable_goals)
    → _count_passed(): schließt _inappl aus (Zähler + Nenner)
    → goals_passed recompute: sum(1 for k,v if k not in _inappl and v >= threshold)
    → AurikErgebnis.goal_applicability: dict[str, bool]  ← für externe Caller (V51)
```

**V51-Invariante**: `RestaurierErgebnis` MUSS `goal_applicability: dict[str, bool]` als Dataclass-Feld haben (default `{}`). `_konvertiere()` MUSS `goal_applicability=dict(getattr(raw, "goal_applicability", None) or {})` setzen. `_goal_app_raw` MUSS vor dem try-Block als `{}` initialisiert werden — fehlt diese Initialisierung, wirft Python bei Exception im try-Block `UnboundLocalError` und blockiert die gesamte Pipeline. `AurikErgebnis` MUSS ebenfalls `goal_applicability`-Feld haben.

**V52-Invariante**: `separation_fidelity` folgt der `spatial_depth`-Near-Mono-Codec-Regel (gleiche `_CODEC_JOINT_STEREO_MATS`, gleicher corr-Schwellwert ≥ 0.83). Fehlt die Parallelregel, bleibt `separation_fidelity` bei Joint-Stereo-Codec fälschlich applicable → false Violation.

**Propagations-Vollständigkeitsinvariante**: Die Kette ist vollständig oder nicht vorhanden. Teilpropagation (GAF feuert, ExzellenzDenker ignoriert) erzeugt systematisch falsche Violation-Counts und Over-Processing. Leer-frozenset → kein Overhead.

## §2.53b [RELEASE_MUST] Denker-Plan-Determinismus in UV3 (v10.0.0)

### Vertrag

Wenn `UnifiedRestorerV3.restore(..., precomputed_phase_plan=[...])` gesetzt ist,
ist dieser Plan der **verbindliche Ausführungsplan**.

### Invarianten

1. UV3 MUSS `_select_phases()` und `_optimize_phase_plan_intelligence()` überspringen.
2. UV3 MUSS `selected_phases = list(precomputed_phase_plan)` als Basis verwenden.
3. UV3 MUSS `phase skipping` in diesem Pfad deaktivieren.
4. Nur normative Sicherheitsinjektionen sind zulässig:
    - §2.50 Stereo-Notfall-Remediation (`phase_14`, `phase_15`)
    - weitere explizit versionsmarkierte RELEASE_MUST-Injektionen
5. Stale-Zustand aus früheren Läufen darf nicht in den precomputed-Pfad leaken
    (`_last_material_priority_phases` ist vor Ausführung zu neutralisieren).

### Verboten

- Denker-Plan laden und anschließend durch UV3-autonome Selektion/Optimierung überschreiben.
- Denker-Plan via `phase skipping` implizit verändern.

### Rationale

Hybrid-Orchestrierung (Denker + UV3-Autoselektion im selben Lauf) erzeugt nicht-deterministische
Planabweichungen und erschwert Reproduzierbarkeit, QA und Root-Cause-Analyse.

**Ergänzung zur Entscheidungslogik:** Die Denker liefern die inhaltliche Intelligenz
für Erkennung, Kontext und Priorisierung. Ihre Ergebnisse fließen in das zentrale
`restoration_policy_profile` ein, das die normative songweite Steuerung bereitstellt.
Die Denker dürfen damit die Qualität maximal verbessern, aber keine zweite, konkurrierende
Song-Steuerquelle aufbauen.

**Denker-Policy-Input:** Vor UV3 wird ein `denker_policy_input` gebildet. Pflichtfelder
sind, sofern verfügbar: `strategy.intervention_budget`, `strategy.listening_experience_targets`,
`strategy.human_hearing_risk_map`, `strategy.human_hearing_comfort_profile`,
`phase_interaction.goal_risk_map`,
`repair_risk_profile`, `reconstruction_risk_profile` und `signal_signature`. UV3 darf
diese Signale nur über den zentralen Policy-Resolver in `restoration_policy_profile`
übernehmen.

**Human-Hearing-Comfort-Profil:** Der `StrategieDenker` MUSS aus der
songindividuellen Signatur (`crest_db`, `hf_ratio`, `transient_ratio`,
`micro_dynamic_db`), dem Eingriffsbudget und den Risiko-Maps ein
`human_hearing_comfort_profile` ableiten. Das Profil steuert u. a.
`peak_overshoot_cap_db`, `hf_loss_tolerance_db`, `hf_lift_cap_db`,
`transient_protection`, `microdynamic_protection`, `fatigue_sensitivity` und
`dynamic_smoothing_tolerance`. Der finale Hoerkomfort-Guard und alle zukuenftigen
komfortbezogenen Phasen-/Post-Gates verwenden nur dieses zentrale Profil.

## §2.54 [RELEASE_MUST] Adaptives Phasen-Optimum — Messen-Handeln-Validieren (v10.0.0, erweitert v10.0.0)

> Dieses Paradigma ist normativ übergeordnet gegenüber allen festen Schwellwerten in §2.48, §2.29d, §2.45.
> Feste Schwellwerte sind **Notbremsen** (letztes Sicherheitsnetz), nicht die Steuerung.

### Grundprinzip

Die Denker-Schicht ist die Intelligenzverstärkung vor UV3: Sie analysiert Material,
Kontext, Defektlage und Rekonstruktionshinweise und verdichtet diese Informationen in
eine bessere zentrale Policy. UV3 bleibt die einzige normative Ausführungs- und
Exportinstanz; `restoration_policy_profile` bleibt die einzige songweite Steuerquelle.

Die Denker optimieren damit nicht gegen technische Messwerte allein, sondern gegen das
positive Klangerlebnis für das menschliche Gehör: Natürlichkeit, Ermüdungsfreiheit,
Vokalnähe, Mikro-Dynamik, Wärme, glaubwürdige Raumanteile und Nicht-Halluzination.

Jeder Song ist einzigartig. Feste Schwellwerte können die Vielfalt an Genre, Ära, Tonträgerkette und
Defekten nicht abbilden. Stattdessen durchläuft jede Phase einen **Messen→Handeln→Validieren-Zyklus**:

1. **MESSEN** — Zustand vor der Phase: Klangtreue, Defekt-Schwere, Energie-Profil
2. **HANDELN** — Phase mit materialadaptiver Stärke ausführen (SongCal × PhaseConductor)
3. **VALIDIEREN** — Zustand nach der Phase messen: Hat sich der Klang verbessert?
4. **ENTSCHEIDEN**:
   - Verbesserung klar hörbar → Phase akzeptieren, weiter
   - Verbesserung marginal → Stärke anpassen, erneut (max 3 Iterationen)
   - Verschlechterung → Stärke reduzieren oder Phase überspringen
   - Katastrophale Beschädigung → Rollback (Notbremse)
5. **BESTES ERGEBNIS BEHALTEN** — Über alle Iterationen das perceptuell beste Resultat wählen

### Steuerungs-Zuordnung

| Komponente | Rolle | NICHT die Rolle |
| --- | --- | --- |
| **Denker** | Plant Phase-Reihenfolge + Initialkonfiguration | Feste Schwellwerte setzen |
| **PhaseConductor** (§2.52) | Misst 4D-Zustand, empfiehlt `strength` | Starres Pass/Fail |
| **PMGG** (§2.29) | Misst Musical-Goals-Delta, steuert Stärke-Iteration | Festes `regression > 0.02` |
| **SongCalibration** (§2.47) | Skaliert alle Stärken material-/song-adaptiv | Universelle Konstante |
| **CumulativeInteractionGuard** (§2.48) | **Nur Notbremse**: kumulative Drift | Routine-Steuerung |
| **GPOptimizer** | Lernt Pareto-optimale Hyperparameter | Erstmalige Parameterwahl |

### Adaptive Drift-Toleranz

Die Drift-Toleranz des CIG wird **berechnet**, nicht fest vorgegeben:

```python
adaptive_drift_tolerance = compute_adaptive_drift_tolerance(
    restorability_score,     # 0–100: wie stark degradiert? → mehr Spielraum
    material_type,           # vinyl/shellac brauchen mehr als cd_digital
    defect_severity_mean,    # hohe mittlere Severity → mehr Toleranz nötig
    n_active_phases,         # mehr Phasen → mehr kumulative Drift normal
)
# Ergebnis: z.B. -0.03 (CD, leicht) bis -0.25 (Shellac-4-Gen, schwer degradiert)
```

**Normative Material-Basis-Toleranzen** (Implementierung: `_MATERIAL_BASE` in `compute_adaptive_drift_tolerance()`):

| Material | Basis | Material | Basis |
| --- | --- | --- | --- |
| `cd_digital` | −0.03 | `vinyl` | −0.10 |
| `dat` | −0.03 | `shellac` | −0.15 |
| `minidisc` | −0.04 | `wax_cylinder` | −0.18 |
| `mp3_high` | −0.04 | `wire_recording` | −0.15 |
| `mp3_low` | −0.06 | `optical_film` | −0.10 |
| `cassette` | −0.07 | `radio_broadcast` | −0.08 |
| `tape` | −0.08 | `unknown` | −0.06 |
| `reel_tape` | −0.09 | | |

**Modifikatoren:**

- `restorability_factor = 1.8 − (restorability / 100)` — niedrige Restorabilität → breiterer Spielraum
- `severity_factor = 1.0 + 0.5 × defect_severity_mean` — schwere Defekte → mehr Toleranz
- `phase_factor = 1.0 + 0.02 × max(0, n_phases − 5)` — mehr Phasen → normaler kumulativer Drift

**Hard-Clamp:** `tolerance ∈ [−0.30, −0.02]` — nie enger als −0.02, nie lockerer als −0.30.

### Invarianten

1. Kein fester Schwellwert darf eine restorative Phase blockieren, wenn das Material den Eingriff braucht
   und die Phase den Defekt messbar reduziert.
2. Checkpoint-Selektion: Guard wählt immer das perceptuell **beste** Audio als Checkpoint.
3. Pipeline-Stopp nur bei echtem Schaden: `should_stop` nur nach materialadaptiver Schwelle UND ohne bessere Stärke.
4. Referenz-Paradoxon (§2.44): Carrier-Repair-Phasen verändern das Signal intentional — Metrik-Drop ≠ Verschlechterung.

### Implementierung

- `compute_adaptive_drift_tolerance()` in `backend/core/cumulative_interaction_guard.py`
- `compute_adaptive_max_rollbacks()` ebenda
- Testpflicht: `tests/unit/test_adaptive_drift_tolerance.py`

### §2.54a PMGG-Blend-Invariante und Pre-Pipeline-Ceiling (v10.0.0)

**Problem**: Fixer 60/40-Blend in PMGG `_run_with_retry()` und UV3 `_effective_goal_thresholds` erzeugt Schwellwerte **über** der physikalischen Ceiling — bei Shellac `brillanz` (canonical 0.78, SGT 0.51): `0.60×0.78 + 0.40×0.51 = 0.71` bei physikalischer Grenze 0.51 → PMGG startet 5-fachen Retry-Zyklus → Stärke sinkt auf 15 % → Restaurierung versagt.

**Normative Lösung — zwei Mechanismen gemeinsam**:

**1. Pre-Pipeline `_pmgg_ceiling_capped_targets`** (Pflicht, UV3 `restore()` vor Phase-Loop):

```python
from backend.core.physical_ceiling_estimator import PhysicalCeilingEstimator
_pce = PhysicalCeilingEstimator().estimate(input_audio, sample_rate, {}, material_key)
_pmgg_ceiling_capped_targets = {
    g: float(min(sgt[g], _pce.ceiling[g]))
    for g in sgt
}
# wird als adaptive_goal_thresholds an jede wrap_phase() übergeben
```

**2. Delta-adaptiver Blend** (Pflicht, identisch in PMGG + Pipeline-Ende, §09.2):

```python
delta = canonical[goal] - adaptive[goal]   # positiv = adaptiv ist niedriger
if delta > 0.10:
    blended = adaptive[goal]               # Ceiling-Fall → adaptiv direkt
elif delta > 0.04:
    blended = 0.40 * canonical[goal] + 0.60 * adaptive[goal]
else:
    blended = 0.60 * canonical[goal] + 0.40 * adaptive[goal]
blended = float(np.clip(blended, 0.30, 0.99))
```

**Invariante**: PMGG `_run_with_retry()` und UV3 `_effective_goal_thresholds`-Block verwenden **identische** delta-adaptive Logik.

### §2.54b Headroom-Scalar für additive Enhancement-Phasen (v10.0.0)

Additive Phase-Familien (`harmonic_reconstruction`, `harmonic_enhancement`, `tonal_enhancement`, `source_enhancement`, `stereo_enhancement`, `stereo_generation`) erhalten vor `wrap_phase()` einen Strength-Scalar proportional zum **absoluten** Headroom bis zur physikalischen Decke:

```python
HR_WINDOW = 0.25   # volle Stärke wenn Headroom >= 0.25; linear gedämpft darunter
HR_GOALS  = ("brillanz", "waerme", "raumtiefe", "bass_kraft", "sep_fidelity")

min_hr = 1.0
for goal in HR_GOALS:
    headroom = max(0.0, _pmgg_ceiling_capped_targets[goal] - current_score[goal])
    hr_ratio  = min(1.0, headroom / HR_WINDOW)
    min_hr    = min(min_hr, hr_ratio)

hr_strength = float(np.clip(min_hr, 0.40, 1.0))   # Minimum 40 % auch nahe der Decke
if hr_strength < 0.95:
    combined_strength = float(np.clip(combined_strength * hr_strength, 0.05, 1.0))
```

**Psychoakustischer Hintergrund**: Grenznutzen sinkt asymptotisch nahe der physikalischen Ceiling; Artefaktrisiko (Over-Processing, spektrale Artefakte) steigt gleichzeitig. Das 0.25-Fenster entspricht typischem JND-Bereich für timbrale Änderungen. Restorative und Pflicht-Phasen (`_is_restorative_phase`, `_is_mandatory_phase`) sind ausgenommen.

**VERBOTEN**: Relative Normierung `(curr - 0.30) / (ceil - 0.30)` — überdämpft CD-Phasen mit großem Headroom (CD `brillanz=0.60, ceil=0.99` → scalar=0.57 trotz 0.39 freiem Raum). Nur absoluter Headroom ist korrekt.

## §2.55 [RELEASE_MUST] PMGG-CIG-Synchronisations-Invariante (v10.0.0)

`CIG._PHASE_SPECIFIC_DRIFT_EXCLUSIONS` und
`PMGG.PHASE_GOAL_EXCLUSIONS` müssen für alle P1/P2-Goals
bidirektional synchron sein.

### Formale Bedingung

Für jede Phase `p` gilt:

- `CIG_excl(p) ∩ P1P2 ⊇ PMGG_excl(p) ∩ P1P2`
- `PMGG_excl(p) ∩ P1P2 ⊇ CIG_excl(p) ∩ P1P2`

### Rationale

Wenn PMGG ein Goal in einer Phase exkludiert, CIG aber nicht, akkumuliert CIG
falschen Drift und kann in späteren Phasen einen fehlerhaften Rollback auslösen.
Die inverse Asymmetrie führt dazu, dass PMGG Goals blockiert, die CIG nicht als
Drift zählt.

### Verboten

- Neue Phase einführen und nur eine der beiden Exclusion-Tabellen erweitern.

### Testpflicht

- CI-Regressionstest: `tests/unit/test_pmgg_cig_sync.py`

---

## §2.57 [RELEASE_MUST] Phase-50-HF-Guard + Phase-09-LPC/AR-Reparatur (v10.0.0 / v10.0.0)

### §2.57a Phase-50 HF-Spike-Schutz für Vorphasen-Harmoniken

**Problem**: Pass-1 Spike-Detektor (11-Bin-Fenster) flaggt durch `phase_07`/`phase_06` restaurierte
Harmoniken als Codec-Spikes und inpaintet sie — Vorphasen-Restaurierung wird rückgängig gemacht.

**Invariante** (`backend/core/phases/phase_50_spectral_repair.py`):

- `_repair_channel(audio, hf_protected_bin_start=0)` — neuer Parameter
- Bins ≥ `hf_protected_bin_start` aus Pass-1 (Spike-Detection) ausgeschlossen
- Pass-2 (Frame-Energy-Dropout) bleibt global aktiv — Frame-RMS reagiert nicht auf isolierte HF-Peaks
- `process()` berechnet `hf_protected_bin_start = material_rolloff × 0.85 / bin_hz`

**Material-Rolloff-Lookup** (analoge Materialtypen, Pass-1 Schutzzone aktiv):

| Material | Rolloff | Material | Rolloff |
| --- | --- | --- | --- |
| `wax_cylinder` | 5 000 Hz | `lacquer_disc` | 10 000 Hz |
| `shellac` | 8 000 Hz | `cassette` | 12 000 Hz |
| `wire_recording` | 6 000 Hz | `vinyl` | 18 000 Hz |
| `tape` / `reel_tape` | 16 000 Hz | `minidisc` | 16 000 Hz |

Digitale Materialien (`cd_digital`, `mp3*`, `dat`, `aac`, `streaming`): keine Schutzzone.

**Metadata**: `hf_protected_bin_start`, `hf_protection_rolloff_hz` in Phase-Metadata (Audit).

**Testpflicht**: `tests/unit/test_phase_50_hf_protection_guard.py` (16 Tests, alle grün).

### §2.57b Phase-09 LPC/AR-Lücken-Interpolation

**Problem**: `_interpolate_hybrid()` rief intern `_interpolate_linear()` auf — kein AR-Verhalten.

**Vollständige LPC/AR-Vorhersage** (`backend/core/phases/phase_09_crackle_removal.py`):

```python
# `_ar_fill_channel(gap_audio, pre_context, post_context, lpc_order=32)`:
# 1. Vorwärts-AR aus Pre-Gap-Kontext (Rabiner & Schafer 1978, Yule-Walker)
# 2. Rückwärts-AR aus Post-Gap-Kontext (gespiegeltes Signal)
# 3. Lineare Überblendung beider Vorhersagen über Lückenlänge
# 4. Pol-Stabilisierung: alle Pole |z| ≥ 0.995 auf 0.994 gespiegelt
# 5. 5 ms Boundary-Crossfade tapern an Lückenrändern
```

**Geltungsbereich**: Shellac-Material (`params["interpolation"] == "hybrid"`) und alle Gaps ≤ 50 ms.

**Wissenschaftliche Referenzen**:

- Lagrange & Marchand (2007) "Long Interpolation of Audio Signals using Linear Prediction"
- Godsill & Rayner (1998) "Digital Audio Restoration"

### §2.57c Phase-50 STFT-Konsistenz-Projektion (POCS)

**Problem**: Pass-2 (Time-Axis-Dropout-Reparatur) verwendete einmalige lineare Interpolation.

**Iterative STFT-Konsistenz-Projektion** (5 Iterationen, POCS-Schema):

```
1. Initialisierung mit linearer Interpolation der Dropout-Frames
2. ISTFT → zeitkontinuierliches Signal
3. STFT → zurück in Spektralraum
4. Undamaged Frames re-ankern (Original-Spektraldaten wiederherstellen)
5. Schritt 2–4 wiederholen (5 Iterationen)
```

Die STFT-Redundanz propagiert Spektralstruktur aus unbeschädigten Frames in Lücken.

**Wissenschaftliche Referenz**: Siedenburg & Dörfler (2013) "Audio Inpainting", JASA.

**Testpflicht**: `tests/unit/test_literature_algorithms.py` (21 Tests: Phase 09 + Phase 50).

---

## §2.58 [RELEASE_MUST] PMGG Passthrough-Erkennung (v10.0.0)

Phasen, die ihr Audio unverändert zurückgeben (z. B. `phase_31` bei CREPE confidence=0.0),
dürfen kein Goal-Scoring, Retry oder StrictConflictDecay auslösen.

**Invariante** (`backend/core/per_phase_musical_goals_gate.py`):

```python
if np.array_equal(phase_input_audio, phase_output_audio):
    # Kein Scoring, kein Retry, kein Decay
    return PhaseGateResult(accepted=True, passthrough=True)
```

**Rationale**: ~51 s überflüssige CREPE/pYIN-Inferenz pro Song bei confidence=0.0 werden eingespart.
Passthrough ist kein Qualitätsmangel — die Phase hat einfach keinen Eingriff für nötig befunden.

**VERBOTEN**: Passthrough-Audio durch alle Goal-Scoring-Pfade schicken.

### §2.58a [RELEASE_MUST] Strict-Conflict Severity-Orchestrierung (v10.0.0)

Strict-Conflict-Decays und Cap-Tightening dürfen nicht mehr rein zählbasiert sein.
Die Schwere muss reason-, psychoakustik- und zielprioritätsbasiert gewichtet werden.

**Kanonische Policy-Felder** (`song_calibration_profile.strict_conflict_policy`):

- `reason_decay_weight: dict[str, float]`
- `reason_cap_tighten_weight: dict[str, float]`
- `rollback_decay_per_family: dict[str, float]`
- `rollback_decay_floor: float`

**Normative Invarianten**:

1. `pmgg_best_effort` wird milder gewichtet als harte Klangschäden.
2. `best_effort_emergency` wird strenger gewichtet als reguläres `best_effort_rX`.
3. `artifact_freedom_rollback` muss mit wachsendem Defizit (`0.95 - artifact_freedom`) strenger werden.
4. `vocal_no_harm_rollback` mit `checks.singer_identity=False` muss strenger gewichtet werden.
5. Goal-Prioritäten müssen in die Schwere einfließen:
     - P1/P2-Regressionen strenger als P4/P5 bei gleicher Magnitude.
6. Vocal-Supremacy-Kopplung:
     - bei `panns_singing >= 0.35` müssen vokalkritische Regressionsziele
         (`natuerlichkeit`, `authentizitaet`, `artikulation`, `timbre_authentizitaet`,
         `vocal_quality`, `formant_fidelity`) zusätzlich verstärkt werden.
7. Disagreement-Guard (Unsicherheitsdämpfung):
         - wenn die letzten Family-Konflikte hohe Reason-Diversität zeigen (`>= 3` unterschiedliche
             Gründe in der jüngsten Verlaufsscheibe) und kein harter Reason vorliegt,
             MUSS `pmgg_best_effort` temporär konservativer gedämpft werden.
         - sobald ein harter Grund im jüngsten Verlauf enthalten ist
             (`artifact_freedom_rollback`, `noise_texture_rollback`,
             `vocal_no_harm_rollback`, `hf_hallucination_rescue`), MUSS diese
             Unsicherheitsdämpfung deaktiviert bleiben.

**Schwere-Normalisierung**:

- Runtime-Decay-Gewicht: Clip auf `[0.0, 1.5]`
- Event-Severity-Score: Clip auf `[0.5, 2.0]`
- Bucket-Mapping:
  - `critical` ab `>= 1.4`
  - `high` ab `>= 1.15`
  - `medium` ab `>= 0.90`
  - sonst `low`

**Runtime-Event-Pflichtfelder** (`_phase_goal_conflict_runtime.events[*]`):

- `severity_score: float`
- `severity_bucket: str`
- `severity_fingerprint: dict`

**Mindestinhalt von `severity_fingerprint`**:

- `reason_weight_base`
- `dynamic_weight`
- `vocal_presence`
- `goal_priority_peak`
- `artifact_deficit`
- `emergency_action`
- `fatigue_rebound_applied`
- `prior_soft_conflict_streak`
- `disagreement_brake_applied`
- `recent_reason_diversity`

**Strict-Report-Aggregation** (`_build_strict_phase_goal_conflict_report`):

- `regressive_phase_events[*].severity` MUSS befüllt sein.
- `regressive_weight_sum` MUSS statt reinem Event-Count in `conflict_score` einfließen.
- `runtime_severity_sum` und `runtime_severity_max` MUSS im Report enthalten sein.

**VERBOTEN**:

- reines Zählschema für `conflict_score` ohne Severity-Gewichtung.
- Vocal-Pfade als ungewichtete Standardkonflikte zu behandeln.
- Priority-Pfade (P1/P2 vs. P4/P5) gleich zu behandeln.

---

## §2.59 [RELEASE_MUST] CausalDefectReasoner Bidirektionale Konsistenz (v10.0.0)

`CAUSES` und `CAUSE_TO_PHASES` müssen bidirektional konsistent sein.

**Problem**: Eine Ursache (z. B. `vocal_harshness`) nur in `CAUSE_TO_PHASES` einzutragen ohne
korrespondierendes `CAUSES`-Feld erzeugt dead code — der Bayes-Loop iteriert **ausschließlich `CAUSES`**.

**Invariante** (`backend/core/causal_defect_reasoner.py`):

- Jeder Schlüssel in `CAUSE_TO_PHASES` MUSS einen entsprechenden Eintrag in `CAUSES` haben.
- Jeder Eintrag in `CAUSES` SOLLTE in `CAUSE_TO_PHASES` abgebildet sein (oder explizit dokumentiert,
  warum er keine direkten Phasen triggert).
- `LIKELIHOOD_FNS` muss jeden `CAUSES`-Eintrag abdecken (bei fehlendem Eintrag: Lambda → 0.0).

**Testpflicht**: Behavioral Guard Test — starkes `vocal_harshness`-Defekt-Score muss
`phase_42_vocal_enhancement` in `recommended_phases` enthalten.

**VERBOTEN**: Neue Ursache nur in einer Richtung eintragen.

---

## §2.2c Denker-Orchestrierung, Hänger-Patterns & Diagnose (konsolidiert aus Skill pipeline-debug)

### Denker-Rollendifferenzierung (§11.7a)

| Stufe | Denker | Domäne | Kurzregel |
| --- | --- | --- | --- |
| 6 | `ReparaturDenker` | Defekt-Beseitigung | Entfernt Clicks, Hum, Clipping |
| 7 | `RekonstruktionsDenker` | Rekonstruktion | Füllt Lücken, annotiert BW-Verlust |
| 8 | `RestaurierDenker` | Restaurierung | Orchestriert UV3, schützt Klangcharakter |

**Kontextfluss**: `defect_result → ReparaturDenker → RekonstruktionsDenker(+defect_result) → RestaurierDenker(+reconstruction_context) → UV3`

### §2.41 Denker-Vollkontext (v10.0.0)

- **ReparaturDenker**: 12 Material-Profile (click_iqr, click_kernel_ms, clip_threshold, hum_detect_db). Shellac IQR=4.0 → CD IQR=9.0. Era-adaptive Hum (≤1940: ≥−42 dB).
- **RekonstruktionsDenker**: 6 Material-Konfigurationen für GapReconstructor (Shellac: max 200 ms, Tape: bis 2000 ms).
- **AurikDenker**: Leitet defect_scores, defect_locations, era_decade, material an alle Stufen weiter.

### Parallelisierung

Tier 0+1 sequenziell; Era+Schlager+Medium parallel (ThreadPoolExecutor max_workers=3); Tier 6 sequenziell.

### Song-Selbstkalibrierung — Berechnungsblöcke (Reihenfolge in `_build_song_calibration_profile`)

1. Era-GP-Warmstart: ≤1940 → ×1.10; ≤1960 → ×1.00; ≥1970 → ×0.88
2. Material-Multiplikatoren (6 Materialien)
3. Per-Defekt-Family-Boost: 28 DefectTypes → 6 Familien, max +12 %
4. Spektral-Fingerprint: rolloff→reconstruction, noise_floor→denoise, wow_flutter→dynamics
5. SOFT_SATURATION-Guard: severity ≥ 0.25 → denoise −12 %, transient −7 %
6. Schlager-Profil: vocal +10 %, transient +5 %, dynamics +5 %, reconstruction ×0.95
7. Diversity-Penalty: ≥8 Defekte → global −1 % je Extra, max −6 %
8. PANNs: vocal_prob/inst_prob → Familien-Skalierung
9. Modus-Post: studio → reconstruction ×1.08, transient/vocal/instrument ×1.05

### Bekannte Hänger-Patterns (aus realen Runs)

**1. Progress stuck bei ~2 % — synchrone Carrier-Analyse in `load_audio_file()`**
`load_audio_file()` ruft intern `analyze_carrier_forensics()` → `classify_medium()` auf vollem Audio.
Bei 225s Stereo (10 M+ Samples) → 6+ Minuten synchroner Block im `BatchProcessingThread`.
Diagnose: UV3-Log "Starting restoration" erscheint nie → Blocker liegt VOR `denke()`.
Fix: `load_audio_file(path, do_carrier_analysis=False)` in allen UI/Thread-Aufrufen.

**2. Phase hängt 2+ Stunden — O(n²)-Autokorrelation im DSP-Fallback**
`np.correlate(signal, signal, mode="full")` bei 10 M+ Samples = ~10¹⁴ Operationen.
Fix: `np.array([np.dot(s[:n-k], s[k:]) for k in range(AR_ORDER+1)])` — O(n·order).
Betroffen: AR/LPC-DSP-Fallbacks in Phase 09, Phase 12 und anderen.

**3. R-Kanal kollabiert zu -111 dBFS — kumulativer Stereo-Drift**
4 Stereo-Phasen à 6–8 dB L/R-Imbalance-Delta → kumulativ > 40 dB Kollaps.
Jede Phase besteht per-Phase δ-Guard (< 0.05 Δ), Gate-Kaskade bleibt blind.
Fix: Post-Pipeline kumulative Stereo-Collapse-Guard (§2.49b).

**4. PlateauStop dämpft fälschlich ab Phase 4 für Stereo-Songs**
`_spectral_quality_score` nutzte `a[0]` statt `a[:, 0]` → immer 0.0 für Stereo →
PlateauStop aktiv. Fix: `mono = a[:, 0] if a.ndim == 2 else a`.

**5. Hauptfortschritt erreicht früh 97 % und wirkt eingefroren**
UV3-Post-Processing-Callbacks (`pct >= 86`) duerfen in der GUI den Hauptbalken nicht bis
97-98 % treiben, solange Defektabschluss, Export-Quality-Gate, Resampling/Datei-Export und
UI-Finalisierung noch folgen. Kanonischer UI-Vertrag: UV3-Analyse/Phasen belegen 9-83 %, UV3-
Post-Processing belegt 83-90 %, und 90-100 % bleiben exklusiv fuer Abschluss-/Export-Schritte.
Heartbeat-Prognosen duerfen diese 90-%-Grenze vor dem expliziten Export-/Finalisierungsblock nicht
ueberschreiten.

### Psychoakustik-Gewichtung für Tiefen-Immersion (§8.3)

| Prinzip | Gewicht | Modul |
| --- | --- | --- |
| Transient-Punch | ~40 % | TDP |
| Mikro-Dynamik | ~25 % | MDEM (400 ms, läuft zuerst) → EmotionalArcCorrection (5 s, läuft danach — nur wenn Bogen nicht erhalten) |
| Klarheit | ~20 % | SGMSE+ / OMLSA |
| Vokal-Präsenz | ~10 % | Phase 42/43 + VocalAI |
| Neurale Synthese | ~5 % | Vocos 48k (Studio, MOS < 4.3) |

---

## §2.60 [RELEASE_MUST] Rollback-Hierarchie (v10.0.0)

Quelle: `[SRC:S06,S07,S12,S13]`

**Wenn ein Gate scheitert, MUSS Aurik die nächste Stufe versuchen — nie sofort abbrechen oder exportieren.**

### [RELEASE_MUST] Vollständige Recovery-Kaskade

| Stufe | Aktion | Trigger |
| --- | --- | --- |
| 1 | **Phase-Rollback**: Einzelphase zurückrollen → vorheriges Audio, Phase-Score negativ markiert | PMGG-Fail oder artifact_freedom < 0.95 für diese Phase |
| 2 | **Strength-Reduktion**: Phase mit 50 % Strength wiederholen → neues PMGG-Check | Phase-Rollback × 2 für dieselbe Phase |
| 3 | **Carrier-Checkpoint**: Rollback auf `best_carrier_checkpoint` (nach Stufe 1–4, vor Enhancement) | HPI ≤ 0 oder artifact_freedom < 0.95 am Pipeline-Ende |
| 4 | **Pre-Pipeline-Checkpoint**: Rollback auf Audio direkt nach TDP (vor allen Phases) | Carrier-Checkpoint nicht verfügbar oder schlechter als Pre-Pipeline |
| 5 | **Input-Export**: Original degradierter Input exportieren, Status: `degraded` | Alle Recovery-Stufen ausgeschöpft |
| — | **VERBOTEN**: Leerer Export, Prozess-Abbruch ohne Ausgabe, Export mit bekanntem Artefakt | — |

### Implementierung (`backend/core/unified_restorer_v3.py`)

```python
def _recovery_cascade(self, gate_fail_reason: str) -> RestorationResult:
    """Vollständige 5-Stufen-Recovery-Kaskade."""

    # Stufe 1: Phase-Rollback (wird pro Phase inline ausgeführt, nicht hier)

    # Stufe 2: Strength-Reduktion (wird inline mit retry ausgeführt)

    # Stufe 3: Carrier-Checkpoint
    if self._best_carrier_checkpoint is not None:
        audio = self._best_carrier_checkpoint
        hpi = self._compute_hpi(audio)
        if hpi > 0:
            return RestorationResult(audio=audio, status="recovered",
                                     metadata={**self.metadata, "recovery_stage": 3,
                                               "recovery_reason": gate_fail_reason})

    # Stufe 4: Pre-Pipeline-Checkpoint
    if self._pre_pipeline_audio is not None:
        audio = self._pre_pipeline_audio
        hpi = self._compute_hpi(audio)
        if hpi > 0:
            return RestorationResult(audio=audio, status="recovered",
                                     metadata={**self.metadata, "recovery_stage": 4,
                                               "recovery_reason": gate_fail_reason})

    # Stufe 5: Input-Export — IMMER besser als Artefakt
    logger.warning("All recovery stages failed — exporting degraded input. reason=%s", gate_fail_reason)
    return RestorationResult(audio=self._original_input, status="degraded",
                             metadata={**self.metadata, "recovery_stage": 5,
                                       "recovery_reason": gate_fail_reason})
```

**Invariante**: `RestorationResult.status ∈ {"success", "recovered", "degraded"}`. Status `degraded` ist kein Fehler — er ist die korrekte Antwort wenn alle Recovery-Versuche scheitern. Status `degraded` ist **immer besser** als ein über-prozessiertes Artefakt.

---

## §2.61 [RELEASE_MUST] Output-Length-Guard (v10.0.0)

**Jede Phase und der finale Export müssen dieselbe Sample-Anzahl wie das Input-Audio haben** (±64 Samples Toleranz für Resampling-Rundung).

### Pflicht-Check in UV3 nach jeder Phase

```python
# In UV3._execute_pipeline(), nach jeder Phase:
if abs(len(output) - len(audio_in)) > 64:
    logger.error("length_mismatch phase=%s delta=%d samples",
                 phase_id, len(output) - len(audio_in))
    # Harter Crop — besser als stilles Padding oder AV-Desync
    if len(output) > len(audio_in):
        output = output[:len(audio_in)]
    else:
        output = np.pad(output, [(0, len(audio_in) - len(output))] + [(0,0)] * (output.ndim - 1))
    metadata.setdefault("length_corrections", []).append(phase_id)
```

**Betroffene Phasen-Typen** (MÜSSEN Ausgabelänge explizit sicherstellen):

- STFT/ISTFT-Phasen (Resampling-Überlauf bei Edge-Frames)
- Resampling-Phasen (Rundungsfehler ± N Samples)
- Chunk-Stitching-Phasen (Overlap-Add-Residuen)

**VERBOTEN**: Stilles Zero-Padding als primäre Längenkorrektur (maskiert den Wurzelbug, erzeugt Stille am Ende). Richtig: Phase muss Länge intern korrekt einhalten; UV3-Guard ist Fangschicht, nicht Lösung.

> Telemetrie: `metadata["length_corrections"]` (Liste betroffener Phase-IDs); CI-Test: `test_output_length_guard.py`

---

## §2.63 [RELEASE_MUST] Intro/Outro-Edge-Safety + Stereo-Lag-Invariante (v10.0.0)

**Pegelexplosionen an Intro/Outro müssen in der Entstehung vermieden werden.**
Nachträgliche Kaschierung (z. B. reiner Crossfade am Ausgang) ist nur sekundäre Fangschicht.

### Primärmechanismus (Pflicht)

Für Boundary-anfällige Phasen (insb. ML, STFT/ISTFT, Chunk-Stitching) gilt:

1. Vor Verarbeitungsaufruf: Kontext-Padding (`reflect` oder `symmetric`) beidseitig anwenden.
2. Nach Verarbeitungsaufruf: deterministisch auf Originallänge strippen.
3. Erst danach optionale sekundäre Edge-Schutzschicht (Edge-Taper) anwenden.

Referenzmuster:

```python
ctx = min(int(1.0 * sr), n_samples // 4)
audio_pad = np.pad(audio, ((0, 0), (ctx, ctx)), mode="reflect")  # channel-first stereo
processed = model(audio_pad)
processed = processed[:, ctx:ctx + n_samples]  # deterministischer Strip
```

### [RELEASE_MUST] Stereo-Lag-Invariante (Pflicht)

- Werden L und R getrennt durch denselben Algorithmus geführt, müssen `ctx`, Strip-Offset und Zielsamplezahl identisch sein.
- **VERBOTEN**: Kanalindividuelles Resampling als primäre Korrektur nach Boundary-Verarbeitung.
- `interchannel_delay_out - interchannel_delay_in` darf nicht ansteigen; §2.51a Hard-Fail bei Delay > 1 ms bleibt bindend.

### Invariante

Restaurierung darf weder neue Intro/Outro-Peaks erzeugen noch neue L/R-Zeitverschiebung einführen.

---

## §2.64 [RELEASE_MUST] Per-Phase-Score-Delta-Loop (v10.0.0)

**Jede Phase wird von einem bindenden Vorher/Nachher-Messrahmen eingeschlossen.** Keine Phase darf ohne Delta-Analyse ausgeführt werden. Das Ergebnis jeder Messung steuert, ob die Phase übernommen, zurückgerollt oder die Stärke gedämpft wird.

### `_fast_goal_snapshot()` — DSP-Proxy-Messung (≤ 200 ms)

```python
# backend/core/unified_restorer_v3.py
def _fast_goal_snapshot(self, audio: np.ndarray, phase_id: str) -> dict[str, float]:
    """DSP-only proxy measurement of all 15 Musical Goals. NO ML inference.
    timbral_ref_vector: extracted from original_audio at pipeline start;
    shifts to best_carrier_checkpoint when carrier_chain_recovery_ratio > 0.15 (§0d).
    Budget: ≤ 200 ms for 5-min stereo audio at SR=48000.
    """
    return self._proxy_goal_checker.measure_fast(
        audio=audio,
        reference=self._timbral_ref_vector,
        sr=48000,
        phase_id=phase_id,
    )
```

**Proxy-Metrik-Mapping** (DSP-only, kein ML):

| Musical Goal | DSP-Proxy | Budget |
| --- | --- | --- |
| `natuerlichkeit` | `1 − spectral_novelty` (§2.46e) + `artifact_freedom_heuristic` | 15 ms |
| `authentizitaet` | MFCC-Korrelation zur `timbral_ref_vector` (13 Koeff.) | 25 ms |
| `tonal_center` | Chroma-Kovarianz zur Referenz | 20 ms |
| `timbre_authentizitaet` | MFCC-Distanz zur Referenz (normiert, RMS-gewichtet) | 20 ms |
| `artikulation` | Onset-Dichte-Verhältnis (SNR-gewichtet) | 15 ms |
| `emotionalitaet` | Spectral-Flux-Korrelation zur Referenz | 10 ms |
| `micro_dynamics` | Crest-Factor-Verhältnis pre/post | 5 ms |
| `groove` | Beat-Alignment-Score (madmom, gecacht aus Analyse-Phase) | 0 ms |
| `transparenz` | §9.7.25 Spektral-Flatness der Oktavband-Energien (250 Hz–8 kHz), identisch-semantisch zu PMGG/finaler `TransparenzMetric` | 5 ms |
| `waerme` | LF-Energie-Anteil (ISO 226:2003, 200–2000 Hz) | 5 ms |
| `bass_kraft` | Sub-Bass-RMS (20–250 Hz) | 5 ms |
| `separation_fidelity` | M/S-Balance-Ratio (Stereo) / 1.0 (Mono) | 5 ms |
| `brillanz` | HF-Energie-Anteil (8–16 kHz) | 5 ms |
| `spatial_depth` | Cross-Correlation-Peak L/R (Stereo) / 1.0 (Mono) | 5 ms |
| `transient_energie` | Onset-Amplitude-Ratio 5 ms gegen Referenz-/Carrier-Anker | 10 ms |

**Implementierung**: `backend/core/dsp/fast_goal_proxy.py` [ROADMAP] — `FastGoalProxy.measure_fast()`

### Delta-Analyse und Entscheidungslogik

```python
def _profiled_phase_call_with_delta(self, phase_fn, audio, phase_id, **kwargs):
    # 1. Pre-Phase-Snapshot
    pre_scores = self._fast_goal_snapshot(audio, phase_id)

    # 2. Phase ausführen
    post_audio, phase_result = phase_fn(audio, **kwargs)

    # 3. Post-Phase-Snapshot
    post_scores = self._fast_goal_snapshot(post_audio, phase_id)

    # 4. Delta berechnen
    delta = {g: post_scores[g] - pre_scores[g] for g in GOAL_NAMES}
    mas_gap = {g: self._mas_targets[g] - post_scores[g] for g in GOAL_NAMES}

    # 5. P1/P2-Regression-Check (außer für _RESTORATIVE_PHASES)
    if phase_id not in _RESTORATIVE_PHASES:
        if any(delta[g] < -0.03 for g in P1P2_GOALS):
            logger.warning("phase=%s p1p2_regression delta=%s → rollback", phase_id, delta)
            self._metadata["phase_deltas"][phase_id] = {"pre": pre_scores, "delta": delta, "action": "rollback"}
            return audio, {"applied": False, "reason": "p1p2_regression"}

    # 6. MAS-Overshoot: Strength-Clamp wenn Goal > MAS + 0.03
    if any(post_scores[g] > self._mas_targets[g] + 0.03 for g in P1P2_GOALS):
        post_audio = np.clip(audio + 0.5 * (post_audio - audio), -1.0, 1.0)
        post_scores = self._fast_goal_snapshot(post_audio, phase_id)
        mas_gap = {g: self._mas_targets[g] - post_scores[g] for g in GOAL_NAMES}

    # 7. MAS-Konvergenz-Check → ggf. Pipeline-Stop
    self._check_mas_convergence(post_scores, mas_gap)

    # 8. Delta-Log (Pflicht)
    self._metadata["phase_deltas"][phase_id] = {
        "pre": pre_scores, "post": post_scores,
        "delta": delta, "mas_gap": mas_gap,
    }
    return post_audio, phase_result
```

### Invarianten

- `_fast_goal_snapshot()` **DARF KEIN** ML-Modell laden oder aufrufen (nur DSP-Proxies).
- `timbral_ref_vector` wird VOR der Pipeline einmalig extrahiert und für alle Phasen wiederverwendet.
- Delta-Log `metadata["phase_deltas"]` ist Pflicht für jeden Phasenaufruf — ohne Log kein konvergenzgesteuertes Stoppen.
- `_RESTORATIVE_PHASES` (phase_02/03/09/18/20/23/24/29/49) sind von P1/P2-Regression-Check ausgenommen (§2.29c Baseline-Capping).

### [RELEASE_MUST] Proxy-Kalibrierungs-Invarianten (v10.0.0)

Drei bekannte Proxy-Calibration-Bugs, die zu systematisch zu niedrigen Delta-Werten für **alle Import-Songs** führen:

1. **Single-Segment-Bias**: FFT-Basis auf einem einzigen Zentrum-Segment (`N//2`) ist nicht repräsentativ für den gesamten Song. Ein lauter Chord oder eine Pause treibt spektrale Proxys auf ≤ 0.20. **Pflicht**: Multi-Segment-Mittelung über 25 %/50 %/75 % des Songs; `_spec = mean([fft(seg25), fft(seg50), fft(seg75)])`.

2. **`authentizitaet`-ACF-Crash nach phase_24**: ACF auf ersten 8192 Samples ist hypersensitiv zu phase_24-Inpainting (Dropout-Repair). Ein reparierter Dropout am Songbeginn kollabiert den ACF-Peak von 0.71 → 0.06. **Pflicht**: ACF auf Zentral-Drittel `mono[N//3 : N//3+8192]` berechnen — vermeidet Intro/Dropout-reparierte Segmente.

3. **`transparenz`-Proxy für komprimiertes Pop/Schlager**: 5th/99th-Perzentil-SNR auf dem Zentrum-Segment gibt ≤ 0.20 für normales Musikmaterial, weil der 5th-Percentile in einem lauten Segment ≈ 0.1–0.2 liegt. **Pflicht**: `_p05_full + _p95_full` auf **Vollsignal** (nicht Segment) + Spectral-Flatness-Blend: `0.70 × log10(p95/p05)/4.0 + 0.30 × (1 − sfm_avg)`.

**VERBOTEN**: Jeden dieser drei Proxys in einer neuen Session auf Single-Segment zurückzuportieren — diese Bugs waren systemische Ursache für PMGG-Fehlsteuerung und persistente Proxy-Werte unter Material-Floor für alle Import-Songs.

---

## §2.65 [RELEASE_MUST] MAS-Convergence-Early-Stop (v10.0.0)

**Sobald alle P1/P2-Goals ihr song-spezifisches Maximum (`MAS`) erreicht haben, stoppt die Pipeline.** Weitere Phasen verletzen §2.45 Minimal-Intervention und riskieren Over-Processing.

### MAS-Konvergenz-Check

```python
MAS_TOLERANCE        = 0.02   # P1/P2: "erreicht" wenn mas_gap[g] ≤ 0.02
MAS_FULL_TOLERANCE   = 0.05   # P3–P5: "erreicht" wenn mas_gap[g] ≤ 0.05
MAS_OVERSHOOT_TOL    = 0.03   # Strength-Clamp wenn post_score > MAS + 0.03

def _check_mas_convergence(self, post_scores: dict, mas_gap: dict) -> None:
    """Signal early pipeline stop when MAS achieved for all P1/P2."""
    p1p2_achieved = all(mas_gap[g] <= MAS_TOLERANCE for g in P1P2_GOALS)
    p3p5_achieved = all(mas_gap[g] <= MAS_FULL_TOLERANCE for g in P3P5_GOALS)

    if p1p2_achieved:
        self._metadata["mas_achieved_at_phase"] = self._current_phase_id
        self._metadata["mas_p3p5_achieved"] = p3p5_achieved
        if p3p5_achieved:
            self._mas_fully_achieved = True   # → Pipeline sofort stoppen
        # sonst: P3–P5-Phasen können noch laufen (Groove, Raumtiefe)
```

### Drei Konvergenzzustände

| Zustand | Bedingung | Pipeline-Reaktion |
| --- | --- | --- |
| **Voll konvergiert** | Alle 15 Goals ≤ MAS_TOLERANCE/MAS_FULL_TOLERANCE | Pipeline sofort stoppen — Export |
| **P1/P2 konvergiert** | P1/P2 konvergiert; P3–P5 noch Lücken | Nur P3–P5-Phasen (Groove, Raumtiefe) weiter |
| **Nicht konvergiert** | P1/P2-Lücken vorhanden | Nächste Defekt-Phase gemäß Plan |

### Pipeline-Loop (UV3)

```python
for phase_id in planned_phases:
    if self._mas_fully_achieved:
        logger.info("MAS fully achieved at phase=%s — pipeline stopped early", self._metadata.get("mas_achieved_at_phase"))
        break
    audio, result = self._profiled_phase_call_with_delta(phase_fn, audio, phase_id)
```

### [RELEASE_MUST] Invarianten

- **VERBOTEN**: Pipeline läuft nach `_mas_fully_achieved = True` weiter.
- **VERBOTEN**: Phase ohne Pre/Post-Delta-Log (kein `_profiled_phase_call_with_delta()`-Aufruf).
- `_mas_targets` = `SongGoalTargets.targets` aus `estimate_song_goal_targets()` (§0k, Spec 09 §09.11).
- MAS-Targets sind ceiling-geclampt auf `PHYSICAL_CEILING[material]` (Spec 09 §09.11) — kein physikalisches Limit überschreiben.

> Spec 09 §09.11 für MAS-Formaldefinition + PHYSICAL_CEILING; §0k für normatives Prinzip

---

## §2.66 [RELEASE_MUST] SharedF0Context — Geteilte F0-Referenz für alle Phasen (v10.0.0)

> **Konzeptuelle Kernlücke geschlossen**: Ohne SharedF0Context schätzt jede Phase die
> Grundfrequenz unabhängig (CREPE oder PYIN). Das führt zu inkonsistenten F0-Karten über
> die Pipeline — Phase_12 (Wow/Flutter) und Phase_42 (Vocal Enhancement) arbeiten
> dann auf abweichenden F0-Spuren und können entgegengesetzte Korrekturen einführen.
> Weltklasse erfordert eine einzige kanonische F0-Spur, auf die alle Phasen zugreifen.

### §2.66a Berechnung (vor Phase-Ausführung)

```python
# Nach VocalFocusAnalyzer, vor GoalApplicabilityFilter:
# Laufzeit-Budget: ≤ 5 s/min Audio (CREPE-tiny auf CPU)
from plugins.crepe_pitch_tracker import get_crepe_pitch_tracker

_f0_tracker = get_crepe_pitch_tracker()
_f0_result = _f0_tracker.estimate(
    audio=_pipeline_input_mono,  # Mono-Downmix für F0-Analyse
    sr=sr,
    hop_length_ms=10,            # 10 ms Hop (= 480 Samples @ 48 kHz)
    model_size="tiny",           # tiny: 4.2 MB, akzeptable Genauigkeit, schnell
    viterbi=True,                # Viterbi-Decoder für glatte F0-Kontur
)
# {
#   "f0_hz":      np.ndarray,    # f₀ pro Frame [Hz], 0.0 = unvoiced
#   "confidence": np.ndarray,    # Konfidenz [0, 1] pro Frame
#   "voiced_flag":np.ndarray,    # bool-Maske voiced/unvoiced
#   "hop_length": int,           # samples
#   "times_s":    np.ndarray,    # Zeitpunkte [s]
# }
_restoration_context["shared_f0"] = _f0_result
logger.info(
    "shared_f0: voiced_ratio=%.2f%%, f0_range=[%.1f, %.1f] Hz",
    _f0_result["voiced_flag"].mean() * 100,
    _f0_result["f0_hz"][_f0_result["voiced_flag"]].min() if _f0_result["voiced_flag"].any() else 0.0,
    _f0_result["f0_hz"][_f0_result["voiced_flag"]].max() if _f0_result["voiced_flag"].any() else 0.0,
)
```

### §2.66b Zugriff durch Phasen

```python
# Jede Phase, die F0 benötigt, MUSS auf shared_f0 zugreifen:
_shared_f0 = kwargs.get("shared_f0") or _restoration_context.get("shared_f0")
if _shared_f0 is None:
    # Fallback: eigene Schätzung (PYIN, ressourcenschonend)
    logger.warning("phase=%s: shared_f0 not available — local F0 fallback", _phase_id)
    _f0_hz = _estimate_f0_local(audio, sr)
else:
    _f0_hz = _shared_f0["f0_hz"]
    _voiced = _shared_f0["voiced_flag"]
```

### [RELEASE_MUST] §2.66c Phasen-Bindung (MUSS shared_f0 nutzen)

| Phase | F0-Nutzung |
| --- | --- |
| `phase_12_wow_flutter` | Pitch-Warp-Referenz; Segmente mit voiced_flag=True |
| `phase_29_tape_hiss` | Masking-Profil pro Voiced-Frame (Harmonik-Schutz) |
| `phase_42_vocal_enhancement` | Formant-Extraktion relativ zu f₀ |
| `phase_03_denoise` | Harmonik-Schutz: preserve bei voiced_flag=True |
| `phase_55_diffusion_inpainting` | AR-Prior: f₀-konditioniertes Inpainting |

**VERBOTEN**: Neue Phase, die F0 benötigt, schätzt F0 eigenständig ohne `shared_f0`-Fallback-Prüfung.

### §2.66d Invarianten

- `shared_f0` ist **read-only** für alle Phasen (kein Schreiben in `_restoration_context["shared_f0"]`).
- F0-Schätzung läuft auf Mono-Downmix, auch bei Stereo-Material (L+R-Mittel).
- Non-blocking: Exception in CREPE → `shared_f0 = None` → lokaler Fallback in betroffener Phase.
- `voiced_ratio < 0.05`: Instrumentalstück — `shared_f0` leer befüllt, keine F0-abhängige Verarbeitung.

### §2.66e RecordingChainProfiler- und RestorationMemory-Priors [RELEASE_MUST]

`RecordingChainProfiler.profile_chain()` und `RestorationMemory.get_prior()` dürfen nicht nur Metadaten erzeugen. Ihre Ausgaben MÜSSEN direkt in `GPParameterOptimizer.propose_pareto(..., chain_hint=..., memory_prior=...)` eingehen.

- `chain_hint["strength_scale"]` skaliert konfliktträchtige Strength-/Boost-/Ratio-Parameter konservativ, wenn mehrere Causes derselben physikalischen Aufnahmekette aktiv sind.
- `memory_prior["phase_params"]` wird bounds-geclampt in den Kandidaten geblendet, gewichtet durch `hpi_achieved`.
- Beide Priors sind non-blocking. Fehlende oder korrupte Daten fallen auf den normalen GP-/UCB-Pfad zurück.
- VERBOTEN: `chain_hint` nur in `_restoration_context` speichern, ohne den GP-Kandidatenraum zu beeinflussen.

---

## §2.67 [RELEASE_MUST] DynamicArcProtection — Schutzmechanismus für den Dynamikbogen (v10.0.0)

> **Konzeptuelle Kernlücke geschlossen**: Jede Phase kann lokal korrekt arbeiten und
> trotzdem den Gesamt-Dynamikbogen des Titels flachdrücken. Akkumulierte Kompression
> über 6–8 NR/EQ-Phasen macht aus einem emotionalen Titel mit Piano-Forte-Kontrast
> eine gleichmäßige "Schachtel" ohne Drama. §2.35e (Spec 01) ist das Mess-Korrelat.

### §2.67a Pre-Pipeline Envelope-Referenz

```python
# Unmittelbar nach Silence-Mask-Computation, vor allen Phasen:
from backend.core.musical_goals.emotional_arc_metric import get_emotional_arc_metric

_arc_metric = get_emotional_arc_metric()
_arc_envelope_ref = _arc_metric.extract_loudness_envelope(
    audio=_pipeline_input,
    sr=sr,
    window_ms=400,    # momentane LUFS, 400 ms Fenster (BS.1770-5)
)
_restoration_context["arc_envelope_ref"] = _arc_envelope_ref
# {
#   "lufs_curve":      np.ndarray,   # momentane LUFS pro Frame
#   "klimax_segments": List[int],    # Top-5% Energie-Frame-Indizes
#   "piano_segments":  List[int],    # Bottom-10% Energie-Frame-Indizes
#   "dynamic_range":   float,        # Differenz max–min in LUFS
# }
```

### §2.67b Post-Pipeline Check

```python
# In UV3._holisticperceptualgate(), nach allen Phasen:
_arc_result = _arc_metric.measure_emotional_arc_preservation(
    audio_orig=_pipeline_input,
    audio_restored=_best_sig,
    sr=sr,
    arc_envelope_ref=_restoration_context["arc_envelope_ref"],
)
metadata["emotional_arc"] = _arc_result

if _arc_result["emotional_arc_score"] < 0.80:
    logger.warning(
        "dynamic_arc_guard: arc_score=%.3f < 0.80 — triggering recovery_cascade",
        _arc_result["emotional_arc_score"],
    )
    _recovery_cascade(reason="dynamic_arc_score_too_low", metadata=metadata)
    # Non-blocking: kein Export-Stopp; beste bisherige Version exportieren
```

### §2.67c Phasen-Level-Arc-Guard

```python
# Nach jeder NR/Dynamik-Phase (phase_29, phase_03, phase_26, phase_35):
_post_arc = _arc_metric.extract_loudness_envelope(_post_phase_audio, sr)
_arc_correlation = np.corrcoef(
    _restoration_context["arc_envelope_ref"]["lufs_curve"],
    _post_arc["lufs_curve"]
)[0, 1]

if _arc_correlation < 0.80:
    logger.warning("phase=%s: arc_correlation=%.3f < 0.80 — rollback phase", phase_id, _arc_correlation)
    return _pre_phase_audio  # Phase-Level-Rollback
```

### §2.67d Invarianten

- Envelope-Referenz wird **einmal** vor der Pipeline berechnet — niemals überschrieben.
- Arc-Check ist **non-blocking** für den Export. Recovery-Kaskade gibt dem System die Chance,
  sich selbst zu korrigieren (Stärke-Reduktion, Checkpoint-Rollback).
- `arc_correlation ≥ 0.80` per Phase; `emotional_arc_score ≥ 0.88` nach Pipeline (Studio 2026: ≥ 0.90).
- Gilt für Restoration UND Studio 2026 — der Dynamikbogen ist heilig in beiden Modi.
- Laufzeit: ≤ 500 ms/min Audio (nur LUFS-Berechnung, kein ML).

### §2.67e Phase-Koalitionen und VFA-Zonen-Steuerung [RELEASE_MUST]

Phasen, die eine gemeinsame physikalische Ursache behandeln, dürfen nicht so bewertet werden, als sei jede einzelne Phase ein unabhängiger Eingriff. UV3 MUSS Koalitions- und VFA-Kontext vor `PMGG.wrap_phase()` wirksam machen:

- Koalitionskandidaten: Tape-Transport (`phase_12`, `phase_25`, `phase_29`), Vinyl-Träger (`phase_04`, `phase_09`, `phase_28`), Reverb/Vokalproduktion (`phase_19`, `phase_20`, `phase_42`, `phase_49`). §0a-verbotene Phasen dürfen in Restoration nie Koalitionsmitglied sein.
- `frisson_zones`, `vibrato_zones`, `passaggio_zones`, `breath_segments` aus `VocalFocusAnalyzer` sind Steuerdaten. Sie müssen Strength-Caps, Energy-Bias oder Dry/Wet-Schutz beeinflussen, nicht nur Logging.
- Vibrato-Zonen: NR/Dereverb/Gate-Phasen max. `strength=0.20`.
- Frisson-Zonen: NR/Dynamics/Inpainting/Dereverb max. `strength=0.45`.
- Passaggio-Zonen: Vocal-/De-Esser-/NR-Phasen max. `strength=0.35` und `passaggio_energy_bias_db=-3.0`.
- Phasen erhalten `vocal_zone_strength_policy` und dürfen diese Policy weiter segmental verfeinern; globale Caps in UV3 sind die Mindestschutzschicht.

---

## §2.68 [RELEASE_MUST] Structural Silence Isolation Protocol (SSIP) — v10.0.0

> **Systemproblem**: Generative und inpainting-basierte Phasen erzeugen katastrophal laute
> Artefakte in Stille-Zonen (Intro, Outro, Fade, Pre-Roll). Konventionelle Silence-Masken,
> Gain-Guards und Post-Processing-Clips greifen nicht, weil das Problem **architekturell**
> ist, nicht parametrisch. Dieser Abschnitt spezifiziert die drei Failure-Modes und ihre
> kanonische Lösung auf Architektur-Ebene.

### §2.68a Failure-Mode-Analyse (warum konventionelle Regeln versagen)

**Failure Mode 1 — Gap-Detektor klassifiziert Stille-Musik-Grenze als Dropout**

```
Ursache:
  Andere Phasen (NR, EQ, BW-Erweiterung) fügen kleine Artefakte in Stille-Zonen ein.
  Die Kontrast-Bedingung im Gap-Detektor (pre_db > gap_db + 8.0 dB) trifft dann zu:
  Musik  : −12 dBFS (hoch) vs. Stille+Artefakt : −55 dBFS → Kontrast = 43 dB ≥ 8 dB
  → Gap-Detektor: "Dropout erkannt" → Inpainting generiert Musik in die Stille.

Versagen der konventionellen Lösung:
  Silence-Mask wird als repaired_gap_samples-Filter NACH Gap-Detektion angewendet.
  Wenn aber die Gap-Grenzen durch verarbeitetes Audio (mit Artefakten) verschoben wurden,
  ist der Overlap-Check unzuverlässig: Gap beginnt bei Sample X+100 statt X,
  liegt außerhalb der Silence-Zone und wird nicht gefiltert.
```

**Failure Mode 2 — Modell-Kontext-Kontamination über Stille-Grenzen**

```
Ursache:
  Diffusions- und AR-Inpainting-Modelle verwenden ein Kontext-Fenster (typisch 2–10 s).
  Liegt eine Gap-Region innerhalb von 2 s einer Stille-Grenze, enthält das Kontext-Fenster
  sowohl Musik als auch Stille. Das Modell generiert musikalische Kontinuität → füllt die
  Stille mit generiertem Inhalt bei voller Musiklautstärke.

Versagen der konventionellen Lösung:
  Eine Post-Processing-Maske kann nur das Output-Signal kappen. Aber das Modell hat
  bereits energie-dominante Inhalte erzeugt, die in benachbarte Zonen (±200 ms der
  Stille-Grenze) "durchbluten" — weil ISTFT/Overlap-Add den generierten Inhalt
  über Fenster-Grenzen hinaus verteilt.
```

**Failure Mode 3 — Silence-Mask-Null-Propagation**

```
Ursache:
  silence_mask ist None wenn UV3 sie nicht explizit in kwargs["silence_mask"] oder
  restoration_context["silence_mask"] injiziert. In diesem Fall überspringt phase_55
  die gesamte Schutzlogik ohne Fehler oder Warnung. Keine Test-Assertion prüft das.

Versagen der konventionellen Lösung:
  Schwellwerte und Guards existieren, werden aber nie ausgeführt wenn silence_mask=None.
  Kein Logging "silence_mask not available — all silence zones unprotected".
```

### §2.68b Kanonische Lösung — Hard Isolation Architecture

**Prinzip**: Stille darf das Inpainting-Modell nie als Eingabe-Kontext erreichen.
Isolation vor der Verarbeitung — kein Post-Processing.

```python
# backend/core/dsp/structural_silence_isolation.py
# Singleton: get_structural_silence_isolator()

class StructuralSilenceIsolator:
    """
    Architekturell trennt Stille von Audio-Inhalt vor jeder generativen Verarbeitung.

    NICHT: Post-Processing-Maske (greift nicht bei Kontextkontamination).
    SONDERN: Vorverarbeitungs-Isolation — das Modell sieht nur Audio-Segmente,
             nie Stille, nie Stille-Musik-Grenzen.
    """

    # Silence-Schwellen — material-adaptiv (Singleton-Cache)
    SILENCE_THRESHOLDS_DBFS = {
        "shellac":    -22.0,
        "vinyl":      -33.0,
        "cassette":   -43.0,
        "reel_tape":  -50.0,
        "cd_digital": -57.0,
        "unknown":    -45.0,
    }

    # Sicherheitspuffer: Gap darf nicht näher als CONTEXT_GUARD_MS an Stille sein
    CONTEXT_GUARD_MS = 1500.0  # 1.5 s — typisches AR-Kontextfenster

    def detect_structural_silence_zones(
        self,
        audio_original: np.ndarray,  # ORIGINAL, vor Verarbeitung
        sr: int,
        material_key: str = "unknown",
    ) -> List[Tuple[int, int]]:
        """
        Erkennt strukturelle Stille aus dem ORIGINAL-Audio (vor allen Phasen).
        Gibt (start_sample, end_sample)-Zonen zurück.

        Bedingungen für "strukturelle Stille" (UND-Verknüpfung):
        1. RMS < SILENCE_THRESHOLDS_DBFS[material_key] (200 ms-Fenster)
        2. Dauer >= 300 ms (kürzere Pausen sind musikalische Pausen, keine Stille)
        3. Zone muss am ANFANG oder ENDE des Signals liegen, ODER > 1000 ms dauern
           (Mitte des Stücks: nur lange Pausen sind "strukturelle Stille")

        WICHTIG: Aus dem ORIGINAL-Audio berechnen — niemals aus verarbeitetem Audio.
        """
        ...

    def split_at_silence_boundaries(
        self,
        audio: np.ndarray,
        sr: int,
        silence_zones: List[Tuple[int, int]],
    ) -> List[Dict]:
        """
        Splittet Audio in Segmente, die KEINE Stille-Zonen enthalten.
        Stille-Zonen werden als separate "silence"-Segmente zurückgegeben.

        Returns: [
            {"type": "audio",   "start": int, "end": int, "data": np.ndarray},
            {"type": "silence", "start": int, "end": int, "data": np.ndarray},
            ...
        ]

        Garantie: Kein "audio"-Segment enthält Stille aus silence_zones.
        Garantie: Kein "audio"-Segment liegt näher als CONTEXT_GUARD_MS an Stille-Grenze.
        → Gap-Detektor und Modell sehen niemals Stille als Nachbar.
        """
        ...

    def reassemble_from_segments(
        self,
        segments: List[Dict],
        original_silence_audio: np.ndarray,
        n_samples_total: int,
    ) -> np.ndarray:
        """
        Fügt verarbeitete Audio-Segmente und ORIGINAL-Stille-Daten zusammen.

        Stille-Segmente: ORIGINAL-Daten (aus pre-Pipeline-Audio) — niemals verarbeitet.
        Audio-Segmente: verarbeitete Daten aus dem Inpainting.

        HARD RULE: In reassembliertem Output an Stille-Zonen-Positionen sind
        AUSSCHLIESSLICH die Original-Samples zulässig. Keine Exception.
        """
        ...

    def post_inpainting_silence_audit(
        self,
        audio_before_inpainting: np.ndarray,
        audio_after_inpainting: np.ndarray,
        silence_zones: List[Tuple[int, int]],
        sr: int,
    ) -> np.ndarray:
        """
        Post-Inpainting-Sicherheitsnetz (letzter Layer).

        Für jede Stille-Zone:
          1. Messe Energie in audio_after_inpainting
          2. Messe Energie in audio_before_inpainting
          3. Wenn after > before + 3 dB: HARD RESET — Original-Samples einsetzen
             (kein Clipping, kein Mixing — vollständiger Reset auf original_audio)

        Reset-Schwelle: +3 dB über Pre-Inpainting-Pegel (kein Gain-Creep erlaubt).
        VERBOTEN: Clamp/Clip als Alternative zu Hard-Reset — Clamp erzeugt
        Signalverzerrung; Hard-Reset reproduziert das Original exakt.

        Returns: audio mit zurückgesetzten Stille-Zonen.
        """
        ...
```

### §2.68c Pflicht-Integration in Phase_55 und Phase_24

```python
# Pflicht-Pattern für JEDE generative/inpainting Phase:

def _run_inpainting_with_ssip(audio, sr, silence_zones, inpainting_fn, **kwargs):
    """
    Wraps jede Inpainting-Funktion mit SSIP.
    Muss in phase_55 und phase_24 verwendet werden — kein direkter Aufruf mehr.
    """
    _isolator = get_structural_silence_isolator()

    # Layer 1: Isolation — Modell sieht keine Stille
    segments = _isolator.split_at_silence_boundaries(audio, sr, silence_zones)

    processed_segments = []
    for seg in segments:
        if seg["type"] == "silence":
            # Stille: immer Original, niemals verarbeiten
            processed_segments.append(seg)
        else:
            # Audio-Segment: Inpainting anwenden
            processed_data = inpainting_fn(seg["data"], sr, **kwargs)
            processed_segments.append({**seg, "data": processed_data})

    # Layer 2: Reassembly — Stille-Zonen aus Original
    result = _isolator.reassemble_from_segments(
        processed_segments,
        original_silence_audio=audio,  # Pre-Inpainting-Audio für Stille-Referenz
        n_samples_total=audio.shape[-1],
    )

    # Layer 3: Post-Audit — letztes Sicherheitsnetz
    result = _isolator.post_inpainting_silence_audit(
        audio_before_inpainting=audio,
        audio_after_inpainting=result,
        silence_zones=silence_zones,
        sr=sr,
    )

    return result
```

### §2.68d Null-Propagation-Guard (Failure Mode 3)

```python
# In phase_55 und phase_24: Pflicht-Assertion bei Stille-Zonen-Verfügbarkeit
_silence_zones = _get_structural_silence_zones(kwargs, audio_original, sr, material_key)
# _get_structural_silence_zones() ist nie None — berechnet eigenständig wenn nötig:

def _get_structural_silence_zones(kwargs, audio_original, sr, material_key):
    """MUSS immer gültige Stille-Zonen zurückgeben — niemals None oder leere Liste
    ohne Berechnung."""
    # Versuch 1: aus kwargs
    silence_zones = kwargs.get("structural_silence_zones")
    if silence_zones is not None:
        return silence_zones

    # Versuch 2: aus restoration_context
    ctx = kwargs.get("restoration_context", {})
    silence_zones = ctx.get("structural_silence_zones")
    if silence_zones is not None:
        return silence_zones

    # Versuch 3: Eigenständig berechnen aus Original-Audio
    # NIEMALS auf None zurückfallen — das deaktiviert den gesamten Schutz.
    logger.warning(
        "SSIP: structural_silence_zones nicht in context — "
        "eigenständige Berechnung aus original_audio (Fallback, non-blocking)"
    )
    isolator = get_structural_silence_isolator()
    return isolator.detect_structural_silence_zones(audio_original, sr, material_key)
```

### §2.68e UV3-Pflichten (Pre-Pipeline)

```python
# In UV3._execute_pipeline(), nach Silence-Mask-Berechnung:
# structural_silence_zones wird EINMALIG aus ORIGINAL-Audio berechnet
# und in _restoration_context injiziert.

_ssip = get_structural_silence_isolator()
_structural_silence_zones = _ssip.detect_structural_silence_zones(
    audio_original=_pipeline_input,  # ORIGINAL vor allen Phasen
    sr=sr,
    material_key=material_type,
)
_restoration_context["structural_silence_zones"] = _structural_silence_zones
logger.info(
    "SSIP: %d strukturelle Stille-Zone(n) detektiert — "
    "Inpainting-Isolation aktiv",
    len(_structural_silence_zones),
)

# Wenn 0 Zonen: kein Fehler — Titel beginnt/endet mit Musik (kein Fade-In/Out)
# VERBOTEN: silence_zones=None als Ergebnis — immer leere Liste [] als Default.
```

### §2.68f Invarianten (absolut, keine Phase-Override)

1. **Isolation vor Verarbeitung**: Kein Inpainting-Modell darf Stille als Kontext erhalten.
2. **Original-Stille ist sakrosankt**: Stille-Zonen im Export MÜSSEN bit-identisch mit dem Pre-Pipeline-Original sein (oder energie-äquivalent innerhalb ±0.5 dB).
3. **Context-Guard**: Kein Gap darf innerhalb `CONTEXT_GUARD_MS` (1500 ms) einer strukturellen Stille-Zone verarbeitet werden — er wird als Boundary-Gap klassifiziert und mit `_conservative_boundary_fill` gefüllt (DSP, kein generatives Modell).
4. **Hard-Reset, kein Clamp**: Wenn Stille-Zone Energie enthält → Reset auf Original-Samples. Clamp produziert Verzerrungsartefakte. Hard-Reset ist immer sicherer.
5. **Material-adaptive Stille-Schwelle**: SILENCE_THRESHOLDS_DBFS pro material_key — Shellac hat anderen Rauschboden als CD.
6. **Gilt für**: phase_55 (Diffusions-Inpainting) + phase_24 (Dropout-Repair) + jede zukünftige Phase mit generativer oder AR-basierter Lücken-Füllung.

### §2.68g CI-Tests (Pflicht, [RELEASE_MUST])

```python
# tests/unit/test_structural_silence_isolation.py

def test_ssip_no_energy_in_silence_zone_after_inpainting():
    """Silence-Zone darf nach Inpainting nicht lauter als Original sein."""
    # Synthetisches Audio: 1 s Stille + 3 s Musik + 1 s Stille
    audio = np.concatenate([np.zeros(48000), np.random.randn(144000) * 0.3, np.zeros(48000)])
    # Inpainting simulieren (worst case: füllt alles mit Energie)
    noisy_inpainted = audio + np.random.randn(len(audio)) * 0.1

    isolator = get_structural_silence_isolator()
    zones = isolator.detect_structural_silence_zones(audio, 48000, "unknown")
    result = isolator.post_inpainting_silence_audit(audio, noisy_inpainted, zones, 48000)

    for start, end in zones:
        energy_result = np.sqrt(np.mean(result[start:end] ** 2) + 1e-12)
        energy_orig = np.sqrt(np.mean(audio[start:end] ** 2) + 1e-12)
        assert energy_result <= energy_orig * 1.12  # max +1 dB Toleranz

def test_ssip_null_propagation_guard():
    """Ohne injizierte silence_zones MUSS eigenständige Berechnung erfolgen."""
    audio = np.concatenate([np.zeros(48000), np.random.randn(144000) * 0.3, np.zeros(48000)])
    zones = _get_structural_silence_zones({}, audio, 48000, "unknown")
    assert zones is not None
    assert len(zones) >= 1  # mindestens eine Zone muss gefunden werden

def test_ssip_context_guard_ms():
    """Gap näher als CONTEXT_GUARD_MS an Stille → kein generatives Inpainting."""
    # Wird implizit durch split_at_silence_boundaries sichergestellt:
    # Kein Audio-Segment enthält eine Gap-Region innerhalb CONTEXT_GUARD_MS von Stille.
    ...

def test_ssip_reassembly_bit_exact_silence():
    """Stille-Zonen in Reassembly sind bit-identisch mit Original."""
    audio = np.concatenate([np.zeros(48000), np.random.randn(144000) * 0.3, np.zeros(48000)])
    isolator = get_structural_silence_isolator()
    zones = isolator.detect_structural_silence_zones(audio, 48000, "unknown")
    segments = isolator.split_at_silence_boundaries(audio, 48000, zones)
    result = isolator.reassemble_from_segments(segments, audio, len(audio))

    for start, end in zones:
        np.testing.assert_array_equal(
            result[start:end], audio[start:end],
            err_msg=f"Stille-Zone [{start}:{end}] ist nicht bit-identisch nach Reassembly"
        )
```

## v10: PIM & RLP

Der **Perceptual Intensity Mapper (PIM)** wird VOR dem Phasen-Loop ausgeführt und kalibriert 10 Frequenzbänder × N Song-Sektionen. Der **Reflective Listening Pass (RLP)** läuft NACH dem Loop und bessert Restprobleme nach.

## v10.0.0: Phasen 51–66 — Erweiterte Restaurierungsphasen

| Phase | Modul | Klasse | Beschreibung |
|-------|-------|--------|-------------|
| 51 | `phase_51_drums_enhancement` | `DrumsEnhancementV1` | Schlagzeug-Transienten-Restaurierung |
| 52 | `phase_52_piano_restoration` | `PianoRestorationV1` | Klavier-Harmonik-Rekonstruktion |
| 53 | `phase_53_semantic_audio` | `SemanticAudioPhase` | BPM/Key/Genre-Analyse (Metadaten) |
| 54 | `phase_54_transparent_dynamics` | `TransparentDynamicsV1` | Transparente Dynamik-Prozessierung |
| 55 | `phase_55_diffusion_inpainting` | `DiffusionInpaintingPhase` | ML-basierte Lückenfüllung |
| 56 | `phase_56_spectral_band_gap_repair` | `SpectralBandGapRepairPhase` | HEAD_WEAR Spektralband-Lücken |
| 57 | `phase_57_print_through_reduction` | `PrintThroughReductionPhase` | Bidirektionale LMS Print-Through |
| 58 | `phase_58_lyrics_guided_enhancement` | `Phase58LyricsGuidedEnhancement` | Lyrics-geführte Stimmoptimierung |
| 59 | `phase_59_modulation_noise_reduction` | `ModulationNoiseReductionPhase` | Tonband-Modulationsrauschen (§6.7) |
| 60 | `phase_60_inner_groove_distortion_repair` | `InnerGrooveDistortionRepairPhase` | Vinyl-Innenrillen-Verzerrung (§6.8) |
| 61 | `phase_61_groove_echo_cancellation` | `GrooveEchoCancellationPhase` | Pre/Post-Echo-Entfernung (§6.9) |
| 62 | `phase_62_crosstalk_cancellation` | `CrosstalkCancellationPhase` | Stereo-Übersprech-Kompensation (§6.10) |
| 63 | `phase_63_intermodulation_reduction` | `IntermodulationReductionPhase` | IMD-Artefakt-Reduktion (§6.11) |
| 64 | `phase_64_tape_splice_repair` | `TapeSpliceRepairPhase` | Physische Klebestellen-Reparatur (§6.12) |
| 65 | `phase_65_vocal_naturalness_restoration` | `VocalNaturalnessRestorationPhase` | Formant/Vibrato-Erhalt (§6.13) |
| 66 | `phase_66_stem_targeted_nr` | `StemTargetedNRPhase` | Source-Separation Rauschminderung (§6.14) |

Alle Phasen werden über `backend/core/phases/__init__.py` mit bedingten Imports exportiert
(try/except ImportError, Graceful Degradation).

## §2.46a: Transferkette in Export-Metadaten (v10.0.0)

**Pflicht**: Jeder Export (WAV, FLAC, MP3, AIFF, OGG) muss die erkannte
Tonträgerkette als Metadaten-Tag einbetten.

- **API**: `backend/exporter.py` → `set_chain_metadata()` / `_build_chain_metadata()`
- **Format**: ID3 `COMM` (MP3/AIFF), Vorbis `COMMENT` (FLAC/OGG)
- **Inhalt**: `"Chain: Vinyl→Cassette→WAV → 3 → light"`
- **Thread-Sicherheit**: `threading.Lock`-geschützter Modul-Speicher
- **BWF**: Zusätzlich BWF `bext`-Chunk via `bwf_writer.write_bwf_chunks()` für WAV/RF64

Die Tonträgerkette wird vor dem Export vom `TontraegerketteDenker` per
`exporter.set_chain_metadata()` gesetzt und während `export_audio()` via
`_build_chain_metadata()` ausgelesen und an `MetadataPreserver.transfer(transfer_chain=...)`
weitergereicht.

## §2.46b: Deep-Transfer-Chain — Mehrstufige Tonträgerketten-Inferenz (v10.0.0)

**Pflicht**: Die vollständige Tonträgerkette (z. B. `reel_tape → vinyl → cassette → mp3_low`)
muss aus allen verfügbaren Quellen rekonstruiert werden. Eine Verkürzung auf die letzte
Stufe (z. B. `mp3_low`) ist unzulässig.

### 2.46b-I: Datenquellen für die Ketten-Rekonstruktion

Drei unabhängige Signal-Pfade speisen die Ketten-Inferenz:

| Quelle | Modul | Signal-Typ | Was sie liefert |
|--------|-------|-----------|-----------------|
| 1. Digitaler Container | `MediumDetector` via `detect(audio, sr, file_ext=…)` | Dateiendung + Spektral-Fingerprint | Letzte Kettenstufe (z. B. `mp3_low`) |
| 2. Physikalische Defekte | `DefectScanner` via `auto_detected_material` | Crackle, Wow, Flutter, Hiss | Analoges Zwischenmedium (z. B. `cassette`) |
| 3. Aufnahme-Ära | `EraClassifier` via `material_prior` | Frequenzgang, Rauschtextur, Produktions-Marker | Ursprünglicher Primärträger (z. B. `reel_tape`) |

### 2.46b-II: DefectScanner auto_detected_material

Der `DefectScanner` detektiert das Material **unabhängig vom externen Hint**:

```python
# DefectScanner.scan() — auto-detection läuft IMMER:
_auto_material = self._auto_detect_material(audio)  # §6.6.1
material_type = hint or _auto_material              # Hint hat Vorrang für Thresholds

# Ergebnis speichert BEIDE Werte:
DefectAnalysisResult(
    material_type=material_type,                    # Für Threshold-Setup verwendet
    auto_detected_material=_auto_material,          # Was der Scanner WIRKLICH sieht
)
```

**Regel**: Wenn `auto_detected_material ≠ material_type` (z. B. Hint=`mp3_low` aber
auto=`cassette`), MUSS die Ketten-Inferenz das auto-detektierte Material als
alternative Kettenstufe prüfen. Der Hint bestimmt die Defekt-Thresholds, aber
**nicht** die Ketten-Rekonstruktion.

### 2.46b-III: Deep-Transfer-Chain Injection in pre_analysis

`backend/core/pre_analysis.py` baut die Kette nach folgendem Algorithmus:

```
1. _chain = MediumDetector.transfer_chain          (z.B. ["mp3_low"])
2. _era_material = EraClassifier.material_prior    (z.B. "reel_tape")
3. _defect_material = DefectAnalysisResult.auto_detected_material  (z.B. "cassette")
   Fallback: DefectAnalysisResult.material_type
4. Für jedes analoge Material aus {2, 3}:
   Wenn nicht in _chain → VOR der ersten digitalen Stufe einfügen
5. Vinyl-Inference (§2.46b-IV): Wenn reel_tape + cassette + 1950≤decade≤1990
   → "vinyl" zwischen reel_tape und cassette einfügen

Ergebnis: reel_tape → vinyl → cassette → mp3_low
```

### 2.46b-IV: Vinyl-Inference (Heuristik, kein Audio-Eingriff)

**Bedingung**: Alle drei Kriterien müssen erfüllt sein:

1. `reel_tape` ist in der Kette (Master-Aufnahme auf Band)
2. `cassette` ist in der Kette (Consumer-Überspielung)
3. Die Aufnahme-Ära liegt zwischen 1950 und 1990 (Vinyl-Ära)

**Aktion**: `"vinyl"` wird VOR `"cassette"` in die Kette eingefügt.

**Begründung**: In der Vinyl-Ära (1950–1990) war die kommerzielle Veröffentlichung
einer Band-Aufnahme praktisch immer auf Vinyl. Die Kassette war das Consumer-Format,
aber die Vinyl-Pressung war der Zwischenschritt vom Masterband zum Consumer-Markt.

**Spezifikationstreue**: Keine Audio-Veränderung — rein logische Inferenz.

### 2.46b-V: Minimal-Reproduktions-Skript

```bash
# Reproduktion der Ketten-Inferenz (ohne GUI):
cd Aurik_Standalone
python3 -c "
from backend.core.pre_analysis import run_pre_analysis
import soundfile as sf
audio, sr = sf.read('test_song.mp3')
result = run_pre_analysis(audio, sr, file_path='test_song.mp3')
md = result.medium
print(f'Chain: {md.chain_label}')
print(f'Primary: {md.primary_material}')
print(f'Multi-Gen: {md.is_multi_generation}')
"
```

## §2.46c: Phase 12 Material-Adaptiver Pitch-Span-Threshold (v10.0.0)

**Pflicht**: Phase 12 (Wow/Flutter) darf bei Cassette/Reel-Tape mit
DefectScanner-bestätigtem Wow/Flutter (Score ≥ 0.70) NICHT durch starren
100-Cent-Threshold blockiert werden.

- **Cassette/Reel-Tape + wow ≥ 0.70**: Threshold = 400 Cents
- **Alle anderen**: Threshold = 100 Cents
- Datenfluss: `process()` → `self._mat_type_for_stretch` → `_calculate_stretch_factors()`
- Datei: `backend/core/phases/phase_12_wow_flutter_fix.py`

### v10-Amendment (2026-07-12)

**Bidirektionale Genre↔Medium-Validierung** (v10.0.3, 2026-07-12):

Die Pre-Analysis-Pipeline führt nach Medium- und Genre-Erkennung eine
bidirektionale Validierung durch:

1. **Medium → Genre** (`get_genre_constraints(chain)`):
   - Nutzt `_MEDIUM_EXCLUDES_GENRES` (8 Medien × Ausschlusslisten)
   - Nutzt `_MEDIUM_PREFERRED_GENRES` (13 Medien × Präferenzlisten)
   - Beispiel: Schellack + Hip-Hop → WARNING "AUSGESCHLOSSEN"
   - Beispiel: Vinyl + Schlager → DEBUG "passt (preferred)"

2. **Genre + Sprache → Kette** (`_best_matching_chain(genre, language)`):
   - Kette wird mit Genre- und Sprach-Parametern neu gematcht
   - `_LANGUAGE_MEDIUM_BONUS`: Deutsch → Schellack+0.15, Japanisch → CD+0.15
   - Verfeinerte Kette ersetzt Original im `MediumResult`

3. **Datenfluss**:

   ```
   Import → MediumDetect → GenreDetect → Bidirektionale Validierung
   → Chain-Refinement → Deep-Transfer-Chain → Export
   ```

**Implementierung**:

- `backend/core/pre_analysis.py` → Bidirektionale Validierung vor Chain-Injection
- `backend/core/genre_classifier.py` → `SchlagerClassificationResult.language_code`
- `forensics/medium_detector.py` → `get_genre_constraints()`, `_best_matching_chain(genre, language)`

**Knowledge Base** (alle in `forensics/medium_detector.py`):

- `_MEDIUM_ORDER`: 20 Tonträger chronologisch (1877→2020)
- `_KNOWN_CHAINS`: 76 Transfer-Ketten-Templates
- `_GENRE_EARLIEST_ORDER`: 195 Genres × 14 Ära-Gruppen
- `_MEDIUM_EXCLUDES_GENRES`: 8 Medien × ausgeschlossene Genres
- `_MEDIUM_PREFERRED_GENRES`: 13 Medien × typische Genres
- `_LANGUAGE_MEDIUM_BONUS`: 7 Sprachen × Medium-Präferenzen
- `_STUDIO_FORMAT_INDICATORS`: 21 Studio-Charakteristiken
- `_MEDIUM_DISPLAY_NAMES`: 20 deutsche GUI-Namen

## §2.46d: AudioSR Recovery-Kette (v10.0.3 final)

**Pflicht**: AudioSR muss Recovery vor Zone Passthrough implementieren.

| Stufe | Verfahren | Steps | Device | Qualität |
|-------|-----------|-------|--------|----------|
| 1 | CPU-DDIM | 25–50 | CPU | ML-optimal |
| 2 | SBR-DSP (`_sbr_extend`) | — | CPU | DSP-Spektralspiegelung |
| 3 | Zone Passthrough | — | — | Unverändert |

### Architektur-Entscheidungen

- **CPU-only** (v10.0.3): `build_model(device="cpu")` — gesamtes Modell auf CPU.
  - ROCm (AMD GPU) produziert NaN im HiFi-GAN-Vocoder via transposed-convolution-Bug.
  - Mixed-Device (GPU-DDIM + CPU-Vocoder) führte zu inkonsistenten Device-Platzierungen.
  - CPU-Inferenz: ~20s pro 1s Audio (25 DDIM-Steps), akzeptabel für Offline-Restoration.

- **DDIM numerische Stabilität** (v10.0.3):
  - `models/audiosr/audiosr/latent_diffusion/models/ddim.py`: Alle unguarded
    `sqrt()`, `log()`, `reciprocal()`-Operationen mit `clip(…, 1e-12/1e-20)` abgesichert.
  - Gleiche Fixes in `ddpm.py` und `plms.py` für Konsistenz.
  - `weight_norm`-NaN beim FP16→FP32-Load: `nan_to_num`-Clean aller 2592 Parameter
    nach `build_model()` (0/2592 NaN nach Clean).
  - `warnings.filterwarnings("ignore", "invalid value encountered in multiply")`
    für harmlose RuntimeWarning beim weight_norm-Load.

- **SBR-DSP-Fallback**: `logger.info` statt `logger.warning`.
  - SBR ist ein voll funktionsfähiger Pfad, kein Fehler.
  - Spektrale HF-Energie steigt nachweislich (z.B. 0.000→0.157).

- AudioSR-Trainings-Limit: 2–8x Extension. Bei Extremfällen (375 Hz→17 kHz=45x)
  ist Stufe 1 instabil, Stufe 2 (SBR-DSP) ist der wahrscheinlichste Pfad.

- Datei: `plugins/audiosr_plugin.py`
- DDIM-Fixes: `models/audiosr/audiosr/latent_diffusion/models/ddim.py`,
  `ddpm.py`, `plms.py`

## §2.35c: LPC-Formant-Tracker Anti-Aliasing (v10.0.0)

**Pflicht**: AA-Filter vor LPC-Dezimation muss immer verfügbar sein.

- `scipy.signal.butter` + `sosfiltfilt` auf Modulebene (nicht inline)
- 2× try/except-Fallback entfernt — kein stiller Degrade-Pfad
- 1-Sample-Off-by-One: `start = min(start, mono.size - max_win)`
- Segment-Trim: `pre_seg`/`post_seg` auf `min(len)` getrimmt
- Filter: Butterworth 4. Ordnung, Zero-Phase, float64, 90% Ziel-Nyquist
- Datei: `backend/core/dsp/lpc_formant_tracker.py`

### v10-Amendment (2026-07-12)

**SFT LEVEL_COLLAPSE: Phasenkorrektur-Phasen → INFO** (v10, 2026-07-12):

Der Signal Flow Tracer (`backend/core/signal_flow_tracer.py`) flaggt `LEVEL_COLLAPSE`
(rms_post < −60 dBFS) per `logger.warning`. Bei Phasenkorrektur-Phasen (phase_14,
phase_25, azimuth, phase_correction) auf fehlausgerichtetem Stereo-Material (Cassette,
Reel-Tape) erzeugt die Azimuth-Korrektur kurze transiente Pegelabfälle bis nahe
−∞ dBFS durch gegenphasige Überlappung. Dies ist mathematisch erwartet und wird vom
PMGG (waerme delta ≈ 0.0001) als nicht hörbar bestätigt.

**Implementierung** (`backend/core/signal_flow_tracer.py`):

```python
_pc_collapse = ("LEVEL_COLLAPSE" in " | ".join(flags)
    and phase_id and any(p in str(phase_id)
    for p in ("phase_14", "phase_25", "azimuth", "phase_correction")))
if _pc_collapse:
    logger.info("§SFT %s FLAGS: %s", phase_id, " | ".join(flags))
else:
    logger.warning("§SFT %s FLAGS: %s", phase_id, " | ".join(flags))
```

**Erwartetes Ergebnis**: Keine WARNING-Flut bei Cassette-Restaurierung. Echte
LEVEL_COLLAPSE in anderen Phasen bleibt als WARNING erhalten.

### v10-Amendment (2026-07-12)

**Phase-Fazit-System** (v10.0.3):
Jede Phase loggt ein menschenlesbares Fazit mit 0-10 Score:

```
┌─ Phase 03 (Entrauschen) ───────────────────────────────────┐
│ ✅ Rauschen um 12.3 dB reduziert                            │
│ 🏆 Score: 8.5 / 10.0  (SNR: 18→30 dB, ML: ja)              │
└─────────────────────────────────────────────────────────────┘
```

- `backend/core/phase_fazit.py`: `log_phase_fazit()` + `log_restoration_summary()`
- Integration in `create_phase_result()` via neue `phase_id`/`phase_name` Parameter
- Score = `quality_estimate × 10`
- Restaurations-Zusammenfassung mit Tonträgerkette als Geschichte in Klartext

**Laienfreundliche Kommunikation** (v10.0.3):

- `backend/core/phase_names.py`: Mapping phase_id → deutsche Anzeigenamen
- Log-Meldungen: `▶ phase_03_denoise (Entrauschen) startet (2/42)`
- Post-Restoration: Box mit Qualitäts-Label (🏆 Weltklasse/✅ Ausgezeichnet/👍 Sehr gut)
- Qualitäts-Labels: 95%+ 🏆, 85%+ ✅, 70%+ 👍, 50%+ ⚡, 30%+ ⚠️

**MediumDetector SOTA Knowledge Base** (v10.0.3):

- `_MEDIUM_ORDER`: 20 Tonträger chronologisch (1877→2020)
- `_KNOWN_CHAINS`: 76 Transfer-Ketten-Templates
- `_GENRE_EARLIEST_ORDER`: 195 Genres × 14 Ära-Gruppen
- `_MEDIUM_EXCLUDES_GENRES`: 8 Medien × ausgeschlossene Genres
- `_MEDIUM_PREFERRED_GENRES`: 13 Medien × typische Genres
- `_LANGUAGE_MEDIUM_BONUS`: 7 Sprachen × Medium-Präferenzen
- `_STUDIO_FORMAT_INDICATORS`: 21 Studio-Charakteristiken
- `_MEDIUM_DISPLAY_NAMES`: 22 deutsche GUI-Namen
- `get_genre_constraints(chain)`: Bidirektionale API für Genre-Validierung

**Chronologische Ketten-Sortierung** (v10.0.3):

- `_MEDIUM_ORDER` korrigiert: reel_tape(6) vor vinyl(7)
- Chronologische Sortierung im MediumDetector (line ~2530)
- "No backward jumps"-Gate entfernt (blockierte sekundäre Träger)
- Sortierung nach Deep-Transfer-Chain-Injection in pre_analysis

**ML-Readiness Flood-Control** (v10.0.3):

- `_FAILURE_CACHE`: WARNING nur einmal pro model+phase
- `clear_readiness_cache()` für neuen Restoration-Run
- ONNX MIOPEN-Warnungen unterdrückt (50+ Zeilen pro Load)
