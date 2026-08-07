# §v10.18 Preset-Learning × Selbstkalibrierung — Synergie-Spezifikation

## Prinzip

> Aurik kombiniert statistisches Preset-Lernen (aus historischen erfolgreichen
> Restaurationen) mit Live-Selbstkalibrierung (pro-Song HPE-Feedback).
> Beide Systeme lernen voneinander: erfolgreiche Kalibrierungen fließen gewichtet
> zurück ins Preset-System.

## Architektur

```
┌──────────────────────────────────────────────────────────────┐
│                    Preset-Learning                            │
│  Built-in Presets  │  User-Presets  │  Material/Ära/Genre    │
│  (6 kuratierte)    │  (lernend)     │  Fuzzy-Scoring          │
└────────────────────────┬─────────────────────────────────────┘
                         │ Preset = Startpunkt
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                  Selbstkalibrierung                           │
│  HPE-Gate (Live)   │  Defekt-Profil  │  Material-Floor       │
│  ±15% Anpassung    │  Strength-Mod   │  Ceiling-Prüfung       │
└────────────────────────┬─────────────────────────────────────┘
                         │ Kalibrierung = Feintuning
                         ▼
              ┌─────────────────────┐
              │  Optimaler Punkt     │
              │  Preset ∩ Selbst     │
              │  = max. Qualität     │
              └─────────────────────┘
```

## Module

| Modul | Rolle | Synergie |
|-------|-------|----------|
| `magic_restore_preset.py` | Preset-Selektion + User-Learning | Built-in + gelernte Presets |
| `mid_pipeline_quality_gate.py` | Live-HPE-Wächter | Selbstkalibrierung via Strength-Reduktion |
| `reference_track_calibrator.py` | Referenz-Track-Analyse | Preset=Referenz, Selbst=Material-Floor |
| `model_warmup_pool.py` | ML-Modelle vorladen | Preset sagt vorher welche Modelle nötig |
| `phase_fingerprint.py` | Änderungs-Erkennung | Preset-Änderung → Phase neu |
| `album_consistency_gate.py` | Album-weite Konsistenz | Track 1=Preset, 2-N=kalibriert |

## Preset-Learning

### Built-in Presets (v10.10)

| Name | Material | Ära | Genre | Besonderheit |
|------|----------|-----|-------|-------------|
| `cassette_schlager` | cassette | 1970–1995 | schlager | HPE-Toleranz 6% |
| `vinyl_jazz` | vinyl | 1950–1980 | jazz | Warm, Stereo 0.40 |
| `shellac_classical` | shellac | 1900–1950 | klassik | Fragil, HPE 8% |
| `cd_pop` | cd_digital | 1985–2025 | pop | Studio 2026, LUFS -12 |
| `reel_tape_rock` | reel_tape | 1960–1990 | rock | Analog, Dynamics 0.55 |
| `default_restoration` | unknown | 1900–2030 | default | Fallback |

### User-Learning

- `learn_from_result(preset, final_hpe, user_rating)` — gewichteter Running-Average
- User-Rating (1–5) beeinflusst HPE-Toleranz
- Erfolgreiche Presets werden in `~/.aurik/magic_presets/` persistiert

## Selbstkalibrierung

### HPE-Gate (Mid-Pipeline)

- Check-Intervall: alle 8 Phasen
- WARNING: HPE-Drop ≥ 3% → leichte Strength-Reduktion (×0.80)
- CRITICAL: HPE-Drop ≥ 7% → starke Strength-Reduktion (×0.60)
- Recovery: nach 2 aufeinanderfolgenden OKs → Learning-Rate 0.15 zurück zu voller Strength

### Defekt-Profil → Strength-Modulation

- Hiss > 0.6 → Denoise +20%
- Noise > 0.7 → Denoise +25%
- Klicks > 0.5 → Denoise +15%
- Ø Defekt < 0.3 → Overall -15%

### Material-Floor (Ceiling-Prüfung)

- Shellac: Brillanz ≤ 0.40, Stereo ≤ 0.20
- Kassette: Brillanz ≤ 0.55, Stereo ≤ 0.60
- CD: Brillanz ≤ 0.90, Stereo ≤ 0.95

## Version

v10.18 — 19. Juli 2026 — Aurik 10.0.10
