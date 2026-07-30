# Depth-Calibration-Shift — Verifikationsplan

## Vor jedem Release auszuführen

### 1. Unit-Tests

```bash
pytest tests/ -q --timeout=60 -k "depth or chain or threshold or artifact or hallucination or quality or interaction"
```

**Erwartet**: 850+ passed, 0 durch unsere Änderungen verursachte Failures.

### 2. Pipeline-Integrationstest

Mindestens ein Song pro Depth-Stufe importieren und verarbeiten:

| Depth | Material | Erwartetes Verhalten |
|-------|----------|---------------------|
| 1 | CD/FLAC | Alle Phasen, AF≥0.90, kein Rollback |
| 2 | Vinyl | Alle Phasen, AF≥0.80, ≤1 Rollback |
| 3 | Vinyl→MP3 | Volle ML-Pipeline, AF≥0.75 |
| 4 | Kassette 4-fach | Alle Phasen, AF≥0.70, ≤3 Rollbacks |
| 5 | Wachs→Schellack→Band→Kassette→MP3 | DSP-Fallbacks aktiv, AF≥0.60 |

### 3. Log-Analyse

Nach jedem Testlauf prüfen:

- `grep "Max consecutive rollbacks.*reached"` → **darf nicht erscheinen**
- `grep "HPI.*0.0000"` → **darf nicht erscheinen** (außer bei komplett zerstörtem Input)
- `grep "CIG_ROLLBACK" | wc -l` → **≤5 pro Depth-4-Lauf**

### 4. Manuelle Hörprobe

Bei Depth-4-Kassette prüfen:

- Kein hörbares Pre-Echo (De-Esser zu aggressiv)
- Keine "sterile" Stille (HallucinationGuard zu strikt)
- Transienten nicht "verwaschen" (NaturalnessOptimizer Blend)
- Keine hörbaren STFT-Artefakte (Phase_29 Group Delay)
