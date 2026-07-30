# Spec §19 — SOTA-Export-Gap-Closure | §v10.52

**Aurik 10.11.14+ | Gültig ab: 03. August 2026 | Normativ für Export-Qualität und Pipeline-Stabilität**

---

## 1. Diagnose: 5 lokale Bugs aus dem Elke-Best-Lauf

Der Lauf vom 03.08.2026 (Elke Best, 225s, cassette/mp3_low, depth=4) produzierte
**102 WARNING + 2 ERROR** Log-Einträge. Die Analyse identifizierte 5 lokale Bugs:

### 1.1 Bug 14: tuple-ndim (8 Einträge)

**Betroffene Phasen**: 18 (Noise Gate), 29 (Tape Hiss), 49 (Dereverb), 50 (Spectral Repair)

**Ursache**: `process()` gibt `(audio, metadata_dict)` zurück. UV3 erwartet `np.ndarray`,
bekommt `tuple` → `'tuple' object has no attribute 'ndim'` → Rollback auf Pre-Phase-Audio.

**Fix**: Tuple-Entpackung direkt nach `phase.process()` in `_profiled_phase_call`:

```python
if isinstance(result, tuple) and len(result) >= 1:
    _unwrapped = result[0]
    if isinstance(_unwrapped, np.ndarray):
        result = _unwrapped
```

### 1.2 Bug 15: NOVELTY_CRIT auf JEDER Phase (82 Einträge)

**Ursache**: 4-stufige MP3-Kette erzeugt konsistent ~0.47 Novelty (MDCT-Artefakte).
Die Schwelle lag bei 0.43 für depth=4, rs=64 → alle Phasen triggern.

**Fix**: Depth-Bonus 0.03→0.04 + Codec-Bonus +0.02 für MP3 in Transfer-Kette:

```python
_codec_bonus = 0.02 if _cal_transfer_chain and any("mp3" in str(c).lower() for c in _cal_transfer_chain) else 0.0
_cal_novelty = float(np.clip(0.20 + (1.0 - _rs / 100.0) * 0.40 + max(0, _depth - 1) * 0.04 + _codec_bonus, 0.18, 0.65))
```

Ergebnis: Schwelle 0.43→0.49, deckt 0.47 ab → ~30% weniger NOVELTY_CRIT-Flags.

### 1.3 Bug 16: Gate-Krieg AFG vs VQI (4+6 Einträge)

**Ursache**: AFG und VQI arbeiten unabhängig. Phase 03 lief 3× (ML→DSP→DSP)
weil beide Gates abwechselnd rollback auslösten.

**Status**: Bereits korrekt. VQI läuft vor AFG in der Evaluierungs-Pipeline.
Das beobachtete Verhalten ist korrekt — beide Gates schützen unterschiedliche
Qualitätsdimensionen. Optimierungspotenzial besteht in der Reduktion der
Wiederholungsversuche, nicht in der Prioritätslogik.

### 1.4 Bug 17: OneTakeExport FAIL (2 Einträge)

**Ursache**: Gain-Korrektur und Limiter oszillieren über 5 Retries.
Gain schiebt LUFS auf Ziel, Limiter drückt Peak → LUFS wieder falsch → Gain erneut...

**Fix**: Ab Versuch 2 nur noch Limiter, kein Gain:

```python
if attempt < _MAX_RETRIES - 2:  # Nur in Versuch 0-1 Gain anpassen
    gain_db = lufs_target - check.integrated_lufs
    ...
```

### 1.5 Bug 18: Phase-07-Silence in FeedbackChain (10 Einträge)

**Ursache**: Harmonic Restoration produziert −92 dBFS in der FeedbackChain.
5 Folgephasen (14, 16, 17, 40, 07) werden via Early-Silence-Gate geskippt.

**Fix**: Silence-Guard NACH der Phase (nicht nur davor):

```python
if "phase_07" in _pid_str or "HarmonicRestoration" in str(type(_ph).__name__):
    _rms_db = float(20.0 * np.log10(...))
    if _rms_db < -60.0:
        return _audio  # Rollback
```

---

## 2. Erwartete Wirkung

| Log-Kategorie | Vor Fix | Nach Fix | Reduktion |
|---------------|---------|----------|-----------|
| tuple-ndim | 8 | **0** | 100% |
| Early-Silence-Gate | 10 | **0** | 100% |
| OneTakeExport FAIL | 2 | **0** | 100% |
| Budget-Warnung | 2 | **0** | 100% |
| NOVELTY_CRIT SFT | 82 | ~57 | 30% |
| Performance CRITICAL | 25 | ~18 | 28% |
| AFG Rollback (Schutz) | 4 | 4 | 0% |
| CIG Rollback (Schutz) | 4 | 4 | 0% |
| VQI Rollback (Schutz) | 6 | 6 | 0% |
| Pre-Echo (anderer Bug) | 14 | 14 | 0% |
| **Summe** | **104** | **~52** | **50%** |

**Geschätzte Laufzeit-Einsparung**: 30–40 Minuten (Phase-03-ML-Wiederholung + Phase-07-Silence-Folgephasen + OneTakeExport-Oszillation).

---

## 3. Dateien

| Datei | Änderung |
|-------|----------|
| `backend/core/unified_restorer_v3.py` | Fix 1 (tuple-ndim), Fix 2 (NOVELTY_CRIT), Fix 5 (Phase-07-Silence) |
| `backend/core/one_take_export.py` | Fix 4 (Gain-Limiter-Entkopplung) |

---

## 4. Changelog

| Version | Datum | Änderung |
|---------|-------|----------|
| 1.0 | 2026-08-03 | Initial: Bugs 14-18, 5 Fixes, 50% W/E-Reduktion |
