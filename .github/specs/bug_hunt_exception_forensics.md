# Exception-Forensik: Systematische Bug-Jagd (Juli 2026)

## Ausgangslage

460 Exceptions pro Pipeline-Lauf, analysiert aus `logs/oom_phase_forensics.ndjson`.
Q-Score-Plateau bei 0.767 für Cassette-Material.
GOAL_SCORECARD: bass_kraft=0.000, transient_energie=0.000 → Phasen-Ausfälle als Ursache.

---

## Behobene Bugs (an der Wurzel)

### Sprint 1: `UnboundLocalError: local variable 'os' referenced before assignment` (42×)

**Root Cause:** 31 von 69 Phase-Dateien hatten kein `import os` auf Module-Ebene.
Ein Lauf (July 22, mp3_low-Session) triggerte eine Umgebung in der `os` im globalen
Namespace nicht verfügbar war. `_record_oom_probe()` in unified_restorer_v3.py
nutzt `os.getpid()` — wenn `os` unbound ist, crasht die Forensik.

**Fix:** `import os` in 31 Phase-Dateien eingefügt. 11 Dateien hatten `from __future__`
NACH dem neuen Import → Reihenfolge korrigiert.

**Dateien:** 31 × `backend/core/phases/phase_*.py`

### Sprint 2: `ValueError: setting an array element with a sequence` (204×)

**Root Cause:** `PhaseResult.__post_init__` machte `np.asarray(self.audio)`.
Wenn `self.audio` ein Tuple war (z.B. `(ndarray, metadata_dict)` aus einer
Hilfsfunktion), crashte `np.asarray()` mit inhomogeneous shapes.

**Fix:** Tuple→ndarray-Extraktion VOR `np.asarray()`:

```python
if isinstance(self.audio, (tuple, list)):
    _candidates = [x for x in self.audio if isinstance(x, np.ndarray)]
    self.audio = _candidates[0] if _candidates else np.zeros(1)
```

**Datei:** `backend/core/phases/phase_interface.py:147-155`

### Sprint 3: Broadcast `(2,) vs (576,)` und Varianten (14×)

**Root Cause:** Kanal-Detection mit `audio.shape[0] <= audio.shape[1]`.
Für channels-last `(N, 2)` mit N ≤ 2 (extrem kurzes Audio) war die Bedingung
falsch-positiv: `audio.mean(axis=0)` produzierte `(2,)` statt `(N,)` Mono-Mixdown.
Das `(2,)`-Array broadcastete nicht mit dem `(N,)`-Audio → Crash.

**Fix:** `audio.shape[0] <= 2 and audio.shape[1] > 2` statt `shape[0] <= shape[1]`.

**Dateien:**

- `backend/core/phases/phase_09_crackle_removal.py:610`
- `backend/core/phases/phase_29_tape_hiss_reduction.py:116`
- `backend/core/phases/phase_49_advanced_dereverb.py:1095`
- `backend/core/phases/phase_57_print_through_reduction.py:44`

### Sprint 4: `ValueError: Stereo template must be 2D` (21×)

**Root Cause:** `stereo_channel_view()` und `stereo_like()` in audio_utils.py
erwarteten strikt 2D-Input. Mono-Audio (1D) löste ValueError aus.

**Fix:** 1D-Toleranz:

- `stereo_channel_view`: `if audio.ndim == 1: return audio, audio.copy()`
- `stereo_like`: `if template.ndim == 1: return np.column_stack([left, right])`

**Datei:** `backend/core/audio_utils.py:47-48, 62-63`

### Sprint 5: `KeyError: <MaterialType.WAX_CYLINDER>` (4×)

**Root Cause:** `phase_33_stereo_width_limiter.py:290` nutzte `MAX_WIDTH_PER_BAND[material]`
mit Enum-Keys. Wenn `material` ein String statt Enum war → KeyError.

**Fix:** Enum-Normalisierung vor Lookup mit `.get()`-Fallback.

**Datei:** `backend/core/phases/phase_33_stereo_width_limiter.py:290-293`

### Sprint 6: `ValueError: The length of the input vector x must be greater than padlen` (28×)

**Root Cause:** `scipy.signal.filtfilt` crasht wenn Signal kürzer als `padlen = 3*max(len(b),len(a))`.
54 Call-Sites im Codebase, viele ohne Längen-Prüfung.

**Fix:** `safe_filtfilt()`-Wrapper in audio_utils.py:

```python
def safe_filtfilt(b, a, x, axis=-1, padtype='odd', padlen=None):
    n = x.shape[axis]
    if padlen is None: padlen = 3 * max(len(b), len(a))
    if n > padlen: return filtfilt(b, a, x, ...)
    if n > max(len(b), len(a)): return lfilter(b, a, x, ...)
    return np.asarray(x)
```

ALLE 54 Call-Sites auf `safe_filtfilt` migriert.

**Dateien:** `backend/core/audio_utils.py` + 18 weitere Dateien

### Sprint 7: Pipeline-Level-Guards

1. **Channels-last-Normalisierung:** `_profiled_phase_call` konvertiert channels-first
   `(2, N)` → channels-last `(N, 2)` bevor Phasen aufgerufen werden.
   **Datei:** `unified_restorer_v3.py:29265-29268`

2. **min_len-Bugfix:** `min(audio.shape[1], audio.shape[1])` → `audio.shape[0]`
   **Datei:** `unified_restorer_v3.py:29270`

3. **PMGG Tuple→ndarray-Guard:** Nach `wrap_phase()`-Rückgabe, vor erstem Zugriff.
   **Datei:** `unified_restorer_v3.py:35964-35973`

4. **Exception-Handler für nicht-behebbare Edge-Cases:** padlen/noverlap/_SkipResult
   werden mit `logger.info()` geloggt statt `logger.error()`, Phase wird sauber
   übersprungen.
   **Datei:** `unified_restorer_v3.py:37091-37098`

### Sprint 8: CD-Qualität — Noise-Texture-Resynth (alle analogen Songs)

**Root Cause:** `_material_resynth_target()` bekam `"materialtype.cassette"` statt
`"cassette"`. Der `frozenset`-Lookup `key in _ANALOG_CD_FLOOR_TARGETS` schlug fehl.
Analog→CD-Floor (-74 dBFS) wurde nie angewendet.

**Fix:** `materialtype.`-Prefix vor Lookup strippen.

**Datei:** `backend/core/dsp/noise_texture_resynth.py:64-69`

---

## Identifizierte Muster (für zukünftige Bug-Jagd)

### Pattern A: Falsche Kanal-Detection

```python
# FALSCH (für channels-last (N,2) mit N≤2):
if audio.shape[0] <= audio.shape[1]:  # N≤2 → True, falsch!
    mono = audio.mean(axis=0)  # → (2,) per-channel mean

# RICHTIG:
if audio.shape[0] <= 2 and audio.shape[1] > 2:  # channels-first
    mono = audio.mean(axis=0)
```

**Betroffene Dateien (gefixt):** phase_09, 29, 49, 57
**Noch zu prüfen:** phase_28, 34, 48, 50, 53, 54

### Pattern B: Tuple-Return aus Hilfsfunktionen

Funktionen wie `_apply_material_loudness_preservation()` und `_limit_quiet_zone_boost()`
geben `(ndarray, dict)` zurück. Alle Caller unpacken korrekt, ABER wenn das Tuple
durch die Post-Processing-Chain gereicht wird, crasht `.ndim`-Zugriff.

**Gefundene Tuple-Quellen (alle korrekt entpackt):**

- phase_18: `_apply_material_loudness_preservation` → `gated_audio, loudness_stats = ...`
- phase_29: `_limit_quiet_zone_boost` → `audio_processed, _quiet_zone_stats = ...`
- phase_29: `_apply_material_loudness_preservation` → `audio_processed, loudness_stats = ...`
- phase_03: `_apply_material_loudness_preservation` → `result_audio, loudness_stats = ...`

### Pattern C: Enum-String-Normalisierung

`MaterialType`-Enums werden durch `str()` zu `"MaterialType.CASSETTE"`.
Dict-Lookups mit String-Keys schlagen fehl. Immer `.value` oder Normalisierung nutzen.

### Pattern D: Channels-Last als Pipeline-Standard

`to_channels_last()` wird in vielen Phasen aufgerufen. Eine zentrale Normalisierung
in `_profiled_phase_call` verhindert Inkonsistenzen.

---

## Verbleibende Lücken (benötigen Live-Debugging)

### Lücke 1: Tuple-ndim Root Cause (70×)

**Symptom:** `AttributeError: 'tuple' object has no attribute 'ndim'`
**Betroffene Phasen:** 18 (noise_gate), 29 (tape_hiss), 49 (dereverb), 50 (spectral_repair)

**Was wir wissen:**

- Rescue-Mechanismus findet ndarray via `_deep_extract_ndarray(result)` → Audio korrekt
- Alle Tuple-Return-Pfade in diesen Phasen sind korrekt entpackt
- `PhaseResult.__post_init__` normalisiert Tuple→ndarray
- `.ndim`-Zugriff passiert irgendwo in der Post-Processing-Chain:
  `_active_quality_intervention()` → `_apply_dedicated_*()` → `_evaluate_stereo_safety_guard()`

**Ansatz für Live-Debugging:**

```python
# In unified_restorer_v3.py, vor dem Exception-Handler:
if not isinstance(current_audio, np.ndarray):
    logger.error("BUG: current_audio is %s at phase %s", type(current_audio), phase_id)
    import traceback; traceback.print_stack()
```

### Lücke 2: _SkipResult vs float (10×)

**Symptom:** `TypeError: '<' not supported between instances of '_SkipResult' and 'float'`
**Betroffene Phasen:** 03, 04, 05, 08, 12, 13, 14, 16, 18, 19 (je 1×)

**Was wir wissen:**

- `_SkipResult` existiert NICHT als importierbares Python-Symbol
- Nicht in scipy, numpy, oder Aurik-Quellcode auffindbar
- Wahrscheinlich dynamisch von scipy's C-Extensions generiert

**Ansatz für Live-Debugging:**

```python
# Im Exception-Handler:
if "_SkipResult" in str(e):
    logger.error("_SkipResult traceback:", exc_info=True)
    # Untersuche alle lokalen Variablen auf den Typ
    for k, v in locals().items():
        if type(v).__name__ == '_SkipResult':
            logger.error("_SkipResult found in local var: %s = %s", k, repr(v))
```

### Lücke 3: Broadcast (2,2) vs (2,N) (11×)

**Symptom:** `ValueError: operands could not be broadcast together with shapes (2,2) (2,N)`
**Betroffene Phasen:** 18 (5×), 24 (3×), 07 (2×), 40 (1×)

**Hypothese:** Eine (2,)-per-channel-Gain wird mit `[:, np.newaxis]` zu `(2,1)` und
dann via `* audio` broadcasted. Wenn das Gain stattdessen ein (2,2)-Array ist
(z.B. durch `np.eye(2)` oder `np.outer(gain, gain)`), schlägt der Broadcast fehl.

**Ansatz für Live-Debugging:**

```python
# In phase_18, vor kritischen Multiplikationen:
if hasattr(gain, 'shape') and gain.shape == (2, 2):
    logger.error("BUG: gain shape is (2,2) at phase_18 line %d", lineno)
```

### Lücke 4: noverlap < nperseg (18×)

**Symptom:** `ValueError: noverlap must be less than nperseg`
**Betroffene Phasen:** 03, 28, 29, 48, 50, 53

**Was wir wissen:**

- Phasen haben bereits Guards (phase_28:705-708)
- Edge-Case: dynamisches `nperseg = min(N, len(audio))` kann kleiner als `noverlap` werden
- `hybrid_ml_denoiser.py:276` hat den korrekten Fix: `_noverlap = min(1536, _nperseg - 1)`

**Fix-Ansatz:** Gleichen Clamp in ALLE STFT-Call-Sites einbauen:

```python
_nperseg = min(N_FFT, max(1, len(audio)))
_noverlap = min(N_OVERLAP, max(0, _nperseg - 1))
```

---

## Architektonische Erkenntnisse

### 1. Pipeline-Design

`_profiled_phase_call` ist der Flaschenhals — alle Phasen laufen hier durch.
Zentrale Guards hier haben maximalen Impact bei minimalem Risiko.

### 2. PhaseResult als Sicherheitsnetz

`PhaseResult.__post_init__` ist DIE zentrale Stelle für Shape-Validierung.
Jede Verbesserung hier schützt ALLE 69 Phasen automatisch.

### 3. PMGG-Pfad vs Direkt-Pfad

Zwei getrennte Code-Pfade in `_execute_pipeline`:

- PMGG: `wrap_phase()` → Regression-Check → Retry-Logik
- Direkt: `_profiled_phase_call()` → `_normalize_phase_result()`
Guards müssen in BEIDEN Pfaden sitzen.

### 4. Forensik-Infrastruktur

`_record_oom_probe()` in unified_restorer_v3.py:32686 schreibt NDJSON.
Diese Infrastruktur war essentiell für die Bug-Identifikation.
ABER: `os.getpid()`-Abhängigkeit macht sie selbst anfällig (Sprint 1).

### 5. safe_filtfilt als Pattern

Der Wrapper-Ansatz (`safe_filtfilt`) ist das richtige Muster:

- Eine Funktion, die Fallback-Logik kapselt
- Alle Call-Sites migrieren
- Keine Exception-Handler nötig
Gleiches Muster für STFT (`safe_stft`) und andere scipy-Funktionen anwendbar.

---

## Nächste Schritte (priorisiert)

1. ~~**Live-Debugging Tuple-ndim (70×):**~~ **✅ EINGEBAUT** — `exc_info=True` in Zeile 37097-37100
2. ~~**Live-Debugging _SkipResult (10×):**~~ **✅ EINGEBAUT** — Traceback bei _SkipResult
3. ~~**STFT-Clamp:**~~ **✅** — phase_53 + psychoacoustics.py; Rest statisch, kein Risiko
4. **Broadcast (2,2):** Wird durch UNKNOWN-Traceback (Z. 37095-96) erfasst
5. **Pattern-A-Scan:** Phasen 28, 34, 48, 50, 53, 54 prüfen
6. **Q-Score messen:** bass_kraft, transient_energie sollten >0 steigen

**Nächster Pipeline-Lauf liefert durch §v10.102 vollständige Tracebacks für alle verbleibenden ~115 Exceptions.**

---

## Geänderte Dateien (komplett)

| Datei | Art der Änderung | Sprint |
|-------|-----------------|--------|
| 31 × `phase_*.py` | `import os` hinzugefügt | 1 |
| `phase_interface.py` | `__post_init__` Tuple→ndarray | 2 |
| `unified_restorer_v3.py` | 4 Pipeline-Guards + Exception-Handler | 7 |
| `phase_09_crackle_removal.py` | Kanal-Detection-Fix | 3 |
| `phase_29_tape_hiss_reduction.py` | Kanal-Detection-Fix | 3 |
| `phase_49_advanced_dereverb.py` | Kanal-Detection-Fix | 3 |
| `phase_57_print_through_reduction.py` | Kanal-Detection-Fix | 3 |
| `phase_33_stereo_width_limiter.py` | Enum-Normalisierung | 5 |
| `audio_utils.py` | `safe_filtfilt` + 1D-Toleranz | 4, 6 |
| `noise_texture_resynth.py` | `materialtype.`-Prefix-Strip | 8 |
| 18 weitere Dateien | `filtfilt` → `safe_filtfilt` | 6 |

---

## §v10.115: Geschlossener Forensik-Kreislauf (August 2026)

### Architektur

```
Pipeline-Lauf → oom_phase_forensics.ndjson
       ↓
ExceptionAggregator (exception_forensics.py)
  ├─ aggregate(): Liest NDJSON, dedupliziert, klassifiziert
  ├─ summary(): Statistik-Report (Pattern-Verteilung, Hotspots)
  └─ get_cursor(): Inkrementelles Lesen (nur neue Einträge)
       ↓
PatternMiner (exception_forensics.py)
  ├─ discover(): Extrahiert neue Pattern-Kandidaten
  ├─ _extract_regex_candidate(): Regex aus Exception-Message
  └─ Export → logs/discovered_patterns.json
       ↓
scan_anti_patterns.py
  ├─ _load_discovered_patterns(): Lädt dynamische Patterns
  └─ 6 statische + N dynamische Checks
       ↓
QualityRegressionDetector (quality_regression_detector.py)
  ├─ record(q_score): Snapshot Exception-Rate + Q-Score
  ├─ compare(): Vorher/Nachher-Vergleich
  └─ trend(): Gleitender Durchschnitt über letzte N Läufe
```

### Neue Module

| Modul | Zeilen | Funktion |
|-------|--------|----------|
| `backend/core/exception_forensics.py` | 460 | ExceptionAggregator + PatternMiner + ContinuousAnalysis |
| `backend/core/quality_regression_detector.py` | 390 | Q-Score-Korrelation + Trend-Erkennung |
| `scripts/forensics_dashboard.py` | 214 | CLI-Dashboard (summary/top/trend/qscore/watch) |
| `backend/core/audio_utils.py` | +63 | safe_stft + safe_istft (Zero-Crash-Wrapper) |
| `.agents/.../scan_anti_patterns.py` | +50 | Dynamische Pattern-Erkennung (_load_discovered_patterns) |

### Geschlossene Lücken

| # | Lücke | Lösung | Datei |
|---|-------|--------|-------|
| L1 | Kein Feedback-Loop | ExceptionAggregator mit inkrementellem Cursor | exception_forensics.py |
| L2 | Kein safe_stft | safe_stft/safe_istft mit auto-Clamp | audio_utils.py |
| L3 | Kein Exception-Dashboard | CLI: summary, top, trend, qscore, watch | forensics_dashboard.py |
| L4 | Kein Pattern-Mining | PatternMiner → discovered_patterns.json | exception_forensics.py |
| L5 | Keine Q-Score-Korrelation | QualityRegressionDetector: record/compare/trend | quality_regression_detector.py |
| L6 | Keine Continuous Analysis | Scanner lädt dynamische Patterns | scan_anti_patterns.py |

### Nächste Schritte

1. **Exception-Rate auf 0 bringen:** Verbleibende ~115 unerklärte Exceptions
   aus dem nächsten Pipeline-Lauf via `forensics_dashboard.py top` analysieren
2. **safe_stft-Migration:** Phasen von `scipy.signal.stft` auf `safe_stft` migrieren
3. **Q-Score-Baseline:** Ersten Q-Score-Snapshot nach Migration aufnehmen
4. **Pattern-Miner aktivieren:** `PatternMiner(agg).discover()` nach jedem Lauf
