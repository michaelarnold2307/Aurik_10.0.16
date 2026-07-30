# §18 — Non-Plus-Ultra: Vom Messen zum Wahrnehmen (§v10.80–§v10.83)

> **Status:** Spec | **Version:** 10.0.80 | **Datum:** 2026-08-03
>
> Aurik misst Qualität mit technischen Metriken. Der Mensch hört mit Ohren.
> Diese Spec schließt die vier architektonischen Lücken zwischen Messen und Wahrnehmen.

---

## §18.1 — Wahrnehmungsmetrik: Presence Embedding (§v10.80)

### Problem

```
Input Quality:  43.0/100  (technisch)
Output Quality: 43.1/100  (technisch)
MUSHRA Score:   95.3/100  (perzeptuell)
HPI:            0.886     (passed)
```

Die technische Qualitätsmetrik sagt „keine Verbesserung". Alle perzeptuellen Metriken sagen „exzellent". Aurik kann nicht messen, was es tatsächlich verbessert hat: **die menschliche Anwesenheit in der Aufnahme.**

### Lösung: `PresenceEmbedding`

Ein neues perceptual embedding, das die Distanz zwischen „Aufnahme" und „Live-Präsenz" misst:

```
PresenceScore = f(
    vocal_formant_coherence,   # Wie "echt" klingt die Stimme?
    transient_immediacy,       # Wie direkt sind die Transienten?
    room_tone_continuity,      # Atmet der Raum natürlich?
    microdynamic_liveliness,   # Lebt die Dynamik?
    spectral_air_authenticity  # Ist die Luft echt oder synthetisch?
)
```

**Berechnung:**

- Vocal Formant Coherence: MERT-basierte Distanz zwischen restaurierten Formanten und einer Datenbank echter Gesangsaufnahmen
- Transient Immediacy: Onset-Stärke-Verteilung im Vergleich zu Live-Referenzen
- Room Tone Continuity: Varianz des Rauschbodens über die Zeit (niedrig = kontinuierlich = echt)
- Microdynamic Liveliness: Crest-Faktor-Verteilung in 200ms-Fenstern
- Spectral Air Authenticity: Korrelation der HF-Hüllkurve (>10 kHz) mit natürlichen Referenzen

**Integration:**

- Läuft NACH allen Restaurierungsphasen, VOR dem Export
- Ersetzt NICHT die technischen Metriken — ergänzt sie
- Wird im Quality Report als eigene Zeile ausgewiesen
- Schwellwert: PresenceScore ≥ 0.70 für „hörbare Verbesserung"

---

## §18.2 — GDD-Budget-Manager: Per-Phase Gruppenlaufzeit (§v10.81)

### Problem

```
§2.48 STFT group delay deviation 39.9 ms > threshold 13.2 ms
after 3 STFT phases → rollback
```

Der CumulativeInteractionGuard erkennt die Grenzüberschreitung, aber er hat keine präventive Kontrolle. Jede STFT-Phase läuft mit voller Stärke, bis die Summe aller GDDs die Schwelle reißt — dann Rollback. Phase_29 wird bestraft für das, was Phase_03 und Phase_23 vor ihr getan haben.

### Lösung: `GddBudgetManager`

```python
class GddBudgetManager:
    """Verteilt Gruppenlaufzeit-Budget pro Phase und überwacht kumulativ."""

    TOTAL_BUDGET_MS = 13.2  # Max kumulative GDD über alle STFT-Phasen

    def allocate(self, phase_id: str, material: str) -> float:
        """Gibt das GDD-Budget für diese Phase zurück (ms)."""

    def consume(self, phase_id: str, actual_gdd_ms: float) -> bool:
        """Verbraucht Budget. Returns False wenn überschritten → reduziere Stärke."""

    def remaining(self) -> float:
        """Verbleibendes Gesamtbudget."""
```

**Budget-Verteilung pro Material:**

| Material | Budget | Pro-Phase-Cap |
|----------|--------|---------------|
| Shellac | 8.0 ms | 4.0 ms |
| Vinyl | 10.0 ms | 5.0 ms |
| Cassette | 13.2 ms | 6.0 ms |
| Reel-Tape | 15.0 ms | 7.0 ms |
| Digital | 5.0 ms | 3.0 ms |

**Integration in `_profiled_phase_call`:**

- Vor jeder STFT-Phase: `budget = gdd_budget.allocate(phase_id, material)`
- Wenn budget < 1.0 ms: Phase auf Stärke 0.25 reduzieren (Passthrough-nah)
- Nach jeder STFT-Phase: `gdd_budget.consume(phase_id, actual_gdd_ms)`
- Wenn consume False returned: Stärke der AKTUELLEN Phase reduzieren, nicht Rollback

---

## §18.3 — Audio-Sanity-Check: Silence-Guard nach Rollback (§v10.82)

### Problem

```
🔇 Early-Silence-Gate phase_07: Audio-RMS=-92.4 dBFS → Phase SKIPPED
```

Nach einem Rollback (Phase_29 → CIG → zurück auf Phase_27) wird das Audio an Phase_07 weitergereicht — aber der Rollback-Pfad hat das Audio beschädigt. Phase_07 bekommt -92.4 dBFS Stille und „skipt". Der Schaden ist bereits passiert.

### Lösung: `RollbackSanityCheck`

```python
def validate_rollback_audio(audio: np.ndarray, source_phase: str) -> bool:
    """Prüft ob das Audio nach einem Rollback noch intakt ist."""
    rms_db = 20 * log10(rms(audio) + 1e-15)
    if rms_db < -60.0:
        logger.critical("Rollback-Sanity: Audio zerstört (RMS=%.1f dBFS) — Checkpoint-Wiederherstellung", rms_db)
        return False
    if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
        logger.critical("Rollback-Sanity: NaN/Inf im Audio — Checkpoint-Wiederherstellung")
        return False
    peak = np.max(np.abs(audio))
    if peak < 1e-6:
        logger.critical("Rollback-Sanity: Kein Signal (Peak=%.1e) — Checkpoint-Wiederherstellung", peak)
        return False
    return True
```

**Integration:**

- Nach JEDEM Rollback (CIG, SFT, AFG): `validate_rollback_audio()` aufrufen
- Bei False: NICHT den Rollback-Punkt verwenden, sondern den LETZTEN BEKANNT GUTEN Checkpoint
- Wenn kein gültiger Checkpoint existiert: Original-Audio (Pre-Phase-01) als Fallback

---

## §18.4 — Tuple-Ndim-Rettung im Profiled-Phase-Pfad (§v10.83)

### Problem

Drei Phasen verlieren ihre Ergebnisse durch tuple-ndim-Fehler INNERHALB von `_profiled_phase_call`. Der bisherige Fix (`_deep_extract_ndarray`) sitzt im Exception-Handler des Main-Loops — aber der Fehler tritt in der Post-Processing-Chain von `_profiled_phase_call` auf, BEVOR die Exception den Main-Loop erreicht.

### Lösung: `_normalize_phase_audio()` in `_profiled_phase_call`

Am ENDE von `_profiled_phase_call`, NACH allen Post-Processing-Schritten (PMGG, SFT, V22, Tilt-Guard), aber VOR der Rückgabe an den Main-Loop:

```python
def _normalize_phase_audio(result, audio_before_phase):
    """Garantiert dass result.audio ein np.ndarray ist."""
    if hasattr(result, "audio"):
        ra = result.audio
        if not isinstance(ra, np.ndarray):
            # Versuche Extraction aus tuple/list/object
            extracted = _deep_extract_ndarray(ra)
            if extracted is not None:
                result.audio = extracted
                return
            # Letzter Fallback: Original-Audio
            result.audio = np.asarray(audio_before_phase, dtype=np.float32)
```

**Integration:**

- Wird in `_profiled_phase_call` als LETZTER Schritt vor `return result` aufgerufen
- Verhindert, dass Tupel/Listen das `result.audio` verlassen
- Logged WARNING wenn Extraction nötig war (für Debugging)

---

## §18.5 — GEBOTE-Integration

| ID | Regel |
|----|-------|
| §G90 | **PresenceEmbedding-Pflicht** — Jeder Export MUSS einen PresenceScore berechnen und im Quality Report ausweisen. PresenceScore ≥ 0.70 definiert „hörbare Verbesserung". |
| §G91 | **GDD-Budget-Pflicht** — Jede STFT-Phase MUSS ein GDD-Budget vom GddBudgetManager anfordern und einhalten. Kumulative GDD darf materialspezifische Grenze nicht überschreiten. |
| §G92 | **Rollback-Sanity-Pflicht** — Nach JEDEM Rollback MUSS `validate_rollback_audio()` das Ziel-Audio auf Integrität prüfen. Beschädigtes Audio wird durch letzten gültigen Checkpoint ersetzt. |
| §G93 | **Tuple-Ndim-Garantie** — `_profiled_phase_call` MUSS garantieren dass `result.audio` IMMER ein `np.ndarray` ist. `_normalize_phase_audio()` als letzter Schritt vor Return. |

---

## §18.6 — VERBOTE-Integration

| Verbot | Beschreibung | Korrektur |
|--------|-------------|-----------|
| Qualitätsmetrik ohne PresenceScore | Technische Metriken (SNR, MCD) ohne perzeptuelle Validierung → „43→43"-Paradox | PresenceEmbedding als ergänzende Metrik |
| STFT-Phase ohne GDD-Budget | Kumulative Gruppenlaufzeit unkontrolliert → Rollback ohne Vorwarnung | GddBudgetManager pro Phase |
| Rollback ohne Audio-Integritätsprüfung | Beschädigtes Audio nach Rollback an Folgephasen weitergegeben | `validate_rollback_audio()` |
| Tuple-ndim im Phase-Result | `result.audio` als tuple/list verlässt `_profiled_phase_call` | `_normalize_phase_audio()` |
