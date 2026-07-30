# Spec 22: Wohlklang-Strategie — Systematische Exzellenz für das menschliche Ohr

> **Version:** Aurik 10.0.16 · **Scope:** Qualitäts-Roadmap aus Tiefenanalyse-Erkenntnissen
> **Status:** Strategie — priorisiert, terminiert, mit Erfolgskriterien
> **Erstellt:** 2026 · **Audit-Datum:** 2026

## Inhaltsverzeichnis

1. [Prämisse](#prämisse)
2. [Phase A — Fehlerquellen eliminieren (Woche 1)](#phase-a--fehlerquellen-eliminieren-woche-1)
3. [Phase B — Qualitativer Sprung (Woche 2)](#phase-b--qualitativer-sprung-woche-2)
4. [Phase C — Architektonische Vollendung (Monat 2)](#phase-c--architektonische-vollendung-monat-2)
5. [Erfolgskriterien](#erfolgskriterien)
6. [Risiken & Gegenmaßnahmen](#risiken--gegenmaßnahmen)

---

## Prämisse

Diese Strategie leitet sich aus **acht unerwarteten Erkenntnissen** einer mehrtägigen
Tiefenanalyse der gesamten Aurik-Codebasis ab. Die Erkenntnisse wurden durch
systematisches Audit aller 65 Phasen, der FeedbackChain-Architektur, der
ExcellenceOptimizer-Kalibrierung und der Vocal-Quality-Pipeline gewonnen.

**Kernziel:** Aurik übertrifft iZotope RX 11 an natürlichem Wohlklang (Naturalness
0.88 → 0.90) für ALLE Importsongs auf ALLEN Materialien — bei gleichzeitiger
struktureller Absicherung gegen Regression.

**Leitprinzip:** Jede Maßnahme muss mindestens eines dieser drei Ziele direkt
verbessern:

1. **Hörbarer Wohlklang** (Naturalness, Formant-Treue, Mikrodynamik)
2. **Systemische Stabilität** (keine Regression, keine unbemerkte Degradation)
3. **Nachhaltige Wartbarkeit** (keine Monkey-Patches, keine versteckten Abhängigkeiten)

---

## Phase A — Fehlerquellen eliminieren (Woche 1)

> Ziel: Die drei häufigsten Fehlerklassen werden strukturell unmöglich gemacht.
> Keine neue Funktionalität — nur Absicherung des Erreichten.

### A1 — Safe-STFT-Wrapper

**Erkenntnis:** Der `scipy.signal.safe_stft`-Monkey-Patch in `backend/__init__.py`
ist eine architektonische Zeitbombe. Fünf Phasen (03, 20, 23, 24, 49) stürzen ab,
wenn die Import-Reihenfolge sich ändert.

**Maßnahme:**

1. `backend/core/audio_utils.py`: `safe_stft()` und `safe_istft()` als öffentliche
   Funktionen deklarieren (existieren bereits, müssen nur exportiert werden).
2. In allen fünf Phasen `signal.safe_stft(...)` durch `from backend.core.audio_utils
   import safe_stft` ersetzen.
3. Monkey-Patch in `backend/__init__.py` mit Deprecation-Warning versehen,
   aber für Rückwärtskompatibilität belassen.
4. Smoke-Test für jede der fünf Phasen mit explizitem `safe_stft`-Aufruf.

**Erfolgskriterium:** Keine Phase importiert mehr `scipy.signal.safe_stft`.
`python3 -c "import scipy.signal; assert not hasattr(scipy.signal, 'safe_stft')"`
muss fehlschlagen (der Monkey-Patch wird entfernt).

**Aufwand:** 2h | **Wohlklang-Impact:** Indirekt — verhindert zukünftige Abstürze.

---

### A2 — Undefined-Variable-Linting in CI

**Erkenntnis:** Zwei Scope-Bugs (`phase_23`: `kwargs is not defined`,
`phase_30`: `_mk is not defined`) waren monatelang in Produktion, weil kein
Linter sie erkannte. Beide hätten mit statischer Analyse gefunden werden können.

**Maßnahme:**

1. `.pylintrc` ergänzen um `--enable=undefined-variable`, `--enable=used-before-assignment`.
2. CI-Pipeline (`pytest --pylint`) in `.github/workflows/` um Lint-Schritt erweitern.
3. Bestehende False-Positives via `# pylint: disable=...` dokumentieren (nicht ignorieren).
4. `tests/unit/test_regression_guards.py::TestScopeFixes` als Regression-Suite
   behalten (existiert bereits mit 35 Tests).

**Erfolgskriterium:** `pylint backend/core/phases/` findet 0 undefined-variable
Fehler. CI bricht bei neuen Scope-Bugs ab.

**Aufwand:** 1h | **Wohlklang-Impact:** Indirekt — verhindert zukünftige Laufzeitfehler.

---

### A3 — Automatisierter Kalibrierungs-Audit

**Erkenntnis:** Aurik wurde über Monate mit defensiven Parametern betrieben, die
explizit als "zu konservativ gegen iZotope RX11" kommentiert waren. Ein automatisierter
Check hätte das sofort erkannt.

**Maßnahme:**

1. `scripts/calibration_audit.py` — neues Skript, das:
   - Alle Parameter in `excellence_optimizer.py` liest
   - Gegen SOTA-Zielwerte prüft (z.B. `_HARM_BOOST_DB` ≥ 3.2)
   - Alle Material-Profile auf Plausibilität prüft (shellac > cd)
   - Warnt bei Werten, die >30% unter dem SOTA-Ziel liegen
2. CI-Integration: `./scripts/calibration_audit.py --ci` mit Exit-Code ≠0 bei
   unterschrittenen Schwellwerten.
3. Wöchentlicher Slack/Log-Report: "Kalibrierungs-Status: alle Parameter innerhalb
   SOTA-Toleranz" oder "⚠️ _MODULATION_STRENGTH 0.42 < 0.55 Soll"

**Erfolgskriterium:** `./scripts/calibration_audit.py` läuft in CI und warnt bei
defensiven Parametern. Kein Parameter fällt unbemerkt unter SOTA-Niveau.

**Aufwand:** 3h | **Wohlklang-Impact:** Direkt — verhindert die defensive
Kalibrierungs-Falle dauerhaft und stellt sicher, dass der ExcellenceOptimizer
immer auf RX11-Niveau arbeitet.

---

## Phase B — Qualitativer Sprung (Woche 2)

> Ziel: Drei architektonische Verbesserungen, die den Wohlklang direkt und hörbar
> steigern — ohne neue Parameter, nur durch intelligentere Entscheidungslogik.

### B1 — Meta-Richter für Metrik-Konflikte

**Erkenntnis:** Wenn Goosebumps-Score und HPI widersprechen ("Original klingt
emotionaler" vs. "Restaurat ist objektiv besser"), gab es keine Schiedsinstanz.
Der Goosebumps-Score gewann immer — und das defekte Original wurde gewählt.

**Maßnahme:**

1. `backend/core/metric_arbiter.py` — neue Klasse `MetricArbiter` mit einer
   Methode `resolve(quantitative_score, qualitative_score, context) → decision`.
2. Regeln (priorisiert):
   - Wenn `HPI ≥ 0.80` ODER `artifact_freedom ≥ 0.95`: **qualitativer Score**
     überschreibt quantitativen → Restaurat wird bevorzugt.
   - Wenn `VQI < 0.70` und Konflikt betrifft Vokalfrequenzen: **VQI** überschreibt
     alles → Gesangsschutz hat absolute Priorität.
   - Sonst: gewichtetes Mittel (60% qualitativ, 40% quantitativ).
3. Integration in Goosebumps-Recovery (`unified_restorer_v3.py`): `MetricArbiter`
   ersetzt die bisherige rein Score-basierte Entscheidung.
4. Test: `tests/unit/test_metric_arbiter.py` mit 6 Konflikt-Szenarien.

**Erfolgskriterium:** Bei HPI≥0.80 wird das defekte Original NIEMALS dem Restaurat
vorgezogen — unabhängig vom Goosebumps-Score. Der §v10.113 HPI-Gate wird durch
den Meta-Richter architektonisch verankert.

**Aufwand:** 4h | **Wohlklang-Impact:** ⬆️⬆️ Direkt hörbar — eliminiert die
letzte verbleibende Situation, in der Aurik das defekte Original dem sauberen
Restaurat vorzieht.

---

### B2 — FeedbackChain-Awareness pro Phase

**Erkenntnis:** Drei Phasen (07, 23, NaturalnessOptimizer) verhielten sich im
FeedbackChain destruktiv, weil sie nicht wussten, dass sie im zweiten Durchlauf
auf bereits sauberem Audio operieren.

**Maßnahme:**

1. `PhaseInterface._safe_process()`: Setzt `kwargs['_feedback_chain_pass'] = True`
   wenn die Phase im FeedbackChain-Kontext läuft (erkennbar via `_fc_active`-Flag
   im `_restoration_context`).
2. Phase 07: Wenn `_feedback_chain_pass=True` und `H2/H1 ≥ 0.35` → sofort
   Passthrough (keine harmonische Synthese auf sauberem Audio).
3. Phase 23: Wenn `_feedback_chain_pass=True` → DSP-only-Modus (kein FlashSR ML,
   das auf sauberem Audio halluziniert).
4. NaturalnessOptimizer: Wenn `_feedback_chain_pass=True` → `_transient_preservation`
   deaktiviert (Transienten sind bereits optimal).
5. `PhaseInterface._safe_process()`: `rms_drop_db`-Metadaten mit `fc_pass`-Tag
   versehen für Tracing.

**Erfolgskriterium:** FeedbackChain-Durchlauf 2 produziert NIE −86 dBFS Stille
oder degradierte Transienten. Der §v10.114 Silence-Guard wird durch architektonische
Vermeidung ersetzt.

**Aufwand:** 3h | **Wohlklang-Impact:** ⬆️⬆️⬆️ Direkt hörbar — der zweite
Durchlauf wird von einer Risikoquelle zu einem reinen Qualitätsgewinn.

---

### B3 — Vocal-System-Dokumentation & Benchmark

**Erkenntnis:** Auriks Gesangs-Restaurierung (VQI, BreathPreserver,
6-Dimensionen-Scorer, Formant-Guard, SingerReferenceLibrary, instrument_formant_db)
ist wissenschaftlich auf Sundberg-Niveau — aber vollständig undokumentiert.

**Maßnahme:**

1. `docs/VOCAL_SYSTEM.md` — umfassende Dokumentation:
   - Architektur-Übersicht (VocalDetector → VocalNaturalnessScorer → VocalQualityGate)
   - Die 6 Dimensionen: Formant-Integrität, Atem-Natürlichkeit, Sibilanz-Erhalt,
     Verständlichkeit, Hörkomfort, Stimmwärme
   - Wissenschaftliche Grundlagen (Sundberg 1987, Klatt 1980, Fant 1960)
   - Vergleich mit iZotope RX11, Nectar, Clarity Vx
2. `benchmarks/vocal_quality/` — Benchmark-Suite:
   - 5 Testsignale (Sopran, Tenor, Bariton, Sprechstimme, Chor)
   - Metriken: VQI, MUSHRA, Formant-Korrelation, Sibilanz-Erhalt
   - Vergleich: Aurik vs. RX11 vs. unbearbeitet
3. Integration in `README.md`: "🎤 Weltklasse-Gesangsrestauration" als Feature-Hervorhebung

**Erfolgskriterium:** `docs/VOCAL_SYSTEM.md` existiert und beschreibt alle
Komponenten. Der Vocal-Benchmark zeigt Aurik ≥ RX11 in ≥4/6 Dimensionen.

**Aufwand:** 5h | **Wohlklang-Impact:** Indirekt — macht Auriks stärkstes
Feature sichtbar und messbar.

---

## Phase C — Architektonische Vollendung (Monat 2)

> Ziel: Die langfristige Wartbarkeit und Erweiterbarkeit wird auf das Niveau
> der bereits erreichten Wohlklang-Qualität gehoben.

### C1 — Parameter-Interaktions-Graph

**Erkenntnis:** Die `_TARGET_CV_MIN → deficit → strength = min(cap, deficit*2.0)`-
Kette zeigte, dass Parameter-Tuning ohne Verständnis der Interaktionen wirkungslos
ist. Diese Ketten müssen dokumentiert und automatisch prüfbar sein.

**Maßnahme:**

1. `scripts/parameter_impact_trace.py` — AST-basierte Analyse:
   - Extrahiert alle Parameter-Definitionen (Modul-Konstanten, Profile-Felder)
   - Verfolgt den Datenfluss bis zur tatsächlichen Anwendung (z.B. `_TARGET_CV_MIN`
     → `deficit` → `strength` → `modulation *=`)
   - Generiert `docs/parameter_graph.json` als machine-readable Dependency-Graph
2. `docs/parameter_graph.md` — menschenlesbare Übersicht pro Parameter:
   - "Wenn Sie X erhöhen, ändert sich Y nur wenn Z > Schwellwert"
   - "X ist ein Cap — der tatsächliche Wert wird durch Y bestimmt"
3. CI-Integration: `./scripts/parameter_impact_trace.py --check` prüft, ob der
   Graph aktuell ist (Hash des Quellcodes vs. gespeicherter Hash).

**Erfolgskriterium:** Jeder Parameter im ExcellenceOptimizer hat einen
dokumentierten Wirkungspfad. Neue Entwickler können in <5 Minuten verstehen,
warum eine Erhöhung von `_MODULATION_STRENGTH` nicht den erwarteten Effekt hat.

**Aufwand:** 6h | **Wohlklang-Impact:** Indirekt — ermöglicht präzise zukünftige
Kalibrierungen ohne Trial-and-Error.

---

### C2 — Guard-Self-Test-Modus

**Erkenntnis:** Die systemischen Guards (§v10.115, §v10.117) sind Auriks
wichtigste Sicherheitsinnovation. Ihre Korrektheit muss automatisch verifizierbar
sein — nicht nur durch manuelle Code-Review.

**Maßnahme:**

1. `PhaseInterface._safe_process(..., guard_test=False)` — neuer Parameter.
2. Wenn `guard_test=True`:
   - Injiziert künstliche Degradation (RMS-Drop, Transient-Shift, Spectral-Novelty,
     Formant-Shift) zwischen `process()` und den Guards
   - Prüft, dass jeder Guard die Degradation erkennt (Warning, Metadata)
   - Prüft, dass der RMS-Guard tatsächlich Rollback durchführt
   - Gibt `GuardTestResult(passed: bool, failures: list[str])` zurück
3. `tests/unit/test_guard_self_test.py` — parametrisierter Test:
   - `test_rms_guard_triggers_on_30db_drop`
   - `test_transient_guard_triggers_on_10ms_shift`
   - `test_hallucination_guard_triggers_on_novelty_0_20`
   - `test_formant_guard_triggers_on_correlation_0_70`
   - `test_all_guards_non_blocking` (kein Crash bei Guard-Fehler)
4. Wöchentlicher Scheduled-Run in CI.

**Erfolgskriterium:** `python3 -m pytest tests/unit/test_guard_self_test.py -v`
besteht mit 100%. Jeder Guard löst bei seiner spezifischen Degradation korrekt aus.

**Aufwand:** 4h | **Wohlklang-Impact:** Direkt — garantiert, dass die Wohlklang-
Schutzmechanismen dauerhaft funktionieren.

---

## Erfolgskriterien

| Ziel | Metrik | Aktuell | Ziel |
|---|---|---|---|
| Naturalness (MUSIC_NAT) | ExcellenceOptimizer | 0.81 | **0.88** (RX11-Niveau) |
| Phase-Crash-Rate | Smoke-Tests | 56/56 | 56/56 + Guard-Self-Tests |
| Undefined-Variable-Bugs | Linter | Unbekannt | **0** in CI |
| Kalibrierungs-Defensivität | Audit-Skript | Manuell | **Automatisch** |
| Vocal-System-Sichtbarkeit | Dokumentation | Keine | **Vollständig** |
| FeedbackChain-Degradation | Log-Analyse | 3 bekannte Fälle | **0** |
| Monkey-Patch-Fragilität | Architektur | 1 kritischer | **0** |

---

## Risiken & Gegenmaßnahmen

| Risiko | Eintrittswkt. | Gegenmaßnahme |
|---|---|---|
| Phase-07 FC-Awareness zu aggressiv → Passthrough verpasst echte Harmonik-Lücken | Niedrig | H2/H1-Schwelle 0.35 ist konservativ; echte Lücken haben H2/H1 < 0.20 |
| Meta-Richter überschreibt legitime Goosebumps-Präferenz | Niedrig | HPI-Schwelle 0.80 ist hoch; darunter greift der Meta-Richter nicht ein |
| `safe_stft`-Wrapper verändert Verhalten (Edge Cases) | Mittel | Smoke-Tests + spezifische STFT-Regression-Tests vor Migration |
| Kalibrierungs-Audit meldet False-Positives (Parameter absichtlich niedrig) | Mittel | `# calibration-audit: intentional` Kommentar-Marker für Ausnahmen |

---

## Zusammenfassung

```
Woche 1 (6h):  M1 Safe-STFT  M2 Linting  M3 Audit     → Fehlerquellen eliminiert
Woche 2 (12h): M4 Meta-Richter  M5 FC-Awareness  M6 Vocal-Doku  → Qualitätssprung
Monat 2 (10h): M7 Parameter-Graph  M8 Guard-Self-Test  → Architektonische Vollendung
                                                    ─────────────────────────────
Gesamt: 28h    170 Tests grün    Naturalness 0.81→0.88    Wartbarkeit verdoppelt
```

Jede Maßnahme ist einzeln umsetzbar, baut nicht kritisch auf anderen auf,
und verbessert mindestens eines der drei Kernziele direkt.
