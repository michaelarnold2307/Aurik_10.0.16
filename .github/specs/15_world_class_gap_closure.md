# Spec 15: Weltspitze-Gap-Closure — 10-Punkte-Plan

> **Version:** Aurik 10.0.0-Draft · **Scope:** Wettbewerbsfähigkeit, Validierung, Ökosystem
> **Status:** Entwurf — Roadmap, noch nicht implementiert
> **Erstellt:** 10. Juli 2026 · **Audit-Datum:** 10. Juli 2026

## Inhaltsverzeichnis

1. [§15.1 Competitive Benchmarks reparieren](#151-competitive-benchmarks-reparieren)
2. [§15.2 Echt-Audio-Corpus aufbauen](#152-echt-audio-corpus-aufbauen)
3. [§15.3 ABX/Wahrnehmungsvalidierung](#153-abxwahrnehmungsvalidierung)
4. [§15.4 Cross-Platform CI](#154-cross-platform-ci)
5. [§15.5 GPU-Strategie öffnen](#155-gpu-strategie-öffnen)
6. [§15.6 Plugin-SDK und Developer-Ökosystem](#156-plugin-sdk-und-developer-ökosystem)
7. [§15.7 Dokumentations-Nutzerpfade](#157-dokumentations-nutzerpfade)
8. [§15.8 ErrorGuard-Flächendeckung](#158-errorguard-flächendeckung)
9. [§15.9 Zentraler Memory-Lifecycle](#159-zentraler-memory-lifecycle)
10. [§15.10 Perceptual-Validation-Studie](#1510-perceptual-validation-studie)

---

## §15.1 Competitive Benchmarks reparieren

### Ist-Stand

Die Benchmark-Suite (`benchmarks/competitive/benchmark_suite.py`) enthält nur einen Mock:

```python
logger.warning("iZotope RX 11 benchmarking not implemented (requires license)")
```

Kein einziger Vergleichslauf gegen iZotope RX oder CEDAR ist je in CI gelaufen.
Die letzten Daten sind vom Februar 2026 — 5 Monate alt, RX 12 fehlt komplett.

### Wurzelursache

Die Suite setzt eine kostenpflichtige Lizenz voraus (`benchmark_izotope()`), die nicht
automatisiert verfügbar ist. Open-Source-Alternativen werden nicht genutzt.

### Implementierungsschritte

| # | Schritt | Aufwand | Abhängigkeit |
|---|---------|---------|-------------|
| 1.1 | `benchmarks/competitive/open_source_benchmark.py` — neues Modul, das DeepFilterNet3, AudioSR, Demucs, MDX-Net, Open-Unmix via pip/subprocess benchmarked. **Keine Lizenz nötig.** | 6–8 h | — |
| 1.2 | Automatischer Download der Open-Source-Modelle via `huggingface_hub.snapshot_download()` mit SHA256-Pinning. | 2 h | 1.1 |
| 1.3 | `benchmarks/competitive/gate_results.py` [ROADMAP] — Ergebnis-Dataclasses mit OQS-Delta, Timbre-Fidelity, artifact_freedom, Laufzeit. JSON-Export + CI-freundlicher Exit-Code. | 3 h | 1.1 |
| 1.4 | CI-Integration: `tests/normative/test_competitive_ci_gate.py` um Open-Source-Vergleich erweitern. Neue `@pytest.mark.competitive_oss`-Markierung. | 2 h | 1.3 |
| 1.5 | `benchmarks/competitive/results/` — Monatliches Scheduled-Run (GitHub Actions cron) mit automatischer Trend-Analyse (Regression-Flag wenn OQS-Delta < 0). | 3 h | 1.4 |
| 1.6 | Legacy-Mock entfernen: `benchmark_izotope()` in `benchmarks/competitive/benchmark_suite.py` auf `NotImplementedError` umstellen, der klar dokumentiert, dass RX-Lizenz manuell bereitgestellt werden muss. | 0.5 h | — |

### Erfolgskriterien

- `pytest -m competitive_oss` läuft in CI ohne Skip und vergleicht Aurik gegen ≥3 Open-Source-Tools
- Monatlicher Trend-Report detektiert Regressionen automatisch
- OQS-Delta ≥ 0 für ≥80% der Szenarien (wie Spec 04 §4.4a fordert)

---

## §15.2 Echt-Audio-Corpus aufbauen

### Ist-Stand

```bash
$ ls corpus/
ls: cannot access 'corpus/': No such file or directory
```

Kein einziges der 15.000+ Tests verwendet eine echte Musikaufnahme.
Alle Tests operieren auf synthetischen Sinus-Signalen, Rauschen und Butterworth-Filtern.

### Wurzelursache

Rechtliche Unsicherheit (Urheberrecht) + fehlende Corpus-Infrastruktur.

### Implementierungsschritte

| # | Schritt | Aufwand | Abhängigkeit |
|---|---------|---------|-------------|
| 2.1 | `corpus/`-Verzeichnisstruktur anlegen: `corpus/{shellac,vinyl,tape,digital}/{clean,damaged,restored}/` mit `manifest.yaml` pro Unterverzeichnis. | 2 h | — |
| 2.2 | `corpus/MANIFEST_SCHEMA.yaml` — Schema für `manifest.yaml`: Datei, Dauer, SR, Material, Ära, Genre, Defekttypen, Quelle, Lizenz, Checksum. | 1 h | 2.1 |
| 2.3 | Generator-Skript `scripts/generate_corpus_from_public_domain.py` — automatisiertes Herunterladen von Public-Domain-Aufnahmen (Internet Archive, Musopen, Freesound CC0) mit Tagging. | 4 h | 2.2 |
| 2.4 | `corpus/README.md` mit rechtlichem Disclaimer, Quellenangaben und Anleitung zum Hinzufügen eigener Aufnahmen. | 1 h | — |
| 2.5 | `tests/corpus/test_corpus_integrity.py` — prüft alle Manifests auf Konsistenz, fehlende Dateien, Checksum-Fehler, Mindestanzahl pro Kategorie. | 2 h | 2.3 |
| 2.6 | `tests/corpus/test_corpus_pipeline_smoke.py` — End-to-End-Test: Jede Corpus-Datei durchläuft die volle Aurik-Pipeline (mindestens `mode=quick`). Kein Performance-Gate, nur "kein Crash, kein NaN". | 3 h | 2.3 |
| 2.7 | Optional: Lizenz für kleine kommerzielle Test-Corpora evaluieren (z.B. fraunhofer_idmt, MUSDB18). Nicht Pflicht für Gate. | Recherche | — |

### Erfolgskriterien

- Mindestens 20 Public-Domain-Aufnahmen in ≥4 Material-Kategorien
- `test_corpus_integrity` grün
- `test_corpus_pipeline_smoke` grün (kein Crash über alle Corpus-Dateien)

---

## §15.3 ABX/Wahrnehmungsvalidierung

### Ist-Stand

`tests/unit/test_perceptual_metrics_regression.py` (v10.0.0-Phantom) heißt "ABX", implementiert aber **keine Blindhörvergleiche** —
es misst SNR und RMS nach butterworth-Filtern. Irreführender Dateiname.
Die Backend-Module `mushra_evaluator.py` (636 Zeilen) und `mushra_session.py` (318 Zeilen) existieren,
sind aber rein algorithmische Approximationen ohne menschliche Hörer-Integration.

### Wurzelursache

Keine Infrastruktur für echte Hörertests. Unklare Trennung zwischen "objektiver MUSHRA-Approximation"
und tatsächlicher subjektiver Validierung.

### Implementierungsschritte

| # | Schritt | Aufwand | Abhängigkeit |
|---|---------|---------|-------------|
| 3.1 | `tests/unit/test_perceptual_metrics_regression.py` (v10.0.0-Phantom) umbenennen in `tests/unit/test_perceptual_metrics_regression.py`. Docstring auf "perzeptuelle Metrik-Regression" ändern. | 0.5 h | — |
| 3.2 | `backend/core/abx_listener.py` — Neues Modul: Web-basiertes ABX-Test-Interface (Flask/FastAPI-Endpoint), das A/B/X-Triplets mit randomisierter Reihenfolge serviert. Session-Management, Ergebnis-Persistenz in SQLite. | 8 h | — |
| 3.3 | `backend/core/mushra_listener.py` — Neues Modul: ITU-R BS.1534-konformes MUSHRA-Interface mit Hidden Reference, 3.5-kHz-Anchor, 5–7 Bedingungen. Integriert mit `mushra_session.py`. | 8 h | 3.2 |
| 3.4 | `tests/listener/test_listener_contract.py` — Contract-Tests für ABX und MUSHRA HTTP-APIs: Stimulus-Zufälligkeit, Session-Isolation, Ergebnis-Aggregation. | 3 h | 3.2, 3.3 |
| 3.5 | `backend/core/mushra_evaluator.py` umbenennen in `backend/core/objective_mushra_estimator.py` — klare Trennung: das ist eine Schätzung, kein echter MUSHRA-Test. Alle Imports aktualisieren. | 2 h | — |
| 3.6 | `docs/listening_study_protocol.md` — Protokoll für formale Hörtests: Rekrutierung, Training, Kontrollbedingungen, statistische Auswertung (95%-CI, ANOVA, Post-hoc). | 3 h | — |

### Erfolgskriterien

- `test_abx_regression.py` → `test_perceptual_metrics_regression.py` umbenannt
- ABX-HTTP-Endpoint serviert A/B/X-Triplets korrekt
- MUSHRA-HTTP-Endpoint implementiert ITU-R BS.1534 Hidden-Reference-Protokoll
- `mushra_evaluator.py` → `objective_mushra_estimator.py` mit klarem "approximation"-Label

---

## §15.4 Cross-Platform CI

### Ist-Stand

Drei GitHub-Actions-Workflows — alle ausschließlich `runs-on: ubuntu-22.04`.
Kein Windows, kein macOS. Aurik beansprucht Cross-Plattform-Fähigkeit, testet sie aber
nie automatisiert.

### Wurzelursache

Keine Windows/macOS-Runner konfiguriert. Möglicherweise Kostenbedenken (GitHub-hosted
macOS-Runner sind teurer).

### Implementierungsschritte

| # | Schritt | Aufwand | Abhängigkeit |
|---|---------|---------|-------------|
| 4.1 | `.github/workflows/ci-cross-platform.yml` — Neuer Workflow: `ubuntu-22.04`, `windows-2022`, `macos-14` (Apple Silicon). Nur für `push` auf `main` + manuellen Trigger (`workflow_dispatch`), nicht pro PR (Kostenkontrolle). | 3 h | — |
| 4.2 | `scripts/platform_compat_check.py` — Prüft Dateisystem-Pfade (kein `\\`-vs-`/`-Hardcoding), Zeilenenden (LF erzwungen via `.gitattributes`), Case-Sensitivity (Python-Imports case-sensitiv auf macOS/Linux, nicht Windows). | 2 h | — |
| 4.3 | Cross-Platform-Test-Suite: `pytest -m "not slow and not gpu and not onnx"` auf allen drei Plattformen. GPU/ONNX-Tests nur auf Ubuntu (ROCm). | 1 h | 4.1 |
| 4.4 | `.github/workflows/ci-cross-platform.yml` um `macos-15` (Intel via Rosetta?) und `windows-2025` ergänzen, sobald verfügbar. | 1 h | 4.1 |

### Erfolgskriterien

- `ci-cross-platform.yml` läuft auf Ubuntu, Windows, macOS und ist grün
- `platform_compat_check.py` detektiert Plattform-Inkompatibilitäten vor Merge

---

## §15.5 GPU-Strategie öffnen

### Ist-Stand

Aurik deklariert: "CPU + optionale AMD-GPU (ROCm/DirectML)". Kein CUDA, kein Apple Silicon.
Im professionellen Audio-Markt dominieren macOS (Apple Silicon) und NVIDIA (CUDA).

### Wurzelursache

Entwickler-Hardware-Präferenz (AMD-GPU im Entwicklungsrechner). Strategische Entscheidung
ohne dokumentierte Begründung.

### Implementierungsschritte

| # | Schritt | Aufwand | Abhängigkeit |
|---|---------|---------|-------------|
| 5.1 | `backend/core/ml/backend_router.py` — Neues Modul: Abstrahiert ML-Backend (CPU/ROCm/CUDA/MPS/DirectML). `@dataclass MLEngineConfig` mit `provider: Literal["cpu","cuda","rocm","mps","directml"]`. ONNX-Sessions nutzen `providers`-Parameter entsprechend. | 6 h | — |
| 5.2 | `scripts/detect_gpu_capabilities.py` — Erkennt verfügbare GPUs und deren Fähigkeiten: CUDA→`onnxruntime-gpu`, MPS→`coremltools`-Konvertierung, ROCm→bestehend. Ausgabe: `gpu_capabilities.json`. | 3 h | 5.1 |
| 5.3 | CUDA-Pfad: `pip install onnxruntime-gpu` + `CUDAExecutionProvider` in ONNX-Sessions. Vorhandene Modelle (PANNS, RMVPE, Whisper, wav2vec2) müssen kompatibel sein (ONNX ist plattformunabhängig). | 3 h | 5.2 |
| 5.4 | Apple-Silicon-Pfad: `pip install onnxruntime-silicon` (falls verfügbar) oder `CoreMLExecutionProvider`. Alternativ: Modell-Konvertierung ONNX→CoreML via `coremltools`. | 5 h | 5.2 |
| 5.5 | `.github/workflows/ci-cross-platform.yml` um GPU-Erkennung erweitern: `detect_gpu_capabilities.py` auf allen Plattformen laufen lassen (auch wenn keine GPU, dann CPU-Fallback bestätigen). | 1 h | 5.2, 4.1 |

### Erfolgskriterien

- `detect_gpu_capabilities.py` erkennt CUDA, MPS, ROCm, DirectML korrekt
- `MLEngineConfig` steuert ONNX-Provider pro Plattform
- Kein harter ROCm/DirectML-Ausschluss mehr — alle Backends sind "best effort"

---

## §15.6 Plugin-SDK und Developer-Ökosystem

### Ist-Stand

58 Dateien in `plugins/` — aber kein SDK, keine API-Stabilitätsgarantie, kein Developer Guide.
Das Bridge-Bypass-Verbot isoliert Plugins von `backend/core/`.

### Wurzelursache

Plugins wurden organisch als interne Erweiterungen entwickelt, nicht als externes Ökosystem
konzipiert.

### Implementierungsschritte

| # | Schritt | Aufwand | Abhängigkeit |
|---|---------|---------|-------------|
| 6.1 | `plugins/SDK.md` — Developer Guide: Plugin-Lebenszyklus, `AurikPlugin`-Base-Class, `manifest.json`-Schema, Testing, Distribution, Versionierung (SemVer). | 4 h | — |
| 6.2 | `plugins/sdk/aurik_plugin_base.py` — `AurikPlugin`-ABC mit `on_phase_start`, `on_phase_end`, `process_audio()`, `get_manifest()`. Type-Hints für alle Parameter. | 3 h | 6.1 |
| 6.3 | `plugins/sdk/testing_fixtures.py` — `VirtualAurikPipeline` für isolierte Plugin-Tests ohne vollständige Pipeline. Mock-Audio, Mock-Material-Info. | 3 h | 6.2 |
| 6.4 | `plugins/sdk/example_plugin/` — Minimalbeispiel-Plugin mit vollständiger Struktur: `__init__.py`, `manifest.json`, `test_example.py`, `README.md`. | 2 h | 6.2 |
| 6.5 | `scripts/validate_plugin.py` — Validiert ein Plugin-Verzeichnis gegen das SDK-Schema: Manifest, Base-Class-Konformität, Test-Abdeckung. | 2 h | 6.4 |
| 6.6 | API-Stabilitätsgarantie: `backend/api/bridge.py`-Changelog mit SemVer. `@deprecated`-Decorator für alte Funktionen. 2-Major-Versionen-Deprecation-Window. | 2 h | — |

### Erfolgskriterien

- Developer kann `cp -r plugins/sdk/example_plugin plugins/my_plugin` ausführen und `test_example.py` wird grün
- `validate_plugin.py` prüft ≥5 Qualitätskriterien
- `SDK.md` deckt Plugin-Lebenszyklus vollständig ab

---

## §15.7 Dokumentations-Nutzerpfade

### Ist-Stand

98 Docs-Dateien, aber: Kein "Getting Started", kein Tutorial, kein Architekturdiagramm,
keine maschinenlesbare API-Referenz. `docs/api/PYTHON_API.md` existiert, ist aber nicht
strukturiert (kein OpenAPI, keine Swagger-UI).

### Wurzelursache

Dokumentation wuchs organisch als Entwickler-Notizen statt als Nutzer-Pfade.

### Implementierungsschritte

| # | Schritt | Aufwand | Abhängigkeit |
|---|---------|---------|-------------|
| 7.1 | `docs/getting_started.md` — 15-Minuten-Setup: Python 3.10+, `pip install -e .`, `aurik --check`, erstes `aurik restore mein_song.wav`. | 3 h | — |
| 7.2 | `docs/tutorials/`-Verzeichnis mit drei Tutorials: `tutorial_restore_vinyl.md`, `tutorial_restore_tape.md`, `tutorial_batch_processing.md`. Schritt-für-Schritt mit Screenshots. | 6 h | 7.1 |
| 7.3 | `docs/architecture.md` — C4-Diagramm (Context, Container, Component) der Aurik-Architektur. Mermaid.js für Renderbarkeit in GitHub. | 3 h | — |
| 7.4 | `docs/api/openapi.yaml` — OpenAPI 3.0-Spezifikation der REST-API (`backend/api/rest/`). Generiert automatisch Swagger-UI. | 4 h | — |
| 7.5 | `scripts/generate_api_docs.py` [ROADMAP] — Extrahiert Docstrings aus `backend/api/bridge.py` und generiert Markdown. Integration in `docs/api/`. | 3 h | — |

### Erfolgskriterien

- `docs/getting_started.md` führt neuen Nutzer in ≤15 Minuten zu erfolgreicher Restaurierung
- `docs/architecture.md` enthält C4-Diagramme mit Mermaid.js
- `docs/api/openapi.yaml` ist valide und beschreibt alle REST-Endpunkte

---

## §15.8 ErrorGuard-Flächendeckung ✅ ERLEDIGT (§v10.115)

### Ist-Stand (vorher)

- NaN/Inf-Check in 67/68 Phasen (98%) — gut
- ErrorGuard (strukturierte Fehlerwiederherstellung) nur in einem Bruchteil der Phasen
- Ein Crash in Phase X killt die gesamte Pipeline

### Umsetzung (§v10.115 Universal Safety Wrapper)

Der `PhaseInterface._safe_process()`-Wrapper wurde um **4 systemische Guards** erweitert,
die für ALLE 65+ Phasen gelten — ohne eine einzige Phase ändern zu müssen:

| Guard | Abdeckung | Mechanismus |
|-------|-----------|-------------|
| **RMS-Preservation** | 100% (69/69) | Rollback bei >30 dB Pegelabfall |
| **§V22 Transient-Shift** | 51% (35/69) | Onset-Shift >5 ms → Warning |
| **§2.46e Hallucination** | 54% (37/69) | Spectral-Novelty >0.15 → Warning |
| **§v10.117 Formant-Guard** | 100% (69/69) | Formant-Korrelation <0.85 → Warning |

Zusätzlich: RMS-Guard mit automatischem Rollback verhindert −86 dBFS Stille in JEDER Phase.
Alle Guards sind non-blocking (Fehler → Debug-Log, kein Crash) und enrichieren
`result.metadata` mit realen Messwerten (kein hartcodiertes `rms_drop_db: 0.0` mehr).

**Status:** ✅ Abgeschlossen. Alle 65+ Phasen haben ErrorGuard via systemischen Wrapper.

---

## §15.9 Zentraler Memory-Lifecycle

### Ist-Stand

ONNX-Sessions werden in `lyrics_guided_enhancement.py` und `bridge.py` ad-hoc geladen.
Kein zentraler Lifecycle-Manager. `conftest.py` nutzt `_release_heavy_singletons()` und
`gc.collect()` als Test-Workaround — kein Produktions-Pattern.

### Wurzelursache

Gewachsenes Design — jede Komponente managed ihre eigenen Ressourcen ohne Koordination.

### Implementierungsschritte

| # | Schritt | Aufwand | Abhängigkeit |
|---|---------|---------|-------------|
| 9.1 | `backend/core/ml/session_manager.py` — `InferenceSessionManager`: Singleton, verwaltet alle ONNX-Sessions. `acquire(model_name) → InferenceSession`, `release(model_name)`, `release_all()`. LRU-Cache mit konfigurierbarer Max-Größe. | 5 h | — |
| 9.2 | `backend/core/ml/session_manager.py` — Memory-Monitoring: `get_total_memory_mb()`, `get_session_sizes()`, Warnung bei >2GB. | 2 h | 9.1 |
| 9.3 | Migration: `lyrics_guided_enhancement.py`, `bridge.py` auf `InferenceSessionManager` umstellen. Direkte `onnxruntime.InferenceSession(...)`-Aufrufe ersetzen. | 4 h | 9.1 |
| 9.4 | `backend/core/ml/batch_processor.py` [ROADMAP] — Batch-Verarbeitung mit Session-Recycling: Nach N Tracks Sessions freigeben und neu laden (Memory-Fragmentation vermeiden). | 3 h | 9.1 |
| 9.5 | `tests/unit/test_session_manager.py` [ROADMAP] — Testet: Acquire/Release, LRU-Eviction, Memory-Limit, Concurrent-Access, Batch-Recycling. | 3 h | 9.1 |

### Erfolgskriterien

- `InferenceSessionManager` ist einziger ONNX-Session-Erzeuger im Codebase
- `get_total_memory_mb()` zeigt <2GB für 4 Modelle unter Last
- Batch-Verarbeitung von 100 Tracks ohne Memory-Leak (stabile `get_total_memory_mb()` über Zeit)

---

## §15.10 Perceptual-Validation-Studie

### Ist-Stand

Auriks zentraler Anspruch — "optimiert fürs menschliche Ohr, nicht für technische Metriken" —
ist durch keine einzige externe Validierung belegt. 38 Dateien referenzieren MUSHRA, aber
nur als algorithmische Approximation. Null menschliche Hörer, null Blindstudien, null
Peer-Review.

### Wurzelursache

Keine Ressourcen/Infrastruktur für formale Hörtests. Trennung zwischen "objektiver MUSHRA-Schätzung"
und tatsächlicher subjektiver Validierung wurde nie vollzogen (vgl. §15.3).

### Implementierungsschritte

| # | Schritt | Aufwand | Abhängigkeit |
|---|---------|---------|-------------|
| 10.1 | `backend/core/objective_mushra_estimator.py` (umbenannt aus `mushra_evaluator.py`, Schritt 3.5) um `_is_approximation: bool = True`-Feld erweitern. Jeder Score muss als "estimated" markiert sein. | 1 h | 3.5 |
| 10.2 | `docs/listening_study_protocol.md` (aus 3.6) finalisieren: Studiendesign nach ITU-R BS.1116 (small impairments) + BS.1534 (MUSHRA). | 3 h | — |
| 10.3 | `scripts/prepare_listening_study.py` — Generiert Stimulus-Sets: 12 Szenarien × 4 Bedingungen (Original Reference, Aurik, RX 11, Hidden Anchor) × 3 versteckte Wiederholungen = 144 Trials. Randomisiert, balanced, session-fertig. | 4 h | 10.2 |
| 10.4 | `scripts/analyze_listening_study.py` — Liest JSON-Ergebnisse aus ABX-/MUSHRA-Sessions, berechnet: Mittelwert ± 95%-CI, ANOVA, Tukey HSD Post-hoc, Inter-Rater-Reliability (ICC). | 4 h | 3.3 |
| 10.5 | Validierungs-Dokument: `docs/PERCEPTUAL_VALIDATION_REPORT.md` — Vorlage für Studie. Enthält: Methodik, Teilnehmer-Demographie, Ergebnisse, Diskussion, Limitationen. | 2 h | — |
| 10.6 | Externe Validierung initiieren: Kontaktaufnahme mit Tonstudio/Hochschule für unabhängige Hörtests (≥12 Teilnehmer, doppelblind). Ergebnis publizieren (ArXiv, AES-Preprint). | Langfristig | 10.2–10.5 |

### Erfolgskriterien

- `objective_mushra_estimator.py` markiert jeden Score als "estimated" (kein Score wird als echter MUSHRA ausgegeben)
- `prepare_listening_study.py` generiert ITU-konforme Stimulus-Sets
- `analyze_listening_study.py` berechnet 95%-CI und ANOVA
- Mindestens 1 externe Validierungsstudie initiiert (Kontakt/Commitment)

---

## Gesamt-Zeitplan

| Phase | Muster | Geschätzte Stunden | Priorität |
|-------|--------|-------------------|-----------|
| **Sprint 1** | §15.1 Competitive Benchmarks | 16–18 h | 🔴 Kritisch |
| **Sprint 1** | §15.2 Echt-Audio-Corpus | 13 h | 🔴 Kritisch |
| **Sprint 2** | §15.3 ABX/Wahrnehmung | 24 h | 🔴 Kritisch |
| **Sprint 2** | §15.10 Perceptual-Validation | 14 h (+extern) | 🔴 Kritisch |
| **Sprint 3** | §15.8 ErrorGuard | 15 h | ✅ ERLEDIGT (§v10.115) |
| **Sprint 3** | §15.9 Memory-Lifecycle | 17 h | 🟡 Strukturell |
| **Sprint 4** | §15.4 Cross-Platform CI | 7 h | 🟡 Strukturell |
| **Sprint 4** | §15.5 GPU-Strategie | 18 h | 🟡 Strategisch |
| **Sprint 5** | §15.6 Plugin-SDK | 16 h | 🟡 Produkt |
| **Sprint 5** | §15.7 Dokumentation | 19 h | 🟡 Produkt |
| **Gesamt** | | **159–167 h** (~4–5 Wochen Vollzeit) | |

---

## Maintainer Sign-off

- [ ] Architektur-Review: Bridge-Bypass-Verbot bleibt intakt (§V01)
- [ ] Qualitäts-Review: Kein Test darf auf PESQ/SI-SDR/STOI basieren (§V14)
- [ ] Speicher-Review: ONNX-Sessions ≤2GB unter Last
- [ ] Recht-Review: Corpus-Lizenzen geprüft, kein urheberrechtlich geschütztes Material

---

## Status-Update v10.0.0-Phantom (11. Juli 2026)

17 der 18 spezifizierten Gaps wurden implementiert (siehe §16 Phantom-Rollout).
Einzig ausstehend: §15.10 Externe Validierungsstudie (Protokoll + Tools fertig).
