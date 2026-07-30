# Aurik 10 — Spec 08: Architektur, Code-Standards & Distribution | §v10 Pleasantness-First

> Softwareschichten, Code-Konventionen, Frontend-Regeln, Plugin-Policy,
> CLI, Distribution (AppImage/NSIS), Out-of-the-Box-Pflicht.

---

## §11 Softwareschichten-Architektur

```text
┌─────────────────────────────────────────────────────────────┐
│  Desktop-UI  (Aurik10/)   PyQt5 · ModernMainWindow          │
│    BatchProcessingThread (Stufe 1, Quality-first Standard)    │
│    MLRefinementThread    (Stufe 2, LIMIT_BACKGROUND=∞)       │
├─────────────────────────────────────────────────────────────┤
│  Frontend-Bridge  backend/api/bridge.py  Lazy-Imports        │
├─────────────────────────────────────────────────────────────┤
│  Orchestrator  denker/aurik_denker.py  8 Stufen (kanonisch)  │
├─────────────────────────────────────────────────────────────┤
│  Backend-Core  backend/core/ · plugins/ · dsp/  DSP + ML   │
│    PerformanceGuard  (Limiten: BALANCED/QUALITY/MAXIMUM/∞)   │
│    MLRefinementQueue (DeferredRefinementJob, thread-safe)    │
└─────────────────────────────────────────────────────────────┘
```

**Verbot:** UI-Module unter `Aurik10/` dürfen `backend/core/`, `dsp/` oder `plugins/` **nicht** direkt importieren.
Kommunikation nur über `backend/api/bridge.py` und Qt-Signals/Slots.

### §11.1 Kanonischer Verarbeitungseinstieg (PFLICHT)

- UI/Batches müssen über `get_aurik_denker_class()` aus der Bridge gehen.
- Aufrufkette: `Aurik10/ui/*` → `backend/api/bridge.py` → `denker/aurik_denker.py` → Core.
- Direkte UI-Aufrufe von `UnifiedRestorerV3.restore(...)` sind nicht zulässig.

```python
from backend.api.bridge import get_aurik_denker_class

AurikDenkerClass = get_aurik_denker_class()
denker = AurikDenkerClass()
result = denker.denke(audio, sr, mode="quality")

# Standardpfad (GUI/CLI/Batch): keine Qualitätsreduktion zugunsten RT
result = denker.denke(audio, sr, mode="quality", no_rt_limit=True)
```

### §11.1b [RELEASE_MUST] Canonical Contract Drift Gate

Alle Release-fähigen Oberflächen müssen denselben Aurik-Vertrag ausführen. Es darf keine funktional ähnliche Parallelwelt mit eigenem Import, eigener Voranalyse, eigenem Modus-Mapping, eigener Pipeline-Instanz oder eigenem Export entstehen.

**Kanonische Kette:**

1. Audio laden: `backend.api.bridge.get_load_audio_fn()`.
2. Voranalyse: `backend.api.bridge.run_pre_analysis()` genau einmal pro Datei, Ergebnis an den Denker weiterreichen.
3. Pipeline: `get_aurik_denker_instance().denke(...)` oder Bridge-äquivalenter Singleton-Accessor; kein direkter `UnifiedRestorerV3.restore()`-Bypass in UI/CLI/Batch-Releasepfaden.
4. Modus: Nutzeroberfläche bietet exakt `Restoration` und `Studio 2026`; interne Aliasbildung muss deterministisch auf `restoration` / `studio2026` führen.
5. Export: `export_guard()` vor Schreiboperation, `validate_export_quality()` / `build_export_quality_gate_payload()` vor Statusentscheidung, `AudioExporter` als Primärpfad, atomic WAV-Fallback nur mit `PCM_24`.
6. Telemetrie: `degradation_status`, `fail_reason`, `fail_reasons`, `quality_gate_payload` oder äquivalente Bridge-Metadata muss erhalten bleiben.

**Legacy-Regel:** Server-/REST-/Experimentpfade gehören nicht zum Desktop-Releasepfad. Wenn sie im Repository verbleiben, müssen sie als `LEGACY_NON_RELEASE` markiert sein. Ohne diese Markierung müssen sie denselben Bridge-Vertrag erfüllen.

### §11.1c [RELEASE_MUST] Frontend-Version-Update-Invariante

Bei jedem Release-Bump muss die sichtbare Frontend-Version automatisch aus der kanonischen
Paketversion kommen und darf nicht in UI-Texten hartkodiert werden.

Pflichtregeln:

1. Kanonische Quelle: `Aurik10/__init__.py::__version__`.
2. Fenstertitel: `Aurik10/ui/modern_window.py` nutzt `_AURIK_VERSION` aus `__version__`.
3. Splashscreen-Badge: `Aurik10/ui/splash_screen.py` nutzt `_VERSION` aus `__version__`.
4. App-Metadaten: `Aurik10/main.py` setzt `app.setApplicationVersion(__version__)`.
5. `ui.app_title` in i18n darf eine Versionsanzeige nur als dynamischen Platzhalter enthalten
    (z. B. `{version}`), niemals als hartkodierte Release-Nummer.

Abweichungen sind Release-Blocker.

### §11.1d [RELEASE_MUST] ROCm TorchAudio-ABI-Invariante

Der GPU-Launcher darf niemals mit einem inkonsistenten `torch`/`torchaudio`-Stack starten.
Ein ABI-Fehler (`undefined symbol`, Build-Tag-Mismatch) muss vor dem UI-Start erkannt und
automatisch repariert oder klar auf CPU-Fallback umgeschaltet werden.

Pflichtregeln:

1. Bei Auswahl des ROCm-venv muss ein Preflight laufen: `import torch`, `import torchaudio`, Build-Tag-Abgleich (`+rocmX.Y` gleich auf beiden Paketen).
2. Fehlerfall: Auto-Reparatur ist Pflicht, indem `torchaudio==<torch.__version__>` aus dem passenden PyTorch-ROCm-Index installiert wird.
3. Wenn Reparatur fehlschlägt und der Fehler auf `torchaudio` begrenzt ist (Import/ABI), bleibt der ROCm-GPU-Pfad aktiv; ausschließlich `torchaudio`-abhängige Phasen/Plugins müssen selektiv auf CPU/DSP fallbacken.
4. Nur wenn `torch` selbst im ROCm-venv nicht nutzbar ist, ist ein globaler CPU-Fallback zulässig.
5. `.pth`-Bridge darf keine CPU-`torchaudio`-Wheels in den ROCm-Interpreter einschleusen; lokales ROCm-venv-Paket hat Vorrang.
6. Jeder Fallback-Grund muss sichtbar im Launcher-Log stehen (kein stilles Degradieren).

Abweichungen sind Release-Blocker.

### §11.1a [RELEASE_MUST] Bridge-Experience-Insights-Kontrakt (v10.0.0)

`backend/api/bridge.py` MUSS eine stabile Extraktionsfunktion bereitstellen:

```python
get_experience_insights(result) -> {
    "joy_index": float,
    "fatigue_index": float,
    "cluster_key": str,
    "cluster_policy": dict,
    "recommendations": list,
    "recommendation_count": int,
}
```

**Invarianten:**

- Frontend nutzt diese Funktion statt ad-hoc-Metadata-Parsen.
- Rückgabe bleibt bei fehlenden Feldern schema-stabil (`0.0`, `""`, `{}`, `[]`).
- NaN/Inf sind in Rückgabe verboten.

---

## §3 Code-Standards

### §3.1 Numerische Robustheit (PFLICHT)

```python
# Nach jeder numerischen Operation:
result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

# Ausgabe-Audio immer clippen:
audio = np.clip(audio, -1.0, 1.0)

# Float-Guard vor Qualitäts-Updates:
if not math.isfinite(score):
    logger.debug("Score ungültig, überspringe Update")
    return
```

**ABSOLUTES VERBOT:** Jede Audio-Ausgabe und jeder Score muss NaN/Inf-frei sein.

### §3.2 Singleton-Pattern (PFLICHT für jedes neue Kernmodul)

```python
import threading
from typing import Optional

_instance: Optional[MyModule] = None
_lock = threading.Lock()

def get_my_module() -> MyModule:
    """Thread-sicherer Singleton (Double-Checked Locking)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MyModule()
    return _instance

def my_convenience_function(audio: np.ndarray, sr: int) -> MyResult:
    return get_my_module().process(audio, sr)
```

**Pflicht-Invariante:** Jeder globale `_instance`-Cache mit `threading.Lock()`.
`_cache = {}` ohne Lock in Produktionscode ist **verboten**.

### §3.3 Docstring-Standard (PFLICHT)

```python
def score_audio(self, reference: np.ndarray, degraded: np.ndarray, sr: int) -> PQSResult:
    """Berechnet perceptuellen Qualitätsscore (referenz-basiert).

    Algorithmus:
        1. Kreuzkorrelationsbasierte Zeitausrichtung
        2. Gammatone-Filterbank (25 Bänder, 50–8000 Hz)
        3. NSIM = SSIM(Gammatone(ref), Gammatone(deg))
        4. MCD = (10/ln10) · √(2·Σᵢ(cᵢ_ref - cᵢ_deg)²)  [dB]
        5. MOS = 1.0 + 4.0·σ((z-0.5)·8)

    Args:
        reference: Referenz-Audio (1D float32/64, normalisiert [-1,1])
        degraded:  Degradiertes Audio (selbe Länge)
        sr:        Sample-Rate (muss 48000 sein)

    Returns:
        PQSResult mit .mos, .nsim, .mcd_db, .spectral_coherence
    """
```

### §3.4 Lazy Imports & Graceful Degradation

```python
def _load_optional_model(model_path: str, plugin_name: str = ""):
    try:
        import onnxruntime as ort
        try:
            from backend.core.ml_device_manager import get_ort_providers as _get_prov
            providers = _get_prov(plugin_name)
        except Exception:
            providers = ["CPUExecutionProvider"]
        return ort.InferenceSession(model_path, providers=providers)
    except (ImportError, FileNotFoundError):
        logger.debug("ONNX nicht verfügbar, nutze DSP-Fallback")
        return None
# Pflicht: ml_device_manager für Device-Dispatch; CPU-Fallback immer gewährleistet
```

### §3.5 Logging-Konventionen

```python
logger = logging.getLogger(__name__)

# Nutzer-Meldungen/UI: DEUTSCH
# Code-Kommentare/Docstrings: ENGLISCH
# Log-Meldungen (technisch): ENGLISCH

logger.info("🧠 CausalReasoner: Ursache=%s Konfidenz=%.2f", cause, conf)
logger.info("📊 PQS-Score: MOS=%.2f NSIM=%.3f MCD=%.1f dB", mos, nsim, mcd)
# KEIN print() in Produktionscode
```

### §3.5a [RELEASE_MUST] Heavy-ML Headroom-Guard-Kontrakt

Heavy-ML-Pfade muessen vor **Load** und vor **Inference** einen phasenlokalen RAM-Headroom-Guard ausfuehren.

```python
if not has_sufficient_ml_headroom(audio, sr, model_name):
    # Structured runtime fallback, never skip entire phase
    metadata.setdefault("ml_guard_events", []).append({...})
    deferred_phases.append("phase_xx")
    return run_dsp_fallback(audio, sr)
```

**Pflichtregeln:**

- `FlashSRPlugin()` / `InferenceSession()` / `torch.load()` nur nach positivem Guard.
- Bei knappem RAM erst proaktiv aufraeumen (`evict_stale_plugins`, `gc.collect`, `malloc_trim`).
- Guard-Fallback darf nicht auf Original-Audio zurueckspringen.
- Log-Meldungen bleiben technisch auf Englisch; Nutzertexte bleiben Deutsch.

### §3.6 Datenklassen für Ergebnisse

```python
from dataclasses import dataclass, field

@dataclass
class MyResult:
    """Immer als @dataclass — niemals als raw dict zurückgeben."""
    primary_metric: float
    metadata: dict[str, float] = field(default_factory=dict)
```

### §3.7 Type-Annotation-Pflicht (ab v10.0.0)

```python
# PFLICHT für alle public APIs:
def process(self, audio: np.ndarray, sr: int, *, mode: str = "restoration") -> ProcessResult: ...

# VERBOTEN:
def process(self, audio, sr, mode="restoration"): ...  # ❌ kein Type

# mypy.ini: strict = true, disallow_untyped_defs = true
```

### §3.8 SHA256-Ergebnis-Cache (für teure Operationen)

```python
import hashlib, threading

_result_cache: dict[str, object] = {}
_cache_lock = threading.Lock()

def audio_sha256(audio: np.ndarray, sr: int) -> str:
    h = hashlib.sha256()
    h.update(audio.tobytes())
    h.update(sr.to_bytes(4, "little"))
    return h.hexdigest()[:16]

# Cache-Regeln:
# - Max. 128 Einträge (FIFO-Trim)
# - Immer mit threading.Lock() (thread-sicher)
# - Kein Disk-Cache (nur RAM, Prozess-Leben)
# - Cache-Keys: Modul-Präfix + SHA256 ("panns:abc123", "scan:def456")
```

---

## §3.9 [RELEASE_MUST] Stabilitäts-Invarianten (v10.0.0)

Ergänzende Invarianten zur Absicherung gegen Abstürze, OOM, Deadlocks, Freezes und undefinierte Zustände.  
Diese Regeln sind orthogonal zu §2.38–§2.41 und fokussieren auf Laufzeit-Systemgrenzen.

### §3.9.1 Per-Phase-Inference-Timeout

**Problem**: `ort.InferenceSession.run()` oder `torch.model()` können bei korruptem Modell oder BLAS-Deadlock unbegrenzt blockieren. `PerformanceGuard` misst nur kumulativen RT-Faktor, nicht wall-clock pro Inferenz.

**Pflicht**: Jede schwere ML-Inferenzphase (≥ 0.5 GB Modell) MUSS in `concurrent.futures.ThreadPoolExecutor.submit()` + `future.result(timeout=PHASE_INFERENCE_TIMEOUT_S)` gewrappt sein.

```python
# Pflicht-Pattern für schwere Inferenz (ONNX / torch)
PHASE_INFERENCE_TIMEOUT_S = 300.0  # 5 Minuten; überschreiten = hängendes Modell

import concurrent.futures

def _run_inference_with_timeout(fn, *args, timeout=PHASE_INFERENCE_TIMEOUT_S, **kwargs):
    """Run ML inference in a daemon thread with wall-clock timeout.

    On timeout: logs error, raises InferenceTimeoutError.
    Caller MUST catch and fall back to DSP path.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="aurik-inf") as exc:
        fut = exc.submit(fn, *args, **kwargs)
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.error("Inference timeout after %.0f s — phase=%s", timeout, fn.__name__)
            raise InferenceTimeoutError(f"Inference timeout: {fn.__name__}")
        except Exception:
            raise
```

**Invarianten:**

- `InferenceTimeoutError` → DSP-Fallback der Phase (kein Phase-Skip auf Original-Audio).
- Betroffene Phase in `deferred_phases` eintragen → KMV Stufe 2 wiederholt ohne Zeitlimit.
- `metadata["fail_reasons"]` erhält strukturierten Eintrag `reason_code="inference_timeout"`.
- VERBOTEN: `threading.Thread.join()` ohne `timeout=` auf ML-Inferenz-Threads.

### §3.9.2 SIGTERM-Handler — Checkpoint bei graceful Shutdown

**Problem**: systemd-oomd sendet SIGKILL (nicht fangbar); `systemctl stop` / Prozessmanager senden SIGTERM (fangbar). §2.39 fängt nur Python `MemoryError`.

**Pflicht**: `main.py` setzt nach `QApplication`-Initialisierung einen SIGTERM-Handler:

```python
import signal, threading

def _sigterm_handler(signum, frame):
    """SIGTERM → emergency checkpoint + graceful Qt shutdown."""
    logger.warning("SIGTERM received — initiating emergency checkpoint")
    # 1. Emergency-Checkpoint versuchen (best-effort, non-blocking)
    _emergency_checkpoint_if_running()
    # 2. Qt-Shutdown aus Main-Thread via QTimer (thread-safe)
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QTimer
    _app = QApplication.instance()
    if _app:
        QTimer.singleShot(0, _app.quit)

signal.signal(signal.SIGTERM, _sigterm_handler)
```

**`_emergency_checkpoint_if_running()`-Invarianten:**

- Non-blocking: `threading.Event.wait(timeout=0)` — kein Warten auf laufende Phase.
- Nur aufgerufen wenn `BatchProcessingThread.isRunning() == True`.
- Schreibt Checkpoint-Datei atomar (`.tmp` → `os.replace`) wenn `audio_original` im Speicher ist.
- SIGKILL kann NICHT abgefangen werden — §2.39 dokumentiert diese Einschränkung explizit.

### §3.9.3 Phase-Output-Guard — strukturelle NaN/Inf-Absicherung

**Problem**: §3.1 schreibt `np.nan_to_num` + `np.clip` per Konvention vor; keine strukturelle Erzwingung. Ein fehlerhafter ML-Output kann NaN-Audio durch alle nachfolgenden Phasen propagieren.

**Pflicht**: `backend/core/phase_output_guard.py` stellt Decorator bereit:

```python
def phase_output_guard(fn):
    """Decorator: wraps phase function and guards output audio array.

    Enforces on every return value:
      1. np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
      2. np.clip(audio, -1.0, 1.0)
      3. assert np.isfinite(audio).all()  — hard fail if guard insufficient
      4. assert audio.dtype == np.float32

    If assertion fails: logs CRITICAL + raises PhaseOutputError.
    Caller (PerPhaseMusicalGoalsGate / UV3) catches + DSP-fallback.
    """
    ...
```

**Anwendung**: Alle Phasen-Funktionen (01–64) MÜSSEN mit `@phase_output_guard` dekoriert ODER manuell äquivalent absichern.

**Invariante**: NaN-Propagation aus ML-Ausgaben ist verboten. Stille (Nullen) ist der sichere Fallback bei korruptierter Inferenz.

### §3.9.4 ThreadPoolExecutor-Lifecycle — kein Orphan-Thread

**Problem**: `ThreadPoolExecutor`-Instanzen ohne explizites `shutdown()` können beim Prozessende Worker-Threads als Zombie hinterlassen, die offene Dateien / Sockets / Modell-Handles halten.

**Pflicht**:

```python
# PFLICHT: Context Manager oder explizites Shutdown in Cleanup
with ThreadPoolExecutor(max_workers=3) as pool:
    results = list(pool.map(fn, items))
# ↑ __exit__ ruft pool.shutdown(wait=True) automatisch

# Falls kein Context Manager möglich: in __del__ oder atexit
def _cleanup(self):
    if hasattr(self, "_executor") and self._executor:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._executor = None
```

**Invarianten:**

- `module_coordinator.py` und alle `ThreadPoolExecutor`-Instanzen in `backend/core/` MÜSSEN `shutdown(wait=True, cancel_futures=True)` in ihrer Cleanup-Methode aufrufen.
- Alle lang-lebigen Executor-Instanzen als Kontext-Manager oder mit `atexit.register(executor.shutdown)`.
- VERBOTEN: dauerhaft laufende Executors ohne Shutdown-Kontrakt.

### §3.9.4a Background-Monitor-Lifecycle — kein Zombie-Daemon

**Problem**: Lang lebige Monitor-Threads (z. B. RAM-/Plugin-Lifecycle-Manager) sind oft als `daemon=True` implementiert. Das verhindert zwar Prozess-Hänger am Exit, ist aber **kein** ausreichender Cleanup-Vertrag: in Tests und bei geordnetem App-Shutdown bleiben sonst Race-Conditions, GIL-Stalls und stale Singleton-Zustände zurück.

**Pflicht**:

```python
class Manager:
    def shutdown(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.0)  # best-effort, non-blocking
        self._thread = None
```

**Invarianten:**

- Jeder lang lebige Monitor-Thread in `backend/core/` MUSS einen **idempotenten** `shutdown()`-Pfad besitzen.
- `shutdown()` darf nie unbounded blockieren; `join(timeout=...)` ist Pflicht, nacktes `join()` ist verboten.
- `daemon=True` ist nur eine zusätzliche Exit-Sicherung, **nicht** der primäre Lifecycle-Kontrakt.
- Test-Infrastruktur darf solche Manager in `pytest_sessionfinish` oder Fixture-Finalizern stoppen und Singleton-Instanzen resetten.
- Implementierungen müssen nach `shutdown()` einen erneuten Prozess-/Session-Start ohne stale Thread-Referenz erlauben.

### §3.9.5 ml_memory_budget Startup-Reconciliation

**Problem**: Bei SIGKILL während eines `try_allocate()`-Aufrufs bleibt die Budget-Buchführung inkonsistent. Nächster Start sieht ggf. ein bereits voll ausgelastetes Budget (stale allocation), obwohl kein Modell mehr geladen ist.

**Pflicht**: `ml_memory_budget.py` führt nach Initialisierung eine **Reconciliation** durch:

```python
def _reconcile_on_startup(self) -> None:
    """Reset allocated budget to 0 on fresh process start.

    Rationale: All allocations from a previous process are gone after OS cleanup.
    Each module re-registers via try_allocate() when it actually loads its model.
    No stale allocation persists across process boundaries.
    """
    with self._lock:
        # Reset to zero: this process has no loaded models yet.
        self._allocated_gb = 0.0
        self._allocations.clear()
    logger.info("ml_memory_budget: startup reconciliation — budget reset to 0.0 GB")
```

**Invariant**: `_reconcile_on_startup()` wird im `__init__` der `MLMemoryBudget`-Singleton-Klasse aufgerufen — genau einmal pro Prozessstart.

### §3.9.6 Structured Exception Logging — kein stilles `except Exception:`

**Problem**: Breite `except Exception:` ohne Structured-Fail-Reason in `genre_classifier.py`, `musikalischer_globalplan.py`, `lyrics_guided_enhancement.py` u.a. schlucken Fehler lautlos; `metadata["fail_reasons"]` bleibt leer.

**Pflicht**: Jedes `except Exception:` in pipeline-kritischen Pfaden (Phasen, Plugins, Denker-Kette) MUSS:

```python
except Exception as exc:
    logger.error("phase=%s error=%s", phase_id, exc, exc_info=True)
    # §2.41 Structured Fail-Reason Pflicht
    metadata.setdefault("fail_reasons", []).append({
        "phase_id": phase_id,
        "reason_code": "phase_exception",
        "severity": "error",
        "action": "fallback",
        "details": {"exc_type": type(exc).__name__, "exc_msg": str(exc)[:200]},
    })
    # Dann: DSP-Fallback oder re-raise (nie silent ignore)
```

**Invarianten:**

- VERBOTEN: `except Exception: pass` in Phasen-Code.
- VERBOTEN: `except Exception as e: return None` ohne vorherigen Log + `fail_reasons`-Eintrag.
- `details.exc_msg` auf 200 Zeichen begrenzen (kein sensitives Log-Overflow).

### §3.9.7 Audio-Buffer-RAM-Guard vor Pipeline-Eintritt

**Problem**: Sehr große Audiodateien (z. B. 8 h / 10 GB WAV) werden von `AudioFileValidator` auf Dateigröße geprüft, aber die `numpy`-Allokation im Speicher kann 4–10× der Dateigröße übersteigen (float32 statt int16 + Stereo-Duplikate).

**Pflicht**: Nach `soundfile.read()` / `pedalboard.read()`, vor Pipeline-Übergabe:

> **Audio-Import-Kaskade (kanonisch, Stand April 2026):** Alle Einstiegspunkte (`batch_processor.py`, `backend/aurik_restore.py`, `backend/meta_router.py`, `Aurik10/ui/modern_window.py`) MÜSSEN `load_audio_file(filepath)` aus `backend.file_import` verwenden — **nicht** `sf.read(path)` oder `librosa.load(path)` direkt. Die Kaskade: `soundfile` (WAV/FLAC/OGG) → `pedalboard/FFmpeg` (MP3/AAC/WMA/Opus) → `pydub` (universell). `sf.read(io.BytesIO(...))` auf interne PCM-Puffer ist zulässig.

```python
MAX_AUDIO_BYTES_RAM: int = 4 * 1024**3  # 4 GB absolutes RAM-Limit für einen Audio-Buffer

def _check_audio_buffer_size(audio: np.ndarray, file_path: str) -> None:
    """Raises AudioTooLargeError if audio array exceeds RAM guard."""
    nbytes = audio.nbytes
    if nbytes > MAX_AUDIO_BYTES_RAM:
        raise AudioTooLargeError(
            f"Audio-Buffer {nbytes / 1024**3:.1f} GB überschreitet RAM-Limit "
            f"({MAX_AUDIO_BYTES_RAM / 1024**3:.0f} GB). "
            f"Bitte kürze '{Path(file_path).name}' oder teile die Datei auf."
        )
```

**Invarianten:**

- Prüfung erfolgt nach Laden, VOR `resample_poly` (Resampling vergrößert Buffer weiter).
- `AudioTooLargeError` → `item_error`-Signal mit verständlicher deutscher Fehlermeldung.
- `MAX_AUDIO_BYTES_RAM` als Konfigurationskonstante in `backend/core/audio_validator.py` [ROADMAP].

**Code-Sync v10.0.0:** Das normative Limit ist auf 4 GB harmonisiert und entspricht
dem aktuellen UV3-Guard in `backend/core/unified_restorer_v3.py`.

### [RELEASE_MUST] §3.9.8 Lock-Acquisition-Order — Deadlock-Prävention zwischen ARM und PLM

**Problem**: `AdaptiveResourceManager` (ARM) und `PluginLifecycleManager` (PLM) halten eigene Locks. Ein zirkulärer Lock-Erwerb (ARM-Lock → PLM-Lock in einem Thread; PLM-Lock → ARM-Lock in einem anderen) ist ein klassisches Deadlock-Muster.

**Bindende Lock-Ordnung:**

| Priorität | Lock | Besitzer |
| --- | --- | --- |
| 1 (zuerst) | `MLMemoryBudget._lock` | `ml_memory_budget.py` |
| 2 | `PluginLifecycleManager._lock` | `plugin_lifecycle_manager.py` |
| 3 | `AdaptiveResourceManager._lock` | `adaptive_resource_manager.py` |

**Invarianten:**

- Ein Thread darf NIEMALS Lock der Priorität N zuerst acquiren, wenn er bereits Lock der Priorität M > N hält.
- `evict_stale_plugins()` (ARM aufgerufen) läuft AUSSERHALB des ARM-Locks — korrekt so, MUSS beibehalten werden.
- Neue Module: Lock-Dokumentation als Docstring (`# Lock-order: Priority N — see §3.9.8`).
- VERBOTEN: verschachteltes Locking über Modulgrenzen hinweg ohne Dokumentation der Ordnung.

### §3.9.9 MLRefinementThread — Buffer-Registrierung + Post-Abbruch-Cleanup

**Problem**: `DeferredRefinementJob.audio_original` hält mehrere GB Audio-Daten. Bei `terminate()` (Watchdog-Kill nach wait(3000)) läuft Python-Cleanup (`__del__`) ggf. nicht — `ml_memory_budget` bleibt fehlerhaft belastet.

**Pflicht**:

```python
class DeferredRefinementJob:
    def __init__(self, audio_original, ...):
        self.audio_original = audio_original
        # §3.9.9: Budget-Registrierung sofort bei Job-Erstellung
        _size_gb = audio_original.nbytes / 1024**3
        if not ml_memory_budget.try_allocate("kmv_job", _size_gb):
            raise MemoryError(f"KMV: Insufficient RAM for job buffer ({_size_gb:.2f} GB)")
        self._registered_size_gb = _size_gb

    def release_buffer(self) -> None:
        """Must be called after Stufe-2-Export OR on cancellation."""
        if getattr(self, "_registered_size_gb", 0) > 0:
            ml_memory_budget.release("kmv_job")
            self._registered_size_gb = 0.0
        self.audio_original = None  # GC-freigabe
```

**`MLRefinementThread.run()`-Cleanup-Invariante:**

```python
try:
    # ... vollständige UV3-Pipeline (Stufe 2) ...
finally:
    # §3.9.9: Buffer IMMER freigeben — auch bei Abbruch/Exception
    if job is not None:
        job.release_buffer()
```

**Invarianten:**

- `DeferredRefinementJob.release_buffer()` wird in `finally`-Block aufgerufen, unabhängig von Erfolg/Abbruch.
- Nach Startup: `_reconcile_on_startup()` (§3.9.5) setzt KMV-Budget automatisch auf 0 zurück — kein manueller Cleanup nötig nach SIGKILL.
- VERBOTEN: `audio_original` im Job halten, nachdem `release_buffer()` aufgerufen wurde.

---

## §10 Verbotene Praktiken

### Code-Qualität

```python
# VERBOTEN:
print(f"Score: {score}")         # → logger.info()
return {"mos": 3.5}              # → return PQSResult(mos=3.5)
if score > 0:                    # NaN-Falle → if math.isfinite(score) and score > 0:
_cache = {}                      # ohne Lock → threading.Lock() Pflicht
```

### Architektur

- Kein direkter `Aurik10/`-Import in Core-Modulen
- Keine hardcodierten Pfade → stets `pathlib.Path.home() / ".aurik" / ...`
- Kein `from module import *`
- Keine sync-Datei-I/O in Hot-Paths (GP-Gedächtnis nur am Anfang/Ende)
- Keine realen Audio-Dateien in Tests
- Keine Sprach-Metriken (PESQ, DNSMOS, NISQA, STOI) für Musikbewertung

### Verbotene Legacy-Algorithmen als Primärverarbeitung

```text
Ephraim & Malah (1984) Wiener-Filter  → Ersatz: OMLSA/IMCRA
Simple Spectral Subtraction            → Ersatz: MMSE-LSA + OMLSA
YIN Pitch-Tracker                      → Ersatz: pYIN / CREPE
Medianfilter-Declicker (primitiv)      → Ersatz: RBME + iterative Konsistenz
AR-Modell ohne spektrale Konsistenz    → Ersatz: NMF-β + Sinusoidal Modeling
```

---

## §10.5 Eingabedatei-Sicherheit (OWASP-konform)

```python
class AudioFileValidator:
    MAX_FILE_SIZE_BYTES: int = 10 * 1024 ** 3   # 10 GB
    MAX_DURATION_HOURS: float = 8.0

    def validate(self, path: pathlib.Path) -> None:
        # 1. Dateigröße ≤ 10 GB
        # 2. Magic-Bytes-Verifikation (WAV: b'RIFF', FLAC: b'fLaC', ...)
        # 3. Länge ≤ 8 Stunden, SR ∈ [8000, 384000]
        # 4. Kanäle 1–2 (> 2 → Downmix)
        # 5. Kein eval() / subprocess mit Metadaten-Inhalt
        # 6. Pfad-Traversal ausgeschlossen: os.path.realpath() vor Zugriff
        # 7. FFmpeg: Dateiname IMMER als Liste (kein Shell-String → Injection-Schutz)
        ...
```

---

## §11.3 Plugin-Policy — bestehende Plugins nutzen, nicht neu schreiben

> **Aktuelle Plugin-Anzahl**: 51 Plugin-Dateien unter `plugins/` (Stand v10.0.0).

```text
✅ = lokal gebündelt, out-of-the-box, kein Download

# Vocoder / Synthese
plugins/vocos_plugin.py               ✅ PRIMÄRER Vocoder (Vocos 48 kHz nativ, Kaskade: 48k→44.1k→24k)
plugins/bigvgan_v2_plugin.py           ✅ BigVGAN-v2 (0,4 GB ONNX/PyTorch, SEKUNDÄRER Vocoder; Studio-2026, GPU-beschleunigt)
plugins/hifigan_plugin.py             ✅ HiFi-GAN (3,6 MB ONNX, Tertiär-Fallback)

# Stem-Separation
plugins/demucs_v4_plugin.py           ✅ MDX23C Kim_Vocal_2/Kim_Inst (2× 64 MB)
plugins/bs_roformer_plugin.py         ✅ BS-RoFormer / Mel-RoFormer (+0.4–0.8 dB SDR)

# Rauschunterdrückung & Dereverb
plugins/deepfilternet_v3_ii_plugin.py ✅ DeepFilterNet3 (Schröter et al. 2023, 37 MB, 3 ONNX) — PRIMÄR NR
                                          # "v3.II" = Aurik-interne Iterations-Bezeichnung (keine offizielle DeepFilterNet-Versionsnummer)
plugins/sgmse_plugin.py               ✅ SGMSE+ (251 MB TorchScript) — PRIMÄR Dereverb/Enhancement
plugins/mp_senet_plugin.py            ✅ MP-SENet 2023 (ONNX) — Music/Vocal Enhancement
                                      Laufzeitvertrag: segmentierte Inferenz in 32-Frame-Chunks
plugins/wpe_plugin.py                 ✅ WPE Dereverb (3-Tier: nara_wpe→NumPy→OMLSA)
# VERBOTEN: dccrn_plugin (deprecated — ersetzt durch mp_senet_plugin)

# Codec-Artefakte
plugins/apollo_plugin.py              ✅ Apollo (65 MB ONNX, PRIMÄR Codec-Artefakte)
plugins/resemble_enhance_plugin.py    ✅ Resemble-Enhance (41 MB ONNX, Fallback)

# Audio-Tagging & MOS
plugins/beats_plugin.py               ✅ BEATs iter3 (90 MB ONNX) — PRIMÄR Audio-Tagging +10.7 % mAP
plugins/panns_plugin.py               ✅ PANNs CNN14 (81 KB ONNX) — Fallback zu BEATs
plugins/versa_plugin.py               ✅ VERSA-MOS (45 MB ONNX) — PRIMÄR MOS-Bewertung 2024
# VERBOTEN: cdpam_plugin (Sprach-Corpus, ersetzt durch versa_plugin §4.4)

# Pitch / Formanten
plugins/rmvpe_plugin.py               ✅ RMVPE (26 MB ONNX) — PRIMÄR Pitch-Tracking 2023
plugins/crepe_plugin.py               ✅ CREPE full (85 MB ONNX) — Fallback zu RMVPE
plugins/fcpe_plugin.py                ✅ FCPE (ONNX) — Fallback zu CREPE
plugins/formant_tracker.py            ✅ LPC-Formanten F1–F4 (DSP)

# Inpainting
plugins/flow_matching_plugin.py       ✅ Flow Matching — PRIMÄR Generatives Inpainting (SOTA)
plugins/cqtdiff_plus_plugin.py        ✅ CQTdiff+ — Dropout-Inpainting ≥ 50 ms
plugins/diffwave_plugin.py            ✅ DiffWave (552 KB ONNX) — Fallback

# Ära / Genre / BW-Erweiterung
plugins/era_classifier_plugin.py      EraClassifier (CLAP + DSP-Rolloff)
core/genre_classifier.py              GermanSchlagerClassifier (6-Schicht)
plugins/flashsr_plugin.py             FlashSR BW-Erweiterung (5,9 GB, lazy load)
plugins/matchering_plugin.py          ✅ Reference Mastering (matchering==2.0.6)
```

**Regel:** Vor jeder Neuimplementierung existierende Plugins prüfen (§9.4 Anti-Parallelwelten).

---

## §11.4 Frontend (PyQt5) — bindende Regeln

```python
# PFLICHT: Frameless Window
window.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)

# Magic Buttons: border-image, KEIN Text / setIcon:
_img_r = (Path(__file__).parent.parent / "resources" / "restoration.png").as_posix()
self.btn_magic_restoration = QPushButton()
self.btn_magic_restoration.setStyleSheet(f"""
    QPushButton {{
        border-image: url("{_img_r}") 0 0 0 0 stretch stretch;
        border: 3px solid transparent;
        border-radius: 16px;
    }}
    QPushButton:hover  {{ border: 3px solid rgba(118, 75, 162, 0.75); }}
""")
```

**Keyboard-Shortcuts:**

| Shortcut | Aktion |
| --- | --- |
| Leertaste | Play / Pause |
| `A` | Original hören |
| `B` | Restauriert hören |
| `L` | Lyrics-Timeline-Overlay an/aus |
| `Ctrl+O` | Datei öffnen |
| `Ctrl+S` | Exportieren |
| `Ctrl+R` | Restaurierung (RESTORATION) |
| `Ctrl+Shift+R` | Restaurierung (STUDIO 2026) |
| `Escape` | Verarbeitung abbrechen |
| `Ctrl+Z` | Pfad-Clipboard |

**A/B-Sync-Loop** (v10.0.0): Checkable `btn_ab_sync`-Button — bei aktiviertem Sync wechseln A/B die Quelle im aktuellen Loop-Punkt, kein Reset auf Anfang. Queue-Drag-&-Drop-Reordering für Batch-Liste.

### §11.4c [RELEASE_MUST] Experience-UI-Propagation (v10.0.0)

Nach `item_finished_with_result` MUSS die Frontend-Oberfläche die neuen Runtime-Signale darstellen:

1. `status_text`: Freude-/Ermüdungsindex (z. B. „Freude 78 % · Ermüdung 14 %“).
2. `info_banner`: Cluster-Policy (inkl. Schlüssel) + Top-Auto-Improve-Empfehlungen.
3. Darstellung ist advisory-only; fehlende Signale dürfen UI nicht blockieren.

**Verboten:**

- Direkter Core-Zugriff aus UI zur Berechnung dieser Signale.
- Stilles Verwerfen vorhandener Experience-Telemetrie im UI-Endpfad.

---

## §11.4a Echtzeit-UX-Features (ab 9.10.57 — bindend)

### Signal-Kontrakt Erweiterung

Drei neue Signale auf `BatchProcessingThread` (nach `ml_status_update`):

```python
phase_progress = pyqtSignal(int)    # sub-phase progress 0–100 within current step
scan_progress  = pyqtSignal(float)  # waveform scan-cursor fraction 0.0–1.0
quality_update = pyqtSignal(float)  # live MOS estimate 0.0–5.0
```

### Feature-Übersicht

| # | Feature | Klasse / Methode | Verhalten |
| --- | --- | --- | --- |
| 1 | Zweistufiger Fortschrittsbalken | `phase_progress_bar` (`QProgressBar`, `setFixedHeight(5)`, lila Gradient) | Unter `progress_bar`; eingeblendet bei Batch-Start, ausgeblendet + `setValue(10000)` in `_on_all_finished` |
| 2 | Defekte hochzählen / herunterzählen | `_update_defects` + `_tick_defect_reveal` | `status=="detected"` → Count-up-Animation (QTimer, 22 Frames × 85 ms); `_PHASE_REDUCES`-Mapping × 0.3 bei passenden Phasen-Keywords → `defect_update.emit` |
| 3 | Varianten-Wettkampf | `multi_pass_strategy.process_with_variants()` + `_on_batch_progress` | Nach jeder Variante: `"Variante X/N: 'name' → MOS 4.12 ✓"`; Frontend baut Rangliste `★name_1 (4.12) › name_2 (3.87)` |
| 4 | Musical-Goals-Meter live | `quality_meter_widget.set_mos()` ← `quality_update` | Startet bei 2.5 MOS, steigt proportional zum Fortschritt auf 4.2 |
| 5 | Phasen-Erklärungstext | `_PHASE_EXPL` (22 Einträge) in `_on_batch_progress` | Phasen-Keyword → Kurzbeschreibung, angehängt als `[Kontext]` in Statuszeile |
| 6 | Waveform-Scan-Cursor | `WaveformWidget.set_scan_pos(frac)` ← `_on_scan_progress` | Oranger gestrichelter Cursor: 12 px Glow `rgba(255,150,30,45)` + 2 px DashLine `rgba(255,178,55,215)`; `set_scan_pos(-1.0)` blendet aus; Reset in `_on_all_finished` |
| 7 | Live-Qualitätszahl | `quality_meter_widget` ← `quality_update` | Eingeblendet bei Batch-Start mit `set_mos(2.5)`; steigt mit Fortschritt |
| 8 | Vorab-Hörprobe | `_auto_preview_restored()` ← `QTimer.singleShot(1400, …)` | Spielt erste 5 s (5×48 000 Samples) nach Fertigstellung; nur wenn kein aktiver Playback-Thread läuft |

### Implementierungsregeln

- **Thread-Safety**: alle Widget-Zugriffe via `_dispatch_to_gui` / `QTimer.singleShot(0, fn)` — kein direkter Widget-Zugriff aus `BatchProcessingThread`
- **Scan-Cursor Skalierung**: `scan_progress.emit(float(pct) / 100.0)` in `_on_batch_progress`
- **Quality-Estimate Skalierung**: `quality_update.emit(2.5 + (pct / 100.0) * 1.7)` — Bereich 2.5–4.2 MOS
- **Sub-Bar Skalierung**: `phase_progress.connect(lambda v: phase_progress_bar.setValue(v * 100))` — Eingang 0–100, Bar intern 0–10000
- **Defekt-Countdown Multiplier**: 0.3 (reduziert auf 30 % des ursprünglichen Scan-Werts)
- **Auto-Preview Guard**: `_play_thread.is_alive()` prüfen → kein doppelter Playback
- **`_tick_defect_reveal`**: 22 Frames × 85 ms = ~1.9 s Zähleranimation; `_frac = frame / 22.0`

### `_PHASE_REDUCES`-Mapping (17 Einträge, bindend)

```python
_PHASE_REDUCES = {
    "tape_hiss": ["crackle", "noise_level", "noise"],
    "denoise": ["noise_level", "noise", "hum"],
    "dropout": ["dropout"],
    "click_repair": ["clicks", "pops"], "declick": ["clicks", "pops"],
    "wow_flutter": ["wow", "flutter"],
    "reverb_reduction": ["reverb_excess"],
    "frequency_restoration": ["bandwidth_loss"],
    "vocal": ["sibilance"],
    "diffusion_inpainting": ["dropout", "bandwidth_loss"],
    "hum_removal": ["hum"], "rumble": ["rumble"], "declip": ["clipping"],
    "dc_offset": ["dc_offset"], "quantization": ["quantization_noise"],
    "compression_artifact": ["compression_artifacts"],
    "transient": ["transient_smearing"],
}
```

---

## §11.4b Schadensmarker-Lebenszyklus (ab 9.10.123 — bindend)

Schadensmarker in der Wellenformvisualisierung haben einen klar definierten Lebenszyklus.
Das visuelle Feedback teilt sich auf zwei Anzeigebereiche auf:

| Status | Wellenform-Marker | Defekt-Chip im Panel |
| --- | --- | --- |
| `detected` | Farbiger Band-Marker je Defekttyp erscheint per Count-up-Animation | Severity-Chip mit Fortschrittsbalken (rot / amber je Schwere) |
| `correcting` (Schaden aktiv) | Marker bleibt sichtbar; pulsiert nicht | `🔧 Defektname` Chip mit orange Hintergrund |
| **Schaden behoben** (score ≤ 0.01) | **Marker verschwindet vollständig** — `_tick_defect_removal()` entfernt 1–2 Instanzen/75 ms. Kein grünes Overlay, kein Residual-Rechteck. | Fortschrittsbalken **entfällt**; Chip wird zu **`✓ Defektname`** (grüner Haken-Chip, Rand `rgba(77,200,120,0.45)`) |
| `completed` | Alle bearbeiteten Marker verschwunden; dezente grüne Gesamtton-Tinte (alpha 14) über die volle Breite | Alle behobenen Defekte als grüne Haken-Chips |

### Normative Implementierungsregeln §11.4b

- `_show_resolved_markers = False` in `_draw_defect_overlay` — keine grünen Overlay-Rechtecke für gelöste Defekte in der Wellenform
- `_tick_defect_removal()` (75 ms QTimer): pro Tick 1–2 Segmente aus `_pending_removal` → `_resolved_locations` verschieben, `_recently_resolved_ts[dk]` setzen
- **Grüner Haken-Chip**: `fix_ratio >= 0.95` (correcting) / `>= 0.75` (completed) → Chip ohne `_bar`, nur `&#10003; name` mit `background:rgba(77,200,120,0.13);border:1px solid rgba(77,200,120,0.45);border-radius:4px`
- **Fortschrittsbalken** (`■■■■■`, `■■■□□`, `■□□□□`) nur bei nicht-aufgelösten Chips sichtbar
- Aktiver Repair-Chip (`_is_active_chip`): oranger Highlight mit `🔧`-Prefix bleibt bis `fix_ratio < _green_threshold`

---

## §11.4d [RELEASE_MUST] Tonträgerketten-Display-Invarianten (v10.0.0)

Das Tonträger-Display-System in `Aurik10/ui/modern_window.py` hat **drei unabhängige Update-Pfade**, die alle auf denselben State schreiben (`detected_medium_label`, `_carrier_bg_label`). Ohne Single Source of Truth können Medien-Mappings divergieren und Anzeigen falsch oder leer werden.

### Drei Update-Pfade (normativ dokumentiert)

| Pfad | Trigger | Code-Ort | Update-Bedingung |
| --- | --- | --- | --- |
| **A — Pre-Analysis** | `_pre_analysis_bg` nach MediumDetector | ~Zeile 13640 | Immer — auch bei Einzel-Medium |
| **B — Live (während Verarbeitung)** | `__carrier_chain__:`-Nachricht von AurikDenker | `_apply_authoritative_chain_display` | Nur wenn `len(chain_keys) >= 2` |
| **C — Post-Processing** | `item_finished_with_result`-Signal | `_on_item_finished_with_result` | Nur wenn `len(chain_keys) >= 2` |

**Invariante**: Pfad B und C überschreiben Pfad A. Pfad A ist Vorläufig-Anzeige.

### [RELEASE_MUST] Single Source of Truth — Modul-Level-Konstanten und -Helfer

```python
# Aurik10/ui/modern_window.py (Modul-Level — NUR HIER definiert)
_CARRIER_MEDIUM_DISPLAY: dict[str, tuple[str, str]]  # (icon_stem, label) pro Medium-Key
_CARRIER_EXT_DISPLAY: dict[str, tuple[str, str]]     # (icon_stem, label) pro Dateiendung
_CARRIER_ANALOG_MEDIA: frozenset[str]                # analoge Materialtypen
_CARRIER_ICONS_DIR: str                              # Icons-Verzeichnis-Pfad

def _render_carrier_html(icon_stem, label, icons_dir=...) -> str: ...  # Icon oder Plaintext
def _build_carrier_chain_html(chain_keys: list[str]) -> str: ...       # Kette kombinieren
```

**VERBOTEN**: Lokal in Methoden/Callbacks, Lambdas, Background-Threads eigene Varianten dieser Dicts oder `_html()`/`_ci_html()` Funktionen zu definieren (§UI-CARRIER-DISPLAY-INVARIANT).

### State-Synchronisations-Invariante

```python
# IMMER gemeinsam schreiben:
self.detected_medium_label.setText(html)
self._carrier_bg_label = html  # ← CRITICAL: Era-Badge-Block liest diesen State

# NIEMALS nur eines der beiden, weil:
# Era-Badge-Block (~Zeile 15680) liest _carrier_bg_label → schreibt detected_medium_label
# → wenn _carrier_bg_label ≠ detected_medium_label → Silent Data Loss bei Era-Badge-Update
```

### Chain-Info-Key-Invariante

`KettenErgebnis.as_dict()` (in `denker/tontraegerkette_denker.py`) exportiert den Key `"chain"` — **nicht** `"transfer_chain"`. Die korrekte Verwendung in `_on_item_finished_with_result`:

```python
# KORREKT:
_chain_keys = _chain_info.get("chain") or []

# VERBOTEN:
_chain_keys = _chain_info.get("transfer_chain") or _chain_info.get("chain") or []
# → "transfer_chain" existiert nie → erstes get() immer None → verdeckt zukünftige Bugs
```

`as_dict()` liefert folgende Keys: `"chain"`, `"chain_string"`, `"is_multi_generation"`, `"generation_count"`, `"primary_medium"`, `"original_medium"`, `"glieder"`, `"combined_phases"`, `"chain_complexity"`, `"confidence"`, `"spectral_evidence"`, `"reasoning"`.

### Debug-Logging-Pflicht bei Guard-Feuern

Wenn `len(chain_keys) < 2` → Chain-Display wird nicht aktualisiert. Dieses stille Überspringen MUSS immer mit `logger.debug` protokolliert werden:

```python
logger.debug(
    "Kettenanzeige übersprungen – len=%d < 2 (chain=%s)",
    len(chain_keys), chain_keys
)
```

### [RELEASE_MUST] Icon-HTML ohne Plaintext-Fallback — Verboten

`_render_carrier_html()` ist die einzige erlaubte Implementierung:

- Prüft `_svg` → `_png` → `return label` (Plaintext-Fallback)
- `except (OSError, TypeError, ValueError): return label`

**VERBOTEN**: Direktes `f'<img src="file:///{path}"...'` ohne Datei-Existenz-Prüfung und ohne `try/except`.

### Datenstrom-Diagramm

```
MediumDetector.detect()
    └→ PreAnalysisResult.medium.transfer_chain
            └→ _pre_analysis_bg [Pfad A]
                    └→ _build_carrier_chain_html(chain_keys)
                            └→ detected_medium_label.setText()
                               _carrier_bg_label = html  ← State sync

TontraegerketteDenker.analysiere()
    └→ kette.chain (list)
            └→ aurik_denker.py: _emit("__carrier_chain__:" + "|".join(kette.chain))
                    └→ _apply_authoritative_chain_display(chain_keys) [Pfad B]
                            └→ _build_carrier_chain_html(chain_keys)
                                    └→ detected_medium_label.setText()
                                       _carrier_bg_label = html  ← State sync

AurikErgebnis.chain_info = kette.as_dict()
    └→ chain_info["chain"] (list)  ← KEIN "transfer_chain"!
            └→ _on_item_finished_with_result [Pfad C]
                    └→ _build_carrier_chain_html(chain_keys)
                            └→ detected_medium_label.setText()
                               _carrier_bg_label = html  ← State sync

Era-Badge-Block (nach Pfad C)
    └→ liest: _carrier_bg_label (MUSS = detected_medium_label.text())
    └→ schreibt: detected_medium_label.setText(_carrier_bg_label + stars + badge)
                  _carrier_bg_label = new_html  ← State sync
```

### Testpflicht (§UI-CARRIER-DISPLAY-INVARIANT)

- Test: Neues Medium in `_CARRIER_MEDIUM_DISPLAY` → erscheint in **allen drei** Pfaden korrekt (CI-Guard)
- Test: `chain_info` mit key `"chain"` → korrekte Anzeige; `chain_info` ohne `"chain"` → Debug-Log, kein Crash
- Test: Fehlendes Icon-Verzeichnis → Plaintext-Fallback (kein `<img>`-Tag in Output)
- Test: `detected_medium_label.setText()` ohne `_carrier_bg_label`-Sync → nachfolgendes Era-Badge-Update überschreibt korrekt

---

## §11.5 CLI (`aurik_cli.py`)

```bash
# Pflicht-Argumente:
--input FILE   --output FILE   --mode {Restoration,"Studio 2026"}

# Optionale Argumente:
-q, --quiet

# Exit-Codes:
# 0 = Erfolg
# 1 = Argument/CLI-Fehler
# 2 = Input-Datei fehlt
# 3 = Audio-Import fehlgeschlagen
# 4 = Pipeline-Fehler
# 5 = Export/Speicher-Fehler
# 6 = SR-Normierung fehlgeschlagen
# 7 = quality_estimate-Gate verletzt
# 8 = P1/P2 Goal-Gate verletzt
# 9 = Loudness-Drift > 2.5 dB (Pegel-Einbruchsschutz)
# 10 = Pre-Analysis-Fehler
```

### Normative CLI-Invarianten (§11.5a)

- CLI nutzt den gleichen Pfad wie GUI/Batch: `run_pre_analysis(...)` genau 1x vor `AurikDenker.denke(...)`.
- Übergabe an Denker erfolgt per `pre_analysis_result` (direktes Handover, kein redundant zweiter MediumDetect).
- Audio-Import erfolgt kanonisch über Bridge/`load_audio_file` (kein lokales `sf.read`/`librosa.load`-Forking).
- Export ist nur zulässig, wenn neben Goal-/Quality-Gates auch der Pegelabfallschutz besteht (`loudness_drop_db <= 2.5`).

---

### v10-Amendment (2026-07-12)

**version.py Multi-Package-Name-Probe** (v10, 2026-07-12):

`backend/core/version.py::get_aurik_version()` probiert jetzt mehrere mögliche Package-Namen (`aurik9`, `aurik10`, `Aurik10`) via `importlib.metadata.version()`, bevor es auf den `pyproject.toml`-Fallback zurückfällt. Der `PackageNotFoundError`-Fallback-Pfad loggt jetzt auf `logger.debug`-Level (statt `logger.warning`), da der `pyproject.toml`-Fallback in allen getesteten Umgebungen zuverlässig funktioniert.

**Implementierung**: `backend/core/version.py` → `for _pkg_name in ("aurik9", "aurik10", "Aurik10"): try: ... except PackageNotFoundError: continue`.

## §13 Distribution & Out-of-the-Box-Pflicht

> Aurik 10 muss auf einem frischen Linux- oder Windows-System **ohne Python,
> ohne Terminal, ohne Vorkenntnisse** sofort lauffähig sein.

### Installer-Ziele

| Plattform | Format | Anforderung |
| --- | --- | --- |
| **Linux** | AppImage (`.AppImage`) | Einzeldatei, keine Root-Rechte |
| **Windows 10/11** | NSIS-Installer (`.exe`) | Signiert, optional ohne Admin |

### ML-Modell-Gewichte — 100 % offline nach Installation

```json
// models/manifest.json (Version 2)
{
  "version": 2,
  "models": [
    {
      "name": "apollo",
      "bundled": true,
      "bundled_path": "models/apollo/apollo_model.onnx",
      "sha256": "440c48b110f66ff6d7b86cf5bb77201a302d1592ea6471a9b5b99791b21762ac",
      "size_bytes": 67713684,
      "required": false,
      "fallback": "resemble_enhance_onnx"
    }
  ]
}
```

**Out-of-the-Box-Garantie:**

- Kein Download beim ersten Start
- Kein Download nach der Installation
- Alle SOTA-Upgrade-Modelle lokal gebündelt in Programmierphase vorab integriert
- `sota_upgrade`-Feld: nur Entwickler-Metadaten (löst **keinen** Laufzeit-Download aus)
- SHA256-Verifikation für jedes gebündelte Modell Pflicht

### Requirements — nur verifizierte PyPI-Pakete

```text
# VERBOTEN (existieren nicht auf PyPI oder falsche Version):
dccrn==0.1.0          diffwave==1.0.0
hifi-gan==0.1.1       nisqa==1.0.0

# PyTorch — GPU-Mixed-Mode (§GPU-Mixed-Mode):
# Heavy plugins nutzen ROCm/DirectML wenn verfügbar (ml_device_manager).
# CPU-only Fallback auf Systemen ohne unterstützte GPU.
torch==2.2.2  --extra-index-url https://download.pytorch.org/whl/cpu
# Für ROCm:   pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
# Für DirectML: pip install torch-directml
```

### §8.7 AMD-GPU-Beschleunigung — Architektur-Erkennung & Tier-System (v10.0.0)

**Singleton**: `backend/core/ml_device_manager.py` — `get_ml_device_manager()`

**§v10.305 Initialisierungs-Zeitpunkt**: Der Singleton MUSS im **Hauptthread** vor
`ModernMainWindow.__init__()` initialisiert werden (`Aurik10/main.py`).
GPU-Erkennung DARF KEINE blockierenden Operationen (`torch.zeros("cuda")`) ausführen.
Nur Property-Queries: `torch.cuda.is_available()`, `torch.version.hip`,
`torch.cuda.get_device_properties()`.

**Architektur-Erkennung**: `_detect_amd_architecture(device_name)` matcht GPU-Marketing-Namen und
GFX-IDs gegen `_AMD_ARCH_PATTERNS` → `AMDArchitecture` (RDNA3, RDNA2, RDNA1, GCN5, GCN4, CDNA3/2/1).

**Tier-System**: `_compute_gpu_tier(arch, vram_gb)` → `GPUTier` (TIER_1..TIER_4):

| Tier | Architektur | VRAM | max_usage | min_free | fp16_auto |
| --- | --- | --- | --- | --- | --- |
| **1** | RDNA3 (≥16 GB), RDNA2 (≥16 GB), CDNA (≥8 GB) | ≥16 GB | 85 % | 512 MB | Ja |
| **2** | RDNA2 (8–15 GB), RDNA1 (≥8 GB), CDNA (<8 GB) | 8–15 GB | 80 % | 640 MB | Ja |
| **3** | RDNA1/2 (4–7 GB), GCN5 (≥8 GB) | 4–7 GB | 70 % | 768 MB | Ja |
| **4** | GCN4, <4 GB VRAM | <4 GB | 50 % | 512 MB | Nein |

**Tier-basierte Plugin-Ausschlüsse**:

- Tier 3: FlashSR, AudioLDM2, MERT-330M-fairseq → CPU-only (VRAM zu knapp)
- Tier 4: + MERT-330M-HF, BSRoFormer, MDXNet, BigVGAN, CQTDiffPlus, SGMSE → CPU-only

**Auto-fp16**: `get_ort_providers(plugin_name)` aktiviert auf ROCm automatisch fp16 für
`_FP16_ELIGIBLE_PLUGINS` wenn `_TIER_VRAM_PARAMS[tier].fp16_auto == 1.0`.
Plugins müssen `get_ort_providers_fp16()` **NICHT** explizit aufrufen.

**Invarianten:**

- Kein Plugin darf `CUDAExecutionProvider` oder `.to("cuda")` direkt verwenden
- Alle GPU-Zugriffe über `get_torch_device("PluginName")` / `get_ort_providers("PluginName")`
- Jeder GPU-Fehler → `report_gpu_error()` → automatische Session-Deaktivierung nach 3 Fehlern
- Architektur/Tier in `gpu_status_summary()` für UI und Diagnostik verfügbar

```bash
# Vor jedem Release:
python -m pip install --dry-run -r requirements/requirements_aurik.txt
# → 0 Fehler, 0 nicht gefundene Pakete
```

### §9.1 Checkliste neue Kernmodule

```text
□ Datei in backend/core/ angelegt mit Singleton-Pattern (§3.2)
□ Thread-safe: threading.Lock() + Double-Checked Locking
□ Vollständige PEP 484 Type-Annotations (§3.7)
□ NaN/Inf-Schutz in JEDER numerischen Ausgabefunktion
□ @dataclass-Ergebnisse (kein raw dict)
□ logger(__name__) — kein print()
□ ≥ 35 Unit-Tests in tests/unit/test_v<x>_<name>.py
□ Musical Goals (alle 15): kein Ziel nach dem Modul schlechter; Pipeline-Ende erfüllt alle effektiven Zielwerte oder dokumentiert physikalische Limitierung
□ GrooveMetric: kein Timing-Flattening (DTW ≤ 8 ms RMS)
□ SOFT_SATURATION: nicht als CLIPPING fehldetektiert
□ Beide Modi (restoration + studio2026) getestet
□ Out-of-the-Box-Pflicht: DSP-Fallback für alle Plugin-Imports
□ torch-Imports: ml_device_manager für heavy plugins, CPUExecutionProvider als Fallback
□ models/manifest.json: neues Modell eingetragen (sha256 + bundled_path + fallback)
□ scripts/verify_requirements.sh fehlerfrei
□ Alle bestehenden Tests weiterhin grün (CI: `pytest --collect-only -q | tail -1`)
```

### §9.4 Anti-Parallelwelten-Workflow (vor jeder Implementierung)

```text
1. grep -r "<Funktionsname>" backend/core/ plugins/ dsp/ → 0 Treffer?
2. Kein Plugin mit gleicher Funktionalität in plugins/?
3. Keine Phase mit gleicher DSP-Logik in backend/core/phases/?
4. Kein Modul in backend/core/ mit ähnlichem Klassenname?
5. Erst wenn alle 4 Checks negativ → Neuentwicklung starten
6. Prüfentscheidung in CHANGELOG.md dokumentieren
```

### §9.7 Performance-Optimierungen (Pflicht)

```python
# §9.7.1: SHA256-Cache für DefectScanner + PANNs (§3.8-Muster, max. 128 Einträge)
# §9.7.2: Parallele Eingangs-Analyse (MediumDetector + EraClassifier + GenreClassifier)
# PFLICHT: _run_medium_detector nutzt get_medium_detector().detect(audio, sr, file_ext=...)
# VERBOTEN: _run_medium_classifier (MediumClassifier.classify_medium() kennt kein file_ext
#           → gibt bei codec-enkodiertem Analog-Material 'unknown' zurück)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    fut_mc  = pool.submit(_run_medium_detector, audio, sr, file_ext)  # get_medium_detector().detect()
    fut_era = pool.submit(_run_era_classifier,  audio, sr)
    fut_sc  = pool.submit(_run_genre_classifier, audio, sr)

# §9.7.3: Phasen-adaptive PMGG-Sample-Dauer
PHASE_SAMPLE_DURATIONS = {
    "phase_30": 1.5, "phase_05": 1.5, "phase_02": 2.0,
    "phase_15": 1.5, "phase_11": 1.5, "phase_18": 2.0,
}  # alle anderen: 5.0 s Standard

# §9.7.4: Modell-Warmup im Hintergrund (2 s Verzögerung nach App-Start)
threading.Thread(target=_warmup_models_background, daemon=True, name="AurikWarmup").start()

# §9.7.5: Non-stationäre Defekte (DROPOUTS/TRANSPORT_BUMP) — Vollständiges Audio zwingend (kein Center-Crop)
#          Stationäre Defekte (Hiss, Hum, Flutter) dürfen Center-Crop nutzen (§9.1a)

# §9.7.6: Metrik-spezifische Audio-Cap in MusicalGoals-Metriken (15–20 s je Metrik, Brillanz/Wärme/Chroma)
#          Implementierung: musical_goals_metrics.py pro Metric-Klasse

# §9.7.7: PMGG Stable-Metric-Invariante — NatuerlichkeitMetric darf NIE in _PRECISE_METRICS stehen
#          CREPE Load-State ändert Gewichte (w_crepe 0.0 → 0.18) zwischen scores_before/scores_after
#          → Pseudo-Regression Δ≈0.15–0.28 auf unverändertem Audio → false P1-Kaskade
#          → phase_03 best-effort @ 5.6 % Wet → Noise Floor −55 dBFS statt −72 dBFS → Immersion zerstört
#          NatuerlichkeitMetric läuft ausschließlich im Export-Gate (MusicalGoalsChecker ≥ 0.90)

# §9.7.8: _apply_precise_metric_overrides kürzt Eingabe-Audio auf max. 2.5 s
#          Ausreichend für stationäre Spektral-/Chroma-/Transient-Metriken
#          Verhindert NMF/Onset-Runs auf Langaudio (> 2 s/Call auf 60 s-Material)
```

---

## §11.4c UI — Echtzeit-UX, Thread-Safety & State-Machines (konsolidiert aus Skill ui-feature)

### Thread-Safety (ABSOLUTES VERBOT)

**Kein Qt-Widget-Zugriff aus Hintergrundthreads.**

```python
# Pattern: Signal-Dispatch
_gui_dispatch = pyqtSignal(object)
# connect: self._gui_dispatch.connect(lambda fn: fn())
# Aufruf aus Thread:
self._dispatch_to_gui(lambda: widget.setText("..."))
```

### [RELEASE_MUST] Progress Bar

- **`setRange(0, 10000)`** immer — 1 Einheit = 0.01 %
- Signale senden 0–100, Slot skaliert `v * 100`
- **VERBOTEN**: `setRange(0, 100)` in ModernMainWindow

### Shortcuts

| Key | Aktion |
| --- | --- |
| Space | Play/Pause |
| A / B | Original / Restauriert |
| Ctrl+O / Ctrl+S | Öffnen / Export |
| Ctrl+R / Ctrl+Shift+R | Restoration / Studio 2026 |
| Escape | Abbruch |
| L | Lyrics-Overlay |

### BatchProcessingThread — Signal-Kontrakt

`item_started(str)`, `item_progress(str,int)`, `item_finished(str)`, `item_finished_with_result(str,object)`,
`item_error(str,str)`, `all_finished()`, `defect_update(dict)`, `phase_update(str)`, `waveform_data(ndarray,int)`,
`mode_update(str)`, `ml_status_update(bool,list)`, `phase_progress(int)`, `scan_progress(float)`, `quality_update(float)`.

`progress_callback`-Signatur: `(pct: int, msg: str, elapsed_s: float = 0.0) → None`

### Echtzeit-UX-Features

| Feature | Implementierung |
| --- | --- |
| Phase-Fortschritt | `phase_progress_bar` (5 px, lila Gradient) unter Hauptleiste |
| Defekte-Animation | Count-up (22 Frames × 85 ms); `_PHASE_REDUCES` senkt Scores ×0.3 |
| Varianten-Wettkampf | `★name_1 (4.12) › name_2 (3.87)` Rangliste |
| Quality-Meter | `quality_meter_widget.set_mos()`, startet 2.5 → 4.2 |
| Phasen-Erklärung | `_PHASE_EXPL`-Dict (22 Einträge) → Statuszeile |
| Waveform-Scan | oranger Cursor (12px Glow + 2px DashLine) |
| Vorab-Hörprobe | `QTimer.singleShot(1400, _auto_preview_restored)` — erste 5 s |

### §11.4b Schadensmarker-Lebenszyklus

| Phase | Waveform | Defekt-Chip |
| --- | --- | --- |
| `detected` | Farbiger Marker per Count-up-Animation | Roter/amber Severity-Chip mit Fortschrittsbalken |
| `correcting` | Marker bleibt sichtbar; verschwindet bei Score ≤ 0.01 (`_tick_defect_removal`, 75 ms) | Amber → orange bei aktivem Repair |
| **Abgeschlossen** (score ≤ 0.01) | **Marker verschwindet** — kein grünes Overlay | **Grüner Haken-Chip** `&#10003;` in `#4DC878`, Rand `rgba(77,200,120,0.45)` |

`_show_resolved_markers = False` — keine grünen Overlay-Rechtecke. Haken-Chip-Rendering: `fix_ratio >= 0.75` → kein `_bar`, nur Haken.

### Async-Analyse-Kette

5 Daemon-Threads: `_bg_load` → `_carrier_bg` → `_detect_era_genre_bg` → `_estimate_restorability_bg` → `_run_defect_scan_bg`

**Magic-Button-Sync-Gate**: Buttons deaktiviert bis `_run_defect_scan_bg` UND `_detect_era_genre_bg` fertig.
Freigabe über `_try_signal_preanalysis_done()` → `_finalize_preanalysis()`.
Timeout: `QTimer.singleShot(15_000, _preanalysis_timeout)`.

**`_carrier_bg` Pflicht**: `get_medium_detector().detect(audio, sr, file_ext=...)` — NICHT `classify_medium()`.

### Watchdog-Timer

```python
_per_file_ms = max(5_400_000, int(audio_dur_s * 64_000) + 3_600_000)  # 64×RT + 60min (vgl. §K)
_watchdog_ms = max(5_400_000, n_files * _per_file_ms)  # Min 90 Min, bis 300 Min pro File
```

### Bridge-Fallback (`_BRIDGE_AVAILABLE`)

Bei fehlendem Backend: `_BRIDGE_AVAILABLE = False` mit 17 Stub-Funktionen.
`_export_guard` vollständig (NaN+Clip), alle anderen: `return None`.

### KMV Stufe-2 UI

- `refinement_progress_bar`: 3 px, türkis `#00BCD4`
- Fertig: `"Export vollständig restauriert ✓ — ML-Qualität"` (5 s Notification)
- Escape → `requestInterruption()` trifft BatchThread UND MLRefinementThread

---

## §11.5a Architektur-Visualisierung mit Mermaid (konsolidiert aus Skill aurik-architecture-diagram)

### Farbschema (verbindlich für alle Diagramme)

| Typ | `classDef` | Fill | Beschreibung |
| --- | --- | --- | --- |
| ML-Modell / Plugin | `ml` | `#7B2FBE` (Lila) | Torch, ONNX, MDX23C |
| DSP-Algorithmus | `dsp` | `#1a6fcf` (Blau) | NumPy, SciPy, OMLSA |
| Qualitätsmetrik | `metric` | `#0f7a3e` (Grün) | MusicalGoals, PQS |
| Persistenz | `mem` | `#a05c10` (Braun) | ~/.aurik/ JSON |
| Gating / Schutz | `gate` | `#c0392b` (Rot) | PMGG, Rollback |
| Ein-/Ausgabe | `io` | `#2c3e50` (Dunkel) | Audio-Eingang, Result |

### [RELEASE_MUST] Software-Schichten (korrigiert)

```
Frontend["PyQt5 Frontend"] --> Bridge["backend/api/bridge.py (direkte Python-Aufrufe)"]
Bridge --> Denker["Denker-Orchestrierung"]
Denker --> Core["core/ · plugins/ · dsp/"]
```

**VERBOTEN**: FastAPI/REST-Referenzen — Aurik ist eine reine Desktop-App ohne Server.

### Diagramm-Komplexitäts-Management

- `flowchart TD` (top-down) bevorzugen
- Module in `subgraph`-Blöcken nach Pipeline-Stufen gruppieren
- Haupt-Datenfluss mit `-->`, Gedächtnis/Plugin-Verbindungen mit `-.->` (gestrichelt)
- Bei > 30 Knoten → aufteilen in Pipeline-Übersicht + Detail-Diagramme je Stufe
- HTML-Entitäten für `<` und `>`: `&lt;` und `&gt;`

---

## §11.4c UI — Echtzeit-UX, Thread-Safety & State-Machines (konsolidiert aus Skill ui-feature)

### Thread-Safety (ABSOLUTES VERBOT)

**Kein Qt-Widget-Zugriff aus Hintergrundthreads.**

```python
# Pattern: Signal-Dispatch
_gui_dispatch = pyqtSignal(object)
# connect: self._gui_dispatch.connect(lambda fn: fn())
# Aufruf aus Thread:
self._dispatch_to_gui(lambda: widget.setText("..."))
```

### [RELEASE_MUST] Progress Bar

- **`setRange(0, 10000)`** immer — 1 Einheit = 0.01 %
- Signale senden 0–100, Slot skaliert `v * 100`
- **VERBOTEN**: `setRange(0, 100)` in ModernMainWindow

### Shortcuts

| Key | Aktion |
| --- | --- |
| Space | Play/Pause |
| A / B | Original / Restauriert |
| Ctrl+O / Ctrl+S | Öffnen / Export |
| Ctrl+R / Ctrl+Shift+R | Restoration / Studio 2026 |
| Escape | Abbruch |
| L | Lyrics-Overlay |

### BatchProcessingThread — Signal-Kontrakt

`item_started(str)`, `item_progress(str,int)`, `item_finished(str)`, `item_finished_with_result(str,object)`,
`item_error(str,str)`, `all_finished()`, `defect_update(dict)`, `phase_update(str)`, `waveform_data(ndarray,int)`,
`mode_update(str)`, `ml_status_update(bool,list)`, `phase_progress(int)`, `scan_progress(float)`, `quality_update(float)`.

`progress_callback`-Signatur: `(pct: int, msg: str, elapsed_s: float = 0.0) → None`

### Echtzeit-UX-Features

| Feature | Implementierung |
| --- | --- |
| Phase-Fortschritt | `phase_progress_bar` (5 px, lila Gradient) unter Hauptleiste |
| Defekte-Animation | Count-up (22 Frames × 85 ms); `_PHASE_REDUCES` senkt Scores ×0.3 |
| Varianten-Wettkampf | `★name_1 (4.12) › name_2 (3.87)` Rangliste |
| Quality-Meter | `quality_meter_widget.set_mos()`, startet 2.5 → 4.2 |
| Phasen-Erklärung | `_PHASE_EXPL`-Dict (22 Einträge) → Statuszeile |
| Waveform-Scan | oranger Cursor (12px Glow + 2px DashLine) |
| Vorab-Hörprobe | `QTimer.singleShot(1400, _auto_preview_restored)` — erste 5 s |

### §11.4b Schadensmarker-Lebenszyklus

| Phase | Waveform | Defekt-Chip |
| --- | --- | --- |
| `detected` | Farbiger Marker per Count-up-Animation | Roter/amber Severity-Chip mit Fortschrittsbalken |
| `correcting` | Marker bleibt sichtbar; verschwindet bei Score ≤ 0.01 (`_tick_defect_removal`, 75 ms) | Amber → orange bei aktivem Repair |
| **Abgeschlossen** (score ≤ 0.01) | **Marker verschwindet** — kein grünes Overlay | **Grüner Haken-Chip** `&#10003;` in `#4DC878`, Rand `rgba(77,200,120,0.45)` |

`_show_resolved_markers = False` — keine grünen Overlay-Rechtecke. Haken-Chip-Rendering: `fix_ratio >= 0.75` → kein `_bar`, nur Haken.

### Async-Analyse-Kette

5 Daemon-Threads: `_bg_load` → `_carrier_bg` → `_detect_era_genre_bg` → `_estimate_restorability_bg` → `_run_defect_scan_bg`

**Magic-Button-Sync-Gate**: Buttons deaktiviert bis `_run_defect_scan_bg` UND `_detect_era_genre_bg` fertig.
Freigabe über `_try_signal_preanalysis_done()` → `_finalize_preanalysis()`.
Timeout: `QTimer.singleShot(15_000, _preanalysis_timeout)`.

**`_carrier_bg` Pflicht**: `get_medium_detector().detect(audio, sr, file_ext=...)` — NICHT `classify_medium()`.

### Watchdog-Timer

```python
_per_file_ms = max(5_400_000, int(audio_dur_s * 64_000) + 3_600_000)  # 64×RT + 60min (vgl. §K)
_watchdog_ms = max(5_400_000, n_files * _per_file_ms)  # Min 90 Min, bis 300 Min pro File
```

### Bridge-Fallback (`_BRIDGE_AVAILABLE`)

Bei fehlendem Backend: `_BRIDGE_AVAILABLE = False` mit 17 Stub-Funktionen.
`_export_guard` vollständig (NaN+Clip), alle anderen: `return None`.

### KMV Stufe-2 UI

- `refinement_progress_bar`: 3 px, türkis `#00BCD4`
- Fertig: `"Export vollständig restauriert ✓ — ML-Qualität"` (5 s Notification)
- Escape → `requestInterruption()` trifft BatchThread UND MLRefinementThread

---

## §11.5a Architektur-Visualisierung mit Mermaid (konsolidiert aus Skill aurik-architecture-diagram)

### Farbschema (verbindlich für alle Diagramme)

| Typ | `classDef` | Fill | Beschreibung |
| --- | --- | --- | --- |
| ML-Modell / Plugin | `ml` | `#7B2FBE` (Lila) | Torch, ONNX, MDX23C |
| DSP-Algorithmus | `dsp` | `#1a6fcf` (Blau) | NumPy, SciPy, OMLSA |
| Qualitätsmetrik | `metric` | `#0f7a3e` (Grün) | MusicalGoals, PQS |
| Persistenz | `mem` | `#a05c10` (Braun) | ~/.aurik/ JSON |
| Gating / Schutz | `gate` | `#c0392b` (Rot) | PMGG, Rollback |
| Ein-/Ausgabe | `io` | `#2c3e50` (Dunkel) | Audio-Eingang, Result |

### [RELEASE_MUST] Software-Schichten (korrigiert)

```
Frontend["PyQt5 Frontend"] --> Bridge["backend/api/bridge.py (direkte Python-Aufrufe)"]
Bridge --> Denker["Denker-Orchestrierung"]
Denker --> Core["core/ · plugins/ · dsp/"]
```

**VERBOTEN**: FastAPI/REST-Referenzen — Aurik ist eine reine Desktop-App ohne Server.

### Diagramm-Komplexitäts-Management

- `flowchart TD` (top-down) bevorzugen
- Module in `subgraph`-Blöcken nach Pipeline-Stufen gruppieren
- Haupt-Datenfluss mit `-->`, Gedächtnis/Plugin-Verbindungen mit `-.->` (gestrichelt)
- Bei > 30 Knoten → aufteilen in Pipeline-Übersicht + Detail-Diagramme je Stufe
- HTML-Entitäten für `<` und `>`: `&lt;` und `&gt;`
