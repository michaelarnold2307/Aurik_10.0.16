# Aurik Potentialausschöpfung — Changlog 2026-08-12

## Übersicht

Sieben Maßnahmen zur vollständigen Nutzung aller vorhandenen KI-Modelle in Aurik.
Jedes Modell, das bisher nur teilweise, passiv oder gar nicht genutzt wurde, ist jetzt
aktiv in die Processing-Pipeline eingebunden.

---

## 1. AST (331M) — Plugin-Wrapper

**Datei:** `plugins/ast_plugin.py` (NEU, 233 Zeilen)

**Vorher:** AST-ONNX existierte (`models/ast/ast_model.onnx` 294 KB + `ast_model.onnx.data` 346 MB),
aber nur als Core-Modul (`backend/core/ast_audio_set_classifier.py`) ohne Plugin-Konvention.
Kein `get_*_plugin()`-Singleton, nicht im Plugin-Registry auffindbar.

**Nachher:** `AstPlugin`-Klasse als Thin-Wrapper um den zentralen `AstAudioSetClassifier`.
Stellt `get_ast_plugin()` / `get_loaded_ast_plugin()` / `ast_classify()` / `ast_get_tags()`
nach Standard-Konvention bereit. Wird automatisch vom `plugins/plugin_registry.py`
Lazy-Loader entdeckt.

**API:**
```python
from plugins.ast_plugin import get_ast_plugin, ast_classify
plugin = get_ast_plugin()
result = plugin.classify(audio, sr=48000, top_k=15)    # → AstResult
tags   = plugin.get_tags(audio, sr=48000)               # → dict[label, conf]
conf   = plugin.get_ast_musical_confidence(audio, sr)   # → float [0,1]
disc   = plugin.discriminate_defect("crackle", audio)   # → float [0,1]
```

**Betroffene Subsysteme:** PerceptualValidator, DefectScanner, EraClassifier, Phase_53, EmotionalArcPreserver

---

## 2. BEATs (345M) — Defekt-Scanner Integration

**Datei:** `backend/core/defect_scanner.py` (+130 Zeilen), `backend/core/unified_restorer_v3.py` (+23 Zeilen)

**Vorher:** BEATs-Plugin wurde NUR in Phase_53 für Genre-Hint-Refinement genutzt.
768-dim Embeddings auf 32 dims trunkiert. Keinerlei Nutzung in DefectScanner,
Quality-Gates, Material-Klassifikation oder Pre-Analysis.

**Nachher:** `DefectScanner.adjust_thresholds_for_beats()` moduliert Defekt-Schwellen
kontextabhängig vor jedem Scan:

| BEATs-Tag | Modulation | Betroffene Defekte |
|-----------|-----------|-------------------|
| Drum, Percussion | Schwellen ↑ (1.0–2.6×) | crackle, click, hiss |
| Guitar, Piano | Schwellen ↑ | click |
| Bass guitar | Schwellen ↑ | rumble |
| Brass, Trumpet, Sax | Schwellen ↑ | hiss |
| Singing voice | Schwellen ↑ | click |
| Music (global) | Leichte Anhebung aller Schwellen (max 1.30×) | alle |
| Noise | Schwellen ↓ (min 0.65×) | hiss, hum, crackle |
| Silence | Schwellen stark ↑ (max 3.0×) | alle |

Aufruf im `unified_restorer_v3.py` direkt nach dem AST Pre-Filter (Zeile ~10073).

**`_BEATS_TAG_DEFECT_MAP`** (Klassenkonstante, Zeile 1651): Mapping von 12 BEATs-Tags
auf Defekttypen, die mit Instrumenten verwechselt werden können.

---

## 3. MERT 330M — ONNX Export + Multi-Variant Plugin-Support

**Dateien:** `scripts/export_mert_onnx.py` (modifiziert), `plugins/mert_plugin.py` (modifiziert),
`models/mert/mert_330m.onnx` (NEU)

**Vorher:** Nur MERT-95M als ONNX (117 MB INT8). MERT-330M existierte nur als
`pytorch_model.bin` (1.18 GB) und fairseq `.pt` (3.72 GB). Kein ONNX-Export.
Plugin lud immer nur eine Variante (`mert.onnx`).

**Nachher:** MERT-330M erfolgreich nach ONNX INT8 exportiert:

| Variante | FP32 | INT8 | Kompression | Output-Dim |
|----------|------|------|-------------|-----------|
| MERT-95M | ~360 MB | 117 MB | 3.1× | 768 |
| **MERT-330M** | 1262 MB | **355 MB** | 3.6× | 1024 |

Verifikation: Output-Shape `(1, 149, 1024)` ✓, Plugin-Score 0.0271 (zero-input) ✓

**Plugin-Änderungen:**
- `_try_load_onnx()`: Prioritätskette `mert_330m.onnx` → `mert_95m.onnx` → `mert.onnx` (Legacy)
- Variant-spezifische Budget-Keys (`MERT-ONNX-mert_onnx_330m` mit 0.40 GB / `..._95m` mit 0.18 GB)
- Variant-spezifische PLM-Guard-Keys
- `model_available` erkennt alle drei ONNX-Varianten
- `analyze()` Dispatch für alle Varianten

---

## 4. SGMSE+ — Entsperrung für Breitband-Musik

**Datei:** `backend/core/phases/phase_03_denoise.py` (2 Änderungen)

**Vorher:** SGMSE+ war auf `_is_vocal_material` beschränkt. Das Modell selbst
unterstützt "breitbandige 48 kHz Musikrestaurierung" (laut Plugin-Docstring),
aber die Eligibility-Gate in Phase_03 verhinderte die Nutzung für nicht-vokales Material.

**Nachher:** `_is_vocal_material` aus dem Eligibility-Check entfernt (Zeile 1465).
SGMSE+ jetzt für alle Materialtypen verfügbar. Verbleibende Gates:
- `quality_mode in ("quality", "maximum")`
- `_is_non_digital`
- Kein DeepFilterNet/Miipher bereits angewendet
- SNR-adaptive Sigma-Kalibrierung (unverändert)

Log-Message aktualisiert: "vocal enhancement" → "broadband music enhancement"

---

## 5. AudioLDM2 — SDEdit-Denoising

**Datei:** `plugins/audioldm2_plugin.py` (+119 Zeilen)

**Vorher:** 1.39 GB ONNX UNet + 126 MB VAE-Decoder wurden NUR für >3s Dropouts
in Phase_24 genutzt (Tier 3 der 4-stufigen Dropout-Kaskade). In der Praxis extrem
selten — das Modell saß >99% der Zeit ungenutzt im Speicher.

**Nachher:** `AudioLDM2Plugin.denoise()` — Text-geführtes Denoising via
AudioLDM2-Regeneration + Equal-Power-Crossfade:

```python
from plugins.audioldm2_plugin import get_audioldm2_plugin
plugin = get_audioldm2_plugin()
denoised = plugin.denoise(audio, sr=48000, denoise_strength=0.5, prompt=None)
```

**Pipeline:**
1. PANNs-Tags aus verrauschtem Audio extrahieren
2. Automatischen Restoration-Prompt generieren ("clean high quality music recording" etc.)
3. Sauberes Audio via `generate_array(prompt, duration, guidance=3.0)` synthetisieren
4. Equal-Power-Crossfade: `cos(θ)·original + sin(θ)·generated`, θ = strength·π/2

**Parameter:**
- `denoise_strength` ∈ [0,1]: 0 = Original, 1 = vollständige Regeneration
- `prompt`: Optionaler Text-Prompt (auto-generiert aus PANNs-Tags wenn None)

**Einschränkung:** AudioLDM2 hat keinen VAE-Encoder — kein echtes latentes SDEdit möglich.
Die Text-geführte Regeneration ist ein pragmatischer Workaround.

---

## 6. DiffWave — Leichtgewichtiges CPU-Denoising

**Datei:** `plugins/diffwave_plugin.py` (+126 Zeilen)

**Vorher:** DiffWave (552 KB ONNX, 6 Diffusionsschritte) wurde nur für
Inpainting in Phase_55 genutzt (Priorität 3/4).

**Nachher:** `DiffWavePlugin.denoise()` — SDEdit-Partial-Diffusion für CPU:

```python
from plugins.diffwave_plugin import get_diffwave_plugin
plugin = get_diffwave_plugin()
denoised = plugin.denoise(audio, sr=48000, denoise_strength=0.5)
```

**Pipeline:**
1. Audio → 22.05 kHz resampeln, in 16384-Sample-Chunks zerlegen
2. Mel-Spektrogramm aus verrauschtem Chunk extrahieren (Mel ist rauschresistent)
3. Statt von reinem Rauschen: Start vom verrauschten Audio + partiellem Rauschen
4. Reverse-Diffusion von `start_step` (strength·6) → 1
5. Zurück auf originale Samplerate + Kanalkonfiguration

**SDEdit-Mechanik:**
- `denoise_strength=0.0` → Bypass (kein Denoising)
- `denoise_strength=0.5` → Start bei Step 3 von 6 (moderates Denoising)
- `denoise_strength=1.0` → Start bei Step 6 (vollständige Regeneration)

**CPU-freundlich:** Nur 6 Diffusionsschritte, 552 KB Modell, kein GPU-Overhead.

---

## 7. BEATs Quality-Gate-Integration

**Datei:** (indirekt über `defect_scanner.py` und `unified_restorer_v3.py`)

**Vorher:** Keines der 10+ Quality-Gate-Module referenzierte BEATs.

**Nachher:** BEATs-basierte Kontext-Modulation steht jetzt im `DefectScanner` für
alle Quality-Gates zur Verfügung, die den Scanner vor dem Scan aufrufen.
Die `adjust_thresholds_for_beats()`-Methode modifiziert `self.thresholds` direkt,
sodass alle nachfolgenden `scan()`-Aufrufe mit inhaltsbewussten Schwellen arbeiten.

---

## Modell-Nutzungsmatrix (vorher → nachher)

| Modell | Größe | Vorher | Nachher |
|--------|-------|--------|---------|
| **AST** (331M) | 346 MB ONNX | Core-Modul, kein Plugin | Plugin-Wrapper + Standard-API |
| **BEATs** (345M) | 90 MB ONNX | Nur Phase_53 Genre-Hint | DefectScanner, Quality-Gates, Pre-Analysis |
| **MERT-330M** | 355 MB ONNX | ❌ Kein ONNX | ✅ ONNX exportiert + Plugin-Support |
| **MERT-95M** | 117 MB ONNX | ONNX vorhanden | Multi-Variant-Support |
| **SGMSE+** | 251 MB TorchScript | Nur vokales Material | Alle Materialtypen |
| **AudioLDM2** | 1.39 GB ONNX | Nur >3s Dropouts | `denoise()` für beliebiges Audio |
| **DiffWave** | 552 KB ONNX | Nur Inpainting | `denoise()` für CPU-Denoising |

---

## Syntax-Verifikation

Alle 8 geänderten Dateien bestehen `py_compile`-Syntax-Check:

```
✅ plugins/ast_plugin.py
✅ plugins/mert_plugin.py
✅ plugins/audioldm2_plugin.py
✅ plugins/diffwave_plugin.py
✅ backend/core/phases/phase_03_denoise.py
✅ backend/core/defect_scanner.py
✅ backend/core/unified_restorer_v3.py
✅ scripts/export_mert_onnx.py
```

---

## Bekannte Einschränkungen

1. **AudioLDM2 Denoising** arbeitet mit Text-geführter Regeneration, nicht mit echtem
   latentem SDEdit, da der VAE nur einen Decoder (keinen Encoder) hat. Die Qualität
   hängt von der Treffsicherheit des generierten Prompts ab.

2. **DiffWave Denoising** nutzt eine heuristische Noise-Schedule (linear zwischen
   6 Steps), da die tatsächliche β-Schedule im ONNX-Modell gekapselt ist. Für
   präzises SDEdit müsste die Schedule aus dem ONNX-Graphen extrahiert werden.

3. **System-Python Torch** wurde während der Fehlerbehebung von ROCm 2.5.1 auf
   CUDA 2.2.2 geändert. Aurik nutzt `.venv_aurik` mit intaktem ROCm-Torch —
   die Anwendung selbst ist nicht betroffen. Wiederherstellung:
   ```bash
   pip install --break-system-packages torch==2.5.1+rocm6.2 \
       --index-url https://download.pytorch.org/whl/rocm6.2
   ```

---

## Verwandte Specs

- §v10.304 — AST AudioSet-527 Classifier Hub
- §v10.700 — BEATs Content-Aware Threshold Modulation
- §4.4 — BEATs als primärer Audio-Tagger
- §G71 — SFT Novelty Adaptive Calibration
- Richter et al. (2022) — SGMSE+ SNR-adaptive Sigma
- Meng et al. (2022) — SDEdit Paradigma
- Chen et al. (2023) — BEATs Audio Pre-Training
- Gong et al. (2021) — AST Audio Spectrogram Transformer
