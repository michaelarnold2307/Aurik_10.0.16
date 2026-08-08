# GUI-Integration — Verdrahtung in modern_window.py

> Stand: 10.0.20 | 7 Module bereit, hier die exakten Integration-Patches

## 1. KeyboardShortcuts (Leertaste, Pfeiltasten)

```python
# In ModernMainWindow.__init__():
from Aurik10.ui.keyboard_shortcuts import KeyboardShortcuts
self._shortcuts = KeyboardShortcuts(player=self._audio_player, window=self)

# In ModernMainWindow:
def keyPressEvent(self, event):
    if self._shortcuts.handle_key_press(event.key(), event.modifiers()):
        return
    super().keyPressEvent(event)
```

## 2. ExpertMode (Toggle)

```python
# In ModernMainWindow.__init__():
from Aurik10.core.expert_mode import get_expert_mode
self._expert_mode = get_expert_mode()

# Menu-Button (View → Expert Mode):
expert_action = menu.addAction("Experten-Modus")
expert_action.setCheckable(True)
expert_action.triggered.connect(lambda checked: self._toggle_expert(checked))

def _toggle_expert(self, enabled: bool):
    self._expert_mode.enabled = enabled
    self._update_expert_mode_visibility()

def _update_expert_mode_visibility(self):
    visible = self._expert_mode.enabled
    # Performance-Widgets
    self._rt_label.setVisible(visible)
    self._phase_timeline.setVisible(visible)
    # Technische Metriken
    self._lufs_label.setVisible(visible and self._expert_mode.is_visible("technical_metrics"))
    self._chroma_label.setVisible(visible and self._expert_mode.is_visible("technical_metrics"))
    self._vqi_label.setVisible(visible and self._expert_mode.is_visible("technical_metrics"))
    # Phase-Report
    self._phase_report_widget.setVisible(visible and self._expert_mode.is_visible("phase_report"))
    # Export-Chain
    self._export_chain_label.setVisible(visible and self._expert_mode.is_visible("export_chain"))
```

## 3. SessionMemory (History + Fenster-Position)

```python
# In ModernMainWindow.__init__():
from Aurik10.core.session_memory import get_session_memory
self._session = get_session_memory()

# Fenster-Position wiederherstellen:
geo = self._session.get_window_geometry()
if geo:
    self.restoreGeometry(geo)

def closeEvent(self, event):
    self._session.save_window_geometry(self.saveGeometry())
    super().closeEvent(event)

# Nach Restaurierung:
def _on_restoration_finished(self, result):
    self._session.add_result(
        file_path=self._current_file,
        quality=result.quality_estimate * 100,
        mode=self._current_mode,
        duration_s=result.audio_duration_s,
    )
    # Empfehlung anzeigen
    rec = self._session.get_recommendation(self._current_file)
    if rec:
        self._status_bar.showMessage(rec, 5000)
```

## 4. ResultEnricher (RT-Faktor, Phase-Report, Export-Chain, Metriken)

```python
# In _on_restoration_finished():
from Aurik10.core.result_enrichment import ResultEnricher

enriched = ResultEnricher.enrich(result)

# Performance
perf = enriched["performance"]
self._rt_label.setText(f"RT: {perf['rt_factor']} | {perf['processing']}")

# Phase-Report
report = enriched["phase_report"]
self._phase_summary.setText(
    f"{report['executed']}/{report['total']} Phasen, {report['skipped']} übersprungen"
)

# Export-Chain (Experten-Modus)
if self._expert_mode.is_visible("export_chain"):
    chain = enriched["export_chain"]
    self._export_chain_label.setText(chain["chain"])

# Technische Metriken (Experten-Modus)
if self._expert_mode.is_visible("technical_metrics"):
    tech = enriched["technical"]
    self._lufs_label.setText(f"LUFS: {tech['lufs_delta']}")
    self._chroma_label.setText(f"Chroma: {tech['chroma']}")
    self._vqi_label.setText(f"VQI: {tech['vqi']}")
    self._goosebumps_label.setText(f"Goosebumps: {tech['goosebumps']}")
    self._presence_label.setText(f"Presence: {tech['presence']}")
```

## 5. SpectrumComparison (Vorher/Nachher)

```python
# In Ergebnis-Dialog:
from Aurik10.core.spectrum_comparison import compute_spectrum_comparison

spec_data = compute_spectrum_comparison(
    self._original_audio,
    result.audio,
    self._original_sr,
)
# spec_data["spectrogram_before_db"] → Matplotlib-Image
# spec_data["spectrogram_delta_db"] → Differenz-Heatmap
```

## 6. BatchOverview (Tabelle + Statistik)

```python
# Nach Batch-Verarbeitung:
from Aurik10.core.batch_overview import BatchOverview, BatchTrackInfo

overview = BatchOverview()
for track_result in batch_results:
    overview.tracks.append(BatchTrackInfo(
        file_path=track_result.path,
        quality_after=track_result.quality,
        duration_s=track_result.duration,
        success=track_result.success,
    ))
display = overview.to_display_dict()
# display["summary"] → Statistik
# display["tracks"] → Tabellen-Zeilen
```

## 7. DefectMap (Vorher/Nachher)

```python
# Nach Restaurierung:
from Aurik10.core.defect_map import DefectMap

defect_map = DefectMap.from_defect_lists(
    self._defects_before,
    result.metadata.get("defects_after", []),
    audio_duration_s=result.audio_duration_s,
)
display = defect_map.to_display_dict()
# display["summary"] → "124 → 3 Defekte (97% reduziert)"
# display["per_type"] → Pro-Defekt-Tabelle
```
