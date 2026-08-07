#!/usr/bin/env python3
"""
Löst alle verbleibenden TODO(de):-Marker in Docstrings auf.

Zweiter Pass nach translate_docstrings_to_de.py.
Strategie:
1. Bereits deutsche Phrasen (kein echter Übersetzungsbedarf) → TODO-Prefix entfernen
2. Bekannte englische Strukturmuster → regelbasierte Übersetzung
3. Direkte Lookup-Tabelle für bekannte Einzelphrasen

Aufruf:
    python scripts/resolve_todo_de.py [--dry-run] [--dirs backend forensics ...]
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

_DE_CHARS = set("äöüÄÖÜß")

_GERMAN_STARTS = {
    # Verben (Infinitiv/konjugiert)
    "Initialisiert",
    "Analysiert",
    "Verarbeitet",
    "Erstellt",
    "Berechnet",
    "Gibt",
    "Ruft",
    "Öffnet",
    "Schließt",
    "Prüft",
    "Registriert",
    "Entfernt",
    "Lädt",
    "Speichert",
    "Listet",
    "Wendet",
    "Setzt",
    "Sortiert",
    "Sendet",
    "Empfängt",
    "Erkennt",
    "Verbindet",
    "Trennt",
    "Aktiviert",
    "Deaktiviert",
    "Mischt",
    "Normalisiert",
    "Optimiert",
    "Validiert",
    "Interpoliert",
    "Kapselt",
    "Schreibt",
    "Liest",
    "Filtert",
    "Scannt",
    "Kalibriert",
    "Dämpft",
    "Verstärkt",
    "Kombiniert",
    "Extrahiert",
    "Dekodiert",
    "Kodiert",
    "Generiert",
    "Schätzt",
    "Bewertet",
    "Führt",
    "Expandiert",
    "Akkumuliert",
    "Transformiert",
    "Komprimiert",
    "Restauriert",
    "Rekonstruiert",
    "Propagiert",
    "Verteilt",
    "Protokolliert",
    "Überwacht",
    "Fügt",
    "Allokiert",
    "Glättet",
    "Verschiebt",
    "Skaliert",
    "Resamplet",
    "Begrenzt",
    "Zählt",
    "Formatiert",
    "Konvertiert",
    "Serialisiert",
    "Findet",
    "Parst",
    "Loggt",
    # Nomen / beschreibende Starts
    "Singleton",
    "Audio",
    "Wrapper",
    "Analyse",
    "Typen",
    "Getter",
    "Zugriff",
    "Thread",
    "Klasse",
    "Basis",
    "Modul",
    "Plugin",
    "Phase",
    "Manager",
    "System",
    "Engine",
    "Daten",
    "Signal",
    "Frequenz",
    "Spektrum",
    "Stereo",
    "Kanal",
    "Pegel",
    "Rauschen",
    "Defekt",
    "Fehler",
    "Ergebnis",
    "Wert",
    "Liste",
    "Index",
    # Sonstige deutsche Einleitungen
    "Der",
    "Die",
    "Das",
    "Ein",
    "Eine",
    "Kein",
    "Keine",
    "Beim",
    "Bei",
    "Mit",
    "Ohne",
    "Nach",
    "Vor",
    "Für",
    "Zu",
    "Aus",
    "Über",
    "Unter",
    "Zwischen",
}


def _is_already_german(text: str) -> bool:
    """Prüft ob Text bereits überwiegend deutsch ist."""
    if any(c in text for c in _DE_CHARS):
        return True
    first_word = text.split()[0].rstrip(":.,!?-") if text.split() else ""
    # Direkt bekannte deutsche Wörter
    if first_word in _GERMAN_STARTS:
        return True
    # Typische deutsche Komposita / Patterns
    if first_word.endswith("iert") or first_word.endswith("iert:"):
        return True
    if "-" in first_word and first_word[0].isupper():
        return True
    return False


# ---------------------------------------------------------------------------
# Lookup-Tabelle: bekannte Phrasen → direkte deutsche Übersetzung
# ---------------------------------------------------------------------------

_LOOKUP: dict[str, str] = {
    # ---------- Analysiert ... (gemischte Phrasen) ----------
    "Analysiert a song and return the top-N cleanest segments.": "Analysiert einen Song und gibt die N saubersten Segmente zurück.",
    "Analysiert audio and compute Bark spectrum.": "Analysiert Audio und berechnet das Bark-Spektrum.",
    "Analysiert audio and compute masking profile.": "Analysiert Audio und berechnet das Maskierungsprofil.",
    "Analysiert audio and return musical segment structure.": "Analysiert Audio und gibt die musikalische Segmentstruktur zurück.",
    'Analysiert audio and return musical segment structure."""': "Analysiert Audio und gibt die musikalische Segmentstruktur zurück.",
    "Analysiert audio context from intrinsic features.": "Analysiert den Audio-Kontext aus intrinsischen Merkmalen.",
    "Analysiert audio for all defects.": "Analysiert Audio auf alle Defekte.",
    "Analysiert audio for all genre-specific authenticity metrics.": "Analysiert Audio auf alle genrespezifischen Authentizitätsmetriken.",
    "Analysiert audio for defects.": "Analysiert Audio auf Defekte.",
    "Analysiert audio quality.": "Analysiert die Audioqualität.",
    "Analysiert audio reverb characteristics.": "Analysiert die Hall-Eigenschaften des Audios.",
    "Analysiert audio semantically without genre classification.": "Analysiert Audio semantisch ohne Genreklassifizierung.",
    "Analysiert audio to determine content types.": "Analysiert Audio zur Bestimmung von Inhaltstypen.",
    'Analysiert audio to determine content types."""': "Analysiert Audio zur Bestimmung von Inhaltstypen.",
    "Analysiert audio to determine if Dolby / DBX NR was applied without decoding.": "Analysiert Audio, um zu prüfen ob Dolby/DBX NR ohne Dekodierung angewendet wurde.",
    "Analysiert azimuth error for a single frequency band.": "Analysiert den Azimutfehler für ein einzelnes Frequenzband.",
    "Analysiert context with zone classification.": "Analysiert den Kontext mit Zonenklassifizierung.",
    "Analysiert cover image using DSP embedding + CLAP similarity.": "Analysiert das Cover-Bild mittels DSP-Einbettung und CLAP-Ähnlichkeit.",
    "Analysiert damage per frequency band.": "Analysiert Schäden pro Frequenzband.",
    "Analysiert direct vs reverb content.": "Analysiert Direkt- vs. Hall-Inhalt.",
    "Analysiert dynamic range characteristics.": "Analysiert Dynamikbereichseigenschaften.",
    "Analysiert even vs. odd harmonics using FFT.": "Analysiert gerade vs. ungerade Obertöne mittels FFT.",
    "Analysiert extent and type of audio damage.": "Analysiert Ausmaß und Art des Audioschadens.",
    "Analysiert frequency content of audio.": "Analysiert den Frequenzinhalt des Audios.",
    "Analysiert frequency masking.": "Analysiert die Frequenzmaskierung.",
    "Analysiert harmonic character.": "Analysiert den harmonischen Charakter.",
    "Analysiert listening fatigue factors.": "Analysiert Hörmüdigkeitsfaktoren.",
    "Analysiert live recording for all potential issues.": "Analysiert Live-Aufnahmen auf alle potenziellen Probleme.",
    "Analysiert loudness metrics per ITU-R BS.1770-4.": "Analysiert Lautstärke-Metriken gemäß ITU-R BS.1770-4.",
    "Analysiert low-end muddiness.": "Analysiert Tiefton-Schwammigkeit.",
    "Analysiert microdynamics.": "Analysiert die Mikrodynamik.",
    "Analysiert micro-dynamics variability.": "Analysiert die Mikrodynamik-Variabilität.",
    "Analysiert music structure and detect segments.": "Analysiert die Musikstruktur und erkennt Segmente.",
    "Analysiert phase alignment via cross-correlation with sub-sample precision.": "Analysiert die Phasenausrichtung via Kreuzkorrelation mit Sub-Sample-Präzision.",
    "Analysiert phase correlation between left and right channels.": "Analysiert die Phasenkorrelation zwischen linkem und rechtem Kanal.",
    "Analysiert reference track to extract Musical Goals.": "Analysiert die Referenzspur zur Extraktion der Musical Goals.",
    "Analysiert room tone characteristics.": "Analysiert Raumton-Eigenschaften.",
    "Analysiert sibilance characteristics in audio.": "Analysiert Sibilanz-Eigenschaften im Audio.",
    "Analysiert single frame for formants using LPC.": "Analysiert einen einzelnen Frame auf Formanten mittels LPC.",
    "Analysiert spectral brightness of audio.": "Analysiert die spektrale Helligkeit des Audios.",
    "Analysiert spectral distribution.": "Analysiert die spektrale Verteilung.",
    "Analysiert spectral-temporal characteristics to separate.": "Analysiert spektral-zeitliche Eigenschaften zur Signaltrennung.",
    "Analysiert spectrum and compute deviation from target.": "Analysiert das Spektrum und berechnet die Zielabweichung.",
    "Analysiert stereo field properties without modification.": "Analysiert Stereofeld-Eigenschaften ohne Modifikation.",
    "Analysiert stereo imaging characteristics of audio signals.": "Analysiert Stereoabbildungs-Eigenschaften von Audiosignalen.",
    "Analysiert temporal/rhythmic characteristics.": "Analysiert temporale/rhythmische Eigenschaften.",
    "Analysiert the emotional content of song lyrics and returns time-stamped.": "Analysiert den emotionalen Inhalt von Liedtexten und gibt zeitgestempelte Ergebnisse zurück.",
    "Analysiert the sentiment timeline from a transcription result.": "Analysiert die Stimmungs-Timeline aus einem Transkriptionsergebnis.",
    "Analysiert vocal vs instrumental content.": "Analysiert Vokal- vs. Instrumental-Inhalt.",
    "Analysiert which harmonics are missing via spectral analysis.": "Analysiert fehlende Obertöne mittels Spektralanalyse.",
    "Analysiert which phases can be skipped.": "Analysiert, welche Phasen übersprungen werden können.",
    # ---------- Process ... ----------
    "Process single channel.": "Verarbeitet einen einzelnen Kanal.",
    'Process single channel."""': "Verarbeitet einen einzelnen Kanal.",
    # ---------- Lazy load ... ----------
    "Lazy load DeepFilterNet v3 II Plugin.": "Lädt DeepFilterNet v3 II Plugin beim ersten Zugriff.",
    # ---------- Merged ... ----------
    "Merged consecutive frames of same type into regions.": "Fasst aufeinanderfolgende Frames gleichen Typs zu Regionen zusammen.",
    'Merged consecutive frames of same type into regions."""': "Fasst aufeinanderfolgende Frames gleichen Typs zu Regionen zusammen.",
    # ---------- Listet ... ----------
    "Listet auf: all registered detector names.": "Listet alle registrierten Detektor-Namen auf.",
    'Listet auf: all registered detector names."""': "Listet alle registrierten Detektor-Namen auf.",
    # ---------- Analysiert audio for defects + triple quotes ----------
    'Analysiert audio for defects."""': "Analysiert Audio auf Defekte.",
    # ---------- Unified API / interface / system ----------
    "Unified Restorer V3 - Defect-First Audio Restoration Engine.": "Einheitlicher Restorer V3 – Defektbasierte Audio-Restaurierungs-Engine.",
    "Unified psychoacoustic processing engine.": "Einheitliche psychoakustische Verarbeitungs-Engine.",
    "Unified workflow manager combining all workflow features.": "Einheitlicher Workflow-Manager mit allen Workflow-Funktionen.",
    "Unified Vocal Enhancement with full AI integration.": "Einheitliche Vokal-Verbesserung mit vollständiger KI-Integration.",
    "Unified Signal Forensics Analyzer.": "Einheitlicher Signal-Forensik-Analysator.",
    "Unified interface for uncertainty quantification.": "Einheitliches Interface für Unsicherheitsquantifizierung.",
    "Unified interface for all live recording tools.": "Einheitliches Interface für alle Live-Aufnahme-Werkzeuge.",
    "Unified Defect Detection System.": "Einheitliches Defekterkennungs-System.",
    "Unified Audio Restoration System.": "Einheitliches Audio-Restaurierungs-System.",
    "Unified Audio Enhancement System.": "Einheitliches Audio-Verbesserungs-System.",
    "Unified audio defect detection system.": "Einheitliches Audio-Defekterkennungs-System.",
    "Unified API for vocal spectral inpainting.": "Einheitliche API für vokales spektrales Inpainting.",
    "Unified API for vocal presence enhancement.": "Einheitliche API für Vokal-Präsenz-Verbesserung.",
    "Unified API for vocal dynamics intelligence.": "Einheitliche API für Vokal-Dynamik-Intelligenz.",
    "Unified API for Transparent Dynamics + Micro-Dynamics Enhancement": "Einheitliche API für transparente Dynamik- und Mikrodynamik-Verbesserung.",
    "Unified API for tape-specific defect removal.": "Einheitliche API für bandspezifische Defektentfernung.",
    "Unified API for stem separation.": "Einheitliche API für Stem-Separation.",
    "Unified API for Spatial Enhancement.": "Einheitliche API für räumliche Klangverbesserung.",
    "Unified API for Piano/Keys Restoration.": "Einheitliche API für Klavier/Tasten-Restaurierung.",
    "Unified API for multi-track and stereo enhancement.": "Einheitliche API für Mehrspuraufnahme- und Stereo-Verbesserung.",
    "Unified API for Guitar/String Enhancement.": "Einheitliche API für Gitarren-/Saiten-Verbesserung.",
    "Unified API for genre-specific authenticity detection.": "Einheitliche API für genrespezifische Authentizitätserkennung.",
    "Unified API for formant tracking, correction, and enhancement.": "Einheitliche API für Formant-Tracking, -Korrektur und -Verbesserung.",
    "Unified API for Drums/Percussion Enhancement.": "Einheitliche API für Schlagzeug-/Percussion-Verbesserung.",
    "Unified API for digital-specific defect removal.": "Einheitliche API für digitalspezifische Defektentfernung.",
    "Unified API for Breath Noise Intelligence.": "Einheitliche API für Atemlärm-Intelligenz.",
    "Unified API for Brass/Wind Enhancement.": "Einheitliche API für Blechbläser-/Blasinstrument-Verbesserung.",
    "Unified API for Bass Enhancement.": "Einheitliche API für Bass-Verbesserung.",
    "Unified API for all three tonal balance restoration modules": "Einheitliche API für alle drei Tonbalance-Restaurierungsmodule.",
    # ---------- Singleton wrappers ----------
    "Singleton wrapper for MTEF measure/morph operations.": "Singleton-Wrapper für MTEF-Mess-/Morph-Operationen.",
    "Singleton wrapper for goosebumps quality assessment (§8.3 Spec).": "Singleton-Wrapper für Gänsehaut-Qualitätsbewertung (§8.3 Spec).",
    "Singleton wrapper for adaptive chunk processing.": "Singleton-Wrapper für adaptive Chunk-Verarbeitung.",
    "Singleton vocal style profiler — thread-safe, all methods non-blocking.": "Singleton-Vokalstil-Profiler – thread-safe, alle Methoden nicht-blockierend.",
    "Singleton planner — thread-safe, non-blocking.": "Singleton-Planer – thread-safe, nicht-blockierend.",
    'Singleton wrapper for MTEF measure/morph operations."""': "Singleton-Wrapper für MTEF-Mess-/Morph-Operationen.",
    'Singleton wrapper for goosebumps quality assessment (§8.3 Spec)."""': "Singleton-Wrapper für Gänsehaut-Qualitätsbewertung (§8.3 Spec).",
    'Singleton wrapper for adaptive chunk processing."""': "Singleton-Wrapper für adaptive Chunk-Verarbeitung.",
    'Singleton vocal style profiler — thread-safe, all methods non-blocking."""': "Singleton-Vokalstil-Profiler – thread-safe, alle Methoden nicht-blockierend.",
    'Singleton planner — thread-safe, non-blocking."""': "Singleton-Planer – thread-safe, nicht-blockierend.",
    # ---------- Simple ... ----------
    "Simple vocal detection based on formant energy.": "Einfache Vokalerkennung auf Basis von Formant-Energie.",
    "Simple TextGrid parser (fallback if textgrid library not available).": "Einfacher TextGrid-Parser (Fallback wenn textgrid-Bibliothek nicht verfügbar).",
    'Simple vocal detection based on formant energy."""': "Einfache Vokalerkennung auf Basis von Formant-Energie.",
    'Simple TextGrid parser (fallback if textgrid library not available)."""': "Einfacher TextGrid-Parser (Fallback wenn textgrid-Bibliothek nicht verfügbar).",
    # ---------- Wrapper ... ----------
    "Wrapper for apply_dsp_chain": "Wrapper für apply_dsp_chain.",
    # ---------- Validator / Validation ----------
    "Validator for AutoReprocessingEngine.": "Validator für die AutoReprocessingEngine.",
    "Validation of predicted vs actual quality.": "Validierung von vorhergesagter vs. tatsächlicher Qualität.",
    # ---------- Updated pipeline ... ----------
    "Updated pipeline using formal ResturationJob data structures.": "Aktualisierte Pipeline mit formalen RestaurationJob-Datenstrukturen.",
    # ---------- Additive synthesis ----------
    "Additive synthesis of missing harmonic overtones (I - Salience Multi-Pitch).": "Additive Synthese fehlender harmonischer Obertöne (I – Salience Multi-Pitch).",
    # ---------- Transformer saturation ----------
    "Transformer saturation (symmetric, balanced harmonics) with ADAA.": "Transformatorsättigung (symmetrisch, ausgewogene Harmonik) mit ADAA.",
    # ---------- Apply ... ----------
    "Apply §3.9.3 invariants to *result*; return sanitised value.": "Wendet §3.9.3-Invarianten auf *result* an und gibt bereinigten Wert zurück.",
    "Apply 3-band parametric EQ with Linkwitz-Riley LR4 crossovers.": "Wendet 3-Band-Parametrik-EQ mit Linkwitz-Riley-LR4-Crossovern an.",
    "Apply 5 ms Hanning crossfade gate at breath segment boundaries (§2.8).": "Wendet 5-ms-Hanning-Überblend-Gate an Atemsegmentgrenzen an (§2.8).",
    "Apply a biquad peak EQ to a short audio frame (DSP helper).": "Wendet einen Biquad-Peak-EQ auf einen kurzen Audio-Frame an (DSP-Hilfsfunktion).",
    "Apply absolute and relative gating per ITU-R BS.1770-4": "Wendet absolutes und relatives Gating gemäß ITU-R BS.1770-4 an.",
    "Apply a conservative K-weighting approximation to one channel.": "Wendet eine konservative K-Gewichtungs-Approximation auf einen Kanal an.",
    "Apply a console EQ curve (frequency-gain breakpoints) via STFT/ISTFT.": "Wendet eine Konsolen-EQ-Kurve (Frequenz-Gain-Stützpunkte) via STFT/ISTFT an.",
    "Apply action (redo).": "Wendet Aktion erneut an (Redo).",
    "Apply adaptive comb filters with side-chain detection.": "Wendet adaptive Kammfilter mit Sidechain-Erkennung an.",
    "Apply adaptive expander gate to band signal.": "Wendet adaptives Expander-Gate auf das Bandsignal an.",
    "Apply adaptive sizes and typography based on current screen + window size.": "Passt Größen und Typografie adaptiv an aktuelle Bildschirm- und Fenstergröße an.",
    "Apply adaptive tab style in one place.": "Wendet adaptativen Tab-Stil einheitlich an.",
    "Apply adaptive tonal balance restoration": "Wendet adaptive Tonbalance-Restaurierung an.",
    "Apply adjustments to thresholds.": "Wendet Schwellwert-Anpassungen an.",
    "Apply a frequency-domain gain mask using STFT overlap-add — vectorised.": "Wendet eine Frequenzbereichs-Gain-Maske via STFT-Overlap-Add an (vektorisiert).",
    "Apply air band enhancement to audio.": "Wendet Air-Band-Verbesserung auf Audio an.",
    "Apply Air & Presence enhancement": "Wendet Air- und Präsenz-Verbesserung an.",
    "Apply album corrections to a single song.": "Wendet Album-Korrekturen auf einen einzelnen Song an.",
    "Apply a linear gain (dB) scaled by *strength* to *audio*.": "Wendet einen linearen Gain (dB) skaliert mit *strength* auf *audio* an.",
    "Apply all actions in forward order.": "Wendet alle Aktionen in Vorwärtsreihenfolge an.",
    "Apply all enabled dynamics processing modules in sequence": "Wendet alle aktivierten Dynamikverarbeitungsmodule der Reihe nach an.",
    "Apply all enabled tonal balance restoration modules in sequence": "Wendet alle aktivierten Tonbalance-Restaurierungsmodule der Reihe nach an.",
    "Apply all-pass filters for phase decorrelation.": "Wendet Allpass-Filter zur Phasen-Dekorrelation an.",
    "Apply _apply_phoneme_dsp() to each transcription segment in-place.": "Wendet _apply_phoneme_dsp() auf jedes Transkriptionssegment in-place an.",
    "Apply approximate Dolby / DBX inverse to audio.": "Wendet approximative Dolby/DBX-Umkehrung auf Audio an.",
    "Apply a signed fractional-sample shift to a 1-D float32 array.": "Wendet eine vorzeichenbehaftete fraktionale Sampleverzögerung auf ein 1-D-float32-Array an.",
    "Apply a simple linear gain (dB), then peak-safe clip.": "Wendet einfachen linearen Gain (dB) an, dann Peak-sicheres Clipping.",
    "Apply a simple three-band downward compression pass.": "Wendet einen einfachen Drei-Band-Abwärts-Kompressionspass an.",
    "Apply a single biquad peak-EQ to mono *audio*.": "Wendet einen einzelnen Biquad-Peak-EQ auf Mono-*audio* an.",
    "Apply attack/release ballistics.": "Wendet Attack/Release-Ballistiken an.",
    "Apply attack/release smoothing to gain — 16× downsampled.": "Wendet Attack/Release-Glättung auf Gain an – 16-fach unterabgetastet.",
    "Apply attenuation-only peak limiting against the 99.9th percentile peak.": "Wendet ausschließlich dämpfendes Peak-Limiting gegen das 99,9%-Perzentil an.",
    "Apply augmentation policy to audio.": "Wendet Augmentierungsrichtlinie auf Audio an.",
    "Apply bass enhancement to audio.": "Wendet Bass-Verbesserung auf Audio an.",
    "Apply Bell (Peaking) EQ": "Wendet Bell-(Peaking)-EQ an.",
    "Apply bell (peaking) filter using IIR.": "Wendet Bell-(Peaking)-Filter mittels IIR an.",
    "Apply biquad filter using time-domain convolution (differentiable).": "Wendet Biquad-Filter via Zeitbereichsfaltung an (differenzierbar).",
    "Apply breath-aware gating.": "Wendet atemgesteuertes Gating an.",
    "Apply cascaded all-pass filters for phase decorrelation.": "Wendet kaskadierte Allpass-Filter zur Phasen-Dekorrelation an.",
    "Apply cascaded biquad peaks to a single mono channel.": "Wendet kaskadierte Biquad-Peaks auf einen einzelnen Mono-Kanal an.",
    "Apply cascade of all-pass filters for phase rotation.": "Wendet Kaskade von Allpass-Filtern zur Phasendrehung an.",
    "Apply checkerboard kernel to SSM and return novelty curve.": "Wendet Schachbrett-Kernel auf SSM an und gibt Novelty-Kurve zurück.",
    "Apply click removal and level crossfade at each splice point.": "Wendet Click-Entfernung und Pegel-Überblendung an jedem Schnittunkt an.",
    "Apply complete de-reverb processing.": "Wendet vollständige Hallentfernung an.",
    "Apply complete mastering chain.": "Wendet vollständige Mastering-Kette an.",
    "Apply compression to a single band.": "Wendet Kompression auf ein einzelnes Band an.",
    "Apply conservative confidence calibration and crop-locality annotations.": "Wendet konservative Konfidenz-Kalibrierung und Crop-Lokalitäts-Annotationen an.",
    "Apply consistent premium styling to info cards.": "Wendet konsistentes Premium-Styling auf Info-Karten an.",
    "Apply dark premium theme for two-column layout.": "Wendet dunkles Premium-Theme für zweispaltiges Layout an.",
    "Apply de-essing to sibilance band.": "Wendet De-Essing auf das Sibilanz-Band an.",
    "Apply dialog intelligibility enhancement.": "Wendet Dialog-Verständlichkeitsverbesserung an.",
    # ---------- Apply ... continuation ----------
    "Apply digital artifact removal.": "Wendet digitale Artefaktentfernung an.",
    "Apply dynamics processing.": "Wendet Dynamikverarbeitung an.",
    "Apply EQ to audio.": "Wendet EQ auf Audio an.",
    "Apply exciter to audio.": "Wendet Exciter auf Audio an.",
    "Apply final output gain and hard limiter.": "Wendet finalen Ausgangsgain und Hard-Limiter an.",
    "Apply first-order high-shelf IIR filter.": "Wendet Hochregal-IIR-Filter erster Ordnung an.",
    "Apply formant correction to audio.": "Wendet Formant-Korrektur auf Audio an.",
    "Apply gain in frequency domain.": "Wendet Gain im Frequenzbereich an.",
    "Apply harmonic enhancement to audio.": "Wendet harmonische Verbesserung auf Audio an.",
    "Apply high-pass filter.": "Wendet Hochpassfilter an.",
    "Apply high-pass filtering to audio.": "Wendet Hochpassfilterung auf Audio an.",
    "Apply HPSS to audio.": "Wendet HPSS auf Audio an.",
    "Apply humming removal to audio.": "Wendet Brummton-Entfernung auf Audio an.",
    "Apply LUFS normalization.": "Wendet LUFS-Normalisierung an.",
    "Apply masking threshold floor to noise reduction gain.": "Wendet Maskierungsschwellwert-Untergrenze auf NR-Gain an.",
    "Apply micro-dynamics enhancement to audio.": "Wendet Mikrodynamik-Verbesserung auf Audio an.",
    "Apply minimal intervention principle.": "Wendet Minimal-Interventions-Prinzip an.",
    "Apply multi-band dynamics processing.": "Wendet Mehrband-Dynamikverarbeitung an.",
    "Apply multi-band noise reduction.": "Wendet Mehrband-Rauschreduzierung an.",
    "Apply noise gate.": "Wendet Noise-Gate an.",
    "Apply noise reduction.": "Wendet Rauschreduzierung an.",
    "Apply noise shaping dither.": "Wendet Rauschformungs-Dither an.",
    "Apply notch filter.": "Wendet Notchfilter an.",
    "Apply one step of the restoration plan.": "Führt einen Schritt des Restaurierungsplans aus.",
    "Apply output gain to audio.": "Wendet Ausgangsgain auf Audio an.",
    "Apply parallel compression.": "Wendet Parallelkompression an.",
    "Apply parametric EQ band.": "Wendet parametrisches EQ-Band an.",
    "Apply peak limiting.": "Wendet Peak-Limiting an.",
    "Apply perceptual enhancement to audio.": "Wendet perzeptuelle Verbesserung auf Audio an.",
    "Apply phase-only processing.": "Wendet ausschließlich Phasenverarbeitung an.",
    "Apply pitch correction.": "Wendet Pitch-Korrektur an.",
    "Apply psychoacoustic masking.": "Wendet psychoakustisches Masking an.",
    "Apply rate conversion.": "Wendet Ratenkonvertierung an.",
    "Apply reverb.": "Wendet Hall an.",
    "Apply rolloff correction to audio.": "Wendet Rolloff-Korrektur auf Audio an.",
    "Apply RIAA de-emphasis to audio.": "Wendet RIAA-De-Emphasis auf Audio an.",
    "Apply saturation correction.": "Wendet Sättigungskorrektur an.",
    "Apply sibilance reduction to audio.": "Wendet Sibilanz-Reduzierung auf Audio an.",
    "Apply silence gate.": "Wendet Stille-Gate an.",
    "Apply spatial enhancement to audio.": "Wendet räumliche Klangverbesserung auf Audio an.",
    "Apply spectral enhancement to audio.": "Wendet spektrale Verbesserung auf Audio an.",
    "Apply spectral noise reduction.": "Wendet spektrale Rauschreduzierung an.",
    "Apply spectral subtraction.": "Wendet spektrale Subtraktion an.",
    "Apply stereo enhancement.": "Wendet Stereo-Verbesserung an.",
    "Apply stereo width correction.": "Wendet Stereobreiten-Korrektur an.",
    "Apply tape noise reduction.": "Wendet Band-Rauschreduzierung an.",
    "Apply the configured DSP blocks to *audio*.": "Wendet die konfigurierten DSP-Blöcke auf *audio* an.",
    "Apply the HPSS decomposition.": "Wendet die HPSS-Zerlegung an.",
    "Apply the phase gate.": "Wendet das Phasen-Gate an.",
    "Apply transient shaping.": "Wendet Transientenformung an.",
    "Apply transient-preserving noise reduction.": "Wendet transienten-erhaltende Rauschreduzierung an.",
    "Apply tonal balance restoration.": "Wendet Tonbalance-Restaurierung an.",
    "Apply tonal correction to audio.": "Wendet Tonal-Korrektur auf Audio an.",
    "Apply vinyl noise reduction to audio.": "Wendet Vinyl-Rauschreduzierung auf Audio an.",
    "Apply vocal enhancement to audio.": "Wendet Vokal-Verbesserung auf Audio an.",
    "Apply warmth enhancement to audio.": "Wendet Wärme-Verbesserung auf Audio an.",
    "Apply window correction factor.": "Wendet Fensterkorrekturfaktor an.",
    # ---------- Verbleibende 86 Phrasen (zweiter Pass) ----------
    "Audiodaten unterbrechungsfrei abspielen (gapless source-switch via StreamingAudioPlayer).": "Audiodaten unterbrechungsfrei abspielen (gapless source-switch via StreamingAudioPlayer).",
    "Calibrated reference confidence in [0, 1] from existing reliability signals.": "Kalibrierte Referenz-Konfidenz in [0, 1] aus vorhandenen Zuverlässigkeitssignalen.",
    "Captures all information needed to re-run deferred phases with no RT limit.": "Speichert alle Informationen zum erneuten Ausführen zurückgestellter Phasen ohne RT-Limit.",
    "Checkpoint during processing.": "Checkpoint während der Verarbeitung.",
    "Clipping Detection — §6.3 Spec: CLIPPING vs. SOFT_SATURATION Discrimination.": "Clipping-Erkennung — §6.3 Spec: CLIPPING vs. SOFT_SATURATION Diskriminierung.",
    "Clipping detection (hard limiting).": "Clipping-Erkennung (hartes Begrenzen).",
    "Clipping Detector": "Clipping-Detektor.",
    "Collection of audio augmentation operations.": "Sammlung von Audio-Augmentierungsoperationen.",
    "Combined evidence score for 1960/1970 transition decisions.": "Kombinierter Evidenz-Score für 1960/1970-Übergangs-Entscheidungen.",
    "Combined WOW+FLUTTER score — max of both sub-detectors.": "Kombinierter WOW+FLUTTER-Score – Maximum beider Sub-Detektoren.",
    "Country genre score: simple diatonic harmony + bright twang + wide dynamics.": "Country-Genre-Score: einfache diatonische Harmonie + helles Twang + weite Dynamik.",
    "Detected breath event.": "Erkanntes Atemereignis.",
    "Detected instrument types.": "Erkannte Instrumententypen.",
    "detected phoneme segment with timing and confidence.": "Erkanntes Phonem-Segment mit Timing und Konfidenz.",
    "Detected plosive event (P, B, T, D, K, G sounds).": "Erkanntes Plosiv-Ereignis (P, B, T, D, K, G Laute).",
    "Detected source media types (Spec 3.1.2)": "Erkannte Quell-Medientypen (Spec 3.1.2).",
    "Detected transient (drum hit, percussive sound).": "Erkannter Transient (Schlagzeugschlag, perkussiver Klang).",
    "Deterministischer SHA256-Fingerabdruck eines Audio-Arrays (16 Hex).": "Deterministischer SHA256-Fingerabdruck eines Audio-Arrays (16 Hex).",
    "Execution strategies for module processing.": "Ausführungsstrategien für Modulverarbeitung.",
    "Extended audio exporter with multi-format support.": "Erweiterter Audio-Exporter mit Mehrformat-Unterstützung.",
    "Extended Audio Export Module for AURIK": "Erweitertes Audio-Export-Modul für AURIK.",
    "Extended ContextAnalyzer that integrates Zone Engine.": "Erweiterter ContextAnalyzer mit Zone-Engine-Integration.",
    "Extracted audio metadata for transfer between files.": "Extrahierte Audio-Metadaten für den Dateitransfer.",
    "Extracted feature vectors (Spec 3.1.6)": "Extrahierte Feature-Vektoren (Spec 3.1.6).",
    "Factory for creating and managing HIPS safety wrappers.": "Factory für Erstellung und Verwaltung von HIPS-Sicherheits-Wrappern.",
    "Factory for FailReason — convenience wrapper.": "Factory für FailReason – Komfort-Wrapper.",
    "Factory function to create appropriate safety wrapper for DSP module.": "Factory-Funktion zum Erstellen eines passenden Sicherheits-Wrappers für DSP-Module.",
    "Factory function to create DialogIntelligibilityEnhancer instance.": "Factory-Funktion zum Erstellen einer DialogIntelligibilityEnhancer-Instanz.",
    "Factory function to create MQA system.": "Factory-Funktion zum Erstellen des MQA-Systems.",
    "Factory function to create PhaseRotator instance.": "Factory-Funktion zum Erstellen einer PhaseRotator-Instanz.",
    "Factory function to create PsychoacousticEnhancer instance.": "Factory-Funktion zum Erstellen einer PsychoacousticEnhancer-Instanz.",
    "Factory function to create quality recovery system.": "Factory-Funktion zum Erstellen des Qualitäts-Recovery-Systems.",
    "Findet all valid OOM checkpoints in the sessions directory.": "Findet alle gültigen OOM-Checkpoints im Sessions-Verzeichnis.",
    "Findet content segment at given timestamp.": "Findet das Inhaltssegment beim gegebenen Zeitstempel.",
    "Findet contiguous regions in binary mask.": "Findet zusammenhängende Regionen in binärer Maske.",
    "Findet critical path (longest path through dependency graph).": "Findet den kritischen Pfad (längsten Pfad durch den Abhängigkeitsgraphen).",
    "Findet pairs of degraded and clean audio files.": "Findet Paare aus degradierten und sauberen Audiodateien.",
    "Findet peak frames, wrap in 2-second zones, merge overlapping zones.": "Findet Peak-Frames, fasst sie in 2-Sekunden-Zonen zusammen und führt überlappende Zonen zusammen.",
    "Findet Pre-Echo- und Post-Echo-Delays via Kreuzkorrelations-Peak-Suche.": "Findet Pre-Echo- und Post-Echo-Delays via Kreuzkorrelations-Peak-Suche.",
    "Findet the best-matching label profile for (era, material).": "Findet das am besten passende Label-Profil für (Ära, Material).",
    "Findet the cleanest segments within a song for use as internal reference.": "Findet die saubersten Segmente eines Songs als interne Referenz.",
    "Findet transient locations in audio.": "Findet Transient-Positionen im Audio.",
    "Initialise AMD ROCm runtime to eliminate cold-start latency before first inference.": "Initialisiert AMD-ROCm-Laufzeitumgebung, um Cold-Start-Latenz vor der ersten Inferenz zu eliminieren.",
    "Initialise ROCm runtime with a minimal dummy inference to amortise cold-start.": "Initialisiert ROCm-Laufzeitumgebung mit minimaler Dummy-Inferenz zur Cold-Start-Amortisierung.",
    "Interpolated ceiling in dB for an arbitrary frequency.": "Interpolierte Decke in dB für eine beliebige Frequenz.",
    "Interpolated studio-day reconstruction target in dB.": "Interpoliertes Studio-Tag-Rekonstruktionsziel in dB.",
    "Lazy wrapper for the MusikalischerGlobalplan entry point.": "Lazy-Wrapper für den MusikalischerGlobalplan-Einstiegspunkt.",
    "Logging configuration for phoneme-aware processing module.": "Logging-Konfiguration für das phonem-bewusste Verarbeitungsmodul.",
    "Logging configuration for pitch correction module.": "Logging-Konfiguration für das Pitch-Korrekturmodul.",
    "Logging configuration for vocal separation module": "Logging-Konfiguration für das Vokal-Separationsmodul.",
    "Mixin: makes dataclass instances behave like dicts for backward compatibility.": "Mixin: Lässt Dataclass-Instanzen wie Dicts verhalten (Rückwärtskompatibilität).",
    "Mixture of Experts Ensemble.": "Mixture-of-Experts-Ensemble.",
    "Normalize/alias defect score keys and sanitize values.": "Normalisiert/aliasiert Defektbewertungs-Schlüssel und bereinigt Werte.",
    "optimization/balanced_processor.py — Balanced audio processor": "optimization/balanced_processor.py – Ausgeglichener Audio-Prozessor.",
    "optimization/priority1_efficiency.py — Algorithmic efficiency optimization": "optimization/priority1_efficiency.py – Algorithmische Effizienzoptimierung.",
    "optimization/priority2_vocals.py — Selective vocal enhancement": "optimization/priority2_vocals.py – Selektive Vokal-Verbesserung.",
    "optimization/priority3_oversampling.py — Adaptive oversampling processor": "optimization/priority3_oversampling.py – Adaptiver Oversampling-Prozessor.",
    "optimization/priority4_phase.py — Multiband phase coherence enhancer": "optimization/priority4_phase.py – Mehrband-Phasenkohärenz-Verbesserer.",
    "optimization/priority5_bass.py — Phase-coherent bass processing": "optimization/priority5_bass.py – Phasenkohärente Bass-Verarbeitung.",
    "optimization/priority6_parameters.py — Genre-optimised parameters and presets": "optimization/priority6_parameters.py – Genreoptimierte Parameter und Presets.",
    "optimization/profiling.py — Performance profiler and quality validator": "optimization/profiling.py – Performance-Profiler und Qualitätsvalidator.",
    "Optimized DSP Module for AURIK v8": "Optimiertes DSP-Modul für AURIK v8.",
    "Parst MFA TextGrid output to extract phoneme alignments.": "Parst MFA-TextGrid-Ausgabe zur Extraktion von Phonem-Ausrichtungen.",
    "Parst 'v10.0.0' or '9.10.77' into comparable tuple.": "Parst 'v10.0.0' oder '9.10.77' in ein vergleichbares Tupel.",
    "phase_rotation.py - Phase Rotation/Alignment for AURIK 6.0": "phase_rotation.py – Phasendrehung/Ausrichtung für AURIK 6.0.",
    "Remove/reduce crowd noise from audio.": "Entfernt/Reduziert Publikumsgeräusche aus Audio.",
    "Resolved alle Konflikte durch Priority-based Adjustments.": "Löst alle Konflikte durch prioritätsbasierte Anpassungen.",
    "Runtime environment selection for developer launches.": "Laufzeitumgebungs-Auswahl für Entwickler-Starts.",
    "Scaled RMS-envelope variance — proxy for phrase contour complexity.": "Skalierte RMS-Hüllkurven-Varianz – Proxy für Phrasenkontur-Komplexität.",
    "Selectively oversample transient regions.": "Überabtastet selektiv transiente Regionen.",
    "Serializable decision returned by :class:`VocalNoHarmGate`.": "Serialisierbares Ergebnis von :class:`VocalNoHarmGate`.",
    "Serializable per-case execution/export evaluation result.": "Serialisierbares Ausführungs-/Export-Bewertungsergebnis pro Fall.",
    "Serializable per-case final-quality evaluation result.": "Serialisierbares Endqualitäts-Bewertungsergebnis pro Fall.",
    "Serializable per-case strategy evaluation result.": "Serialisierbares Strategie-Bewertungsergebnis pro Fall.",
    "Serializable result of a real-audio execution/export gate run.": "Serialisierbares Ergebnis eines Real-Audio-Ausführungs-/Export-Gate-Laufs.",
    "Serializable result of a real-audio Golden-Set gate run.": "Serialisierbares Ergebnis eines Real-Audio-Golden-Set-Gate-Laufs.",
    "Serializable result of a real-audio strategy Golden-Set gate run.": "Serialisierbares Ergebnis eines Real-Audio-Strategie-Golden-Set-Gate-Laufs.",
    "Serializable result of the real-audio final-quality gate.": "Serialisierbares Ergebnis des Real-Audio-Endqualitäts-Gates.",
    "Serialization format for logging and persistence.": "Serialisierungsformat für Logging und Persistenz.",
    "Setup optimizer for training.": "Richtet Optimizer für das Training ein.",
    "Setup Title Bar UI": "Richtet Titelleisten-UI ein.",
    "Setup UI elements": "Richtet UI-Elemente ein.",
}

# ---------------------------------------------------------------------------
# Regel-basierte Übersetzer (für nicht im Lookup enthaltene Phrasen)
# ---------------------------------------------------------------------------

# Verb-Corrections (Regex-Bugs aus dem ersten Lauf)
_VERB_FIXES: list[tuple[re.Pattern, str]] = [
    # Apply/Applies
    (re.compile(r"^Apply(ing)? ", re.IGNORECASE), "Wendet an: "),
    (re.compile(r"^Applies ", re.IGNORECASE), "Wendet an: "),
    # Process/Processes/Processing
    (re.compile(r"^Process(es|ing)? ", re.IGNORECASE), "Verarbeitet "),
    # Fetch/Fetches
    (re.compile(r"^Fetch(es|ing)? ", re.IGNORECASE), "Ruft ab: "),
    # Dispatch/Dispatches
    (re.compile(r"^Dispatch(es|ing)? ", re.IGNORECASE), "Verteilt "),
    # Mix/Mixes
    (re.compile(r"^Mix(es|ing)? ", re.IGNORECASE), "Mischt "),
    # Notify/Notifies
    (re.compile(r"^Notif(?:y|ies|ying) ", re.IGNORECASE), "Benachrichtigt "),
    # Lazy load
    (re.compile(r"^Lazy[ -]load(?:s|ing)? ", re.IGNORECASE), "Lädt beim ersten Zugriff: "),
    # Initialise (britische Schreibweise)
    (re.compile(r"^Initialise(?:s|d)? ", re.IGNORECASE), "Initialisiert "),
    # Setup
    (re.compile(r"^Setup ", re.IGNORECASE), "Richtet ein: "),
    # Remove/reduce
    (re.compile(r"^Remove(?:/reduce)? ", re.IGNORECASE), "Entfernt "),
    # Extended
    (re.compile(r"^Extended? ", re.IGNORECASE), "Erweiterter/Erweitertes "),
    # Extracted
    (re.compile(r"^Extracted? ", re.IGNORECASE), "Extrahiertes "),
    # Detected
    (re.compile(r"^Detected? ", re.IGNORECASE), "Erkanntes "),
    # Serializable
    (re.compile(r"^Serializable ", re.IGNORECASE), "Serialisierbares "),
    # Combined
    (re.compile(r"^Combined? ", re.IGNORECASE), "Kombiniertes "),
    # Collection of
    (re.compile(r"^Collection of ", re.IGNORECASE), "Sammlung von "),
    # Captures
    (re.compile(r"^Captures? ", re.IGNORECASE), "Speichert "),
    # Factory for/function
    (re.compile(r"^Factory (for|function) (to )?", re.IGNORECASE), "Factory-Funktion zum "),
    # Logging configuration
    (re.compile(r"^Logging configuration for ", re.IGNORECASE), "Logging-Konfiguration für "),
    # Mixin:
    (re.compile(r"^Mixin: ", re.IGNORECASE), "Mixin: "),
    # Unified
    (re.compile(r"^Unified (API|interface) for ", re.IGNORECASE), "Einheitliche \\1 für "),
    (re.compile(r"^Unified ([\w\s]+)\.", re.IGNORECASE), "Einheitliches \\1."),
    # Singleton
    (re.compile(r"^Singleton wrapper for ", re.IGNORECASE), "Singleton-Wrapper für "),
    (re.compile(r"^Singleton[ -]getter ", re.IGNORECASE), "Singleton-Getter "),
    # Simple
    (re.compile(r"^Simple ", re.IGNORECASE), "Einfaches "),
    # Lightweight
    (re.compile(r"^Lightweight ", re.IGNORECASE), "Leichtgewichtiges "),
    # Wrapper for
    (re.compile(r"^Wrapper for ", re.IGNORECASE), "Wrapper für "),
    # Validation of
    (re.compile(r"^Validation of ", re.IGNORECASE), "Validierung von "),
    # Validator for
    (re.compile(r"^Validator for ", re.IGNORECASE), "Validator für "),
    # Core
    (re.compile(r"^Core ", re.IGNORECASE), "Kern-"),
    # Helper
    (re.compile(r"^Helper ", re.IGNORECASE), "Hilfsfunktion: "),
    # Internal
    (re.compile(r"^Internal ", re.IGNORECASE), "Intern: "),
    # Base
    (re.compile(r"^Base ", re.IGNORECASE), "Basis-"),
    # Main
    (re.compile(r"^Main ", re.IGNORECASE), "Haupt-"),
    # Additive synthesis
    (re.compile(r"^Additive synthesis of ", re.IGNORECASE), "Additive Synthese von "),
    # Merged
    (re.compile(r"^Merged ", re.IGNORECASE), "Fasst zusammen: "),
    # Updated
    (re.compile(r"^Updated ", re.IGNORECASE), "Aktualisierter/Aktualisierte "),
]

# Muster für "Analysiert ENGLISH_REST" → deutsch vervollständigen
_RuleItem = tuple[re.Pattern, str | Any]
_ANALYSIERT_REST: list[_RuleItem] = [
    (
        re.compile(r"Analysiert audio for all (.+)\.$", re.IGNORECASE),
        lambda m: f"Analysiert Audio auf alle {m.group(1)}.",
    ),
    (re.compile(r"Analysiert audio for (.+)\.$", re.IGNORECASE), lambda m: f"Analysiert Audio auf {m.group(1)}."),
    (
        re.compile(r"Analysiert audio and compute (.+)\.$", re.IGNORECASE),
        lambda m: f"Analysiert Audio und berechnet {m.group(1)}.",
    ),
    (
        re.compile(r"Analysiert audio and return (.+)\.$", re.IGNORECASE),
        lambda m: f"Analysiert Audio und gibt {m.group(1)} zurück.",
    ),
    (
        re.compile(r"Analysiert audio to determine (.+)\.$", re.IGNORECASE),
        lambda m: f"Analysiert Audio zur Bestimmung von {m.group(1)}.",
    ),
    (
        re.compile(r"Analysiert audio context from (.+)\.$", re.IGNORECASE),
        lambda m: f"Analysiert den Audio-Kontext aus {m.group(1)}.",
    ),
    (re.compile(r"Analysiert audio (.+)\.$", re.IGNORECASE), lambda m: f"Analysiert Audio: {m.group(1)}."),
    (re.compile(r"Analysiert a song and (.+)\.$", re.IGNORECASE), lambda m: f"Analysiert einen Song und {m.group(1)}."),
    (re.compile(r"Analysiert the (.+)\.$", re.IGNORECASE), lambda m: f"Analysiert {m.group(1)}."),
    (re.compile(r"Analysiert which (.+)\.$", re.IGNORECASE), lambda m: f"Analysiert, {m.group(1)}."),
]


def _apply_verb_fixes(text: str) -> str | None:
    """Versucht Verb-Korrekturen auf text anzuwenden. None wenn kein Match."""
    for pat, repl in _VERB_FIXES:
        if pat.match(text):
            result = pat.sub(repl, text, count=1)
            if not result.rstrip().endswith((".", "!", "?")):
                result = result.rstrip() + "."
            return result
    return None


def _apply_analysiert_rest(text: str) -> str | None:
    """Versucht 'Analysiert ENGLISH_REST' zu vervollständigen."""
    for pat, repl in _ANALYSIERT_REST:
        m = pat.match(text)
        if m:
            if callable(repl):
                return str(repl(m))  # type: ignore[operator]
            return pat.sub(str(repl), text, count=1)
    return None


def _translate_todo_content(raw: str) -> str:
    """
    Übersetzt den Inhalt hinter 'TODO(de): '.
    Reihenfolge: Lookup → bereits-deutsch → Verb-Fix → Regel-Rest → Fallback.
    """
    # Trailing triple-quotes aus dem raw entfernen (Artefakt aus grep)
    text = raw.strip()
    text_clean = text.rstrip('"')

    # 1. Direkte Lookup-Tabelle (exakt oder ohne trailing quotes)
    if text in _LOOKUP:
        return _LOOKUP[text]
    if text_clean in _LOOKUP:
        return _LOOKUP[text_clean]

    # 2. Bereits deutsch → nur TODO-Prefix entfernen
    if _is_already_german(text_clean):
        clean = text_clean
        if not clean.rstrip().endswith((".", "!", "?", '"', "'")):
            clean = clean.rstrip() + "."
        return clean

    # 3. Verb-Fix (Apply, Process, Fetch, ...)
    verb_fix: str | None = _apply_verb_fixes(text_clean)
    if verb_fix:
        return verb_fix

    # 4. Analysiert + englischer Rest
    analysiert_fix: str | None = _apply_analysiert_rest(text_clean)
    if analysiert_fix:
        return analysiert_fix

    # 5. Fallback: unveränderPrefix bleibt, wird nicht entfernt)
    return f"TODO(de): {text_clean}"


# ---------------------------------------------------------------------------
# Datei-Verarbeitung
# ---------------------------------------------------------------------------

# Findet TODO(de):-Marker in Docstrings
_TODO_RE = re.compile(r'TODO\(de\): ([^\n]+?)(?="""|\'{3}|\n|$)')


def _resolve_todos_in_source(source: str) -> tuple[str, int]:
    """Ersetzt alle TODO(de):-Marker im Quelltext. Gibt (neuer_text, count) zurück."""
    count = 0
    result = []
    last_end = 0

    for m in _TODO_RE.finditer(source):
        raw = m.group(1)
        translated = _translate_todo_content(raw)
        if translated.startswith("TODO(de): "):
            # Nicht übersetzt — unverändert lassen
            result.append(source[last_end : m.end()])
        else:
            result.append(source[last_end : m.start()])
            result.append(translated)
            count += 1
        last_end = m.end()

    result.append(source[last_end:])
    return "".join(result), count


def process_file(path: Path, dry_run: bool = False) -> int:
    """Verarbeitet eine Datei. Gibt Anzahl der aufgelösten TODOs zurück."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0

    if "TODO(de):" not in source:
        return 0

    new_source, count = _resolve_todos_in_source(source)
    if count == 0:
        return 0

    if not dry_run:
        path.write_text(new_source, encoding="utf-8")

    return count


def process_directory(directory: Path, dry_run: bool, verbose: bool) -> tuple[int, int]:
    """Verarbeitet alle .py-Dateien in directory rekursiv."""
    total_files = 0
    total_count = 0
    for path in sorted(directory.rglob("*.py")):
        count = process_file(path, dry_run=dry_run)
        if count > 0:
            total_files += 1
            total_count += count
            if verbose:
                mode = "[DRY] " if dry_run else ""
                print(f"{mode}{path}: {count} aufgelöst")
    return total_files, total_count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Haupteinstiegspunkt."""
    parser = argparse.ArgumentParser(description="Löst TODO(de):-Marker auf.")
    parser.add_argument("--dry-run", action="store_true", help="Änderungen nur anzeigen, nicht schreiben")
    parser.add_argument("--verbose", action="store_true", help="Verarbeitete Dateien ausgeben")
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=[
            "backend",
            "forensics",
            "plugins",
            "Aurik10",
            "cli",
            "denker",
            "dsp",
            "export",
            "processing",
            "workflow",
        ],
        help="Verzeichnisse zum Verarbeiten",
    )
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    total_files = 0
    total_count = 0

    for d in args.dirs:
        directory = base / d
        if not directory.exists():
            continue
        files, count = process_directory(directory, dry_run=args.dry_run, verbose=args.verbose)
        total_files += files
        total_count += count

    mode = "DRY RUN — " if args.dry_run else ""
    print(f"{mode}{total_count} TODO(de):-Marker aufgelöst in {total_files} Dateien.")

    # Verbleibende TODOs zählen
    if not args.dry_run:
        remaining = 0
        for d in args.dirs:
            directory = base / d
            if not directory.exists():
                continue
            for path in directory.rglob("*.py"):
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                    remaining += content.count("TODO(de):")
                except OSError:
                    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
        print(f"Verbleibende TODO(de):-Marker: {remaining}")


if __name__ == "__main__":
    main()
