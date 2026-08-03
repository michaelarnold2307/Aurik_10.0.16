# Aurik 10 — GEBOTE & VERBOTE (Normativer Katalog)

> **Status:** Normativ | **Version:** 10.16.0 | **Stand:** August 2026
>
> Dieser Katalog definiert alle unverhandelbaren GEBOTE (positiv, was Aurik TUN MUSS)
> und VERBOTE (negativ, was Aurik NIEMALS tun darf). Jedes Gebot und Verbot ist mit
> einer eindeutigen ID versehen (§G1, §V1 usw.) und wird im Code per Kommentar
> referenziert. Bei Widerspruch zwischen Specs und diesem Katalog gilt dieser Katalog.

---

## Kategorie I — Individuelle Song-Maximierung (§G1–§G9)

Jeder importierte Song wird individuell maximal für das menschliche Ohr verbessert.

| ID | Regel | Beschreibung |
| ---- | ------- | ------------- |
| §G1 | **Pro-Song-Kalibrierung** | Jeder Song durchläuft eine vollständige, isolierte SongCalibration (global_scalar, family_scalars, ALLE Guards). Kein Parameter aus einem vorherigen Song darf ungeprüft übernommen werden. |
| §G2 | **Defekt-Vollständigkeit** | Alle 62 DefectTypes werden pro Song gescannt. Defekte werden über die gesamte Songdauer präzise behoben – nicht nur an Stichproben/Checkpoints. |
| §G3 | **Gesangsintegrität** | Gesang darf NIE verzerrt, verschliffen oder mit Artefakten (Ghost-Echo, Phasing) versehen werden. Der Vocal-Safety-Wrapper muss in jeder Phase aktiv sein, die Frequenzen zwischen 80 Hz und 8 kHz bearbeitet. |
| §G4 | **Ghost-Echo-Freiheit** | Kein hörbares Echo oder Pre-Echo durch Phasenverschiebungen, asymmetrische Fensterung oder STFT-Überlappungsartefakte. §2.60 STCG muss in allen Modi laufen. |
| §G5 | **Konsistenz-Mandat** | Alle Maßnahmen müssen über das gesamte Projekt konsistent sein. Kein phasespezifischer Schwellwert ohne zentrale Definition. |
| §G6 | **Null-Toleranz für Phasen-Leckage** | Parameter, Zustände und Circuit-Breaker aus Phase 12, 21, 35, 42 werden pro Song zurückgesetzt (§C3). |
| §G7 | **Interchannel-Lag** | GCC-PHAT-High-Band (§v10.0.4) wird an LAG_PROBE_0B/1/2a/3 gemessen. L/R-Zeitversatz > 50 samples wird vor Phase 1 global korrigiert. Residuale werden von STCG per-Chunk behandelt. |
| §G8 | **CD-Rauschprofil-Pflicht** | Jeder Export (Restoration + Studio 2026) erhält ein CD-charakteristisches Rauschprofil. Das Profil wird NUR dort appliziert, wo es das menschliche Ohr wahrnimmt (psychoakustische Maskierungsschwelle). |
| §G9 | **Quellmaterial-Unabhängigkeit** | Das CD-Rauschprofil wird unabhängig vom Quellmaterial appliziert. Die Charakteristik ist deterministisch und von der CD-Ära (1982–2000) abgeleitet. |

## Kategorie II — Psychoakustik & Natürlichkeit (§G10–§G19)

| ID | Regel | Beschreibung |
| ---- | ------- | ------------- |
| §G10 | **ERB-Masking-First** | Jede spektrale Entscheidung muss das ERB-Masking-Modell (Equivalent Rectangular Bandwidth) konsultieren. Kein Gain, kein Filter, kein Dither ohne Masking-Check. |
| §G11 | **Natürlicher Wohlklang** | Das Ziel jedes Processing-Schritts ist der Wohlklang für das menschliche Ohr – nicht mathematische Optimalität. Eine Verschlechterung des PQS-MOS < 3.0 löst Rollback aus. |
| §G12 | **Lautheitskonsistenz** | LUFS-integrated nach EBU R128. Restoration-Ziel: −23 LUFS. Studio-2026-Ziel: −14 LUFS. Kein Hard-Limit ohne ISP-geschützten True-Peak-Limiter. |
| §G13 | **Multi-Point-Lag** | Interchannel-Lag wird an ≥3 Positionen gemessen (Start, Mitte, Ende). Konsistenz-Check: Streuung ≤ 50 samples → globale Korrektur; sonst Median + STCG. |
| §G14 | **Spectral-Tilt-Guard** | Nach jeder Phase wird die spektrale Neigung geprüft. Tilt-Änderung > 1.5 dB/Oktave oder HF-Drop > 3 dB löst Korrektur aus. |
| §G15 | **Rauschprofil-Maskierung** | Das CD-Rauschprofil wird frequenzabhängig und zeitabhängig appliziert. In jedem ERB-Band wird nur dann Rauschen addiert, wenn der Signalpegel unter der simultanen Maskierungsschwelle liegt. |
| §G16 | **Rauschprofil-Charakteristik** | Die Rauschprofil-Charakteristik entspricht einer CD-Neuauflage: −96 dBFS Flat-Noise-Floor (16-bit) mit POW-r-Type-3-Shaping → äquivalente Rauschspannung von −110 dBFS(A) bewertet. |
| §G17 | **Stille-Respekt** | Absolute Stille (digital black) wird NICHT verrauscht. Nur Segmente mit Signalenergie erhalten das Profil. |
| §G18 | **Spektrale Kohärenz** | Frequenzantwort des Rauschprofils folgt dem Langzeit-Leistungsdichtespektrum von CD-Mastern: flach von 20 Hz–16 kHz, −3 dB/Oktave Rolloff ab 16 kHz. |
| §G19 | **Dither-Doppelung-Verbot** | Das CD-Rauschprofil und das Export-Dithering dürfen sich nicht additiv aufschaukeln. Das Rauschprofil wird VOR dem Dithering appliziert; das Dithering berücksichtigt den bereits vorhandenen Rauschpegel. |

## Kategorie III — Architektur & Datenfluss (§G20–§G29)

| ID | Regel | Beschreibung |
| ---- | ------- | ------------- |
| §G20 | **Bridge-Bypass-Verbot** | Kein UI-/Frontend-Code importiert `backend/core/` direkt. Nur über `backend/api/bridge.py`. |
| §G21 | **Denker-Zentralität** | Alle Stärke-Entscheidungen fließen zentral im Denker. Keine dezentralen "Magic Numbers" in Phasen. |
| §G22 | **Determinismus** | Derselbe Input → derselbe Output. Jeder Zufallsgenerator wird mit fixem Seed aus dem Datei-Hash initialisiert. |
| §G23 | **ML-Fallback-Logging** | Jeder ML→DSP-Fallback MUSS mit `logger.warning()` protokolliert werden. Silent-Failures sind VERBOTEN. |
| §G24 | **NaN/Inf-Schutz** | Jede der 71 Phasen (68 + 3 Phase-0) MUSS `np.nan_to_num()` oder `np.isfinite()` auf Ausgabe-Audio anwenden (§0a). |
| §G25 | **Logger-Pflicht** | Jede Python-Datei mit `logger`-Verwendung MUSS `import logging` und `logger = logging.getLogger(__name__)` definieren. |
| §G26 | **Guard-Counter-Lebendigkeit** | Jeder deklarierte Guard-Counter MUSS auch inkrementiert werden. Deklaration ohne `+= 1` ist toter Code. |
| §G27 | **Messschleifen-Plateau** | Jede Messschleife mit ≥3 Kandidaten MUSS Plateau-Erkennung haben. |
| §G28 | **PIM-first, RLP-last** | Vor jedem Phasen-Loop wird PIM berechnet. Nach jedem Loop wird RLP ausgeführt. |
| §G29 | **Artistic Intent vor Defect-Scan** | `get_artistic_intent()` wird VOR dem Defect-Scan aufgerufen. |

## Kategorie IV — CD-Rauschprofil & Export (§G30–§G39)

| ID | Regel | Beschreibung |
| ---- | ------- | ------------- |
| §G30 | **L/R-Unkorreliertheit** | Das Rauschsignal für linken und rechten Kanal MUSS statistisch unabhängig (unkorreliert) sein. Korreliertes Rauschen erzeugt ein hörbares Mono-Rauschzentrum in der Stereomitte — das klingt unnatürlich und ist für CD-Wiedergabe untypisch. |
| §G31 | **Maskierungs-Kanten-Glättung** | An Übergängen zwischen maskierten und unmaskierten Zeit-Frequenz-Regionen MUSS ein 500 ms Cosine-Fade-In/Out erfolgen. Abrupte Rauschpegel-Änderungen sind als "Pumpen" hörbar und verletzen §V1, §V2. |
| §G32 | **ML-Device-Detection** | `next(model.parameters()).device` statt `model.device`. Letzteres ist nach partiellen `.cpu()`/`.to()`-Aufrufen auf Sub-Modulen unzuverlässig und verursacht NaN-Werte auf ROCm. |
| §G33 | **ML-Recovery-API-Äquivalenz** | Recovery-Pfad nach GPU-Fehler MUSS dieselbe API wie der Hauptpfad verwenden (z.B. `model.generate_batch()`), nur mit reduzierten Steps. Niemals komplett andere Funktionssignatur im Retry. |
| §G34 | **Test-Assertion-Konvention** | `np.testing.assert_allclose` nimmt Toleranzen (`rtol`, `atol`). NIEMALS Toleranzen an NumPy-Mathefunktionen übergeben (`np.abs(x, rtol=1e-5)` → `np.abs(x)`). |
| §G35 | **Export-Atomizität** | Jeder Datei-Export MUSS atomar erfolgen: erst in `.tmp`-Datei schreiben, dann `os.replace(tmp, target)`. Bei Abbruch entsteht keine korrupte Datei. |
| §G36 | **True-Peak-Grenze** | Kein Export darf True-Peak > 0 dBTP enthalten. ISP-Interpolation nach ITU-R BS.1770-4 Annex 2 zählt. Oversampling ×4 Minimum. |
| §G37 | **Feedback-Chain-Guards** | Die Feedback-Chain (Phase 12 retry, Phase 35 re-run) MUSS alle Quality-Gates, STCG post-feedbackchain und Spectral-Tilt-Guard durchlaufen. Kein "nackter" Re-Run ohne Guard-Schutz. |
| §G38 | **Modus-Parameter-Isolation** | Parameter eines Modus (Restoration vs. Studio 2026) dürfen nicht in den anderen Modus durchsickern. Die `ProcessingConfig` ist unveränderlich nach Konstruktion; abweichende Parameter werden über `kwargs` nur für den aktuellen Run gesetzt. |
| §G39 | **Rauschprofil-Monitoring** | Jede Rauschprofil-Injektion MUSS im Log vermerken: SNR vorher, SNR nachher, aktive Samples mit Rauschzugabe, maximaler Rauschpegel in dBFS, Onset-Stärke an Übergängen. |

## Kategorie V — Rauschprofil-Zeitpunkt & Übergänge (§G40–§G45)

| ID | Regel | Beschreibung |
| ---- | ------- | ------------- |
| §G40 | **Rauschprofil-Zeitpunkt** | Das CD-Rauschprofil wird NACH allen 68 Restaurierungsphasen und VOR dem Dithering appliziert. Dies ist wissenschaftlich der optimale Zeitpunkt: Wird Rauschen früher injiziert, wird es von nachfolgenden Phasen (Denoising, Kompression, EQ) verändert oder verstärkt. Nach der Pipeline ist das Signal stabil und das Rauschen bleibt unverfälscht. |
| §G41 | **Übergangs-Verifikation** | Jeder Übergang zwischen Rauschen und Stille/Musik MUSS verifiziert werden: Die Onset-Stärke (spectral-flux-basiert) darf 0.1 nicht überschreiten. Überschreitung → automatische Verbreiterung des Crossfades auf 500 ms und erneute Prüfung. |
| §G42 | **CD-Produktions-Kohärenz** | Die komplette Export-Kette (Rauschprofil → Dither → Metadaten) MUSS ein Ergebnis liefern, das für einen geschulten Hörer von einer CD-Produktion (1982–2000) nicht unterscheidbar ist. A/B-Blindtest als Validierung. |
| §G43 | **Rauschprofil-Pegel-Anpassung** | Der Rauschpegel passt sich automatisch der Ziel-Bittiefe an: 16-bit → −96 dBFS (CD-Standard), 24-bit → −120 dBFS (Hi-Res-Äquivalent). Kein fester Pegel unabhängig vom Exportformat. |
| §G44 | **Maskierungs-Wissenschaft** | Die Maskierungsschwelle folgt Zwicker & Fastl (1999): −70 dBFS Signalpegel maskiert −96 dBFS breitbandiges Rauschen in ruhiger Umgebung vollständig. Die 50-ms-RMS-Fensterung entspricht der zeitlichen Integration des menschlichen Gehörs. |
| §G45 | **Digital-Black-Integrität** | Exakte Null-Samples (digital black) werden NIE verrauscht — weder durch die Maskierungs-Hüllkurve noch durch Window-Smearing. Sample-genaue Durchsetzung als letzte Verteidigungslinie (§V12). |

---

## VERBOTE — Katalog absoluter Verbote (§V1–§V24)

| ID | Verbot | Beschreibung |
| ---- | -------- | ------------- |
| §V1 | **Gesangsverzerrung** | Es ist VERBOTEN, Gesang zu verzerren, zu verschleifen, zu robotisieren oder mit Vocoder-artigen Artefakten zu versehen. |
| §V2 | **Ghost-Echo** | Es ist VERBOTEN, hörbare Echos, Pre-Echos oder Phasing-Artefakte in das restaurierte Signal einzutragen. |
| §V3 | **Hard-Clamp auf Audio** | Es ist VERBOTEN, einen Hard-Clamp (`np.clip(audio, -1, 1)`) ohne Soft-Knee-Übergang (6 dB) auf das finale Audio anzuwenden. |
| §V4 | **Truncation ohne Dither** | Es ist VERBOTEN, Integer-Quantisierung (16-bit, 24-bit) ohne vorheriges Dithering durchzuführen. |
| §V5 | **Dither-Doppelung** | Es ist VERBOTEN, zweimal zu ditheren. Wenn das CD-Rauschprofil bereits appliziert wurde, muss der Dither-Prozess dies berücksichtigen. |
| §V6 | **Silent-Failure** | Es ist VERBOTEN, dass ML→DSP-Fallbacks ohne `logger.warning()` stattfinden. |
| §V7 | **Toter Guard-Code** | Es ist VERBOTEN, einen Guard-Counter zu deklarieren, der nie inkrementiert wird. |
| §V8 | **Globaler Phasen-Zustand** | Es ist VERBOTEN, dass Phasen-Zustände (Circuit-Breaker, Cache, Session-Daten) zwischen verschiedenen Songs persistieren. |
| §V9 | **Workarounds** | Es ist VERBOTEN, Symptome zu umgehen statt Ursachen zu beheben. |
| §V10 | **Phasen-Individuelle Schwellwerte** | Es ist VERBOTEN, Schwellwerte pro Phase zu definieren, die nicht von `global_scalar` oder der zentralen Decision Intelligence abgeleitet sind. |
| §V11 | **Rauschprofil-Flächendeckung** | Es ist VERBOTEN, das CD-Rauschprofil pauschal über den gesamten Song zu legen. Es darf nur dort appliziert werden, wo das menschliche Ohr es wahrnimmt. |
| §V12 | **Stille-Verfälschung** | Es ist VERBOTEN, digital black (absolute Stille) mit Rauschen zu versehen. |
| §V13 | **Spektrale Verfärbung** | Es ist VERBOTEN, das Rauschprofil so zu formen, dass es den spektralen Charakter des Originals verfärbt. Das Profil muss sich unterhalb der Maskierungsschwelle des Signals bewegen. |
| §V14 | **Modus-Ignoranz** | Es ist VERBOTEN, das CD-Rauschprofil nur in einem Modus zu applizieren. Es gilt für Restoration UND Studio 2026. |
| §V15 | **Nicht-deterministisches Rauschen** | Es ist VERBOTEN, nicht-reproduzierbares Rauschen zu verwenden. Der Rauschgenerator wird mit einem deterministischen Seed pro Song initialisiert (SHA256 der ersten 4096 Samples). |
| §V16 | **Übersteuerndes Rauschen** | Es ist VERBOTEN, dass der Rauschpegel −85 dBFS überschreitet. CD-Noise-Floor = −96 dBFS; mit Shaping max. −90 dBFS in den höchsten Bändern. |
| §V17 | **Quellmaterial-Extraktion** | Es ist VERBOTEN, Rauschen aus dem degradierten Quellmaterial zu extrahieren und wieder einzufügen. Das CD-Rauschprofil wird frisch generiert. Quellrauschen ist ein DEFEKT und wird entfernt. |
| §V18 | **Bridge-Bypass** | Es ist VERBOTEN, dass UI-/Frontend-Code `backend/core/` direkt importiert. Nur über `backend/api/bridge.py`. |
| §V19 | **Nicht-atomarer Export** | Es ist VERBOTEN, die Zieldatei direkt zu überschreiben. Export MUSS atomar sein: `.tmp` → `os.replace`. |
| §V20 | **True-Peak-Überschreitung** | Es ist VERBOTEN, dass ein Export True-Peak > 0 dBTP enthält. ISP-Interpolation nach ITU-R BS.1770-4 Annex 2. Oversampling ×4. |
| §V21 | **ML-Device-Fehlgriff** | Es ist VERBOTEN, `model.device` nach `.cpu()`/`.to()` auf Sub-Modulen zu verwenden. Statthaft: `next(model.parameters()).device`. |
| §V22 | **ML-Recovery-Signaturbruch** | Es ist VERBOTEN, im Recovery-Pfad eine komplett andere API-Signatur zu verwenden. Dieselbe Methode, reduzierte Steps. |
| §V23 | **Diffusionsmodell-Rauschen** | Es ist VERBOTEN, dass Diffusionsmodell-Artefakte im Noise Floor unerkannt bleiben. Der Authenticity-Validator MUSS sie als Artefakt markieren. |
| §V24 | **Falsche Test-Toleranzen** | Es ist VERBOTEN, Toleranzen an NumPy-Mathefunktionen zu übergeben (`np.abs(x, rtol=1e-5)` ist FALSCH). Statthaft: `np.testing.assert_allclose(actual, desired, rtol=...)`. |
| §V25 | **Zwischenphasen-Rauschen** | Es ist VERBOTEN, das CD-Rauschprofil VOR Abschluss aller 68 Restaurierungsphasen zu injizieren. Frühe Injektion führt zu unkontrollierbarer Verstärkung/Modifikation durch nachfolgende Phasen (§G40). |
| §V26 | **Hörbare Übergänge** | Es ist VERBOTEN, dass Übergänge an Rauschprofil-Kanten hörbar sind. Die Onset-Stärke (spectral-flux-basiert) muss < 0.1 sein. Überschreitung → Crossfade-Verbreiterung (§G41). |

---

## Referenz-System

Jedes Gebot und Verbot wird im Code als Kommentar referenziert:

```python
# §G8: CD-Rauschprofil-Pflicht — Rauschen nur unterhalb der Maskierungsschwelle
# §V11: Rauschprofil-Flächendeckung verboten
audio = _apply_cd_noise_profile(audio, sr, mask=erb_mask)
```

**ID-Konventionen:**

- `§G1`–`§G99`: GEBOTE (positiv, was getan werden MUSS)
- `§V1`–`§V99`: VERBOTE (negativ, was NIEMALS getan werden DARF)
- `§C1`–`§C99`: Circuit-Breaker / Schutzschaltungen
- `§F1`–`§F99`: Forensische Regeln
- `§D1`–`§D99`: DSP-Regeln

**Prioritäten:**

- Kategorie I (§G1–§G9): Höchste Priorität — Song-Individualität
- Kategorie II (§G10–§G19): Zweithöchste — Psychoakustik
- Kategorie III (§G20–§G29): Architektur-Invarianten
- Kategorie IV (§G30–§G39): CD-Rauschprofil & Export
- Kategorie V (§G40–§G45): Rauschprofil-Zeitpunkt & Übergänge
- Kategorie VI (§G46–§G59): Metriken & Qualitätssicherung
- Kategorie VII (§G60–§G67): Stereo-Lag-Integrität
- Kategorie VIII (§V27–§V33): Neue VERBOTE Stereo-Lag
- Kategorie IX (§G68–§G75): SFT-Adaptivität & Defekt-Audibilität
- Kategorie X (§G76–§G81): Kalibrierungs-Dispatch
- Kategorie XI (§G82–§G86): Laufzeit-Rekalibrierung
- Kategorie XIX (§G131–§G137): SOTA-Reproduzierbarkeit B1/B2/B3
- Kategorie XX (§G138–§G141): Perzeptueller Autopilot — Wohlklang-Garantien
- Kategorie XXI (§G142–§G145): Perzeptueller Closed-Loop — Per-Band-Hören
- Kategorie XXII (§G150–§G155): Metrik-Hierarchie & Guard-Disziplin
- Kategorie XXIII (§G156–§G166): Restorability-Gate & Bugfixes B19–B30 (§v10.704)
- Kategorie XII (§G87): Noise-Floor-Brücke
- Kategorie XIII (§G88): Defektbehebungs-Module
- Kategorie XIV (§G89): Unsichtbare Signalintegrität
- Kategorie XV (§G90–§G99): Non-Plus-Ultra
- Kategorie XVI (§G100–§G112): Perzeptuelle Architektur §v10.101
- VERBOTE (§V1–§V38): Absolute Verbote, gelten immer und überall

---

## Kategorie VI — Metriken & Qualitätssicherung (§G46–§G59)

| ID | Regel | Beschreibung |
| ---- | ------- | ------------- |
| §G46 | **Harmonic Preservation Score** | HNR-basierte Metrik. Detektiert Obertonschäden durch Überglättung. |
| §G47 | **Transient Preservation Score** | Crest-Faktor + Onset-Positionsabgleich. Detektiert Transienten-Verschleifung. |
| §G48 | **Formant Preservation Score** | Cepstrale Hüllkurvendistanz. Detektiert Vokalcharakter-Änderungen. |
| §G49 | **ABX Test Harness** | Double-Blind A/B/X mit Binomial-Signifikanztest. |
| §G50 | **MUSHRA Proxy Scorer** | 6-Dimensionen-Ensemble 0–100 Skala. |
| §G51 | **Statistical Report** | Binomialtest für Listening-Panel-Signifikanz. |
| §G52 | **Micro-Dynamics Score** | Crest-Faktor-Verteilung in 200ms-Fenstern. |
| §G53 | **Artifact Detector** | Clicks, Spectral Holes, Pre-Echo, Stereo-Anomalien. |
| §G54 | **Emotional Arc Score** | Lautheitskontur + Sektionskontrast + Spektralbewegung + Stille. |
| §G55 | **Blind Reference-Free Quality** | 6 Single-Ended-Features. Bewertet ohne Originalvergleich. |
| §G56 | **Noise Floor Continuity** | −20 dB Minimum-Floor. Verhindert Noise-Gate-Artefakte. |
| §G57 | **Sliding ERB Gain** | Multi-Segment-ERB-Maske. Adaptiert an spektrale Änderungen. |
| §G58 | **Vocal Repair Module** | Bandbreiten-Erweiterung + Verzerrungs-Reparatur vor Phase 42. |
| §G59 | **Restoration Quality Report** | Integriert alle Metriken in einen Aufruf. Blindtest-Readiness-Verdikt. |

---

## Kategorie VII — Stereo-Lag-Integrität (§G60–§G67)

> **Alle Erkenntnisse aus der Lag-Root-Cause-Analyse vom 2026-07-13.**
> 13 Commits, 8 Root Causes identifiziert und behoben.

| ID | Regel | Beschreibung |
| ---- | ------- | ------------- |
| §G60 | **STCG Multi-Point-Primär** | STCG MUSS Multi-Point-GCC-PHAT (≥3 Song-Positionen, Median) als PRIMÄRE Messmethode verwenden. Single-Mid-Window nur als Fallback bei Audio < 30s. |
| §G61 | **Chunk-Phasen-STCG-Pflicht** | Jede Chunk-basierte Phase (Phase 12, Phase 24 u.a.) MUSS für Lag-Erkennung und -Korrektur den zentralen STCG verwenden. Eigene Korrelations-Implementierungen (signal.correlate) sind VERBOTEN (§V27). |
| §G62 | **Sub-Sample-Lag-Korrektur** | Lag-Korrektur MUSS `scipy.ndimage.shift` (cubic spline, Sub-Sample-Präzision) oder STCG direkt verwenden. `np.roll` (zirkulär), `np.concatenate` (ganzzahlig), und Audio-Trunkierung sind VERBOTEN (§V32). |
| §G63 | **Lag-Messung-Orientierungsfrei** | Alle Lag-Messfunktionen MÜSSEN sowohl channels-first `(2, N)` als auch channels-last `(N, 2)` korrekt erkennen und messen. `arr.shape[0]` ohne Orientierungs-Check ist VERBOTEN (§V33). |
| §G64 | **STCG-Singleton-Konsistenz** | Alle Lag-Korrekturen MÜSSEN den zentralen STCG-Singleton verwenden. Keine ad-hoc GCC-PHAT-Reimplementierung in einzelnen Phasen. |
| §G65 | **Post-Chunk-Global-STCG** | Nach ABSCHLUSS aller Chunk-basierten Phasen MUSS ein globaler STCG-Check mit Multi-Point-Verifikation erfolgen. Per-Chunk-Korrekturen ohne globalen Abschluss sind VERBOTEN (§V28). |
| §G66 | **Keine konkurrierenden Lag-Fixes** | Nach einer erfolgreichen STCG-Korrektur darf KEINE zweite, unabhängige Lag-"Korrektur" (Onset-Energy-Fallback, manuelle np.concat) durchgeführt werden (§V29). Nur bei STCG-Fehlschlag ist ein Fallback erlaubt. |
| §G67 | **STFT-Input-Length-Guard** | Jeder Aufruf von `scipy.signal.stft` MUSS durch einen zentralen Längen-Guard geschützt sein, der `nperseg > input_length` abfängt. Der Guard ist in `backend/__init__.py` installiert. |

## Kategorie VIII — Neue VERBOTE Stereo-Lag (§V27–§V33)

| ID | Verbot | Beschreibung |
| ---- | -------- | ------------- |
| §V27 | **Kein signal.correlate für Lag** | Es ist VERBOTEN, `scipy.signal.correlate` (Standard-Kreuzkorrelation ohne PHAT-Whitening) für Stereo-Lag-Messung zu verwenden. Nur GCC-PHAT (via STCG) ist statthaft. |
| §V28 | **Kein begrenzter Lag-Suchraum** | Es ist VERBOTEN, den Lag-Suchraum für Stereo-Messungen auf < ±200ms (±9600 samples @48kHz) zu begrenzen. Kleinere Limits (z.B. 960 samples = 20ms) verfehlen echte Kanalversätze. |
| §V29 | **Keine konkurrierenden Lag-Korrekturen** | Es ist VERBOTEN, nach erfolgreicher STCG-Korrektur eine zweite Lag-"Korrektur" durchzuführen. Der Onset-Energy-Fallback in `_preserve_phase_loudness` ist NUR bei STCG-Exception aktiv. |
| §V30 | **Kein Single-Window-Lag** | Es ist VERBOTEN, Stereo-Lag nur an EINER Song-Position (z.B. Mid-Window 10s) zu messen, wenn die Song-Dauer > 30s beträgt. Multi-Point (≥3 Positionen) ist Pflicht. |
| §V31 | **Kein np.roll für Lag-Korrektur** | Es ist VERBOTEN, `np.roll` (zirkuläre Verschiebung mit Sample-Wrapping) für Stereo-Lag-Korrektur zu verwenden. Nur `scipy.ndimage.shift` (Zero-Padding, Sub-Sample) oder STCG sind statthaft. |
| §V32 | **Kein Audio-Trunkieren für Lag** | Es ist VERBOTEN, Audio zu trunkieren (`audio[:, :N - lag]`), um Lag zu korrigieren. Die Korrektur MUSS die Originallänge durch Zero-Padding erhalten. |
| §V33 | **Kein shape[0] ohne Orientierungs-Check** | Es ist VERBOTEN, `audio.shape[0]` als Sample-Anzahl zu interpretieren, ohne vorher zu prüfen ob `(2,N)` oder `(N,2)` vorliegt. Die Multi-Point-Funktion MUSS beide Orientierungen unterstützen. |

## Kategorie IX — SFT-Adaptivität & Defekt-Audibilität (§G68–§G75)

| ID | Regel | Beschreibung |
| ---- | ------- | ------------- |
| §G68 | **SFT-Novelty-Schwelle adaptiv pro Song** | Die NOVELTY_CRIT-Schwelle MUSS pro Song aus Transfer-Chain-Tiefe und Restorability-Tier kalibriert werden (§v10.40). Statische Schwellen sind VERBOTEN — ein fair-quality Kassette-Song mit 4-stufiger Kette hat fundamental andere Neuheits-Erwartungen als ein excellent Studio-Master mit 1-stufiger Kette. |
| §G69 | **Defekt-Reparatur-Phasen-Klassifikation** | Jede Phase, die Defekte füllt/ersetzt/repariert (nicht nur entfernt), MUSS als Repair-Phase klassifiziert sein. Die Klassifikation steuert SFT-Wet-Minimum und Strength-Floor. Folgende Phasen sind MINDESTENS Repair: 01, 02, 09, 12, 23, 24, 27, 50, 56, 60, 61, 64. |
| §G70 | **SFT-Prioritätskette: Zerstörung vor Neuheit** | Die SFT-ArtifactRescue MUSS in dieser Reihenfolge prüfen: LEVEL_COLLAPSE (wet=0.0) → ECHO_ARTIFACT (wet=0.30) → PEGELEXPLOSION_CRIT (wet=0.22) → NOVELTY_CRIT (adaptiv). LEVEL_COLLAPSE hat ABSOLUTEN Vorrang — zerstörtes Audio darf NIEMALS in die Pipeline getragen werden. |
| §G71 | **Unhörbare Defekte als Qualitätsziel** | Transport Bumps, Tape Head Level Dips und alle anderen chirurgischen Defekte MÜSSEN nach der Restaurierung für das menschliche Ohr unhörbar sein. Die effektive Reparatur-Wirkung (strength × SFT-wet) muss ≥ 0.15 betragen — darunter ist der Defekt hörbar. |
| §G72 | **Keine pauschalen Wet-Werte** | Es ist VERBOTEN, SFT-Wet-Werte pauschal für alle Songs zu setzen. Die Wet-Werte sind Sicherheitsnetze für Phasen, die die adaptiv kalibrierte NOVELTY_CRIT-Schwelle überschreiten. Die primäre Steuerung erfolgt über die Schwelle, nicht über die Wet-Werte. |
| §G73 | **Joint-Calibration Minimum** | Die minimale Phasen-Stärke (min_strength) MUSS ≥ 0.20 betragen. Phasen mit utility ≤ 0.001 (durch Codec-Diskont oder kleine Goal-Gaps) erhalten sonst keine messbare Wirkung. PROTECTED_PHASES MÜSSEN mindestens 0.35 Floor haben. |
| §G74 | **OneTakeExport-Garantie** | Jeder Export MUSS nach spätestens 5 Auto-Korrektur-Versuchen erfolgreich sein. Der letzte Versuch MUSS eine Gain-Reduktion (−0.5 dB) VOR dem Limiter anwenden, um Inter-Sample-Peaks garantiert zu eliminieren. Ein Export-FAIL wegen True Peak ist VERBOTEN. |
| §G75 | **Tuple-ndim Recovery** | Wenn eine Phase einen `'tuple' object has no attribute 'ndim'` Fehler wirft (Post-Processing-Typfehler, Phase-Logik war korrekt), MUSS die Phase als executed markiert werden — nicht als skipped. Der Audio-Stand bleibt auf dem Pre-Phase-Wert (Phase-Logik lief ja korrekt). |

## Kategorie X — Kalibrierungs-Dispatch: Zentrales Nervensystem (§G76–§G81)

| ID | Regel | Beschreibung |
| ---- | ------- | ------------- |
| §G76 | **Zentraler Kalibrierungs-Kontext** | Es MUSS einen einzigen, zentralen `CalibrationContext` geben, der ALLE Pre-Analysis-Messwerte (restorability_score, transfer_chain_depth, material_type, SNR, bandwidth, era_decade, genre, vocal_confidence) in EINEM Objekt bündelt. JEDES Modul, das einen Schwellwert benötigt, MUSS diesen Kontext als Quelle verwenden — NIE eine eigene Konstante. |
| §G77 | **Kontinuierliche Ableitung** | JEDER Schwellwert MUSS über eine kontinuierliche Funktion aus dem CalibrationContext abgeleitet werden. Die Funktion MUSS für jeden kontinuierlichen Eingabewert einen kontinuierlichen Ausgabewert liefern. Es ist VERBOTEN, diskrete Buckets (`if x > 0.4: ... elif x > 0.25: ...`) oder Lookup-Tabellen (`{1:0.25, 2:0.35}`) zu verwenden. |
| §G78 | **Vollständigkeit der Kalibrierung** | ALLE Schwellwerte, Caps, Floors und Blend-Faktoren in der gesamten Pipeline MÜSSEN kalibriert sein. Kein Parameter darf auf einem nicht aus dem CalibrationContext abgeleiteten Default verharren. Ausnahme: Physikalische Konstanten (z.B. −60 dBFS = digital black, −0.3 dBTP = ITU-R BS.1770 Ceiling). |
| §G79 | **Kalibrierungs-Audit** | Jeder kalibrierte Schwellwert MUSS im Log dokumentiert werden: `"§CALIB %s: rs=%.0f depth=%d → %s=%.4f"`. Dies ermöglicht die Rückverfolgbarkeit jeder Entscheidung auf Auriks eigene Messwerte. |
| §G80 | **Unkalibrierter-Fallback-Warnung** | Wenn ein Schwellwert nicht aus dem CalibrationContext abgeleitet werden kann (z.B. weil die Pre-Analysis noch nicht abgeschlossen ist), MUSS ein Default verwendet werden — aber NUR mit einer WARNING: `"⚠️ uncalibrated fallback: %s=%.4f (reason: %s)"`. Unkalibrierte Fallbacks sind als technische Schuld zu behandeln. |
| §G81 | **Einzige Quelle der Wahrheit** | Der CalibrationContext ist die EINZIGE Quelle für alle Schwellwerte. Wenn zwei Module unterschiedliche Werte für denselben Parameter berechnen, ist das ein Architekturfehler. Die Kalibrierungs-Matrix (`calibration_matrix.py`) ist der zentrale Berechnungspunkt — Module rufen ab, sie berechnen nicht selbst. |

## Kategorie XI — Laufzeit-Rekalibrierung (§G82–§G86)

> **Prämisse:** Die Pre-Pipeline-Kalibrierung basiert auf Messwerten des DEGRADIERTEN Eingangssignals. Während der Pipeline verbessert sich das Audio jedoch — SNR steigt, Bandbreite wächst, Defekte verschwinden. Eine Kalibrierung, die nach Phase 03 (denoise) noch mit dem ursprünglichen SNR rechnet, ist FALSCH. Die Pipeline MUSS ihre Sicherheitsparameter kontinuierlich an den verbesserten Audio-Zustand anpassen.

| ID | Regel | Beschreibung |
| ---- | ------- | ------------- |
| §G82 | **Lebendiger CalibrationContext** | Der CalibrationContext ist NICHT statisch. Nach JEDER Phase MUSS Aurik prüfen, ob sich die für die Kalibrierung relevanten Messwerte (SNR, Bandbreite, Noise-Floor, Stereo-Kohärenz) signifikant geändert haben. Bei Änderung > Schwellwert MUSS der CalibrationContext aktualisiert und ALLE davon abhängigen Parameter neu berechnet werden. |
| §G83 | **NOVELTY_CRIT-Rekalibrierung** | Die NOVELTY_CRIT-Schwelle MUSS nach jeder signifikanten Audio-Verbesserung (SNR +3 dB, Bandbreite +1 kHz) NEU berechnet werden. Ein saubereres Signal rechtfertigt eine NIEDRIGERE Toleranz — was vorher „erwartete Neuheit" war, ist jetzt „verdächtige Veränderung". Die Formel bleibt dieselbe (§v10.41), aber die Eingabewerte (insbesondere restorability_score und effektive Bandbreite) sind die AKTUELLEN, nicht die initialen. |
| §G84 | **Phasen-Stärke-Drift-Korrektur** | Die Joint-Calibration berechnet Phasen-Stärken aus Goal-Gaps. Nach jeder Phase ändern sich die Goal-Proxies. Die Stärken der VERBLEIBENDEN Phasen MÜSSEN aus den AKTUELLEN Goal-Gaps neu berechnet werden — nicht aus den initialen. Der MidCalibrate-Mechanismus (33%/66%) ist ein MINIMUM — kritische Parameter (NOVELTY_CRIT, ECHO_THRESH) müssen nach JEDER Phase geprüft werden. |
| §G85 | **Rekalibrierungs-Audit** | Jede Rekalibrierung MUSS im Log dokumentiert werden: `"§RECALIB phase=%s: rs %.1f→%.1f SNR %.1f→%.1f dB → NOVELTY_CRIT %.3f→%.3f"`. Dies macht sichtbar, WIE sich Auriks Sicherheitsparameter während der Pipeline an das zunehmend sauberere Audio anpassen. |
| §G86 | **Monotonie-Garantie** | Die NOVELTY_CRIT-Schwelle darf während der Pipeline NUR sinken (konservativer werden) oder gleich bleiben — NIE steigen. Ein saubereres Signal rechtfertigt keine LASCHERE Toleranz. Die Monotonie MUSS im CalibrationContext erzwungen werden: `_NOVELTY_CRIT = min(current_calculation, previous_value)`. |

## Kategorie XI-b — Maschinelle Durchsetzung (§G122–§G124)

> **Prämisse:** Die GEBOTE der Kategorien X und XI formulieren architektonische Wahrheiten. Aber ohne maschinelle Durchsetzung sind sie Appelle. Jeder Default-Parameter `transfer_chain_depth: int = 1` ist eine Verletzung von §G76 und §G78, die darauf wartet, bei der nächsten Refaktorierung stillschweigend zuzuschlagen (§v10.131).

| ID | Regel | Beschreibung |
| ---- | ------- | ------------- |
| §G122 | **CalibrationContext-Dataclass** | Es MUSS eine einzige, zentrale `CalibrationContext`-Dataclass in `backend/core/calibration_context.py` geben, die ALLE Pre-Analysis-Messwerte als Felder deklariert. JEDE Funktion, die einen Schwellwert berechnet, MUSS einen `CalibrationContext` (oder die benötigten Einzelfelder daraus) als explizites Argument erhalten. Der Default `=1` für `transfer_chain_depth` ist AUSSCHLIESSLICH in dieser Dataclass erlaubt (§G76, §G78). |
| §G123 | **Linter-Baseline** | Ein automatisierter Test (`test_calibration_context_linter.py`) MUSS den gesamten Code auf verbotene Default-Parameter (`transfer_chain_depth: int = 1`) scannen und gegen eine Baseline-Datei abgleichen. NEUE Verstöße (nicht in der Baseline) lassen den Test FEHLSCHLAGEN. Die Baseline wird bei bewusster Schuldenreduktion aktualisiert. Dies verhindert, dass der nächste Refactor denselben Fehler neu einführt. |
| §G124 | **Cross-Depth-Validierung** | Ein parametrisierter Test (`test_cross_depth_validation.py`) MUSS für JEDE Chain-Depth (1–5) und JEDES Material validieren, dass ALLE depth-abhängigen Schwellwerte (GDD, artifact_freedom, REGRESSION_THRESHOLD) physikalisch plausible, monotone Werte liefern. Keine Depth-Stufe darf eine LASCHERE Toleranz haben als die vorherige. Der Test MUSS nach jeder Änderung an chain_factor-Formeln ausgeführt werden. |
| §G125 | **Kalibrierte Konstanten** | ALLE Schwellwerte, Caps, Floors und Blend-Faktoren MÜSSEN in `calibrated_constants.py` als Properties der `CalibratedConstants`-Klasse definiert sein. JEDE Property MUSS ihren Wert AUSSCHLIESSLICH aus dem übergebenen `CalibrationContext` ableiten. Numerische Literale sind NUR in dieser Datei erlaubt. Module importieren `get_constants(ctx)` statt eigene Konstanten zu definieren. |
| §G126 | **Blindtest-Pflicht** | JEDE Änderung an einer chain_factor-Formel oder einem depth-abhängigen Schwellwert MUSS durch einen automatisierten Blindtest (`blindtest_framework.py`) validiert werden. Der Test MUSS zeigen, dass die neue Formel auf synthetisch degradiertem Material eine messbare Verbesserung (PESQ ≥ 0.05 Zuwachs) gegenüber der alten Formel erzielt. Ohne diesen Nachweis darf die Formel nicht geändert werden. |

## Kategorie XII — Noise-Floor-Brücke Phase_03→Phase_26 (§G87)

| ID | Regel | Beschreibung |
| ---- | ------- | ------------- |
| §G87 | **Phase_26 Per-Band-Noise-Floor-Guard** | Phase_26 (DR-Expansion) MUSS die Lücke zwischen Phase_03 (Denoise) und dem finalen CD-Rauschprofil schließen. Die Downward-Expansion wird durch einen dreidimensionalen Guard kontrolliert: **(D1) Per-Band spektrale Floor-Targets**: Jedes der 4 Frequenzbänder hat einen eigenen Studio-Raumton — Bass −65 dBFS (Raumresonanz), Low-Mid −72 dBFS (Wärme), Mid-High −76 dBFS (Präsenz), High −70 dBFS (Luft). **(D2) Psychoakustische Maskierung**: Der Floor wird adaptiv um +8/+5/+2/0 dB relaxiert, wenn die Band-Energie > −20/−30/−40 dBFS beträgt — laute Bänder maskieren ihren eigenen Rauschboden, leise exponierte Bänder sind streng. **(D3) Temporale EMA-Glättung**: Floor-Anstieg (Entspannung) folgt mit α=0.15 (Attack ~50ms), Floor-Abfall (Verschärfung) mit α=0.05 (Release ~200ms). Kein Hard-Clamp — der Floor-Approach ist asymptotisch (correction = deficit × exp(−deficit/knee), knee=4 dB). Ergebnis: klingt nach Neuaufnahme, nicht nach Vinyl mit aufgezwungener CD-Stille. |

## Kategorie XIII — Defektbehebungs-Module auf höchster Qualitätsstufe (§G88)

| ID | Regel | Beschreibung |
| ---- | ------- | ------------- |
| §G88 | **Defektbehebung mit Depth-adaptiven DSP-Fallbacks** | Die vier Defektbehebungs-Module MÜSSEN bei transfer_depth≥5 (extreme chain, §v10.120) und/oder unsicherer Gender-Detektion robuste, konservative DSP-Fallbacks verwenden — NIEMALS ungeprüfte ML-Inferenz auf extrem degradierten Ketten oder gender-spezifische Annahmen ohne Fallback. **(1) Phase_07 Harmonic Restoration**: Tilt-Cap-Floor von 0.50 auf 0.35 absenken bei depth≥5 (§v10.60, §v10.120). Mehr harmonische Synthese durchlassen, da extreme Ketten extreme Tilt-Abweichungen ohnehin erwarten. **(2) Phase_23 Spectral Repair**: FlashSR ML deaktivieren bei depth≥5 (§v10.60, §v10.120). ML halluciniert Frequenzen auf bereits 5× degradiertem Material. DSP-only spectral inpainting (PGHI + Wiener + NMF) ist robuster. Depth 4 (deep cassette, Novelty 0.55) bekommt volle ML-Repair. **(3) Phase_19 De-Esser**: Bei Gender="unknown"/"" freq-agnostisches Band [4500–8000 Hz] statt gender-spezifischem Band (§v10.60). Verhindert Fehlklassifikation von männlichen Stimmen als weiblich (und umgekehrt) mit konsekutiver Über-/Unterbearbeitung. **(4) Phase_43 ML-DeEsser**: GENDER_FREQ_MAP["unknown"] = (5000, 9000 Hz) als konservativer Fallback (§v10.60). Breiteres, tieferes Band als gender-spezifische Bänder — fängt Sibilanz sicher ein, vermeidet aber Überbearbeitung. |

## Kategorie XIV — Unsichtbare Signalintegrität (§G89)

| ID | Regel | Beschreibung |
| ---- | ------- | ------------- |
| §G89 | **Soft-Clipping-Pflicht für alle 71 Phasen (68 + 3 Phase-0)** | Jede Phase MUSS ihre Ausgabe via `apply_soft_clip()` (tanh-basiert, material-adaptiv) statt `np.clip(audio, -1.0, 1.0)` begrenzen (§v10.62). Hard-Clipping auf ±1.0 erzeugt ein Rechteck-Fenster im Zeitbereich → sinc-Spektrum mit hörbaren Obertönen bis Nyquist. Tanh-Soft-Clipping erzeugt nur ungerade Harmonische, die das Ohr als „analoge Sättigung" statt „digitalen Clip" wahrnimmt. Die zentrale Durchsetzung erfolgt in `PhaseResult.__post_init__` und `create_phase_result()` — damit sind alle Phasen-Ausgaben automatisch geschützt. Material-adaptive Knee: Shellac/Vinyl 1.2 dB, Tape/Cassette 0.8 dB, Digital 0.4 dB. |

## Kategorie XVIII — Startup-Integration & Kommunikation (§G71–§G80 spec_constitution, §v10.305)

| ID | Regel | Beschreibung |
| ---- | ------- | ------------- |
| §SC-G71 | **Event-Garantie** | Jedes `threading.Event` MUSS in `finally` oder garantiertem Exception-Handler gesetzt werden. Kein Codepfad darf das Event ungesetzt lassen. Betrifft: `_detection_complete` in `MLDeviceManager.__init__`. |
| §SC-G72 | **Lock-freie Importe** | `threading.Lock` DARF NICHT während `import`-Statements gehalten werden. Importe dauern 5–10s (pkg_resources, webrtcvad). Währenddessen sind alle anderen Lock-Warter blockiert. Betrifft: `try_allocate` in `ml_memory_budget.py`. |
| §SC-G73 | **Plugin-Namen-Validierung** | Jeder Zugriffsname in `warmup_models_background._plugins` MUSS mit einer tatsächlichen Funktion im Zielmodul übereinstimmen. `_failed > 0` nach Warmup MUSS ein `logger.warning` auslösen. Betrifft: `bridge.py`. |
| §SC-G74 | **Watchdog-Selbsttest** | Jeder Watchdog MUSS prüfen, dass seine Aktivierungsbedingung tatsächlich erreichbar ist. `getattr(self, "_preanalysis_pending", False)` muss auf `self` gesetzt sein. Betrifft: `_preanalysis_liveness_check`. |
| §SC-G75 | **Cache-Safety** | Launcher MUSS mit `python3 -B` starten. `.pyc`-Caches können nach Source-Änderungen veralteten Code ausführen. Betrifft: `main.py`, Shell-Launcher. |
| §SC-G76 | **Happy-Path-Gate** | Mindestens ein Codepfad MUSS die Analyse-Labels (`detected_medium_label`, `restorability_banner`, `mode_recommendation_label`) setzen. Jeder Guard, der einen Pfad verwirft, MUSS ein `logger.warning` ausgeben. Betrifft: `_update_all`. |
| §SC-G77 | **Startup-Smoke-Test** | Ein schneller (<60s) Test MUSS GPU-Erkennung, Warmup und Pre-Analysis prüfen. Betrifft: `tests/test_startup_smoke.py` (6 Assertions). |
| §SC-G78 | **Import-Check** | Jedes Modul MUSS alle verwendeten Standard-Imports haben. `ruff F821` (undefined name) ist Null-Toleranz. Betrifft: `ml_device_manager.py` (`import os` fehlte). |
| §SC-G79 | **GPU-Detection Safety** | GPU-Erkennung DARF KEINE blockierenden Operationen ausführen. `torch.zeros(device="cuda")` ist VERBOTEN. Nur Properties: `is_available()`, `version.hip`, `get_device_properties()`. Betrifft: `ml_device_manager.py._detect_cuda_or_rocm`. |
| §SC-G80 | **Unified Progress** | Beide Fortschrittsbalken MÜSSEN aus derselben Methode (`_sync_unified_progress()`) aktualisiert werden. Fragmentierte Update-Pfade sind VERBOTEN. Defekt-Counts in Chips alle ~800ms refreshen. Betrifft: `_tick_heartbeat`. |

---

---

## Kategorie XXII — Architektonische Qualitäts-Garantien: Metrik-Hierarchie & Guard-Disziplin (§G150–§G155)

> §v10.704 — Prämisse: 20 Bugs in der Produktionsanalyse haben drei systemische
> Schwachstellen offengelegt: (1) Guards widersprechen sich mangels Metrik-Hierarchie,
> (2) Guards kennen den Phasen-Kontext nicht, (3) Schwellwerte sind hartcodiert statt
> adaptiv. Diese Kategorie kodifiziert die architektonischen Heilungsmaßnahmen.

| ID | Regel | Beschreibung |
| ---- | ------- | ------------- |
| §G150 | **Metrik-Hierarchie-Pflicht** | Aurik MUSS eine definitive Hierarchie der Qualitätsmetriken einhalten: Priorität 1 = MUSHRA/HPI (perzeptuell), Priorität 2 = Defekt-Reduktion (B2-Daten), Priorität 3 = artifact_freedom (nur VETO bei >5 Artefakten), Priorität 4 = BlindQuality (NUR Diagnostik, NIE Quality-Gate). Keine niedrigere Metrik darf das Urteil einer höheren überschreiben. |
| §G151 | **MUSHRA-Primat** | Wenn MUSHRA > 0 (perzeptuelle Daten verfügbar), MUSS MUSHRA die primäre Ground Truth für ALLE Quality-Gate-Entscheidungen sein. artifact_freedom und BlindQuality dürfen NIEMALS ein positives MUSHRA-Urteil negieren. |
| §G152 | **BlindQuality-Diagnostik-Verbot** | Der BlindQualityScore (SNR/THD/Bandbreite) darf AUSSCHLIESSLICH für technische Diagnostik verwendet werden. Er darf NIEMALS in `quality_guaranteed`, `verdict` oder `export_gate` einfließen. Verstoß → §V40 (bestehend). |
| §G153 | **Guard-Phasen-Whitelist-Pflicht** | Jeder Quality-Guard (AFG, VocalNoHarm, FormantGuard, CIG, SFT, PMGG) MUSS deklarieren, für welche Phasen-Familien er zuständig ist. Eine Phase, deren Familie nicht in der Whitelist des Guards steht, wird von diesem Guard ÜBERSPRUNGEN. Kein Guard läuft盲 auf allen Phasen. |
| §G154 | **Adaptive-Schwellwert-Pflicht** | Jeder Schwellwert in Quality-Gates MUSS aus Material-Typ, Transfer-Chain-Depth und Phasen-Familie ABGELEITET werden — NIEMALS hartcodiert. Formel: `threshold = base_threshold × material_factor × depth_factor × phase_factor`. Die Tabelle der Material/Depth/Phase-Faktoren MUSS zentral in `calibrated_constants.py` definiert sein. |
| §G155 | **Quality-Entscheidungs-Narrativ-Pflicht** | Jede Quality-Gate-Entscheidung (Verdict, Rollback, Skip) MUSS im GUI-Narrativ BEGRÜNDET werden — mit Bezug auf die konkrete Metrik, den Schwellwert und die Phasen-Familie. Kein „NO IMPROVEMENT" ohne Erklärung. Kein Rollback ohne „Warum". Der Nutzer MUSS verstehen, WARUM Aurik so entschieden hat. |

### Neue VERBOTE — Architektonische Qualitäts-Garantien (§V47–§V49)

| ID | Verbot | Beschreibung |
| ---- | -------- | ------------- |
| §V47 | **Metrik-Unterordnung-Verbot** | Es ist VERBOTEN, dass eine niedrigprioritäre Metrik (artifact_freedom, BlindQuality) das Urteil einer höherprioritären Metrik (MUSHRA, HPI) überschreibt oder negiert. |
| §V48 | **Guard-Kontext-Ignoranz-Verbot** | Es ist VERBOTEN, einen Quality-Guard auf einer Phase laufen zu lassen, deren Phasen-Familie nicht in der Whitelist des Guards deklariert ist. |
| §V49 | **Hartcodierter-Schwellwert-Verbot** | Es ist VERBOTEN, einen Quality-Gate-Schwellwert hart zu codieren. Alle Schwellwerte MÜSSEN via `calibrated_constants.py` aus Material+Depth+Phase abgeleitet werden. |

## Kategorie XXIII — Restorability-Gate & Bugfixes B19–B30 (§G156–§G166)

> §v10.704 — Prämisse: Zwei Produktionsläufe mit 5-stufiger Transferkette
> (restorability=66) zeigten eine systematische Failure-Kaskade: De-Esser produzierte
> artifact_freedom=0.494 → HPI-Gate verwarf gesamte Restauration → 0 Phasen wirksam.
> Sechs Root-Cause-Fixes (B19–B30) durchbrechen diese Kaskade.

| ID | Regel | Beschreibung |
| ---- | ------- | ------------- |
| §G156 | **Depth+Restorability-adaptiver HPI-Gate** | Der artifact_freedom-Mindestwert MUSS aus Depth UND Restorability abgeleitet werden: `af_min = base(depth) − (100−rs) × 0.0045`, floor=0.32. Depth≥5→0.48, depth≥4→0.58, depth=3→0.72, depth=2→0.82, depth=1→0.88. (§v10.704 B30) |
| §G157 | **Sample-Axis-Robustheit für B3-Phase-2** | Der Defect-Presence-Scan MUSS beide Audio-Layouts korrekt erkennen: (2,N) und (N,2). `shape[-1]` bei (N,2) = Kanäle, nicht Samples. (§v10.704 B26) |
| §G158 | **MUSHRA/HPI-Forwarding an MQA** | MushraEvaluator und HPI MÜSSEN ihre Scores an MQA weiterleiten. Ohne Forwarding: MQA sieht 0 → "NO IMPROVEMENT" trotz OQS=84.0. (§v10.704 B27) |
| §G159 | **De-Esser-Dynamics-Threshold** | AFG-Threshold für De-Esser: 0.40 (Schwelle 0.38). Alte 0.55 erzwang Rollback trotz unhörbarer Artefakte auf Kassettenmaterial. (§v10.704 B19) |
| §G160 | **Chunked-Mode-Längenwarnung** | Post-Pipeline-Check MUSS `self._in_chunked` erkennen und bei >100k Samples Diff von WARNING auf DEBUG herabstufen. (§v10.704 B28) |
| §G161 | **P5-Exception-Traceback-Pflicht** | "setting an array element"-Exceptions MÜSSEN mit vollständigem Traceback (2000 Zeichen) als WARNING geloggt werden. (§v10.704 B29) |
| §G162 | **HPI-Gate-Restorability-Kontinuität** | `(100−rs) × 0.0045`. Diskrete Buckets sind VERBOTEN. (§G77, §v10.704 B30) |
| §G163 | **Floor-Absolut-Garantie** | artifact_freedom-Mindestwert NIEMALS < 0.32. (§v10.704 B30) |
| §G164 | **Studio-Master-Floor-Invariante** | depth=1, rs=100: af_min UNVERÄNDERT 0.88. (§v10.704 B30) |
| §G165 | **Spec-Constitution-Synchronisation** | Jede AF/HPI-Änderung MUSS in `check_violations`, `is_export_blocked` UND `_get_depth_adaptive_af_min` nachgezogen werden. (§v10.704) |
| §G166 | **Drei-Quellen-Synchronisation** | AF/HPI-Schwellwerte existieren an DREI Orten. Alle MÜSSEN identische Formeln verwenden. (§v10.704) |

### Neue VERBOTE — Restorability-Gate (§V50–§V51)

| ID | Verbot | Beschreibung |
| ---- | -------- | ------------- |
| §V50 | **Restorability-Ignoranz-Verbot** | Es ist VERBOTEN, einen HPI-Gate-Schwellwert ohne `restorability_score` zu berechnen. |
| §V51 | **Sample-Axis-Raten-Verbot** | Es ist VERBOTEN, `audio.shape[-1]` als Sample-Anzahl zu interpretieren, ohne (N,2) vs (2,N) zu prüfen. |

### Neue GEBOTE — Denker-IQ & Material-Awareness (§G167–§G172, §v10.706)

| ID | Gebot | Beschreibung |
| ---- | ------- | ------------- |
| §G167 | **Export-Gate-B30-Komplettierung** | ALLE Export-Gates (ArtifactFreedomGate, FinalExportAudioGate, QualityGateRegistry) MÜSSEN die B30-Formel `af_min = base(depth) − (100−rs)×0.0045` verwenden. Hartcodierte 0.95 sind VERBOTEN. |
| §G168 | **SourceMediumProfile-Kalibrierungspflicht** | SongCalibration MUSS `get_medium_profile()` konsultieren. `is_compressed`, `hiss_reduction_max_strength`, `harmonic_max_order`, `deesser_skip_saturation_conf` fließen in `family_scalars` ein. |
| §G169 | **Per-Phase-SMP-Cap** | Jede Phase mit medium-spezifischen Limits MUSS `SourceMediumProfile` als Hard-Cap konsultieren: Phase_29 (`hiss_reduction_max_strength`), Phase_07 (`harmonic_max_order`), Phase_26 (`is_compressed`), Phase_36 (`has_soft_saturation`), Phase_54 (`is_compressed`). |
| §G170 | **Chain-Depth-Budget-Adaption** | StrategieDenker MUSS das Zeitbudget mit `chain_depth` skalieren: +15% pro Generation, +0.5% pro Restorability-Punkt unter 100. |
| §G171 | **Material-Fremdlauf-Transparenz** | Läuft eine für Medium X designte Phase auf Medium Y (durch Chain-Injection), MUSS dies als INFO geloggt werden. Kein Block, kein Strip — ActiveIntervention entscheidet. |
| §G172 | **OneTakeExport-ISP-Margin** | Brickwall-Ceiling MUSS −1.0 dBTP betragen (ISP-Margin für inter-sample peaks). −0.3 dBTP ist VERBOTEN — TruePeak-Messung erfasst ISP, die der Sample-Peak-Limiter nicht eliminiert. |

### Neues VERBOT — Material-Awareness (§V52)

| ID | Verbot | Beschreibung |
| ---- | -------- | ------------- |
| §V52 | **Material-Nivellierungs-Verbot** | Es ist VERBOTEN, `"tape"`, `"cassette"` und `"reel_tape"` in Kalibrierung oder Phasen-Logik identisch zu behandeln. Jedes Medium hat eigene physikalische Limiten via SourceMediumProfile. |

## Änderungshistorie

| Version | Datum | Änderung |
| --------- | ------- | ---------- |
| 10.0.13 | 2026-08-03 | §G89: Soft-Clipping-Pflicht für alle 71 Phasen (68 + 3 Phase-0) (§v10.62). `apply_soft_clip()` + `crossfade_to_bypass()` in audio_utils.py. Kategorie XIV. |
| 10.0.12 | 2026-08-03 | §G88: Defektbehebungs-Module (Phase_07/19/23/43) mit Depth-adaptiven DSP-Fallbacks. Kategorie XIII. |
| 10.0.11 | 2026-08-03 | §G87: Phase_26 Per-Band-Noise-Floor-Guard (D1–D3). Schließt Phase_03→Phase_26 Noise-Floor-Lücke. Kategorie XII. |
| 10.0.10 | 2026-07-19 | §G82–§G86: Laufzeit-Rekalibrierung. Lebendiger CalibrationContext, NOVELTY_CRIT-Nachführung, Monotonie-Garantie. Kategorie XI. |
| 10.0.9 | 2026-07-19 | §G76–§G81: Kalibrierungs-Dispatch. Zentraler CalibrationContext, kontinuierliche Ableitung aller Schwellwerte, Kalibrierungs-Audit. Kategorie X. |
| 10.14.0 | 2026-07-19 | §G68–§G75: SFT-Adaptivität, Defekt-Audibilität, Repair-Klassifikation. Kategorie IX. |
| 10.0.7 | 2026-07-13 | §G60–§G67 + §V27–§V33. Lag-Integritäts-Architektur nach Root-Cause-Analyse (8 Bugs, 13 Commits). Kategorie VII + VIII. |
| 10.0.6 | 2026-07-13 | §G46–§G59 (Metriken & Qualitätssicherung). Kategorie VI. |
| 10.0.5 | 2026-07-13 | §G30–§G39 (CD-Rauschprofil & Export, ML-Device, Test-Assertion). §V16–§V24. |
| 10.0.4 | 2026-07-13 | Initiale Formalisierung. CD-Rauschprofil (§G8, §G15–§G19, §V5, §V11–§V15). Kategorie I–III strukturiert. |

---

## Kategorie XV — Non-Plus-Ultra: Strukturelle Qualitäts-Deckel beseitigt (§G90–§G99)

> **Prämisse:** Die vier unabhängigen Root-Causes für „43→43" (keine messbare Qualitätsverbesserung) sind identifiziert und behoben. Diese Kategorie kodifiziert die architektonischen Garantien, die verhindern, dass Aurik jemals wieder gegen den defekten Input vergleicht, Exception-Schlucker ohne Logging verwendet oder Phasen ohne Cross-Phase-Koordination laufen.

| ID | Regel | Beschreibung |
| ---- | ------- | ------------- |
| §G90 | **Blinder-Referenz-Vektor-Pflicht** | Der HPI MUSS einen blinden Referenz-Vektor (Mel-Embedding des saubersten 5s-Fensters via BlindInternalReference) als timbral_ref verwenden, wenn der GP-Memory keinen Referenz-Vektor für die aktuelle Genre×Material×Ära-Kombination hat. Es ist VERBOTEN, `reference_audio=None` still auf `original` (degraded_input) zurückfallen zu lassen, ohne mindestens den blinden Vektor versucht zu haben. (§v10.91, `holistic_perceptual_gate.py:_compute_blind_reference_vector`) |
| §G91 | **Embedding-basierte-Referenz-Pflicht** | Audio-Referenzen für den HPI-Vergleich MÜSSEN als Embedding-Vektoren verwendet werden, NICHT als direkte Audio-Samples. Ein 5s-Audio-Slice als Vergleichsreferenz erzeugt Shape-Mismatch mit dem vollständigen restaurierten Audio (3–5 Min) → falsche Mel-Cosinus-Werte und Spektral-Proxies. (§v10.91) |
| §G92 | **Material-adaptive-Confidence-Pflicht** | Die Confidence in `feasibility_controller.estimate_goal_feasibility()` MUSS `predict_quality_score()` aus `calibration_matrix` verwenden — KEINEN harten 0.95-Deckel. Shellac (Ceiling 0.70) erhält proportional niedrigere Confidence als CD (Ceiling 0.95). (§v10.92) |
| §G93 | **Exception-Proxy-Pflicht** | Jeder `return 0.5`-Exception-Fallback in scoring-Funktionen MUSS durch einen Zeitdomain-Proxy ersetzt werden, der aus den verfügbaren Daten eine informierte Schätzung ableitet. Mindestens: `logger.warning(...)` mit `exc_info=True` VOR dem Fallback. Harte 0.5-Defaults ohne Logging sind VERBOTEN. (§v10.92, §v10.93) |
| §G94 | **Cross-Phase-Metadata-Pflicht** | Phasen, die auf denselben Frequenzbändern operieren, MÜSSEN ihre Ergebnisse via `_restoration_context` teilen. Konkret: **(a)** P02 (Hum-Removal) MUSS `hum_notch_freqs` (detektierte Grundfrequenzen) via `_restoration_context` an P37 (Bass-Enhancement) übergeben. P37 MUSS `sub_harmonic_gain` proportional zur Überlappung reduzieren. **(b)** P10 (Compression) MUSS `per_band_gain_db` (max_gain_reduction pro Band) via `_restoration_context` an P26 (Dynamic-Range-Expansion) übergeben. P26 MUSS `max_expansion_db` proportional zur P10-Kompression reduzieren. (§v10.94) |
| §G95 | **Phase-02-vor-Phase-03-Pflicht** | Der Phase-DAG MUSS `HARD_BEFORE(phase_02_hum_removal, phase_03_denoise)` deklarieren. P03 (ML-Denoising) trainiert auf dem Eingangssignal — ohne vorherige Hum-Entfernung lernt das ML-Modell 50/60-Hz-Brumm + Harmonische als „Nutzsignal" und entfernt Musikinhalt in den betroffenen Bändern. (§v10.94, `phase_dag.py`) |
| §G96 | **HPI-NaN-Guard-Pflicht** | Der HPI-Produkt-Term (`mert_sim * timbral * artifact_freedom * emotional_arc`) MUSS durch `np.nan_to_num` VOR `max(..., 0.5)` geschützt werden, da `max(nan, 0.5) == nan` in Python. Zusätzlich MUSS das finale HPI-Produkt via `np.isfinite()` geprüft und bei NaN/Inf auf Floor 0.5 gesetzt werden — mit explizitem Warning-Log aller vier Faktor-Werte. (§v10.93) |
| §G97 | **log10-Null-Guard-Pflicht** | Jede `np.log10(x)`-Verwendung in der Quality-Evaluation-Pipeline MUSS durch `max(x, 1e-10)` geschützt werden, wenn `x` aus `np.percentile()` oder anderen Funktionen stammt, die bei Stille/Leersignal 0.0 zurückgeben können. (§v10.93, `excellence_optimizer.py`, `difficulty_estimator.py`) |
| §G98 | **AUTHENTIC_CHARACTER-Vollständigkeit** | JEDES in der Pipeline unterstützte Material MUSS einen Eintrag in `AUTHENTIC_CHARACTER` (`intentional_artifact_classifier.py`) und `_MATERIAL_THRESHOLD_BONUS` (`per_phase_musical_goals_gate.py`) haben. Fehlende Einträge führen zu `return 1.0` (keine Preservation) bzw. 0.003-Default — beides Qualitätsverlust. (§v10.92) |
| §G99 | **Equality-of-Materials-Pflicht** | Jedes Material (cassette, kassette, lp, aac, streaming, minidisc, dat, wire_recording, lacquer_disc) MUSS in ALLEN Kalibrierungs-Tabellen (`AUTHENTIC_CHARACTER`, `_MATERIAL_THRESHOLD_BONUS`, `_MATERIAL_CLASS`, `_MATERIAL_QUALITY_CEILING`) einen Eintrag haben. Aliase (kassette→cassette, lp→vinyl) sind explizit zu deklarieren, nicht via Default. (§v10.92) |

---

## Kategorie XVI — Perzeptuelle Architektur: Das menschliche Ohr als Richter (§G100–§G112)

> §v10.101 — Prämisse: Auriks Architektur wurde fundamental umgebaut.
> Vorher: DSP-Pipeline mit technischen Metriken zur Validierung.
> Nachher: JEDE Verarbeitungsentscheidung fragt „Ist der Unterschied hörbar?",
> bevor sie handelt. Das menschliche Ohr ist der einzige Richter über Qualität.

| ID | Regel | Beschreibung |
| ---- | ------- | ------------- |
| §G100 | **Hörbarkeit vor Mathematik** | JEDE Verarbeitungsentscheidung MUSS die Frage „Ist der Unterschied für das menschliche Ohr hörbar?" VOR der Frage „Ist der Unterschied mathematisch signifikant?" stellen. Eine unhörbare Verbesserung ist keine Verbesserung. Ein unhörbarer Defekt ist kein Defekt. |
| §G101 | **Perzeptueller Wet/Dry-Blend** | Jeder Wet/Dry-Mix MUSS `perceptual_blend()` aus `backend.core.dsp.perceptual_blend.py` verwenden. Der Blend erfolgt frequenzabhängig nach Bark-Bändern: Nur in den kritischen Bändern, wo die Änderung oberhalb der simultanen Maskierungsschwelle (ISO 11172-3) liegt, wird das Wet-Signal übernommen. In maskierten Bändern bleibt das Dry-Signal erhalten — dort ist die Änderung unhörbar und birgt nur Artefakt-Risiko. |
| §G102 | **Bark-Band-Verarbeitung** | Jede frequenzabhängige Verarbeitung (EQ, Dynamik, Spektralreparatur) MUSS in 24 kritischen Bark-Bändern (Zwicker 1961) arbeiten — NICHT in linearen Hz-Bändern. Das menschliche Ohr hat logarithmische Frequenzauflösung: 100 Hz Unterschied bei 100 Hz sind hörbar, 100 Hz Unterschied bei 10 kHz sind unhörbar. |
| §G103 | **LUFS-basierte Lautheit** | Jede Dynamik-Entscheidung (Kompression, Expansion, Limiting) MUSS auf ITU-R BS.1770-4 LUFS (Loudness Units relative to Full Scale) basieren — NICHT auf RMS oder Peak. RMS korreliert schwach mit wahrgenommener Lautheit; LUFS modelliert die menschliche Lautheitswahrnehmung mit K-Weighting und Gating. |
| §G104 | **JND-Gate nach jeder Phase** | Nach JEDER Phasen-Ausführung MUSS `should_skip_phase()` aus `backend.core.dsp.perceptual_gate.py` geprüft werden. Wenn die tatsächliche Änderung in weniger als 2 Bark-Bändern die Just-Noticeable-Difference überschreitet → Audio wird auf Pre-Phase-Zustand zurückgesetzt. Verhindert, dass unhörbare Änderungen Artefakt-Risiko tragen. |
| §G105 | **ISO-226-Hörschwellen-Integration** | JEDE Pegel-Entscheidung MUSS die frequenzabhängige absolute Hörschwelle nach ISO 226:2003 berücksichtigen. Ein −60 dBFS-Signal bei 4 kHz ist deutlich hörbar; bei 50 Hz unhörbar. Die Hörschwelle variiert um >40 dB. |
| §G106 | **Perzeptuelle Qualitätsgewichtung** | Der QualityAnalyzer MUSS perzeptuelle Metriken (MUSHRA/OQS, Naturalness, Warmth, Clarity) mit ≥70% gewichten. Technische Metriken (SNR, THD, DR) ≤30%. MUSHRA wird mit 35% als Ground-Truth gewichtet. |
| §G107 | **Ermüdungsfreier Klang** | Jede Verarbeitung MUSS auf Langzeit-Hörkomfort optimieren. Spektrale Balance folgt ISO-226 für Ziel-Abhörpegel. Harsche Frequenzspitzen (>6 dB) werden per Bark-Band-Glättung abgefangen. Kein HF-Boost ohne Maskierungsprüfung. |
| §G108 | **Stille als psychoakustischer Raum** | Absolute Stille im Signal MUSS als psychoakustischer Raum respektiert werden. Kein Noise-Gate mit Pump-Artefakten. Die Entscheidung ob „still" basiert auf LUFS (−70 LUFS), nicht RMS. |
| §G109 | **Binaurale Natürlichkeit** | Stereo-Entscheidungen MÜSSEN binaurale Wahrnehmung respektieren. IACC (Interaural Cross-Correlation) nach Blauert (1997) ist primäre Phantom-Center-Metrik. Kein künstliches Stereo-Widening ohne Quellmaterial-Rechtfertigung. |
| §G110 | **Transiente Hörbarkeit** | Transienten-Verarbeitung MUSS zeitliche Maskierung (Pre-Masking 20ms, Post-Masking 100ms nach ISO 11172-3) berücksichtigen. Attack/Release-Zeiten auf psychoakustische Konstanten abstimmen. |
| §G111 | **Adaptiver Frequenzgang** | Zielfrequenzgang passt sich der Abhörlautstärke an (Fletcher-Munson/ISO 226). −23 LUFS → leichte Bass-/Höhenanhebung. −14 LUFS → flacher. Verhindert „leise=kraftlos"-Eindruck. |
| §G112 | **Perzeptuelles Monitoring** | Jeder Pipeline-Run MUSS DREI perzeptuelle Metriken in der Final-Summary ausweisen: 📊 Signalqualität (technisch), 🎧 Hörerlebnis (MUSHRA), 🧠 Restaurations-Index (HPI). |

### Neue VERBOTE — Perzeptuelle Architektur (§V34–§V38)

| ID | Verbot | Beschreibung |
| ---- | -------- | ------------- |
| §V34 | **Skalarer-Blend-Verbot** | Es ist VERBOTEN, einen skalaren Wet/Dry-Faktor (eine Zahl × alle Frequenzen) zu verwenden wenn `perceptual_blend()` verfügbar ist. |
| §V35 | **Lineare-Frequenzband-Verbot** | Es ist VERBOTEN, neue Phasen mit linearen Frequenzbändern zu implementieren. Neue Phasen MÜSSEN `split_into_bark_bands()` verwenden. |
| §V36 | **RMS-Lautheit-Verbot** | Es ist VERBOTEN, RMS als Proxy für wahrgenommene Lautheit zu verwenden, wenn LUFS via `measure_lufs_per_bark()` verfügbar ist. |
| §V37 | **JND-Ignoranz-Verbot** | Es ist VERBOTEN, das Ergebnis einer Phase ohne `should_skip_phase()`-Prüfung zu akzeptieren. Die Prüfung erfolgt post-hoc: war die Änderung unhörbar → Rollback auf Pre-Phase-Audio. |
| §V38 | **Hörschwellen-Ignoranz-Verbot** | Es ist VERBOTEN, Pegel-Entscheidungen ohne ISO-226-Hörschwellen-Konsultation zu treffen. |

---

## Kategorie XVII — Universelle Phasen-Sicherheit & Excellence-Kalibrierung (§G113–§G120)

> §v10.112–§v10.117 — Prämisse: Alle 65+ Phasen werden durch systemische Guards auf
das gleiche SOTA-Sicherheitsniveau gehoben. Keine Phase kann mehr unbemerkt Stille,
Transienten-Verschiebungen, HF-Halluzinationen oder Formant-Degradation verursachen.
Der ExcellenceOptimizer ist auf iZotope-RX11-Niveau kalibriert (Naturalness 0.86–0.90).

| ID | Regel | Beschreibung |
| ---- | ------- | ------------- |
| §G113 | **Universal RMS-Guard** | JEDE Phase MUSS nach der Ausführung einen RMS-Vergleich durchführen. RMS-Drop >30 dB → automatischer Rollback auf Eingangs-Audio. Der Guard ist in `PhaseInterface._safe_process()` zentral implementiert (§v10.115) und gilt für alle 65+ Phasen. |
| §G114 | **Transient-Shift-Detektion** | Alle additiven Phasen (ENHANCEMENT, RESTORATION, Harmonic, Exciter, Air-Band, Bass, Presence, Transient, Spectral, Frequency, Drums, Guitar, Brass, Piano, Vocal, Saturation, Spatial) MÜSSEN nach der Ausführung `detect_transient_shifts()` aufrufen. Onset-Shift >5 ms → Warning. (§V22, §v10.115) |
| §G115 | **Hallucination-Guard** | Alle Synthese-Phasen (Harmonic, Spectral-Repair, Inpainting, Exciter, Frequency-Restoration, Air-Band, Diffusion, Band-Gap, Dropout) MÜSSEN nach der Ausführung Spectral-Novelty prüfen. Novelty >0.15 → Warning. (§2.46e, §v10.115) |
| §G116 | **Formant-Stabilitäts-Guard** | JEDE Phase MUSS `formant_stability` (spektrale Band-Vektor-Korrelation 300–3500 Hz, 10 Bänder logarithmisch) berechnen. Korrelation <0.85 → Warning: mögliche Gesangsdegradation. (§v10.117) |
| §G117 | **Groove-Guard** | Der NaturalnessOptimizer MUSS die Transientendichte (attacks/sec) vor dem Blending berechnen. Bei Dichte >5/s: Blend-Stärke 0.05 (95% Restaurat erhalten). Bei >3/s: 0.10. Bei >1.5/s: 0.18. Sonst: 0.30. Verhindert, dass groovige Songs ihre restaurierten Attack-Transienten durch Original-Blending verlieren. (§v10.112) |
| §G118 | **HPI-Gate im Goosebumps-Recovery** | Wenn die Restauration bereits gute Qualität erreicht hat (HPI-Checkpoint existiert ODER Artifact-Freedom ≥0.90), MUSS der Original-Penalty von 0.030 auf 0.120 vervierfacht werden. Verhindert, dass Artefakt-Transienten des defekten Originals die Goosebumps-Metrik täuschen und das unverarbeitete Original dem Restaurat vorgezogen wird. (§v10.113) |
| §G119 | **FeedbackChain-Silence-Guard** | Phase 07 MUSS vor der harmonischen Synthese H2/H1 prüfen. H2/H1 ≥0.50 → Strength auf ≤0.10 drosseln. H2/H1 ≥0.35 → Strength auf 50% drosseln (Frühwarn-Schwelle). Zusätzlich MUSS nach der Synthese ein RMS-Vergleich erfolgen: Output-RMS < Input-RMS/100 (−40 dB) → Rollback auf Eingangs-Audio. (§v10.114) |
| §G120 | **ExcellenceOptimizer RX11-Kalibrierung** | Der ExcellenceOptimizer MUSS auf iZotope-RX11-Niveau kalibriert sein: `_MODULATION_STRENGTH`=0.55, `_HARM_BOOST_DB`=3.2, `_HARM_MAX_ORDER`=10, `_TARGET_CV_MIN`=0.07, `_FLUX_SMOOTHING_MAX`=0.65. Alle 8 Material-Profile (auto, vinyl, tape, shellac, broadcast, mp3_low, mp3_high, cd_digital) MÜSSEN proportional skalierte Werte haben. (§v10.116) |
| §G121 | **Mode-Differenzierung: RESTORATION = Do No Harm** | Kreative Audio-Eingriffe (Noise-Gate, Spektral-Balance, Stereo-Fokus) dürfen NUR im STUDIO_2026-Mode laufen. In RESTORATION MUSS die Original-Dynamik unangetastet bleiben — der NaturalnessOptimizer darf das Originalsignal NICHT gate-bearbeiten, spektral umbalancieren oder im Stereo-Feld verschieben. (§v10.119) |
| §V39 | **Noise-Gate-Verbot in RESTORATION** | Es ist VERBOTEN, das Noise-Gate, die Spektral-Balance oder den Stereo-Fokus im RESTORATION-Mode auszuführen. Diese sind kreative Werkzeuge, die die Original-Dynamik verändern — ein Verstoß gegen §0 „Do No Harm". (§v10.119) |

## Kategorie XVIII — Laufzeit-Qualitätsgarantien (§G122–§G130)

> §v10.702 — Prämissen aus dem Produktionslauf „Elke Best" (Kassette, 4-stufige Kette, 224 s, 44 Phasen, QualityGate Δ=0.0).
> Diese Kategorie definiert die aus 10 identifizierten Laufzeit-Regressionen abgeleiteten unverhandelbaren GEBOTE.

| ID | Regel | Beschreibung |
| ---- | ------- | ------------- |
| §G122 | **LUFS-Δ-Cap-Pflicht** | Phase_40 MUSS material-adaptives LUFS-Cap anwenden: Kassette ≤8 LU, Shellac ≤5 LU, Vinyl ≤6 LU, CD ≤20 LU. Überschreitung → WARNING + Begrenzung. Verhindert MUSHRA-Zerstörung durch extreme Loudness-Normalisierung. (§v10.702 R2) |
| §G123 | **Closed-Loop-Empfindlichkeit** | Der Closed-Loop-Calibrator MUSS Verbesserungen ab Δ≥0.015 erkennen (vorher 0.04). Regressionen MÜSSEN ab Δ≤−0.03 erkannt werden (vorher −0.06). Adapt-Step 0.04 (vorher 0.08). Kumulativer Δ-Tracker nach 5 Phasen mit ΣΔ>0.05 → Boost. (§v10.702 R3) |
| §G124 | **ExcellenceOptimizer-Hysterese** | Core-Guard des ExcellenceOptimizers MUSS Hysterese 0.05 einhalten (vorher 0.015). Regressionen innerhalb PMGG-Messungenauigkeit (±0.03) → KEIN Rollback. (§v10.702 R4) |
| §G125 | **MDEM-Per-5-Phasen-Prüfung** | Nach jeder 5. erfolgreich ausgeführten Phase MUSS RMS-Hüllkurven-Pearson-Prüfung gegen Pre-Pipeline-Audio erfolgen. Pearson <0.85 → WARNING: kumulativer Dynamik-Verlust. (§v10.702 R5) |
| §G126 | **De-Esser-Soft-Saturation-Skip** | Phase_19 MUSS vor Ausführung soft_saturation in Defect-Scores prüfen. Confidence >0.5 → Phase vollständig überspringen. (§v10.702 R7) |
| §G127 | **Unbound-Variable-Scope-Garantie** | Jede lokale Variable, die in return-Dict referenziert wird, MUSS VOR dem umgebenden if-Block initialisiert sein. Scope-Bugs (Chunked-Streaming-Crash, 1699 s Verlust) sind zu verhindern. (§v10.702 R1) |
| §G128 | **GDD-Budget-Proaktivität** | GDD-Budget-Manager MUSS VOR jeder STFT-Phase allocate() aufrufen. <1.0 ms → Stärke auf 25% drosseln. Nach jeder STFT-Phase MUSS consume() laufen. (§v10.701 D2) |
| §G129 | **Rollback-Sanity-Pflicht** | Nach JEDEM Rollback (HPI, AFG, CIG, SFT) MUSS validate_rollback_audio() das Ziel-Audio prüfen. RMS <−60 dBFS, NaN, Inf oder Peak <1e−6 → CRITICAL + Fallback auf Original. (§v10.701 D4) |
| §G130 | **PresenceEmbedding-Export-Pflicht** | Jeder Export MUSS PresenceScore berechnen (5 Sub-Scorer: Vocal Formant Coherence, Transient Immediacy, Room Tone Continuity, Microdynamic Liveliness, Spectral Air Authenticity). Score ≥0.70 = hörbare Verbesserung. Im Quality Report ausweisen. (§v10.701 D3) |

## Kategorie XIX — SOTA-Reproduzierbarkeit: Kritische Bugfixes B1/B2/B3 (§G131–§G136)

> §v10.702 B1/B2/B3 — Prämisse: Drei kritische Bugs verhinderten die vollständige
> Reproduzierbarkeit und korrekte Qualitätsbewertung von Importsongs. Diese Kategorie
> kodifiziert die architektonischen Garantien für perzeptuelle Verbesserungs-Metrik,
> Defekt-Transparenz und Chunked-Streaming-Determinismus.

| ID | Regel | Beschreibung |
| ---- | ------- | ------------- |
| §G131 | **Perzeptuelle-Verbesserungs-Metrik-Pflicht** | Der MQA-`_minimal_improvement`-Check MUSS `musical_improvement` (40% Tech + 60% MUSHRA/HPI) als primäre Metrik verwenden, wenn perzeptuelle Daten verfügbar sind. Der BlindQualityScore (SNR/THD/bandbreite) bestraft Defekt-ENTFERNUNG (Rauschen=HF-Energie) und darf NUR als Fallback ohne MUSHRA/HPI dienen. (§v10.702 B1) |
| §G132 | **Composite-Score-Schwelle** | Bei verfügbaren perzeptuellen Metriken: `musical_improvement > 0.005` (0.5% Composite = hörbare Verbesserung). Ohne perzeptuelle Daten: `output_score/input_score ≥ 1.015` (1.5% Ratio-Fallback). (§v10.702 B1) |
| §G133 | **Per-Defekt-Reduktions-Pflicht** | Nach der Restauration MUSS ein Pre-vs-Post-DefectScan pro Defekttyp die Severity-Reduktion berechnen (pre, post, reduction, reduction_pct). Das Ergebnis MUSS unter `RestorationResult.metadata["defect_reduction_per_type"]` und im `_mqa_result` gespeichert werden. (§v10.702 B2) |
| §G134 | **Defekt-Transparenz** | Die GUI MUSS für jeden Defekttyp die individuelle Reduktionsrate anzeigen können. Kein "Blackbox"-Qualitätswert — jeder Defekt wird einzeln ausgewiesen. Der Logger MUSS die Reduktion pro Defekttyp protokollieren. (§v10.702 B2) |
| §G135 | **Chunked-Streaming-Determinismus-Pflicht** | `_restore_chunked()` MUSS nach Chunk 0 den kompletten Pre-Analysis-State einfrieren (CalibrationProfile, RestorationContext, AutosetupPolicy, ExecutedPhases, RestorabilityScore) und für alle Folge-Chunks via `_b3_frozen_*` kwargs injizieren. Alle Chunks MÜSSEN identische Phasen und Parameter wie Chunk 0 verwenden. (§v10.702 B3) |
| §G136 | **Wiederholungs-Reproduzierbarkeit** | Derselbe Input (Audio-Datei) MUSS bei wiederholter Ausführung denselben Output liefern. Jeder Zufallsgenerator wird mit deterministischem Seed initialisiert (§G22). Chunked-Streaming ist keine Ausnahme: gleiche Phasen, gleiche Parameter, gleiches Ergebnis. (§v10.702 B3) |
| §G137 | **Full-Song-Defekt-Presence-Pflicht** | Vor dem Chunked-Streaming-Chunking MUSS `scan_defect_presence()` das GESAMTE Audio auf Defekt-Typen scannen (stratifizierte Fenster, max. 10 × 60s). Fehlende Defekt-Typen MÜSSEN VOR CausalReasoner/Denker in `defect_result.scores` gemerged werden (severity=0.06, confidence=0.30). Der Phasen-Plan MUSS Phasen für ALLE im Gesamt-Song vorhandenen Defekt-Typen enthalten — nicht nur für die, die Chunk 0 zufällig sieht. (§v10.702 B3-P2) |

### Neue VERBOTE — SOTA-Reproduzierbarkeit (§V40–§V42)

| ID | Verbot | Beschreibung |
| ---- | -------- | ------------- |
| §V40 | **BlindQuality-als-Ground-Truth-Verbot** | Es ist VERBOTEN, den BlindQualityScore (SNR/THD/bandbreite) als alleinige Verbesserungs-Metrik zu verwenden, wenn MUSHRA/HPI-Daten verfügbar sind. BlindQuality bestraft Rauschentfernung und darf die perzeptuelle Bewertung nicht überschreiben. (§v10.702 B1) |
| §V41 | **Blackbox-Qualitätswert-Verbot** | Es ist VERBOTEN, einen aggregierten Einzelwert als einzige Qualitätsaussage zu präsentieren, wenn `defect_reduction_per_type` mit Pre-vs-Post-Reduktion pro Defekttyp verfügbar ist. Die Reduktion MUSS pro Defekt nachvollziehbar sein. (§v10.702 B2) |
| §V42 | **Nicht-deterministisches-Chunking-Verbot** | Es ist VERBOTEN, dass verschiedene Chunks eines Chunked-Streaming-Laufs unterschiedliche Phasen-Selektionen oder Kalibrierungs-Parameter erhalten. Der Pre-Analysis-State von Chunk 0 MUSS deterministisch auf alle Folge-Chunks übertragen werden. (§v10.702 B3) |

---

## Kategorie XX — Perzeptueller Autopilot: Wohlklang-Garantien (§G138–§G141)

> §v10.703 — Prämisse: Aurik positioniert sich als „Perzeptueller Autopilot".
> Nicht: Werkzeugkasten für Toningenieure (RX 11). Sondern: Ein-Knopf-Garantie
> für hörbare Verbesserung, messbare Defekt-Reduktion und garantierten Wohlklang.
> Diese Kategorie kodifiziert die vier architektonischen Pfeiler.

| ID | Regel | Beschreibung |
| ---- | ------- | ------------- |
| §G138 | **BlindQuality-Verbot-im-Gate** | Der BlindQualityScore (SNR/THD) darf IM QUALITY-GATE NICHT als Verbesserungs-Metrik verwendet werden. Die EINZIGE zulässige Metrik ist `musical_improvement` (perzeptuell gewichtet). Ohne MUSHRA/HPI: QUALITY_UNCERTAIN — keine falsche Garantie bei fehlenden perzeptuellen Daten. (§v10.703 Step 3, §V40) |
| §G139 | **Defekt-Countdown-Pflicht** | Nach der Restauration MUSS ein `defect_countdown`-Dict berechnet werden: total_detected, audible_before, audible_after, resolved, reduced, zero_audible_defects. Die GUI MUSS diesen Countdown anzeigen: „3 Defekte → 2 behoben → 0 hörbar ✅". (§v10.703 Step 1) |
| §G140 | **Export-Gate-Pflicht** | Vor JEDEM Export MUSS `export_gate()` aufgerufen werden. Das Gate prüft: Zero Audible Defects, Quality Guaranteed, Wohlklang-Garantie. Es MUSS Garantien (✅) und Warnungen (⚠️) ausweisen — darf den Export aber NICHT blockieren. (§v10.703 Step 2) |
| §G141 | **Wohlklang-Garantie-Pflicht** | Nach JEDER Restauration MUSS `wohlklang_garantie_check()` den MUSHRA-Score prüfen. MUSHRA ≥ 80 → Wohlklang garantiert. MUSHRA < 80 → ReRun-Empfehlung mit berechneten sanfteren Parametern (global_scalar -Δ, strength × Faktor). Der Nutzer MUSS die Empfehlung in der GUI sehen. (§v10.703 Step 4) |

### Neue VERBOTE — Perzeptueller Autopilot (§V43–§V44)

| ID | Verbot | Beschreibung |
| ---- | -------- | ------------- |
| §V43 | **Export-ohne-Gate-Verbot** | Es ist VERBOTEN, einen Export ohne vorherigen `export_gate()`-Aufruf durchzuführen. Das Gate muss gelaufen sein und seine Ergebnisse müssen im Export-Log erscheinen. |
| §V44 | **MUSHRA-Ignoranz-Verbot** | Es ist VERBOTEN, einen MUSHRA-Score < 80 ohne Wohlklang-Warnung an den Nutzer zu akzeptieren. Der Nutzer MUSS informiert werden, dass die Wohlklang-Garantie nicht erfüllt ist und ein ReRun empfohlen wird. |

---

## Kategorie XXI — Perzeptueller Closed-Loop: Per-Band-Hören (§G142–§G145)

> §v10.703 Steps 5+6 — Prämisse: Aurik simuliert das menschliche Ohr AUF DER EBENE
> DER KRITISCHEN BÄNDER. Kein skalarer Score. Keine mathematische Metrik.
> Jede Phase fragt: „Hat sich der Klang pro Bark-Band verbessert oder verschlechtert?"

| ID | Regel | Beschreibung |
| ---- | ------- | ------------- |
| §G142 | **Per-Band-MUSHRA-Pflicht** | Die MQA MUSS `compute_per_band_mushra()` aus `backend.core.per_band_mushra` aufrufen. Das Ergebnis (24 Bark-Bänder × MUSHRA 0–100) MUSS in `RestorationResult.metadata["per_band_mushra"]` gespeichert werden. Die GUI MUSS eine spektrale Wohlklang-Heatmap anzeigen können. (§v10.703 Step 5) |
| §G143 | **Bark-Band-Blend-Pflicht** | Wenn Per-Band-MUSHRA verfügbar ist, MUSS der Wet/Dry-Mix `perceptual_blend_per_band()` verwenden — NICHT `perceptual_blend()` (skalar). Nur in Bark-Bändern mit perzeptueller Verbesserung wird das Wet-Signal übernommen. In allen anderen Bändern bleibt das Dry-Signal erhalten. (§v10.703 Step 5) |
| §G144 | **MUSHRA-Proxy-Pflicht** | JEDE Phase MUSS nach der Ausführung den `MUSHRAProxy` konsultieren. Der Proxy schätzt MUSHRA vorher/nachher in <100ms via leichten MERT-Embedding-Vergleich. Das Delta (post−pre) MUSS in `result.metadata["mushra_proxy_delta"]` gespeichert werden. (§v10.703 Step 6) |
| §G145 | **Perzeptueller-Rollback-Pflicht** | Wenn `mushra_proxy_delta ≤ 0` (keine hörbare Verbesserung oder Verschlechterung), MUSS die Phase auf Pre-Phase-Audio zurückgesetzt werden. `result.audio = pre_phase_audio.copy()`. `result.metadata["mushra_proxy_rollback"] = True`. Der Rollback MUSS als Warning geloggt werden — er ist kein Fehler, sondern ein Qualitätsmerkmal. (§v10.703 Step 6) |

### Neue VERBOTE — Perzeptueller Closed-Loop (§V45–§V46)

| ID | Verbot | Beschreibung |
| ---- | -------- | ------------- |
| §V45 | **Ungeprüfte-Phase-Verbot** | Es ist VERBOTEN, eine Phase ohne MUSHRA-Proxy-Prüfung abzuschließen, wenn `_perceptual_closed_loop=True` (Default). Der Proxy muss gelaufen sein und das Delta muss in den Metadaten stehen. |
| §V46 | **Delta-Ignoranz-Verbot** | Es ist VERBOTEN, ein negatives MUSHRA-Proxy-Delta (perzeptuelle Verschlechterung) zu ignorieren. Bei Δ ≤ 0 MUSS ein Rollback erfolgen. Kein „es war nur eine kleine Verschlechterung" — das menschliche Ohr ist der Richter. |

---

## Kategorie XXII — Architektonische Qualitäts-Garantien: Metrik-Hierarchie & Guard-Disziplin (§G150–§G155)

> §v10.704 — Prämisse: 20 Bugs in der Produktionsanalyse haben drei systemische
> Schwachstellen offengelegt: (1) Guards widersprechen sich mangels Metrik-Hierarchie,
> (2) Guards kennen den Phasen-Kontext nicht, (3) Schwellwerte sind hartcodiert statt
> adaptiv. Diese Kategorie kodifiziert die architektonischen Heilungsmaßnahmen.

| ID | Regel | Beschreibung |
| ---- | ------- | ------------- |
| §G150 | **Metrik-Hierarchie-Pflicht** | Aurik MUSS eine definitive Hierarchie der Qualitätsmetriken einhalten: Priorität 1 = MUSHRA/HPI (perzeptuell), Priorität 2 = Defekt-Reduktion (B2-Daten), Priorität 3 = artifact_freedom (nur VETO bei >5 Artefakten), Priorität 4 = BlindQuality (NUR Diagnostik, NIE Quality-Gate). Keine niedrigere Metrik darf das Urteil einer höheren überschreiben. |
| §G151 | **MUSHRA-Primat** | Wenn MUSHRA > 0 (perzeptuelle Daten verfügbar), MUSS MUSHRA die primäre Ground Truth für ALLE Quality-Gate-Entscheidungen sein. artifact_freedom und BlindQuality dürfen NIEMALS ein positives MUSHRA-Urteil negieren. |
| §G152 | **BlindQuality-Diagnostik-Verbot** | Der BlindQualityScore (SNR/THD/Bandbreite) darf AUSSCHLIESSLICH für technische Diagnostik verwendet werden. Er darf NIEMALS in `quality_guaranteed`, `verdict` oder `export_gate` einfließen. Verstoß → §V40 (bestehend). |
| §G153 | **Guard-Phasen-Whitelist-Pflicht** | Jeder Quality-Guard (AFG, VocalNoHarm, FormantGuard, CIG, SFT, PMGG) MUSS deklarieren, für welche Phasen-Familien er zuständig ist. Eine Phase, deren Familie nicht in der Whitelist des Guards steht, wird von diesem Guard ÜBERSPRUNGEN. Kein Guard läuft盲 auf allen Phasen. |
| §G154 | **Adaptive-Schwellwert-Pflicht** | Jeder Schwellwert in Quality-Gates MUSS aus Material-Typ, Transfer-Chain-Depth und Phasen-Familie ABGELEITET werden — NIEMALS hartcodiert. Formel: `threshold = base_threshold × material_factor × depth_factor × phase_factor`. Die Tabelle der Material/Depth/Phase-Faktoren MUSS zentral in `calibrated_constants.py` definiert sein. |
| §G155 | **Quality-Entscheidungs-Narrativ-Pflicht** | Jede Quality-Gate-Entscheidung (Verdict, Rollback, Skip) MUSS im GUI-Narrativ BEGRÜNDET werden — mit Bezug auf die konkrete Metrik, den Schwellwert und die Phasen-Familie. Kein „NO IMPROVEMENT" ohne Erklärung. Kein Rollback ohne „Warum". Der Nutzer MUSS verstehen, WARUM Aurik so entschieden hat. |

### Neue VERBOTE — Architektonische Qualitäts-Garantien (§V47–§V49)

| ID | Verbot | Beschreibung |
| ---- | -------- | ------------- |
| §V47 | **Metrik-Unterordnung-Verbot** | Es ist VERBOTEN, dass eine niedrigprioritäre Metrik (artifact_freedom, BlindQuality) das Urteil einer höherprioritären Metrik (MUSHRA, HPI) überschreibt oder negiert. |
| §V48 | **Guard-Kontext-Ignoranz-Verbot** | Es ist VERBOTEN, einen Quality-Guard auf einer Phase laufen zu lassen, deren Phasen-Familie nicht in der Whitelist des Guards deklariert ist. |
| §V49 | **Hartcodierter-Schwellwert-Verbot** | Es ist VERBOTEN, einen Quality-Gate-Schwellwert hart zu codieren. Alle Schwellwerte MÜSSEN via `calibrated_constants.py` aus Material+Depth+Phase abgeleitet werden. |

## Änderungshistorie

| Version | Datum | Änderung |
| --------- | ------- | ---------- |
| 10.17.0 | 2026-08-03 | **§G167–§G172 + §V52: Denker-IQ & Material-Awareness.** B30-Komplettierung (3 Export-Gates). B5: MERT-MUSHRA-Fix. B6–B10: Chain-Injection + PID-Validierung + StrategieDenker-Budget. B11: SourceMediumProfile → SongCalibration. B12–B16: SMP in Phase_29/07/26/54/36. B17: OneTakeExport ISP-Margin. Drei-Schicht-Material-Intelligenz. 10 Dateien. (§v10.706) |
| 10.16.1 | 2026-08-02 | **§G156–§G166 + §V50–§V51: Restorability-Gate & Bugfixes B19–B30.** B30: Depth+Restorability-adaptiver HPI-Gate. B26: Sample-Axis-Fix. B27: MUSHRA/HPI→MQA. B19: AFG-Threshold. B28: Chunked-Längenwarnung. B29: P5-Traceback-Diagnostik. Drei-Quellen-Synchronisation. Kategorie XXIII. (§v10.704) |
| 10.16.0 | 2026-08-03 | **§G150–§G155 + §V47–§V49: Metrik-Hierarchie & Guard-Disziplin.** S1: Metrik-Hierarchie in MQA. S2: Guard-Phasen-Whitelist (AFG). S3: Adaptive-Schwellwert-Pflicht. S4: Quality-Entscheidungs-Narrativ. Kategorie XXII. (§v10.704) |
| 10.15.0 | 2026-08-03 | **§G142–§G145 + §V45–§V46: Perzeptueller Closed-Loop.** Step 5: Per-Band-MUSHRA (24 Bark-Bänder). Step 6: MUSHRA-Proxy in PhaseInterface._safe_process(). Perzeptueller Rollback bei Δ≤0. Kategorie XXI. (§v10.703) |
| 10.15.0 | 2026-08-03 | **§G138–§G141 + §V43–§V44: Perzeptueller Autopilot.** Step 1: Defekt-Countdown. Step 2: Export-Garantie. Step 3: BlindQuality aus MQA entfernt. Step 4: Wohlklang-Garantie. Kategorie XX. (§v10.703) |
| 10.14.2 | 2026-08-03 | **§G131–§G137 + §V40–§V42: SOTA-Reproduzierbarkeit.** B1: Perzeptuelle Verbesserungs-Metrik statt BlindQuality. B2: Per-Defekt-Reduktions-Transparenz. B3: Chunked-Streaming-Determinismus. B3-P2: Full-Song Defect-Presence Pre-Scan (§G137). 3 Compliance-Fixes (R01/R02/R11). Kategorie XIX. (§v10.702) |
| 10.14.1 | 2026-08-02 | **§G122–§G130: Laufzeit-Qualitätsgarantien.** LUFS-Cap, Closed-Loop-Empfindlichkeit, ExcellenceOptimizer-Hysterese, MDEM-Per-5-Phasen, De-Esser-Skip, Scope-Garantie, GDD-Budget, Rollback-Sanity, PresenceEmbedding. Kategorie XVIII. (§v10.702) |
| 10.0.18 | 2026-08 | **§v10.120–§v10.124: Depth-Threshold Calibration-Shift & Major-Version-Upgrade.** Chain-Depth-Paradigma §G71 flächendeckend: depth≥3=moderat, depth≥4=deep cassette, depth≥5=extrem. Alle 6 Quality-Gates depth-adaptiv. 18 Guard-Migrationen, 8 CIG-Exclusions, 17 Phasen mit bw_extension_context. MERT-Referenzspeicher depth-adaptiv (AF≥0.75 für Depth≥4). NaturalnessOptimizer Attack-Perzentil 90→95 adaptiv. Goosebumps-Recovery-HPI-Fix. 46 Dateien. Specs: §18, §19. |
| 10.0.16 | 2026 | §G113–§G120: Universelle Phasen-Sicherheit & Excellence-Kalibrierung. RMS-Guard, Transient-Shift, Hallucination-Guard, Formant-Guard, Groove-Guard, HPI-Gate, Silence-Guard, RX11-Kalibrierung. Kategorie XVII. (§v10.112–§v10.117) |
| 10.14.0 | 2026-08-10 | §G100–§G112 + §V34–§V38: Perzeptuelle Architektur §v10.101. |
| 10.0.14 | 2026-08-10 | §G90–§G99: Non-Plus-Ultra. Blinder Referenz-Vektor, Exception-Proxies, Cross-Phase-Koordination, NaN-Guards, Material-Vollständigkeit. Kategorie XV. |
| 10.0.14 | 2026-07-30 | **§G71–§G80 (spec_constitution): Startup-Integration & Kommunikation.** GPU-Detection failsafe, Lock-Disziplin, Plugin-Namen-Validierung, Watchdog-Selbsttest, Cache-Safety, Happy-Path-Gate, Startup-Smoke-Test, Import-Check, Event-Garantie, Probe-Invocation. Kategorie XII. (§v10.305) |
