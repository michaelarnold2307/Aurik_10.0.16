# Aurik 10 — GEBOTE & VERBOTE (Normativer Katalog)

> **Status:** Normativ | **Version:** 10.0.17 | **Stand:** §v10.119 Mode-Differenzierung: RESTORATION = Do No Harm, STUDIO_2026 = kreativ
>
> Dieser Katalog definiert alle unverhandelbaren GEBOTE (positiv, was Aurik TUN MUSS)
> und VERBOTE (negativ, was Aurik NIEMALS tun darf). Jedes Gebot und Verbot ist mit
> einer eindeutigen ID versehen (§G1, §V1 usw.) und wird im Code per Kommentar
> referenziert. Bei Widerspruch zwischen Specs und diesem Katalog gilt dieser Katalog.

---

## Kategorie I — Individuelle Song-Maximierung (§G1–§G9)

Jeder importierte Song wird individuell maximal für das menschliche Ohr verbessert.

| ID | Regel | Beschreibung |
|----|-------|-------------|
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
|----|-------|-------------|
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
|----|-------|-------------|
| §G20 | **Bridge-Bypass-Verbot** | Kein UI-/Frontend-Code importiert `backend/core/` direkt. Nur über `backend/api/bridge.py`. |
| §G21 | **Denker-Zentralität** | Alle Stärke-Entscheidungen fließen zentral im Denker. Keine dezentralen "Magic Numbers" in Phasen. |
| §G22 | **Determinismus** | Derselbe Input → derselbe Output. Jeder Zufallsgenerator wird mit fixem Seed aus dem Datei-Hash initialisiert. |
| §G23 | **ML-Fallback-Logging** | Jeder ML→DSP-Fallback MUSS mit `logger.warning()` protokolliert werden. Silent-Failures sind VERBOTEN. |
| §G24 | **NaN/Inf-Schutz** | Jede der 68 Phasen MUSS `np.nan_to_num()` oder `np.isfinite()` auf Ausgabe-Audio anwenden (§0a). |
| §G25 | **Logger-Pflicht** | Jede Python-Datei mit `logger`-Verwendung MUSS `import logging` und `logger = logging.getLogger(__name__)` definieren. |
| §G26 | **Guard-Counter-Lebendigkeit** | Jeder deklarierte Guard-Counter MUSS auch inkrementiert werden. Deklaration ohne `+= 1` ist toter Code. |
| §G27 | **Messschleifen-Plateau** | Jede Messschleife mit ≥3 Kandidaten MUSS Plateau-Erkennung haben. |
| §G28 | **PIM-first, RLP-last** | Vor jedem Phasen-Loop wird PIM berechnet. Nach jedem Loop wird RLP ausgeführt. |
| §G29 | **Artistic Intent vor Defect-Scan** | `get_artistic_intent()` wird VOR dem Defect-Scan aufgerufen. |

## Kategorie IV — CD-Rauschprofil & Export (§G30–§G39)

| ID | Regel | Beschreibung |
|----|-------|-------------|
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
|----|-------|-------------|
| §G40 | **Rauschprofil-Zeitpunkt** | Das CD-Rauschprofil wird NACH allen 68 Restaurierungsphasen und VOR dem Dithering appliziert. Dies ist wissenschaftlich der optimale Zeitpunkt: Wird Rauschen früher injiziert, wird es von nachfolgenden Phasen (Denoising, Kompression, EQ) verändert oder verstärkt. Nach der Pipeline ist das Signal stabil und das Rauschen bleibt unverfälscht. |
| §G41 | **Übergangs-Verifikation** | Jeder Übergang zwischen Rauschen und Stille/Musik MUSS verifiziert werden: Die Onset-Stärke (spectral-flux-basiert) darf 0.1 nicht überschreiten. Überschreitung → automatische Verbreiterung des Crossfades auf 500 ms und erneute Prüfung. |
| §G42 | **CD-Produktions-Kohärenz** | Die komplette Export-Kette (Rauschprofil → Dither → Metadaten) MUSS ein Ergebnis liefern, das für einen geschulten Hörer von einer CD-Produktion (1982–2000) nicht unterscheidbar ist. A/B-Blindtest als Validierung. |
| §G43 | **Rauschprofil-Pegel-Anpassung** | Der Rauschpegel passt sich automatisch der Ziel-Bittiefe an: 16-bit → −96 dBFS (CD-Standard), 24-bit → −120 dBFS (Hi-Res-Äquivalent). Kein fester Pegel unabhängig vom Exportformat. |
| §G44 | **Maskierungs-Wissenschaft** | Die Maskierungsschwelle folgt Zwicker & Fastl (1999): −70 dBFS Signalpegel maskiert −96 dBFS breitbandiges Rauschen in ruhiger Umgebung vollständig. Die 50-ms-RMS-Fensterung entspricht der zeitlichen Integration des menschlichen Gehörs. |
| §G45 | **Digital-Black-Integrität** | Exakte Null-Samples (digital black) werden NIE verrauscht — weder durch die Maskierungs-Hüllkurve noch durch Window-Smearing. Sample-genaue Durchsetzung als letzte Verteidigungslinie (§V12). |

---

## VERBOTE — Katalog absoluter Verbote (§V1–§V24)

| ID | Verbot | Beschreibung |
|----|--------|-------------|
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
- Kategorie XII (§G87): Noise-Floor-Brücke
- Kategorie XIII (§G88): Defektbehebungs-Module
- Kategorie XIV (§G89): Unsichtbare Signalintegrität
- Kategorie XV (§G90–§G99): Non-Plus-Ultra
- Kategorie XVI (§G100–§G112): Perzeptuelle Architektur §v10.101
- VERBOTE (§V1–§V38): Absolute Verbote, gelten immer und überall

---

## Kategorie VI — Metriken & Qualitätssicherung (§G46–§G59)

| ID | Regel | Beschreibung |
|----|-------|-------------|
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
|----|-------|-------------|
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
|----|--------|-------------|
| §V27 | **Kein signal.correlate für Lag** | Es ist VERBOTEN, `scipy.signal.correlate` (Standard-Kreuzkorrelation ohne PHAT-Whitening) für Stereo-Lag-Messung zu verwenden. Nur GCC-PHAT (via STCG) ist statthaft. |
| §V28 | **Kein begrenzter Lag-Suchraum** | Es ist VERBOTEN, den Lag-Suchraum für Stereo-Messungen auf < ±200ms (±9600 samples @48kHz) zu begrenzen. Kleinere Limits (z.B. 960 samples = 20ms) verfehlen echte Kanalversätze. |
| §V29 | **Keine konkurrierenden Lag-Korrekturen** | Es ist VERBOTEN, nach erfolgreicher STCG-Korrektur eine zweite Lag-"Korrektur" durchzuführen. Der Onset-Energy-Fallback in `_preserve_phase_loudness` ist NUR bei STCG-Exception aktiv. |
| §V30 | **Kein Single-Window-Lag** | Es ist VERBOTEN, Stereo-Lag nur an EINER Song-Position (z.B. Mid-Window 10s) zu messen, wenn die Song-Dauer > 30s beträgt. Multi-Point (≥3 Positionen) ist Pflicht. |
| §V31 | **Kein np.roll für Lag-Korrektur** | Es ist VERBOTEN, `np.roll` (zirkuläre Verschiebung mit Sample-Wrapping) für Stereo-Lag-Korrektur zu verwenden. Nur `scipy.ndimage.shift` (Zero-Padding, Sub-Sample) oder STCG sind statthaft. |
| §V32 | **Kein Audio-Trunkieren für Lag** | Es ist VERBOTEN, Audio zu trunkieren (`audio[:, :N - lag]`), um Lag zu korrigieren. Die Korrektur MUSS die Originallänge durch Zero-Padding erhalten. |
| §V33 | **Kein shape[0] ohne Orientierungs-Check** | Es ist VERBOTEN, `audio.shape[0]` als Sample-Anzahl zu interpretieren, ohne vorher zu prüfen ob `(2,N)` oder `(N,2)` vorliegt. Die Multi-Point-Funktion MUSS beide Orientierungen unterstützen. |

## Kategorie IX — SFT-Adaptivität & Defekt-Audibilität (§G68–§G75)

| ID | Regel | Beschreibung |
|----|-------|-------------|
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
|----|-------|-------------|
| §G76 | **Zentraler Kalibrierungs-Kontext** | Es MUSS einen einzigen, zentralen `CalibrationContext` geben, der ALLE Pre-Analysis-Messwerte (restorability_score, transfer_chain_depth, material_type, SNR, bandwidth, era_decade, genre, vocal_confidence) in EINEM Objekt bündelt. JEDES Modul, das einen Schwellwert benötigt, MUSS diesen Kontext als Quelle verwenden — NIE eine eigene Konstante. |
| §G77 | **Kontinuierliche Ableitung** | JEDER Schwellwert MUSS über eine kontinuierliche Funktion aus dem CalibrationContext abgeleitet werden. Die Funktion MUSS für jeden kontinuierlichen Eingabewert einen kontinuierlichen Ausgabewert liefern. Es ist VERBOTEN, diskrete Buckets (`if x > 0.4: ... elif x > 0.25: ...`) oder Lookup-Tabellen (`{1:0.25, 2:0.35}`) zu verwenden. |
| §G78 | **Vollständigkeit der Kalibrierung** | ALLE Schwellwerte, Caps, Floors und Blend-Faktoren in der gesamten Pipeline MÜSSEN kalibriert sein. Kein Parameter darf auf einem nicht aus dem CalibrationContext abgeleiteten Default verharren. Ausnahme: Physikalische Konstanten (z.B. −60 dBFS = digital black, −0.3 dBTP = ITU-R BS.1770 Ceiling). |
| §G79 | **Kalibrierungs-Audit** | Jeder kalibrierte Schwellwert MUSS im Log dokumentiert werden: `"§CALIB %s: rs=%.0f depth=%d → %s=%.4f"`. Dies ermöglicht die Rückverfolgbarkeit jeder Entscheidung auf Auriks eigene Messwerte. |
| §G80 | **Unkalibrierter-Fallback-Warnung** | Wenn ein Schwellwert nicht aus dem CalibrationContext abgeleitet werden kann (z.B. weil die Pre-Analysis noch nicht abgeschlossen ist), MUSS ein Default verwendet werden — aber NUR mit einer WARNING: `"⚠️ uncalibrated fallback: %s=%.4f (reason: %s)"`. Unkalibrierte Fallbacks sind als technische Schuld zu behandeln. |
| §G81 | **Einzige Quelle der Wahrheit** | Der CalibrationContext ist die EINZIGE Quelle für alle Schwellwerte. Wenn zwei Module unterschiedliche Werte für denselben Parameter berechnen, ist das ein Architekturfehler. Die Kalibrierungs-Matrix (`calibration_matrix.py`) ist der zentrale Berechnungspunkt — Module rufen ab, sie berechnen nicht selbst. |

## Kategorie XI — Laufzeit-Rekalibrierung (§G82–§G86)

> **Prämisse:** Die Pre-Pipeline-Kalibrierung basiert auf Messwerten des DEGRADIERTEN Eingangssignals. Während der Pipeline verbessert sich das Audio jedoch — SNR steigt, Bandbreite wächst, Defekte verschwinden. Eine Kalibrierung, die nach Phase 03 (denoise) noch mit dem ursprünglichen SNR rechnet, ist FALSCH. Die Pipeline MUSS ihre Sicherheitsparameter kontinuierlich an den verbesserten Audio-Zustand anpassen.

| ID | Regel | Beschreibung |
|----|-------|-------------|
| §G82 | **Lebendiger CalibrationContext** | Der CalibrationContext ist NICHT statisch. Nach JEDER Phase MUSS Aurik prüfen, ob sich die für die Kalibrierung relevanten Messwerte (SNR, Bandbreite, Noise-Floor, Stereo-Kohärenz) signifikant geändert haben. Bei Änderung > Schwellwert MUSS der CalibrationContext aktualisiert und ALLE davon abhängigen Parameter neu berechnet werden. |
| §G83 | **NOVELTY_CRIT-Rekalibrierung** | Die NOVELTY_CRIT-Schwelle MUSS nach jeder signifikanten Audio-Verbesserung (SNR +3 dB, Bandbreite +1 kHz) NEU berechnet werden. Ein saubereres Signal rechtfertigt eine NIEDRIGERE Toleranz — was vorher „erwartete Neuheit" war, ist jetzt „verdächtige Veränderung". Die Formel bleibt dieselbe (§v10.41), aber die Eingabewerte (insbesondere restorability_score und effektive Bandbreite) sind die AKTUELLEN, nicht die initialen. |
| §G84 | **Phasen-Stärke-Drift-Korrektur** | Die Joint-Calibration berechnet Phasen-Stärken aus Goal-Gaps. Nach jeder Phase ändern sich die Goal-Proxies. Die Stärken der VERBLEIBENDEN Phasen MÜSSEN aus den AKTUELLEN Goal-Gaps neu berechnet werden — nicht aus den initialen. Der MidCalibrate-Mechanismus (33%/66%) ist ein MINIMUM — kritische Parameter (NOVELTY_CRIT, ECHO_THRESH) müssen nach JEDER Phase geprüft werden. |
| §G85 | **Rekalibrierungs-Audit** | Jede Rekalibrierung MUSS im Log dokumentiert werden: `"§RECALIB phase=%s: rs %.1f→%.1f SNR %.1f→%.1f dB → NOVELTY_CRIT %.3f→%.3f"`. Dies macht sichtbar, WIE sich Auriks Sicherheitsparameter während der Pipeline an das zunehmend sauberere Audio anpassen. |
| §G86 | **Monotonie-Garantie** | Die NOVELTY_CRIT-Schwelle darf während der Pipeline NUR sinken (konservativer werden) oder gleich bleiben — NIE steigen. Ein saubereres Signal rechtfertigt keine LASCHERE Toleranz. Die Monotonie MUSS im CalibrationContext erzwungen werden: `_NOVELTY_CRIT = min(current_calculation, previous_value)`. |

## Kategorie XII — Noise-Floor-Brücke Phase_03→Phase_26 (§G87)

| ID | Regel | Beschreibung |
|----|-------|-------------|
| §G87 | **Phase_26 Per-Band-Noise-Floor-Guard** | Phase_26 (DR-Expansion) MUSS die Lücke zwischen Phase_03 (Denoise) und dem finalen CD-Rauschprofil schließen. Die Downward-Expansion wird durch einen dreidimensionalen Guard kontrolliert: **(D1) Per-Band spektrale Floor-Targets**: Jedes der 4 Frequenzbänder hat einen eigenen Studio-Raumton — Bass −65 dBFS (Raumresonanz), Low-Mid −72 dBFS (Wärme), Mid-High −76 dBFS (Präsenz), High −70 dBFS (Luft). **(D2) Psychoakustische Maskierung**: Der Floor wird adaptiv um +8/+5/+2/0 dB relaxiert, wenn die Band-Energie > −20/−30/−40 dBFS beträgt — laute Bänder maskieren ihren eigenen Rauschboden, leise exponierte Bänder sind streng. **(D3) Temporale EMA-Glättung**: Floor-Anstieg (Entspannung) folgt mit α=0.15 (Attack ~50ms), Floor-Abfall (Verschärfung) mit α=0.05 (Release ~200ms). Kein Hard-Clamp — der Floor-Approach ist asymptotisch (correction = deficit × exp(−deficit/knee), knee=4 dB). Ergebnis: klingt nach Neuaufnahme, nicht nach Vinyl mit aufgezwungener CD-Stille. |

## Kategorie XIII — Defektbehebungs-Module auf höchster Qualitätsstufe (§G88)

| ID | Regel | Beschreibung |
|----|-------|-------------|
| §G88 | **Defektbehebung mit Depth-adaptiven DSP-Fallbacks** | Die vier Defektbehebungs-Module MÜSSEN bei transfer_depth≥3 und/oder unsicherer Gender-Detektion robuste, konservative DSP-Fallbacks verwenden — NIEMALS ungeprüfte ML-Inferenz auf degradierten Ketten oder gender-spezifische Annahmen ohne Fallback. **(1) Phase_07 Harmonic Restoration**: Tilt-Cap-Floor von 0.50 auf 0.35 absenken bei depth≥3 (§v10.60). Mehr harmonische Synthese durchlassen, da tiefe Ketten extreme Tilt-Abweichungen ohnehin erwarten. **(2) Phase_23 Spectral Repair**: FlashSR ML deaktivieren bei depth≥3 (§v10.60). ML halluciniert Frequenzen auf bereits 3× degradiertem Material. DSP-only spectral inpainting (PGHI + Wiener + NMF) ist robuster. **(3) Phase_19 De-Esser**: Bei Gender="unknown"/"" freq-agnostisches Band [4500–8000 Hz] statt gender-spezifischem Band (§v10.60). Verhindert Fehlklassifikation von männlichen Stimmen als weiblich (und umgekehrt) mit konsekutiver Über-/Unterbearbeitung. **(4) Phase_43 ML-DeEsser**: GENDER_FREQ_MAP["unknown"] = (5000, 9000 Hz) als konservativer Fallback (§v10.60). Breiteres, tieferes Band als gender-spezifische Bänder — fängt Sibilanz sicher ein, vermeidet aber Überbearbeitung. |

## Kategorie XIV — Unsichtbare Signalintegrität (§G89)

| ID | Regel | Beschreibung |
|----|-------|-------------|
| §G89 | **Soft-Clipping-Pflicht für alle 68 Phasen** | Jede Phase MUSS ihre Ausgabe via `apply_soft_clip()` (tanh-basiert, material-adaptiv) statt `np.clip(audio, -1.0, 1.0)` begrenzen (§v10.62). Hard-Clipping auf ±1.0 erzeugt ein Rechteck-Fenster im Zeitbereich → sinc-Spektrum mit hörbaren Obertönen bis Nyquist. Tanh-Soft-Clipping erzeugt nur ungerade Harmonische, die das Ohr als „analoge Sättigung" statt „digitalen Clip" wahrnimmt. Die zentrale Durchsetzung erfolgt in `PhaseResult.__post_init__` und `create_phase_result()` — damit sind alle Phasen-Ausgaben automatisch geschützt. Material-adaptive Knee: Shellac/Vinyl 1.2 dB, Tape/Cassette 0.8 dB, Digital 0.4 dB. |

---

## Änderungshistorie

| Version | Datum | Änderung |
|---------|-------|----------|
| 10.0.13 | 2026-08-03 | §G89: Soft-Clipping-Pflicht für alle 68 Phasen (§v10.62). `apply_soft_clip()` + `crossfade_to_bypass()` in audio_utils.py. Kategorie XIV. |
| 10.0.12 | 2026-08-03 | §G88: Defektbehebungs-Module (Phase_07/19/23/43) mit Depth-adaptiven DSP-Fallbacks. Kategorie XIII. |
| 10.0.11 | 2026-08-03 | §G87: Phase_26 Per-Band-Noise-Floor-Guard (D1–D3). Schließt Phase_03→Phase_26 Noise-Floor-Lücke. Kategorie XII. |
| 10.0.10 | 2026-07-19 | §G82–§G86: Laufzeit-Rekalibrierung. Lebendiger CalibrationContext, NOVELTY_CRIT-Nachführung, Monotonie-Garantie. Kategorie XI. |
| 10.0.9 | 2026-07-19 | §G76–§G81: Kalibrierungs-Dispatch. Zentraler CalibrationContext, kontinuierliche Ableitung aller Schwellwerte, Kalibrierungs-Audit. Kategorie X. |
| 10.0.8 | 2026-07-19 | §G68–§G75: SFT-Adaptivität, Defekt-Audibilität, Repair-Klassifikation. Kategorie IX. |
| 10.0.7 | 2026-07-13 | §G60–§G67 + §V27–§V33. Lag-Integritäts-Architektur nach Root-Cause-Analyse (8 Bugs, 13 Commits). Kategorie VII + VIII. |
| 10.0.6 | 2026-07-13 | §G46–§G59 (Metriken & Qualitätssicherung). Kategorie VI. |
| 10.0.5 | 2026-07-13 | §G30–§G39 (CD-Rauschprofil & Export, ML-Device, Test-Assertion). §V16–§V24. |
| 10.0.4 | 2026-07-13 | Initiale Formalisierung. CD-Rauschprofil (§G8, §G15–§G19, §V5, §V11–§V15). Kategorie I–III strukturiert. |

---

## Kategorie XV — Non-Plus-Ultra: Strukturelle Qualitäts-Deckel beseitigt (§G90–§G99)

> **Prämisse:** Die vier unabhängigen Root-Causes für „43→43" (keine messbare Qualitätsverbesserung) sind identifiziert und behoben. Diese Kategorie kodifiziert die architektonischen Garantien, die verhindern, dass Aurik jemals wieder gegen den defekten Input vergleicht, Exception-Schlucker ohne Logging verwendet oder Phasen ohne Cross-Phase-Koordination laufen.

| ID | Regel | Beschreibung |
|----|-------|-------------|
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
|----|-------|-------------|
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
|----|--------|-------------|
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
|----|-------|-------------|
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

## Änderungshistorie

| Version | Datum | Änderung |
|---------|-------|----------|
| 10.0.17 | 2026 | §G121 + §V39: Mode-Differenzierung RESTORATION/STUDIO_2026. Noise-Gate, Spektral-Balance, Stereo-Fokus NUR in STUDIO_2026. DoNoHarm-Referenzkorrektur (carrier_checkpoint→degraded_input). Depth-adaptiver ArtifactFreedom-Schwellwert. (§v10.119) |
| 10.0.16 | 2026 | §G113–§G120: Universelle Phasen-Sicherheit & Excellence-Kalibrierung. RMS-Guard, Transient-Shift, Hallucination-Guard, Formant-Guard, Groove-Guard, HPI-Gate, Silence-Guard, RX11-Kalibrierung. Kategorie XVII. (§v10.112–§v10.117) |
| 10.0.15 | 2026-08-10 | §G100–§G112 + §V34–§V38: Perzeptuelle Architektur §v10.101. |
| 10.0.14 | 2026-08-10 | §G90–§G99: Non-Plus-Ultra. Blinder Referenz-Vektor, Exception-Proxies, Cross-Phase-Koordination, NaN-Guards, Material-Vollständigkeit. Kategorie XV. |
