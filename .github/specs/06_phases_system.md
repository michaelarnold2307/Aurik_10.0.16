# Aurik 10 — Spec 06: Phasen-System | §v10 Pleasantness-First

> Vollständige Phase-Liste (Phase 01–64), CAUSE_TO_PHASES-Mapping,
> PhaseInterface-Pattern, Namenskonvention für neue Phasen.

---

## §7.1 Vollständige Phase-Liste (exakte Dateinamen in `backend/core/phases/`)

```text
phase_01_click_removal.py           Clicks/Impulse (Median-Detektion)
phase_02_hum_removal.py             Brumm 50/60 Hz + Obertöne (Kammfilter)
phase_03_denoise.py                 Breitrauschen (OMLSA + DeepFilterNet)
phase_04_eq_correction.py           Frequenzgang-Korrektur (parametrisch)
phase_05_rumble_filter.py           Tieffrequenzrumpeln (< 20 Hz Hochpass)
phase_06_frequency_restoration.py   Bandbreitenerweiterung / Shelving-EQ + SourceFidelityEQ (§2.42)
phase_07_harmonic_restoration.py    Oberton-Rekonstruktion
phase_08_transient_preservation.py  Transientenerhalt (Attack-Schutz)
phase_09_crackle_removal.py         Vinyl-Crackle (Impulsanteil)
phase_10_compression.py             Dynamikkompression
phase_11_limiting.py                Limiting (True-Peak-Schutz)
phase_12_wow_flutter_fix.py         Wow/Flutter-Korrektur (Pitch-Instabilität)
phase_13_stereo_enhancement.py      Stereo-Erweiterung
phase_14_phase_correction.py        Phasen-/Azimuth-Korrektur
phase_15_stereo_balance.py          Stereo-Balance L/R
phase_16_final_eq.py                Finales EQ-Trimming
phase_17_mastering_polish.py        Mastering-Politur
phase_18_noise_gate.py              Noise-Gate (Stille-Segmente)
phase_19_de_esser.py                De-Esser (Sibilanten-Reduktion)
phase_20_reverb_reduction.py        Dereverb / Nachhall-Reduktion
phase_21_exciter.py                 Harmonischer Exciter  ⚠ VERBOTEN in Restoration (§0a, UV3 _restoration_forbidden_stem_enhancement)
phase_22_tape_saturation.py         Tape-Sättigungs-Emulation
phase_23_spectral_repair.py         Spektrale Lücken-Reparatur (Apollo primär)
phase_24_dropout_repair.py          Dropout-Interpolation
phase_25_azimuth_correction.py      Azimuth-Korrektur (Bandlaufwerk)
phase_26_dynamic_range_expansion.py Dynamikbereich-Expansion
phase_27_click_pop_removal.py       Click/Pop-Entfernung (2. Pass)
phase_28_surface_noise_profiling.py Oberflächenrauschen-Profil (Vinyl)
phase_29_tape_hiss_reduction.py     Tape-Hiss + Print-Through (LMS-Adaptivfilter)
phase_30_dc_offset_removal.py       DC-Offset-Entfernung (Hochpass 5 Hz)
phase_31_speed_pitch_correction.py  Geschwindigkeit/Pitch-Korrektur
phase_32_mono_to_stereo.py          Mono→Stereo-Aufweitung
phase_33_stereo_width_limiter.py    Stereo-Breiten-Begrenzer
phase_34_mid_side_processing.py     Mid/Side-Verarbeitung
phase_35_multiband_compression.py   Multibandkompression (transparent)
phase_36_transient_shaper.py        Transient-Shaper
phase_37_bass_enhancement.py        Bass-Fundament-Anhebung
phase_38_presence_boost.py          Präsenz-Boost (Ära-bewusst: 2–6 kHz, SourceFidelity Mic-Center §2.42)
phase_39_air_band_enhancement.py    Air-Band-Enhancement (Ära-bewusst, BW-Cap §2.42, > 12 kHz)
                                    **[BUG-FIX v10.0.0] RESTORATION-EINSCHRÄNKUNG**: Phase_39 ist im
                                    Restoration-Modus für **alle analogen Materialien VERBOTEN**:
                                    vinyl, shellac, wax_cylinder, wire_recording, tape, reel_tape,
                                    cassette, lacquer_disc. Grund: Air-Band-Erweiterung über das
                                    physikalische BW-Ceiling (§0a §6.2c) des Originals erzeugt
                                    Halluzinationen (+18 dB Air-Band-Energie bei Vinyl > 16 kHz).
                                    Erlaubt: Studio 2026 (beide Modi), digitale Quellen (cd_digital,
                                    mp3_low, mp3_high, dat, md) in Restoration wenn BW-Ceiling passt.
phase_40_loudness_normalization.py  LUFS-Normierung + Pegeldrift-Korrektur (§2.46 Stufe 4.5)
phase_41_output_format_optimization.py  Ausgabe-Format-Optimierung
phase_42_vocal_enhancement.py       Gesangs-Enhancement
phase_43_ml_deesser.py              ML-gestützter De-Esser
phase_44_guitar_enhancement.py      Gitarren-Enhancement (PANNs conf ≥ 0.50)
phase_45_brass_enhancement.py       Blechbläser-Enhancement (PANNs conf ≥ 0.50)
phase_46_spatial_enhancement.py     Spatial-Enhancement (Raumklang)
phase_47_truepeak_limiter.py        True-Peak-Limiter (EBU R128, −1.0 dBTP)
phase_48_stereo_width_enhancer.py   Stereo-Breiten-Enhancer (MS, Blumlein, Schroeder)
phase_49_advanced_dereverb.py       Advanced Dereverb (Blind-RIR → WPE)
phase_50_spectral_repair.py         Spektrale Gesamt-Reparatur
phase_51_drums_enhancement.py       Schlagzeug-Enhancement (PANNs conf ≥ 0.5)
phase_52_piano_restoration.py       Klavier-Restaurierung
phase_53_semantic_audio.py          Semantische Audio-Analyse
phase_54_transparent_dynamics.py    Transparente Dynamik-Verarbeitung
phase_55_diffusion_inpainting.py    DiffWave / CQTdiff+ / FlowMatching Inpainting
                                    (adaptiver AR-Order: AR(64) < 50 ms, AR(192) ≥ 50 ms)
phase_56_spectral_band_gap_repair.py HEAD_WEAR: Frequenzband-Lücken-Reparatur
                                    (SpectralBandGapRepair — nur bei conf ≥ 0.55)
phase_57_print_through_reduction.py  Print-Through-Reduktion
                                    (bidirektionale LMS: Pre-/Post-Echo getrennt)
phase_58_lyrics_guided_enhancement.py Lyrics-gestütztes Enhancement
                                    (Whisper-Tiny ONNX → wav2vec2-Phonem-Alignment →
                                     ContentAwareProcessor; Latenz ≤ 8 s/min;
                                     produktiver Pfad ausschließlich über
                                     backend/core/lyrics_guided_enhancement.py;
                                     aktiviert wenn Vocals erkannt; §2.36 PFLICHT)
phase_59_modulation_noise_reduction.py Modulationsrauschen-Reduktion
                                    (signalabhängige Rauschminderung bei Bandaufnahmen)
phase_60_inner_groove_distortion_repair.py Inner-Groove-Distortion-Reparatur
                                    (positionsadaptive THD/Asymmetrie-Korrektur)
phase_61_groove_echo_cancellation.py Groove-Echo-Kompensation
                                    (template-basierte Vinyl-Vorecho-Unterdrückung)
phase_62_crosstalk_cancellation.py   Crosstalk-Kompensation
                                    (BSS-basierte Kanalentflechtung)
phase_63_intermodulation_reduction.py Intermodulations-Reduktion
                                    (Volterra-basierte IMD-Tilgung)
phase_64_tape_splice_repair.py       Tape-Splice-Reparatur
                                    (Klick-, Pegel- und Phasendiskontinuität an Klebestellen)
phase_65_vocal_naturalness_restoration.py  DSP-Vocal-Naturalness-Restaurierung
                                    (HNR-Blend + Spektral-Tilt-Korrektur + Formant-Tilt-Korrektur;
                                     nur Restoration, §0a-konform; kein ML, kein Enhancement)
```

**Phase-58-Datenvertrag (bindend ab v10.0.0):**

- Persistiert oder geloggt werden dürfen nur Segmentzeiten, `phoneme_type`, Konfidenzen, Fallback-Flags und aggregierte Zähler.
- Verboten sind Worttext, Transkript, Voll-Lyrics und Roh-Alignment-Tokens in `RestorationResult.metadata`, Checkpoints, Logger-Ausgaben und Debug-UI.
- Legacy-/Forschungsimplementierungen unter `backend/lyrics_guided/` sind nicht Teil des produktiven Phase-58-Vertrags.

---

### v10-Amendment (2026-07-12)

**Phase 24 stereo-lag safety: WARNING→INFO** (v10, 2026-07-12):

Die Stereo-Lag-Safety in `phase_24_dropout_repair.py` korrigiert erfolgreich eingeführte L/R-Kanal-Versätze, die durch unabhängige Dropout-Reparatur pro Kanal entstehen (typisch bei Cassette-Material mit Azimuth-Fehlausrichtung). Die Korrektur ist eine **erfolgreiche Fix-Operation**, kein Fehler.

`logger.warning` → `logger.info`.

**SSIP Fallback-Flut: INFO→DEBUG** (v10, 2026-07-12):

Die Meldung »`structural_silence_zones nicht in context`« in `structural_silence_isolation.py` feuerte einmal pro Dropout-Region (~70× pro Phase-24-Durchlauf). Der Fallback-Pfad ist der erwartete Normalpfad (der Context enthält diese Zonen nur in Ausnahmefällen).

`logger.info` → `logger.debug`.

## §7.1d [RELEASE_MUST] 64-Phasen-SOTA-Bindung und Strength-Oracle-Matrix (v10.0.0)

Die folgende Matrix ist die bindende Vorgabe fuer Phase 01-64. Sie definiert pro Phase:

1. den primaeren SOTA-Algorithmus im produktiven Zusammenspiel,
2. den zulaessigen Fallback,
3. die verpflichtende Oracle-Klasse fuer die optimale Interventionsstaerke,
4. den primaren Teamwork-Beitrag zum 15er-Zielvektor.

**Wichtig:** Die Matrix ersetzt keine Safety-Gates. Sie bestimmt nur den besten lokalen
Werkzeug- und Staerkepfad im Gesamtsystem.

### §7.1d.1 Rollout-Contract fuer Strength-Oracles (off/pilot/all)

Zur sicheren Produktivierung ist der Rollout-Modus kanonisch als `phase_strength_oracle_rollout`
zu fuehren. Gueltige Werte sind:

- `off`: Strength-Oracles deaktiviert, Phasen laufen mit klassischer Parametrik.
- `pilot`: Strength-Oracles nur fuer `_PHASE_STRENGTH_ORACLE_PILOT_PHASES`.
- `all`: Strength-Oracles fuer alle in `_PHASE_INTERVENTION_CLASS` gebundenen Phasen.

Normative Anforderungen:

1. Der Modus MUSS entlang des kanonischen Pfads durchgereicht werden:
    CLI/Bridge -> `AurikDenker.denke(...)` -> `RestaurierDenker.restauriere(...)` -> `UnifiedRestorerV3.restore(...)`.
2. Prioritaet bei der Aufloesung: explizite Runtime-kwargs vor Context/Config-Default.
3. Pro Phase MUSS Telemetrie gesetzt werden:
    `phase_strength_oracle_rollout_mode` und `phase_strength_oracle_enabled_for_phase`.
4. Ungueltige Werte duerfen keinen Abbruch erzeugen; sie fallen auf den sicheren Default (`all`) zurueck.

### Phasen 01-16

| Phase | Primär-SOTA | Fallback | Oracle | Hauptbeitrag im 15er-Team |
| --- | --- | --- | --- | --- |
| `phase_01_click_removal` | RBME-Net + sparse outlier surgery + lokales Patch-Matching | median+LPC repair | `O1_impulse` | Natürlichkeit, Authentizität, Artikulation |
| `phase_02_hum_removal` | harmonische Kalman-Notch-Kette mit Netzfrequenz-Tracking | adaptive comb + notch bank | `O9_periodic_cancellation` | Transparenz, Wärme, BassKraft |
| `phase_03_denoise` | DeepFilterNet v3.II + OMLSA/IMCRA, vokal-lastig MIIPHER only as last-resort stem rescue | OMLSA/MMSE-LSA | `O2_subtractive` | Natürlichkeit, Transparenz, VocalQuality |
| `phase_04_eq_correction` | optimal-transport spectral matching + parametrische Minimum-Phase-EQ | parametric target EQ | `O3_spectral_balance` | Timbre, TonalCenter, Wärme |
| `phase_05_rumble_filter` | psychoakustisch begrenzter linear-phase HPF + subharmonic rumble estimator | Butterworth HPF | `O2_subtractive` | BassKraft, Natürlichkeit, Transparenz |
| `phase_06_frequency_restoration` | AudioSR + SourceFidelityEQ + BW-ceiling aware shelving | sinusoidal+stochastic bandwidth restoration | `O3_spectral_balance` | Brillanz, Transparenz, Timbre |
| `phase_07_harmonic_restoration` | harmonic lattice + DDSP partial reconstruction; Tilt-Cap-Floor 0.35 at depth≥5 (§v10.60, §G88-1, §v10.120) | sinusoidal+stochastic harmonic fill | `O3_spectral_balance` | Timbre, BassKraft, Brillanz |
| `phase_08_transient_preservation` | onset-protection mask + HPSS-rescue + transient re-injection | linked transient shaper | `O1_impulse` | Artikulation, Groove, MikroDynamik |
| `phase_09_crackle_removal` | BANQUET/RBME hybrid + LPC/AR micro-gap interpolation | iterative sparse Bayes decrackle | `O1_impulse` | Natürlichkeit, Transparenz, Authentizität |
| `phase_10_compression` | crest-aware broadband compressor with loudness-sidechain | transparent VCA-style compressor | `O6_dynamics` | MikroDynamik, Emotionalität, Transparenz |
| `phase_11_limiting` | 8x oversampled ISP limiter | 4x true-peak limiter | `O10_output` | ArtifactFreedom, Transparenz, Export-Sicherheit |
| `phase_12_wow_flutter_fix` | FCPE/RMVPE-guided wow/flutter solver + PSOLA/phase correction + head-level stabilizer | pYIN + WSOLA | `O4_time_pitch` | TonalCenter, Groove, Authentizität |
| `phase_13_stereo_enhancement` | correlation-bounded M/S ambience lift | linked stereo shelf widening | `O5_stereo_field` | Raumtiefe, Transparenz, Mono-Kompatibilität |
| `phase_14_phase_correction` | GCC-PHAT + complex all-pass azimuth solver | phase rotator + coherence maximization | `O4_time_pitch` | TonalCenter, Raumtiefe, Authentizität |
| `phase_15_stereo_balance` | loudness-matched linked L/R rebalancer | RMS-energy rebalance | `O5_stereo_field` | Raumtiefe, Transparenz, Authentizität |
| `phase_16_final_eq` | Pareto-constrained linear-phase finishing EQ | minimum-phase trim EQ | `O3_spectral_balance` | Timbre, Wärme, Brillanz |

### Phasen 17-32

| Phase | Primär-SOTA | Fallback | Oracle | Hauptbeitrag im 15er-Team |
| --- | --- | --- | --- | --- |
| `phase_17_mastering_polish` | source-fidelity contour polish + micro-tilt trim | gentle mastering EQ chain | `O3_spectral_balance` | Authentizität, Brillanz, Wärme |
| `phase_18_noise_gate` | vocal-/phoneme-aware soft expander with linked stereo | adaptive noise gate | `O2_subtractive` | Natürlichkeit, Artikulation, MikroDynamik |
| `phase_19_de_esser` | multiband phoneme-aware de-esser; freq-agnostic band [4500–8000 Hz] on gender=unknown (§v10.60, §G88-3) | dynamic EQ de-esser | `O7_vocal_articulation` | Artikulation, VocalQuality, Natürlichkeit |
| `phase_20_reverb_reduction` | SGMSE+ + WPE hybrid dereverb | WPE + OMLSA tail trim | `O2_subtractive` | Transparenz, Raumtiefe, Natürlichkeit |
| `phase_21_exciter` | hallucination-guarded harmonic exciter with material ceiling | harmonic shelf + soft saturation | `O3_spectral_balance` | Brillanz, Präsenz, Separation |
| `phase_22_tape_saturation` | hysteresis/Volterra tape curve with era profile | soft saturation model | `O6_dynamics` | Wärme, Authentizität, MikroDynamik |
| `phase_23_spectral_repair` | Apollo v2 + AudioSR + PGHI; DSP-only (FlashSR ML deaktiviert) at depth≥5 (§v10.60, §G88-2, §v10.120) | consistent Wiener + NMF repair | `O8_generative_repair` | Transparenz, Brillanz, Authentizität |
| `phase_24_dropout_repair` | boundary-aware AudioSR/CQTdiff with SSIP | sinusoidal-stochastic/LPC fill | `O8_generative_repair` | Artikulation, Authentizität, Natürlichkeit |
| `phase_25_azimuth_correction` | coherence-maximizing tape azimuth solver | sample-delay + phase rotator | `O4_time_pitch` | Raumtiefe, Authentizität, Timbre |
| `phase_26_dynamic_range_expansion` | masking-aware upward/downward expansion with per-band noise-floor guard (§G87, §v10.61) | broadband expander with per-band floor targets | `O6_dynamics` | MikroDynamik, Emotionalität, Transparenz, CD-Raumton |
| `phase_27_click_pop_removal` | second-pass sparse impulse classifier + waveform patching | adaptive median + interpolation | `O1_impulse` | Natürlichkeit, Transparenz, Authentizität |
| `phase_28_surface_noise_profiling` | noise-texture fingerprint estimation with carrier prior | spectral noise profile estimator | `O2_subtractive` | Authentizität, Wärme, Transparenz |
| `phase_29_tape_hiss_reduction` | DeepFilterNet v3.II HF + OMLSA low band + harmonic mask | OMLSA/IMCRA fullband | `O2_subtractive` | Transparenz, VocalQuality, Natürlichkeit |
| `phase_30_dc_offset_removal` | robust DC servo with percentile anchor | 5 Hz HPF | `O10_output` | Authentizität, Headroom, Export-Sicherheit |
| `phase_31_speed_pitch_correction` | RMVPE/FCPE + DTW/phase-locked correction | pYIN + phase vocoder | `O4_time_pitch` | TonalCenter, Groove, Artikulation |
| `phase_32_mono_to_stereo` | decorrelated ambience synthesis with mono guard | Haas/filtered pseudo-stereo | `O5_stereo_field` | Raumtiefe, Separation, Transparenz |

### Phasen 33-48

| Phase | Primär-SOTA | Fallback | Oracle | Hauptbeitrag im 15er-Team |
| --- | --- | --- | --- | --- |
| `phase_33_stereo_width_limiter` | mono-compatible M/S width clamp | linked width reduction | `O5_stereo_field` | Mono-Kompatibilität, Raumtiefe, Authentizität |
| `phase_34_mid_side_processing` | goal-aware M/S energy reweighting | static M/S processing | `O5_stereo_field` | Separation, Raumtiefe, Wärme |
| `phase_35_multiband_compression` | linked multiband compression with psychoacoustic sidechains | transparent multiband comp | `O6_dynamics` | Transparenz, MikroDynamik, BassKraft |
| `phase_36_transient_shaper` | onset-masked transient shaping | linked attack/sustain shaper | `O6_dynamics` | Artikulation, Groove, MikroDynamik |
| `phase_37_bass_enhancement` | virtual-pitch bass reconstruction + harmonic bass fill | low-band saturation + shelving | `O3_spectral_balance` | BassKraft, Wärme, Groove |
| `phase_38_presence_boost` | SourceFidelity mic-center presence sculpting | constrained bell/shelf EQ | `O3_spectral_balance` | Artikulation, Transparenz, VocalQuality |
| `phase_39_air_band_enhancement` | ceiling-aware SBR/NVSR air restoration | constrained air shelf | `O3_spectral_balance` | Brillanz, Raumtiefe, Transparenz |
| `phase_40_loudness_normalization` | ITU-R BS.1770-5 + quiet-zone clamp + true-peak aware gain + musikgegated inverse Pegeldrift-Hüllkurve | LUFS-only normalization | `O10_output` | Export-Kohärenz, Transparenz, MikroDynamik, Pegelstabilität |
| `phase_41_output_format_optimization` | high-order noise-shaped dither + sinc resample + codec-safe export prep | TPDF dither + standard resample | `O10_output` | Export-Sicherheit, Authentizität, Transparenz |
| `phase_42_vocal_enhancement` | stem-aware vocal enhancement with formant/vibrato guards | MIIPHER-lite DSP-assisted vocal polish | `O7_vocal_articulation` | VocalQuality, Artikulation, Emotionalität |
| `phase_43_ml_deesser` | phoneme-aware second-pass ML/DSP de-esser; GENDER_FREQ_MAP unknown=(5000,9000) Hz fallback (§v10.60, §G88-4) | dynamic EQ de-esser | `O7_vocal_articulation` | VocalQuality, Artikulation, Natürlichkeit |
| `phase_44_guitar_enhancement` | DDSP string-resonance refinement + pick-articulation contour | source-specific EQ/transient contour | `O3_spectral_balance` | Artikulation, Transparenz, Separation |
| `phase_45_brass_enhancement` | formant-/resonance-preserving brass contouring | dynamic EQ + harmonic tilt | `O3_spectral_balance` | Brillanz, Authentizität, Artikulation |
| `phase_46_spatial_enhancement` | binaural-cue-consistent spatial enhancement | M/S ambience widening | `O5_stereo_field` | Raumtiefe, Separation, Emotionalität |
| `phase_47_truepeak_limiter` | oversampled EBU true-peak limiter | ISP-safe limiter | `O10_output` | Export-Sicherheit, Transparenz, ArtifactFreedom |
| `phase_48_stereo_width_enhancer` | mono-guarded Blumlein/MS width enhancer | correlation-limited width enhancer | `O5_stereo_field` | Raumtiefe, Separation, Transparenz |

### Phasen 49-64

| Phase | Primär-SOTA | Fallback | Oracle | Hauptbeitrag im 15er-Team |
| --- | --- | --- | --- | --- |
| `phase_49_advanced_dereverb` | blind-RIR estimation + WPE + late-tail suppression | deterministic WPE | `O2_subtractive` | Transparenz, Raumtiefe, Natürlichkeit |
| `phase_50_spectral_repair` | consistency Wiener + anomaly detector + PGHI | NMF + spectral interpolation | `O8_generative_repair` | Transparenz, Authentizität, Brillanz |
| `phase_51_drums_enhancement` | drum-transient model + punch-preserving contour | transient shaper + EQ | `O6_dynamics` | Groove, Artikulation, MikroDynamik |
| `phase_52_piano_restoration` | inharmonic partial tracker + pedal-resonance aware reconstruction | harmonic EQ + transient rescue | `O3_spectral_balance` | Authentizität, TonalCenter, Raumtiefe |
| `phase_53_semantic_audio` | BEATs iter3 + CLAP + semantic scene fusion | PANNs + spectral fingerprint ensemble | `O10_output` | Kontextautorität fuer alle 15 Ziele, keine Dominanz |
| `phase_54_transparent_dynamics` | program-dependent transparent dynamics repair with envelope smoothing | broadband transparent comp/exp | `O6_dynamics` | MikroDynamik, Emotionalität, Transparenz |
| `phase_55_diffusion_inpainting` | Flow Matching primary + CQTdiff+ secondary with SSIP | DiffWave + NMF-beta | `O8_generative_repair` | Natürlichkeit, Authentizität, Artikulation |
| `phase_56_spectral_band_gap_repair` | harmonic continuation + FCPE-guided band-gap synthesis | LPC + spectral interpolation | `O8_generative_repair` | Brillanz, Timbre, Authentizität |
| `phase_57_print_through_reduction` | bidirectional predictive echo suppression + periodic template cancel | LMS pre/post-echo subtraction | `O9_periodic_cancellation` | Transparenz, Authentizität, Natürlichkeit |
| `phase_58_lyrics_guided_enhancement` | Whisper-Tiny + wav2vec2 alignment + ContentAwareProcessor | phoneme-boundary DSP only | `O7_vocal_articulation` | Artikulation, VocalQuality, Emotionalität |
| `phase_59_modulation_noise_reduction` | modulation-spectrogram suppression + cyclostationary tracking | adaptive modulation filter | `O9_periodic_cancellation` | Natürlichkeit, Transparenz, Wärme |
| `phase_60_inner_groove_distortion_repair` | position-aware Volterra THD inversion | asymmetric harmonic de-warping | `O9_periodic_cancellation` | Authentizität, Artikulation, Brillanz |
| `phase_61_groove_echo_cancellation` | one-revolution template subtraction + OT alignment | periodic pre-echo suppressor | `O9_periodic_cancellation` | Transparenz, Authentizität, Groove |
| `phase_62_crosstalk_cancellation` | constrained BSS/de-mixing with mono guard | least-squares crosstalk inversion | `O5_stereo_field` | Separation, Raumtiefe, Mono-Kompatibilität |
| `phase_63_intermodulation_reduction` | Volterra-kernel inverse + sideband suppressor | IMD notch/spectral surgery | `O9_periodic_cancellation` | Transparenz, Authentizität, Brillanz |
| `phase_64_tape_splice_repair` | discontinuity classifier + phase-/level-matched spline patch | local click patch + gain/phase crossfade | `O1_impulse` | Authentizität, Natürlichkeit, MikroDynamik |

**Appendix:** `phase_65_vocal_naturalness_restoration` bleibt eine Restoration-only Sonderphase und
folgt `O7_vocal_articulation`; sie ist nicht Teil der 64er-Release-Matrix, aber unterliegt
denselben Teamwork- und Oracle-Invarianten.

**Transfer-Chain-Invariante [RELEASE_MUST]:** Oracle-Klasse und Phase bestimmen das lokale
Parameterprofil, die finale Interventionsstaerke MUSS zusaetzlich durch den
`transfer_chain`-konformen `chain_factor` konditioniert werden. Gleiche Defektlast bei
`vinyl -> cassette -> mp3_low` darf nie staerker eingreifen als bei reinem `vinyl`. `[SRC:S03,S04]`

**Bindende Interaktionsregeln:**

- SUBTRAKTIVE Phasen muessen zuerst auf Weighted-Gap-Closure ohne P1/P2-Schaden optimieren.
- ADDITIVE/ENHANCEMENT-Phasen duerren nur Headroom bis zum Material-/DR-/Hallucination-Ceiling nutzen.
- DYNAMICS-/STEREO-/OUTPUT-Phasen sind Team-Glue: sie stabilisieren die bereits erarbeiteten Goal-Gewinne,
  nicht ueberschreiben sie.
- `phase_53_semantic_audio` liefert Kontextautoritaet und darf nie selbst ein Einzelziel priorisieren.

---

## §7.1b [RELEASE_MUST] Phase 12 — Tape-Head-Level-Stabilizer v2 (v10.0.0)

`phase_12_wow_flutter_fix` enthält neben Wow/Flutter-Korrektur einen dedizierten
`TAPE_HEAD_LEVEL_DIP`-Pfad mit frequenzabhängiger Kompensation.

### Pflichtverhalten

1. Dip-Erkennung auf RMS-Hüllkurve (20 ms Fenster / 10 ms Hop) mit lokaler p75-Referenz.
2. Spektrale Korrektur in STFT-Domain:
    - Broadband-Gain aus Dip-Defizit
    - zusätzlicher HF-Tilt aus Kontext-vs.-Dip-Spektralverlust
3. SNR-Guard pro Frequenzbin: Bins nahe Noise-Floor dürfen nicht geboostet werden.
4. Asymmetrische Gain-Hüllkurve je Dip:
    - langsamer Onset (~30 %)
    - schnelle Recovery (~10 %)
5. §7.1a/§2.51 Stereo-Kohärenz: linked Stereo-Maske (identische Gain-Maske auf L/R).

### Sicherheitsinvarianten

- `strength < 0.01` → passthrough.
- `max_gain_db <= 15 dB`.
- Dips in quasi-Stille (`< -55 dBFS`) werden nicht repariert.
- NaN/Inf-Guard + hartes Clipping auf `[-1, 1]` am Phasenende.

### Detektor-Kopplung

Dieser Pfad ist primär an `tape_head_contact_instability` gebunden und wird bei
`TAPE_HEAD_LEVEL_DIP`-Severity automatisch aktiviert; siehe Spec 05 §6.4b für
Cross-Material-Fallback und Periodizitäts-Marker.

---

## §7.1a [RELEASE_MUST] Stereo-Kohärenz-Pflicht für Phasen (v10.0.0)

Phasen, die auf Stereo-Audio operieren, dürfen L und R **nicht unabhängig** mit signal-modifizierendem DSP verarbeiten (separates Gate, separater Kompressor, separate Spektralreparatur). Dies erzeugt anti-phasige Transient-Artefakte in 2–3 Frame-Grenzen, die §2.49 korrekt als Phase-Cancellation flaggt und zurückrollt — mit direkter OQS-Auswirkung.

**Verbindliche Implementierungsstrategie pro Phase** (detailliert in §2.51 Spec 02):

| Phase | Pflicht-Strategie | Verbot |
| --- | --- | --- |
| `phase_07_harmonic_restoration` | **M/S-Domain**: Harmonics nur auf Mid, Side unverändert | Separate Oberton-Synthese für L und R |
| `phase_18_noise_gate` | **Linked Stereo**: Gate öffnet wenn `max(L_rms, R_rms) > threshold` | Unabhängige Gate-Entscheidung pro Kanal |
| `phase_23_spectral_repair` | **M/S-Domain**: Spektralreparatur auf Mid; Side minimal/nicht bearbeiten | Separate Lückenfüllung für L und R |
| `phase_24_dropout_repair` | **Linked Stereo**: Dropout-Grenze erkannt wenn BEIDE Kanäle unterfallen; Füllung kohärent | Unabhängige L/R-Dropout-Erkennung |
| `phase_35_multiband_compression` | **Linked Stereo**: Gain auf `√(L²+R²)/√2`; gleicher Gain für L und R | Separate GR-Berechnung pro Kanal |

**Zusatzpflicht für `phase_23_spectral_repair` (ML-Pfad, RELEASE_MUST):**

- AudioSR/Apollo-Inference auf Stereo MUSS Mid-zentriert erfolgen (M/S-Domain), Side bleibt unverändert oder nur minimal skaliert.
- **VERBOTEN**: Unabhängige ML-Inferenz auf L und R mit separater Nachkorrektur der Längen.
- **VERBOTEN**: Kanalweises Resampling als primäre Längenkorrektur nach ML-Inferenz.

**Code-Pattern für M/S-Domain** (Referenz-Implementierung):

```python
def _process_ms_domain(audio: np.ndarray, process_fn, side_strength: float = 0.0) -> np.ndarray:
    """Process stereo audio in M/S domain. process_fn anwenden auf Mid;
    Side mit side_strength skaliert (0.0 = unverändert, 1.0 = voll verarbeitet)."""
    assert audio.shape[0] == 2  # channel-first (2, N)
    mid = (audio[0] + audio[1]) / 2.0
    side = (audio[0] - audio[1]) / 2.0
    mid_processed = process_fn(mid)
    if side_strength > 0.0:
        side_processed = process_fn(side) * side_strength + side * (1.0 - side_strength)
    else:
        side_processed = side
    l = np.clip(mid_processed + side_processed, -1.0, 1.0)
    r = np.clip(mid_processed - side_processed, -1.0, 1.0)
    return np.stack([l, r], axis=0)
```

**Code-Pattern für Linked Stereo** (Referenz-Implementierung):

```python
def _compute_linked_gain(l_audio, r_audio, gain_fn) -> np.ndarray:
    """Gain-Kurve auf kombiniertem RMS berechnen; identisch auf L und R anwenden."""
    combined_rms = np.sqrt((l_audio ** 2 + r_audio ** 2) / 2.0 + 1e-12)
    gain_curve = gain_fn(combined_rms)  # z.B. Gate-Öffnen, Kompressor-GR
    l_out = np.clip(l_audio * gain_curve, -1.0, 1.0)
    r_out = np.clip(r_audio * gain_curve, -1.0, 1.0)
    return np.stack([l_out, r_out], axis=0)
```

---

## §7.1c [RELEASE_MUST] Phasen-Familien-Taxonomie — ADDITIVE / SUBTRAKTIVE / DYNAMICS (v10.0.0)

Jede Phase gehört zu **genau einer** Phasen-Familie. Die Zuordnung steuert:

- `_post_additive_bw_guard()` (§6.2c): BW-Ceiling-Guard NUR nach ADDITIVE-Block
- `NoiseTextureCoherenceGuard` (§4.7): nur nach SUBTRAKTIVE-Phasen
- `ArtifactFreedomGate` Roughness/Sharpness-Penalty (§2.49c): nur DYNAMICS/ADDITIVE/ENHANCEMENT

| Familie | Phase-IDs | Beschreibung |
| --- | --- | --- |
| **SUBTRAKTIVE** | phase_01, phase_02, phase_03, phase_05, phase_09, phase_10, phase_17, phase_18, phase_19, phase_20, phase_22, phase_24, phase_27, phase_28, phase_29, phase_30, phase_31, phase_49, phase_50, phase_60, phase_63 | Entfernt Energie (Rauschen, Knistern, Klicks, Reverb, DC-Offset) |
| **ADDITIVE** | phase_06, phase_07, phase_21, phase_23, phase_37, phase_38, phase_39, phase_55, phase_56 | Fügt Frequenzinhalt hinzu (BW-Extension, Harmonik, Inpainting) |
| **DYNAMICS** | phase_11, phase_26, phase_33, phase_34, phase_35, phase_36, phase_40 | Verändert Dynamik/Lautstärke (Kompression, Expansion, Normalisierung) |
| **CORRECTION** | phase_04, phase_08, phase_12, phase_13, phase_14, phase_15, phase_25, phase_32, phase_41 | Korrigiert technische Parameter (EQ, Speed, Phase, Azimuth, Stereo) |
| **ENHANCEMENT** | phase_42, phase_43, phase_44, phase_45, phase_46, phase_47, phase_48, phase_51, phase_52, phase_53, phase_54, phase_57, phase_58, phase_59, phase_61, phase_62, phase_64 | Instrument-spezifisch, Stem-Enhancement, Lyrics-guided |

**Invariante (§6.2c)**: UV3 `_post_additive_bw_guard()` MUSS nach der letzten ADDITIVE-Phase im Pipeline-Block laufen. Neue Phasen MÜSSEN hier zugeordnet werden — fehlende Zuordnung ist ein RELEASE-Blocker.

**[RELEASE_MUST] Ceiling-Pflicht für ADDITIVE- und DYNAMICS-Phasen**:

Jede ADDITIVE-Phase MUSS `_MATERIAL_BW_CEILING_HZ` (Spec 05 §6.2c) respektieren:

| ADDITIVE-Phase | Ceiling-Referenz | Guard |
| --- | --- | --- |
| `phase_06` (BW-Extension) | `_MATERIAL_BW_CEILING_HZ[material]` | Hard-Cap auf generiertes HF-Band |
| `phase_07` (Harmonik) | `_MATERIAL_BW_CEILING_HZ[material]` | Obertöne oberhalb Ceiling unterdrücken |
| `phase_23` (Spectral Repair) | `_MATERIAL_BW_CEILING_HZ[material]` | Apollo/AudioSR-Output BW-begrenzen |
| `phase_39` (Air-Band) | `_MATERIAL_BW_CEILING_HZ[material]` | Air-Band nur bis Ceiling |
| `phase_55` (Inpainting) | `_MATERIAL_BW_CEILING_HZ[material]` | Diffusion-Output BW-begrenzen |
| `phase_56` (Band-Gap) | `_MATERIAL_BW_CEILING_HZ[material]` | Gap-Fill nur bis Ceiling |

Jede DYNAMICS-Phase mit Expansion MUSS `_MATERIAL_DR_CEILING_DB` (Spec 05 §6.2b) respektieren:

| DYNAMICS-Phase | Ceiling-Referenz | Guard |
| --- | --- | --- |
| `phase_26` (DR-Expansion) | `_MATERIAL_DR_CEILING_DB[material]` | Expansion bis Ceiling, nicht darüber |
| `phase_54` (Transparent Dynamics) | `_MATERIAL_DR_CEILING_DB[material]` | DR-Change begrenzt auf Ceiling |

**Invariante**: Fehlender Ceiling-Check in einer ADDITIVE/DYNAMICS-Phase ist ein §0a-Verstoß (Material-Ceiling-Überschreitung = Artefakt).

**Implementierung**: `backend/core/phase_ontology.py` — `PHASE_FAMILY_MAP: dict[str, str]`

---

## §7.2 CAUSE_TO_PHASES-Mapping (CausalDefectReasoner)

**Merge-Regel mit `_MATERIAL_PRIORITY_PHASES` (§6.2a)**:

Beide Dicts liefern unabhängig voneinander Phase-Listen. Die resultierende Phase-Selektion
ist die **Vereinigung (Union)** beider Quellen:

```
selected = set(MATERIAL_PRIORITY_PHASES[material]) | set()
for cause in detected_causes:
    selected |= set(CAUSE_TO_PHASES[cause])
```

- Material-Phasen sind **immer aktiv** (§6.2a Invariante), auch wenn kein Defect sie triggert.
- Cause-Phasen fügen kontextspezifische Reparatur hinzu (z. B. `vocal_harshness` → `phase_43`).
- Duplikate werden durch die Vereinigung automatisch entfernt.
- `GoalApplicabilityFilter` darf einzelne Phasen nachträglich deaktivieren (z. B. phase_48 bei Mono).

```python
CAUSE_TO_PHASES = {
    "tape_dropout":              ["phase_24_dropout_repair", "phase_55_diffusion_inpainting",
                                  "phase_01_click_removal", "phase_03_denoise"],
    "tape_hiss":                 ["phase_29_tape_hiss_reduction", "phase_03_denoise",
                                  "phase_04_eq_correction", "phase_40_loudness_normalization"],
    "vinyl_crackle":             ["phase_09_crackle_removal", "phase_01_click_removal",
                                  "phase_28_surface_noise_profiling", "phase_03_denoise"],
    "vinyl_warp":                ["phase_12_wow_flutter_fix", "phase_31_speed_pitch_correction",
                                  "phase_04_eq_correction", "phase_03_denoise"],
    "wow_flutter":                ["phase_12_wow_flutter_fix", "phase_31_speed_pitch_correction",
                                  "phase_04_eq_correction", "phase_03_denoise"],
    "wow":                        ["phase_12_wow_flutter_fix", "phase_31_speed_pitch_correction",
                                  "phase_04_eq_correction"],
    "flutter":                    ["phase_12_wow_flutter_fix", "phase_08_transient_preservation",
                                  "phase_31_speed_pitch_correction", "phase_03_denoise"],
    "electrical_hum":            ["phase_02_hum_removal", "phase_03_denoise", "phase_04_eq_correction"],
    "head_misalignment":         ["phase_06_frequency_restoration", "phase_04_eq_correction",
                                  "phase_14_phase_correction", "phase_25_azimuth_correction",
                                  "phase_03_denoise"],
    "dc_offset":                 ["phase_30_dc_offset_removal", "phase_40_loudness_normalization"],
    "digital_clip":              ["phase_23_spectral_repair", "phase_06_frequency_restoration",
                                  "phase_40_loudness_normalization"],
    "bandwidth_loss":            ["phase_06_frequency_restoration", "phase_07_harmonic_restoration"],
                                  # phase_39 ENTFERNT (BUG-FIX v10.0.0 §6.2c): Air-Band-Erweiterung
                                  # über BW-Ceiling analoger Materialien erzeugt Halluzinationen
                                  # im Restoration-Modus. Studio-2026 und digitale Quellen bleiben unberührt.
    "high_freq_noise":           ["phase_29_tape_hiss_reduction", "phase_03_denoise",
                                  "phase_18_noise_gate"],
    "stereo_imbalance":          ["phase_15_stereo_balance", "phase_33_stereo_width_limiter",
                                  "phase_34_mid_side_processing"],
    "phase_issues":              ["phase_14_phase_correction", "phase_25_azimuth_correction"],
    "pitch_drift":               ["phase_31_speed_pitch_correction", "phase_12_wow_flutter_fix"],
    "reverb_excess":             ["phase_20_reverb_reduction", "phase_49_advanced_dereverb"],
    "print_through":             ["phase_57_print_through_reduction", "phase_29_tape_hiss_reduction",
                                  "phase_24_dropout_repair", "phase_03_denoise", "phase_23_spectral_repair"],
    "digital_artifacts":         ["phase_23_spectral_repair", "phase_50_spectral_repair",
                                  "phase_06_frequency_restoration"],
    "compression_artifacts":     ["phase_23_spectral_repair", "phase_50_spectral_repair",
                                  "phase_26_dynamic_range_expansion", "phase_06_frequency_restoration",
                                  "phase_54_transparent_dynamics"],
    "quantization_noise":        ["phase_23_spectral_repair", "phase_03_denoise",
                                  "phase_06_frequency_restoration"],
    "jitter_artifacts":          ["phase_23_spectral_repair", "phase_14_phase_correction"],
    "dynamic_compression_excess":["phase_26_dynamic_range_expansion", "phase_54_transparent_dynamics"],
                                  # phase_35_multiband_compression ENTFERNT (BUG-FIX v10.0.0 §0a): Stem-Enhancement VERBOTEN in Restoration
    "head_wear":                 ["phase_56_spectral_band_gap_repair", "phase_14_phase_correction",
                                  "phase_06_frequency_restoration"],
    # BUG-FIX v10.0.0 §2.59/V12: "azimuth_error" ENTFERNT — kein CAUSES-Gegenstück in Code.
    # "azimuth_error" ist ein DefectScanner-Messwert (Eingabe in Likelihood-Funktionen
    # für head_misalignment/head_wear), keine eigenständige Kausalursache.
    # Azimuth-Korrektur wird über head_misalignment + tape_start_instability getriggert.
    "soft_saturation":           [],  # BEWAHREN — kein destruktiver Eingriff
    "pre_echo":                  ["phase_23_spectral_repair", "phase_50_spectral_repair",
                                  "phase_08_transient_preservation"],
    "low_freq_rumble":           ["phase_05_rumble_filter", "phase_03_denoise",
                                  "phase_04_eq_correction"],
    "transient_smearing":        ["phase_08_transient_preservation", "phase_36_transient_shaper",
                                  "phase_23_spectral_repair"],
    "clipping":                  ["phase_23_spectral_repair", "phase_06_frequency_restoration"],
    # Neu v10.0.0:
    "riaa_curve_error":          ["phase_04_eq_correction", "phase_06_frequency_restoration",
                                  "phase_07_harmonic_restoration"],
    "aliasing":                  ["phase_23_spectral_repair",
                                  "phase_50_spectral_repair"],
    "bias_error":                ["phase_04_eq_correction", "phase_03_denoise",
                                  "phase_06_frequency_restoration", "phase_29_tape_hiss_reduction"],
    # Sibilanten (§6.3 v10.0.0):
    "sibilance":                 ["phase_19_de_esser", "phase_43_ml_deesser"],
                                  # phase_42_vocal_enhancement ENTFERNT (BUG-FIX v10.0.0 §0a): Stem-Enhancement VERBOTEN in Restoration
    # Transport-Bump (v10.0.0b — Kassetten-Holpern):
    "transport_bump":            ["phase_12_wow_flutter_fix", "phase_24_dropout_repair",
                                  "phase_31_speed_pitch_correction"],
    # Vocal-Harshness (v10.0.0 — Vokal-Härte/Übersteuerung/Kratzigkeit):
    "vocal_harshness":           ["phase_19_de_esser", "phase_43_ml_deesser",
                                  "phase_23_spectral_repair"],
                                  # phase_42_vocal_enhancement ENTFERNT (BUG-FIX v10.0.0 §0a): Stem-Enhancement VERBOTEN in Restoration
    # Neu v10.0.0/98:
    "tape_start_instability":    ["phase_12_wow_flutter_fix", "phase_25_azimuth_correction",
                                  "phase_31_speed_pitch_correction", "phase_14_phase_correction",
                                  "phase_24_dropout_repair"],
    "tape_head_contact_instability": ["phase_12_wow_flutter_fix", "phase_24_dropout_repair",
                                  "phase_26_dynamic_range_expansion"],
    "modulation_noise":          ["phase_59_modulation_noise_reduction", "phase_03_denoise",
                                  "phase_29_tape_hiss_reduction"],
    "inner_groove_distortion":   ["phase_60_inner_groove_distortion_repair", "phase_23_spectral_repair",
                                  "phase_04_eq_correction"],
    "groove_echo":               ["phase_61_groove_echo_cancellation", "phase_20_reverb_reduction",
                                  "phase_03_denoise"],
    "crosstalk":                 ["phase_62_crosstalk_cancellation", "phase_14_phase_correction",
                                  "phase_34_mid_side_processing"],
    "intermodulation_distortion": ["phase_63_intermodulation_reduction", "phase_23_spectral_repair",
                                  "phase_04_eq_correction"],
    "tape_splice_artifact":      ["phase_64_tape_splice_repair", "phase_01_click_removal",
                                  "phase_24_dropout_repair"],
    "hf_remanence_loss":         ["phase_06_frequency_restoration", "phase_23_spectral_repair",
                                  "phase_04_eq_correction"],
    "stylus_damage":             ["phase_09_crackle_removal", "phase_23_spectral_repair",
                                  "phase_60_inner_groove_distortion_repair"],
    "sticky_shed_residue":       ["phase_24_dropout_repair", "phase_29_tape_hiss_reduction",
                                  "phase_03_denoise"],
    "multiband_wow_flutter":     ["phase_12_wow_flutter_fix", "phase_31_speed_pitch_correction",
                                  "phase_08_transient_preservation"],
    "generation_loss":           ["phase_06_frequency_restoration", "phase_03_denoise",
                                  "phase_23_spectral_repair", "phase_04_eq_correction"],
    "motor_interference":        ["phase_02_hum_removal", "phase_03_denoise",
                                  "phase_29_tape_hiss_reduction", "phase_04_eq_correction"],
    # ── v10.0.0: Pegelveränderung ──────────────────────────────────────────────
    "amplitude_drift":           ["phase_40_loudness_normalization"],
    # ── v10.0.0: DefectType→CAUSE-Lücken ───────────────────────────────────────
    "clicks":                    ["phase_01_click_removal", "phase_09_crackle_removal"],
    "dolby_nr_mismatch":         ["phase_04_eq_correction", "phase_29_tape_hiss_reduction",
                                  "phase_03_denoise"],
    "tape_head_level_dip":       ["phase_12_wow_flutter_fix", "phase_24_dropout_repair",
                                  "phase_40_loudness_normalization"],
    "scrape_flutter":            ["phase_12_wow_flutter_fix", "phase_31_speed_pitch_correction",
                                  "phase_08_transient_preservation"],
    "tape_head_clog":            ["phase_56_spectral_band_gap_repair", "phase_25_azimuth_correction",
                                  "phase_24_dropout_repair"],
    # ── v10.0.0: Erweiterte Kausal-Ursachen ───────────────────────────────────
    "proximity_effect_excess":   ["phase_04_eq_correction", "phase_05_rumble_filter"],
    "room_mode_resonance":       ["phase_04_eq_correction", "phase_16_final_eq",
                                  "phase_05_rumble_filter"],
    "nr_breathing_artifact":     ["phase_54_transparent_dynamics", "phase_08_transient_preservation"],
    "flutter_spectral_sidebands": ["phase_12_wow_flutter_fix", "phase_23_spectral_repair",
                                   "phase_08_transient_preservation"],
    "speed_calibration_error":   ["phase_12_wow_flutter_fix", "phase_31_speed_pitch_correction"],
    "overload_distortion":       ["phase_09_crackle_removal", "phase_23_spectral_repair",
                                  "phase_14_phase_correction"],
    "lacquer_disc_degradation":  ["phase_03_denoise", "phase_09_crackle_removal",
                                  "phase_01_click_removal", "phase_06_frequency_restoration"],
    "cassette_azimuth_tolerance": ["phase_14_phase_correction", "phase_25_azimuth_correction",
                                   "phase_06_frequency_restoration"],
    "wire_recording_specific":   ["phase_12_wow_flutter_fix", "phase_24_dropout_repair",
                                  "phase_03_denoise", "phase_01_click_removal"],
    "vocal_quality_degradation": ["phase_65_vocal_naturalness_restoration", "phase_03_denoise",
                                  "phase_19_de_esser"],
    "vocal_stem_noise":          ["phase_66_stem_targeted_nr",
                                  # Primary: BSRoFormer + stem-spezifische DFN-NR
                                  "phase_03_denoise",  # Sekundär: wideband-NR als Ergänzung
                                  "phase_65_vocal_naturalness_restoration"],  # Tertiär: VQI-Korrektiv wenn nötig
}
# PFLICHT: Jede neue Ursache → Eintrag hier UND in allen Material-Prior-Tabellen des DefectScanners.

# [RELEASE_MUST] §7.2a Severity-Weighted Phase-Reorder bei ≥3 Simultandefekten (v10.0.0+):

#
### §6.10 Phantom-Guards in PhaseInterface (v10.0.0-Phantom)

`PhaseInterface._safe_process()` integriert automatisch:
- **BreathPreserver**: Atemschutz vor/nach Noise-Reduction-Phasen
- **ComfortGuard**: Hörmüdungs-Prävention nach jeder Phase
- **VocalQualityGate**: Delta-basierte Gesangsqualität mit Rollback

Diese Guards laufen ohne manuelle Konfiguration — jede Phase profitiert.

## §6.9b [RELEASE_MUST] Phase-50 Team-Kohärenz + CONFLICT_REGISTRY (v10.0.0, erweitert v10.0.0)

`phase_50_spectral_repair` ist nach `phase_06_frequency_restoration`,
`phase_07_harmonic_restoration` oder `phase_23_spectral_repair` als
**kooperative Folgephase** zu behandeln.

**Normative Invarianten**:

- UV3 muss `prior_phase_context` an Folgephasen durchreichen.
- UV3 muss für jede erfolgreiche Phase den Ontologie-Typ in den Kontext schreiben
  (`last_phase_type`, Typ-Counter, angewendete Typ-Familien).
- PMGG muss diesen Kontext **für alle Phasen** über eine zentrale Übergangs-Policy
  auswerten (Spec 02 §2.29e).
    wenn die erkannte Regression dem Team-Policy-Grund
    `phase50_after_hf_restoration` entspricht.

**CONFLICT_REGISTRY (v10.0.0)** — `backend/core/phase_ontology.py`:

Für alle aktiven Phasen gilt: UV3 prüft vor jeder Phasen-Ausführung, ob im
`CONFLICT_REGISTRY` ein Eintrag existiert, der die aktuelle Phase als potentiell
neutralisierend einstuft. Falls ja, erhält die Phase `conflict_with_prior_phases: list[str]`.

Die Phase selbst entscheidet, wie sie mit `conflict_with_prior_phases` umgeht —
typischerweise: konservativere Threshold, Schutz bestimmter Frequenzbereiche.
`phase_50` nutzt bereits `hf_protected_bin_start` (v10.0.0) für genau diesen Zweck.

**Team-Telemetrie (v10.0.0)**:

UV3 schreibt `metadata["team_coordination"]` nach jeder Pipeline:
- `event_count`: Anzahl Phasen, bei denen Team-Policy aktiv war
- `events`: Liste mit `phase_id`, `action`, `reason`, `excluded_goals`, `threshold_mult`, `strength_cap`
- `phase_type_summary`: Häufigkeiten der Phase-Operationstypen (SUBTRACTIVE, ADDITIVE, etc.)

**Rationale**: Die Phasenkette arbeitet als Team; spätere Reparaturphasen
dürfen frühere restaurative Interventionen nicht indirekt neutralisieren.
# Wenn CausalDefectReasoner für ein Audio-Segment ≥3 Ursachen mit
# max_severity ≥ 0.70 identifiziert, DARF die kanonische CAUSE_TO_PHASES-Reihenfolge
# durch folgende Regel überschrieben werden:
#
#   phase_order = sorted(all_activated_phases,
#       key=lambda p: (severity_for_phase(p), -phase_number(p)), reverse=True)
#
# Rationale: Ein schwerer Dropout (severity=0.92) generiert nach phase_03_denoise mehr
# Information (Stille statt Rauschen) als wenn Hum (severity=0.45) zuerst bearbeitet wird.
# Die Defektinteraktion ist nicht linearkumulativ — frühe Schlüsselphasen erhöhen die
# Qualität nachfolgender Phasen messbar.
#
# Implementierung in `CausalDefectReasoner.build_restoration_plan()`:
#   if len([d for d in defects if d.severity >= 0.70]) >= 3:
#       plan.phases = _severity_reorder(plan.phases, defect_severity_map)
# Dabei gilt: PMGG-Exclusions und Phase-Validity-Regeln bleiben unverändert.
# VERBOTEN: Reorder bei < 3 Hochseverity-Defekten (destabilisiert gut getestete Reihenfolge).
```

---

## §7.3 Instrument-Kontexterkennung (PANNs-Aktivierungsmatrix)

| PANNs-Kategorie | Aktivierte Phase | Confidence-Schwelle |
| --- | --- | --- |
| Guitar / Electric Guitar | `phase_44_guitar_enhancement` | ≥ 0.50 |
| Brass / Trumpet / Saxophone | `phase_45_brass_enhancement` | ≥ 0.50 |
| Drum / Percussion | `phase_51_drums_enhancement` | ≥ 0.50 |
| Piano / Keyboard | `phase_52_piano_restoration` | ≥ 0.50 |
| Singing / Vocals | `phase_19_de_esser` + `phase_42_vocal_enhancement` + `phase_43_ml_deesser` + VocalAIEnhancement | ≥ 0.40 (Soft 0.35–0.40: 50 % Strength) |

> **Invariante** (v10.0.0): Instrument-Schwelle ist einheitlich **0.50** für alle Instrumente. Höherer Wert (z.B. 0.60) blockiert Enhancement bei Ensemble-Aufnahmen mit mehreren gleichzeitigen Instrumenten. Änderungen hier → immer auch `backend/core/unified_restorer_v3.py` L≈5822 + `plugins/panns_plugin.py` Docstring anpassen.

**Regel**: Instrument-Phasen IMMER nach Defektkorrektur, VOR Mastering.

---

## §7.4 PhaseInterface — Pflicht-Pattern für neue Phasen

Jede neue Phase **muss**:

- In `backend/core/phases/phase_NN_<beschreibung>.py` angelegt werden
- `PhaseInterface` aus `backend/core/phases/phase_interface.py` implementieren
- `process(audio: np.ndarray, **kwargs) -> PhaseResult` bereitstellen
- In `backend/core/phases/__init__.py` exportiert werden
- Ausgang: immer `np.clip(result.audio, -1.0, 1.0)` vor `PhaseResult`-Erzeugung
- SR-Invariante: `assert sample_rate == 48000`

```python
# Checkliste neue Phase:
# □ PhaseInterface implementiert
# □ process() → PhaseResult (kein raw ndarray)
# □ NaN/Inf-Guard: np.clip(audio, -1.0, 1.0), nan_to_num
# □ assert sample_rate == 48000
# □ Export in __init__.py
# □ CAUSES + CAUSE_TO_PHASES bidirektional ergänzen (V12 §2.59) — §0a-verbotene Phasen nie eintragen
# □ ≥ 3 Unit-Tests (Erkennung, False-Positive-Rate, Material-Prior)
# □ Stereo-Kohärenz (§2.51 Spec 02 / §7.1a Spec 06):
#     Wenn Stereo-Audio verarbeitet wird: M/S-Domain ODER Linked-Stereo —
#     KEIN unabhängiges L/R-Processing mit gain- oder zeitvarianter Operation
# □ ADDITIVE Phase: hallucination_guard.py (§2.46e) + _MATERIAL_BW_CEILING_HZ einhalten (§6.2c)
# □ ML-NR Phase: energy_bias=−6 dB Vokal / −9 dB Instrumental (§0j); G_floor ≥ 0.10 (§2.62)
# □ DYNAMICS-Expansion: _MATERIAL_DR_CEILING_DB prüfen (§6.2b)
```

---

## §7.5 Parallelisierungs-Invariante (Pipeline-Tiers)

```text
TIER 0 + TIER 1: IMMER sequenziell (TransientDecoupledProcessing, Click-Removal)
TIER 2–4: Dürfen parallelisieren; Merge via np.mean NUR wenn gleiche Frequenzzone
TIER 6:   IMMER sequenziell (EQ → Polish → LUFS → TruePeak → Format)
```

---

## §7.5a [RELEASE_MUST] Phasen-DAG — Formaler Abhängigkeitsgraph

**Motivation**: Die bisherige Parallelisierungs-Invariante (§7.5) ist grob-granular. Ein formaler DAG (Directed Acyclic Graph) der Phasen-Abhängigkeiten ermöglicht:

1. **Korrekte Optimierung**: Welche Phasen können wirklich parallel laufen?
2. **Regression-Prävention**: Phasen werden nicht in falsche Reihenfolge gebracht (§2.46 Stufe-Reihenfolge)
3. **MultiPassScheduler-Sicherheit**: Pass 2 darf nur Phasen wiederholen, die keine Vorläufer überschreiben

**Abhängigkeitstypen**:

| Typ | Bedeutung | Beispiel |
| --- | --- | --- |
| `HARD_BEFORE` | A muss vor B fertig sein | phase_03 → phase_07 (NR vor Harmonik-Erweiterung) |
| `SOFT_BEFORE` | A sollte vor B laufen; B darf ohne A laufen | phase_09 → phase_18 (Crackle vor Noise-Gate empfohlen) |
| `INDEPENDENT` | Keine Abhängigkeit — können parallel laufen | phase_14, phase_25 (parallel) |
| `CONFLICT` | Nicht gemeinsam aktiv (`CONFLICT_REGISTRY` §2.29e) | phase_06 ⊗ phase_07 (nicht beide) |

**Kritische HARD_BEFORE-Ketten** (normativ):

```
phase_01 (DC-Offset) → ALLE anderen Phasen
phase_09 (Crackle)  → phase_18 (NR/Gate, damit nicht Crackle-Residuen als Rauschen bleiben)
phase_03 (ML-NR)    → phase_06 (BW-Extension, damit nicht Rauschen = Extended-Harmonics)
phase_03            → phase_07 (Harmonic Enhancement)
phase_06            → phase_07 (BW vor Harmonik-Aufbau)
phase_12 (Wow/Flutter) → phase_25 (Azimuth), Azimuth vor Bandbreite
phase_29 (Band-NR)  → phase_07 (Harmonik-Erweiterung erst nach Bandrauschen-Entfernung)
phase_24 (Dropout)  → phase_06 (BW-Extension überschreibt keine Dropout-Lücken)
phase_30 (DC-ADC)   → phase_31 (Quant-NR) → alle weiteren
```

**Unabhängige Parallelisierungs-Klassen**:

| Klasse | Phasen | Bedingung |
| --- | --- | --- |
| **Klasse A** (Stereo/Phase) | phase_14, phase_15, phase_25 | Nach phase_12; keine Frequenz-Abhängigkeiten |
| **Klasse B** (lokale Defekte) | phase_09, phase_24, phase_31 | Nach phase_01; Defekttypen überlappen nicht |
| **Klasse C** (Analysis-only) | phase_05, phase_21 (wenn aktiv) | Lesend; kein Audio-Write |

**DAG-Validierung** (`backend/core/phase_dag.py`):

```python
def validate_phase_order(phase_list: list[str]) -> list[str]:
    """
    Prüft phase_list gegen HARD_BEFORE-Constraints.
    Gibt Liste der Constraint-Verletzungen zurück (leer = korrekt).
    VERBOTEN: phase_07 vor phase_03, phase_06 vor phase_29.
    """
    violations = []
    for constraint in HARD_BEFORE_CONSTRAINTS:
        if constraint.before in phase_list and constraint.after in phase_list:
            if phase_list.index(constraint.before) > phase_list.index(constraint.after):
                violations.append(f"{constraint.after} kommt vor {constraint.before}")
    return violations
```

**CI-Invariante**: `tests/unit/test_phase_dag_validation.py` prüft `validate_phase_order()` für alle HARD_BEFORE-Constraints. Test MUSS grün sein.

> Implementierung: `backend/core/phase_dag.py`

---

## §7.6 Adaptive Chunk-Verarbeitung (ab 5 Minuten Dateilänge)

| Defektdichte (lokal) | Chunk-Größe | Begründung |
| --- | --- | --- |
| Hoch (severity ≥ 0.6) | **5 s** | Feingranulare Kontrolle |
| Mittel (0.3 ≤ severity < 0.6) | **15 s** | Balance Qualität/Rechenzeit |
| Niedrig (severity < 0.3) | **60 s** | Kontextkohärenz |
| Stille-Segmente | **120 s** | Passthrough ohne DSP |

**Implementierung**: `backend/core/adaptive_chunk_processor.py`

```python
from backend.core.adaptive_chunk_processor import process_in_adaptive_chunks

# Opt-in für Phasen, die von feinerer Chunk-Verarbeitung profitieren:
result = process_in_adaptive_chunks(
    phase_fn=my_phase_process,
    audio=audio,
    sr=48000,
    max_severity=defect_severity,
    phase_kwargs={"material_type": material, ...},
)
# Crossfade: Hanning-Fenster 10 ms @ 50% Overlap — COLA-konform (Lücke-E-Fix v10.0.0)
#   COLA-Bedingung (Constant Overlap-Add): sum(w[n-H], w[n]) = 1.0 für alle n
#   → Hanning + 50 % Overlap (Hop = fsize/2) ist COLA-konform — kein Amplitudeneinbruch
#   → Konkret: fsize = 480 Samples (10 ms @ 48 kHz), hop = 240 Samples (5 ms)
#   → Jeder Chunk-Rand-Sample ist genau einmal von Hanning = 1.0-Summe abgedeckt
#   VERBOTEN: Hanning mit Hop > fsize/2 (erzeugt periodische −0.5–1.5 dB Dips @ jeder
#             Chunk-Grenze — besonders hörbar bei 5-s-Chunks mit hoher Defektdichte)
# Minimum: 2 s | Maximum: 120 s
# Segment-Grenzen (SegmentAdaptiveProcessor) haben Vorrang vor Chunk-Grenzen
#
# [RELEASE_MUST] §7.6a Chunk-Boundary-Transient-Guard (v10.0.0+):
# Liegt ein Transient (onset_strength > 0.35, aus librosa.onset.onset_strength) innerhalb
# von ±20 ms einer geplanten Chunk-Grenze, MUSS die Grenze um +25 ms nach vorne verschoben
# werden (weg vom Transient).
# Begründung: Ein Drum-Hit oder Plosiv exakt auf der Crossfade-Grenze führt selbst bei
# COLA-konformem Hanning zu hörbarem Transient-Smearing (±0.3–0.8 dB Amplitudenabweichung
# im ersten Attack-Frame, ca. 1–22 ms). Beträgt der Shift mehr als halbe Chunk-Größe,
# Grenze stattdessen um −25 ms nach hinten verschieben.
# Implementierung: `adaptive_chunk_processor._find_safe_boundary(pos_samples, audio, sr)`
```

**Defect-Locations-Flow** (v10.0.0): `_execute_pipeline` extrahiert `defect_locations` (dict[str, list[tuple[float,float]]]) und `max_defect_severity` (float) aus DefectScanner-Ergebnissen und übergibt sie als kwargs an jede Phase. Phasen können Locations als Hints für gezieltere Verarbeitung nutzen (opt-in), erkennen Defekte weiterhin auch eigenständig intern (Redundanz-Prinzip).

**[RELEASE_MUST] Location-Completeness-Invariante**:

- Im Core-Analyse- und Reparaturpfad sind harte Obergrenzen auf `defect_locations` unzulässig.
- Bei hoher Ereignisdichte (auch >1000 Events) muss die vollständige Eventliste für Phasenrouting, PMGG und Nachverarbeitung erhalten bleiben.
- Marker-Reduktion ist nur im UI-Layer erlaubt und darf nie die intern verwendete Defektliste verändern.

---

## §7.7 [RELEASE_MUST] PMGG Inference-Caching bei Retries (§2.29a — ML-deterministische Phasen)

**Kernprinzip**: ML-Modelle sind deterministisch (gleicher Input → gleicher Output). Bei PMGG-Retries wird ML-Inferenz **NICHT** wiederholt. Erster Aufruf mit `strength=1.0` (volle Inferenz) → Cache `audio_full`. Retries variieren ausschließlich Wet/Dry-Blending: `audio_retry = dry + strength × (audio_full − dry)`.

### ML-deterministische Phasen (gecachte Inferenz, nur Wet/Dry-Reblend bei Retry)

| Phase | ML-Modell | Begründung |
| --- | --- | --- |
| `phase_03_denoise` | SGMSE+ (Tier-0, Vokal) / ResembleEnhance / DeepFilterNetV3 / OMLSA (DSP) | ML-Hybrid: Inferenz-Output identisch bei gleichem Input |
| `phase_06_frequency_restoration` | AudioSR | Neurale Bandwidth-Extension deterministisch |
| `phase_09_crackle_removal` | BANQUET ONNX | Blind-Denoising deterministisch |
| `phase_12_wow_flutter_fix` | FCPE/CREPE/pYIN | f₀-Schätzung deterministisch (Timing-Phase: kein Wet/Dry) |
| `phase_18_noise_gate` | Silero VAD | Binary-Mask deterministisch |
| `phase_20_reverb_reduction` | SGMSE+ (Primärpfad) | Reverb-Speech-Separation deterministisch (WPE-DSP-Fallback: muss re-run) |
| `phase_23_spectral_repair` | Apollo (primär) + AudioSR (Fallback) | Spektral-Lückenfüllung deterministisch |
| `phase_24_dropout_repair` | AudioSR | Audio-Generierung deterministisch |
| `phase_29_tape_hiss_reduction` | DeepFilterNet v3 II | HF-Denoising deterministisch (OMLSA-DSP <2 kHz: muss re-run) |
| `phase_42_vocal_enhancement` | BSRoFormer | Stem-Separation deterministisch |
| `phase_55_diffusion_inpainting` | CQTdiff/FlowMatching | Diffusions-Inpainting deterministisch |
| `phase_56_spectral_band_gap` | FCPE/CREPE + Synthese | Noten-Synthese deterministisch |

### Strength-abhängige DSP-Phasen (MÜSSEN bei jedem Retry neu ausgeführt werden)

Alle übrigen Phasen, bei denen `strength` Algorithmus-Parameter steuert (z. B. Filterfrequenz, Kompressionsratio, Sättigungsgrad). Beispiele: `phase_01`, `phase_02`, `phase_04`, `phase_10`, `phase_14`, `phase_17`, `phase_19`, `phase_22`, `phase_25`–`phase_28`, `phase_31`–`phase_41`, `phase_43`–`phase_54`.

**Implementierung**: `PerPhaseMusicalGoalsGate._run_with_retry()` führt `_run_phase(phase, audio, 1.0, kwargs)` genau einmal aus. Retries nutzen `_wet_dry_blend(audio, audio_full, strength, phase)`.

```python
# Wet/Dry-Blend bei Retry (PMGG §2.29a Referenzimplementierung):
def _wet_dry_blend(dry: np.ndarray, wet: np.ndarray, strength: float) -> np.ndarray:
    return np.clip(dry + strength * (wet - dry), -1.0, 1.0)

# Erster Aufruf (strength=1.0, gecacht):
audio_full = _run_phase(phase, audio, strength=1.0, kwargs)  # einmal
# Retries (nur Blend, kein erneuter ML-Run):
audio_retry = _wet_dry_blend(audio, audio_full, retry_strength)
```

> **Invariante**: `phase_58_lyrics_guided_enhancement` ist NICHT ML-deterministisch im PMGG-Sinne — sie operiert als Post-Processing-Modul nach der PMGG-Kette und unterliegt eigenen Retry-Regeln (§2.36).

---

## §7.8 [RELEASE_MUST] Phase-50 HF-Spike-Schutz nach Vorphasen-Restauration (v10.0.0)

Pass-1 Spike-Detektor (11-Bin-Fenster, Threshold-Factor 3.0–4.5) darf durch `phase_07`/`phase_06`
restaurierte Harmoniken **nicht** als Codec-Spikes flaggen.

**Invariante**: `_repair_channel` erhält `hf_protected_bin_start` aus Material-Rolloff-Tabelle
(nur analoge Materialtypen). Bins ≥ `hf_protected_bin_start` sind in Pass-1 ausgeschlossen.
Pass-2 (Frame-Energy-Dropout) bleibt global aktiv.

**VERBOTEN**: Spike-Detektion ohne HF-Schutzzone für analoge Materialien nach Harmonik-Restauration.

> Vollständige Invariante: Spec 02 §2.57a — Algorithmus: Spec 04 §4.7a (Lookup-Tabelle)

## §7.9 [RELEASE_MUST] Phase-09 LPC/AR-Lücken-Interpolation (v10.0.0)

`_interpolate_hybrid()` ist eine **vollständige LPC/AR-Vorhersage** — kein Stub.

**Invariante**: Vorwärts-AR aus Pre-Gap + Rückwärts-AR aus Post-Gap, linear überblendet.
Pol-Stabilisierung (|z| ≥ 0.995 → 0.994). 5 ms Boundary-Crossfade.

**VERBOTEN**: `_interpolate_hybrid()` als Alias für `_interpolate_linear()`.

> Vollständiger Algorithmus: Spec 04 §4.7a

---

## §7.10 [RELEASE_MUST] Phase_65 — DSP-Vocal-Naturalness-Restaurierung (Restoration, v10.0.0)

> **Kontext**: `phase_42_vocal_enhancement` ist in Restoration-Modus per §0a Crossfire-Invariante
> verboten (Stem-Enhancement = Halluzination). Dies hinterlässt eine Lücke: Nach aggressivem NR
> (phase_03, phase_29) verlieren Vokale natürliche Wärme und HNR-Charakter. VQI sinkt auf
> 0.70–0.74 → Recovery-Cascade, aber keine Recovery-Phase kann das Defizit schließen — alle
> Vokal-Enhancement-Phasen sind Studio-Only. Phase_63 schließt diese Lücke durch einen
> **ausschließlich subtraktiven/korrektiven DSP-Ansatz** ohne jede generative Komponente.

### §7.10a Algorithmus — Drei Stufen

**Stufe 1: Spektral-Tilt-Korrektur (Korrektiv, kein Enhancement)**

```python
# Vergleich Input-Spektral-Tilt (Pre-NR) vs. Output-Tilt (Post-NR):
# NR neigt dazu, den Spektral-Tilt im 1–4 kHz-Bereich zu verändern (Resonanzboden dämpfen).
# Korrektur: Shelving-EQ bringt Tilt zurück — nicht als Enhancement, sondern als Inversion.
tilt_delta = estimate_spectral_tilt(pre_nr_audio) - estimate_spectral_tilt(post_nr_audio)
# Nur Korrektur anwenden wenn delta > 1.5 dB (perceptuell relevant):
if abs(tilt_delta) > 1.5:
    audio = apply_spectral_tilt_correction(audio, tilt_delta, sr)
    # Limit: max ±3.0 dB Shelving — kein Enhancement über pre_nr_tilt hinaus
```

**Stufe 2: HNR-Blend (Vokal-spezifisch, nur wenn `panns_singing ≥ 0.35`)**

```python
# Misst HNR Differenz pre_nr vs. post_nr:
delta_hnr = compute_hnr(pre_nr_audio, sr) - compute_hnr(post_nr_audio, sr)
if delta_hnr > 2.5:   # Mehr als 2.5 dB HNR-Verlust durch NR
    blend = np.clip(delta_hnr / 10.0, 0.0, 0.35)  # Max 35 % Dry-Blend
    audio = (1.0 - blend) * audio + blend * pre_nr_audio
    # Kanonisch: apply_hnr_blend() aus §0p — NICHT neu implementieren
```

**Stufe 3: Formant-Tilt-Korrektur (F1–F4, nur wenn Formant-Shift > 1.5 dB)**

```python
# LPC-Formant-Tracking vor/nach NR:
formants_pre  = track_lpc_formants(pre_nr_audio, sr, order=16)
formants_post = track_lpc_formants(post_nr_audio, sr, order=16)
for fn in range(4):
    delta_db = formants_post[fn].energy_db - formants_pre[fn].energy_db
    if abs(delta_db) > 1.5:
        audio = apply_narrow_shelf(audio, sr,
                                   center_hz=formants_post[fn].freq,
                                   gain_db=np.clip(-delta_db, -2.5, +2.5),
                                   q=6.0)
# Limit: max resolve_formant_tolerance_db(...) pro Formant — §0p Formant-Integrität-Guard bleibt aktiv
```

**Phasenpflicht für Vokal-Formant-Gates** [RELEASE_MUST]: Lokale Formant-Guards in Vokal-/NR-/Dereverb-Phasen dürfen keinen festen `2.0 dB`-Schwellwert hardcoden. Die Toleranz kommt aus `resolve_formant_tolerance_db(era_decade, era_profile)`, optional über `kwargs["formant_tolerance_db"]` von UV3 injiziert. Neue Phasen müssen `era_vocal_profile`, `vocal_zone_strength_policy` und `passaggio_energy_bias_db` akzeptieren, wenn sie Vokalregionen verändern.

### §7.10b Invarianten

- **Familien-Taxonomie**: KORREKTIV (nicht ADDITIVE, nicht SUBTRAKTIV)
- **§0a Crossfire-Invariante**: Nie in Studio-2026-Run (dort ist phase_42 richtig)
- **Activation Gate**: `panns_singing ≥ 0.25` UND (`delta_hnr > 2.5` ODER `tilt_delta > 1.5`) — kein Eingriff wenn Vokal nach NR bereits natürlich klingt
- **pre_nr_audio-Pflicht**: Phase_63 MUSS `pre_nr_audio` (Audio vor NR-Phasen) aus `restoration_context["pre_nr_checkpoint"]` beziehen — kein Pre-Pipeline-Input (wäre zu laut/verrauscht)
- **Reihenfolge**: Phase_63 läuft **nach** allen NR-Phasen (phase_03, phase_29, phase_20) — als Korrektiv-Post-NR
- **VQI-Check**: Nach Phase_63 MUSS `compute_vqi()` aufgerufen werden. Wenn VQI nach Phase_63 < VQI vor Phase_63: Rollback
- **PMGG-Exclusions**: `{"natuerlichkeit", "brillanz", "tonal_center"}` — diese Goals werden absichtlich verändert; ausschließlich VQI ist relevante Metrik

### §7.10c CAUSE_TO_PHASES-Mapping

```python
# CAUSES:
"vocal_naturalness_loss" → definiert durch:
    panns_singing ≥ 0.35
    AND delta_hnr_after_nr > 2.5 dB  (DefectScanner muss pre/post-NR HNR vergleichen)
    AND mode == "restoration"

# CAUSE_TO_PHASES:
"vocal_naturalness_loss": ["phase_65_vocal_naturalness_restoration"]
```

### §7.10d Reihenfolge in `_MATERIAL_PRIORITY_PHASES`

Phase_63 läuft in Tier 4 (Post-NR, Pre-Enhancement):

```
... → phase_29 (NR) → phase_20 (Dereverb) → phase_65 (Vocal Naturalness) → phase_08 (Transient) → ...
```

> **Kreuzreferenz**: §0a (Crossfire-Invariante), §0p (HNR-Blend, VQI-Gate), §4.5 (MMSE-LSA),
> Spec 09 §09.11 (VocalQuality Recovery-Phase-Mapping auf phase_65), copilot-instructions §0a

---

## §7.3a Phase-Implementierung — Patterns, Caching & Checkliste (konsolidiert aus Skill new-phase)

### Phasen-Interface (Pflicht für jede Phase)

Jede Phase liegt in `backend/core/phases/phase_XX_<name>.py` und MUSS:

- `def execute(audio, sr, strength=1.0, **kwargs) -> tuple[np.ndarray, dict]` exportieren
- `assert sr == 48000` am Eingang
- `audio = np.clip(audio, -1.0, 1.0)` am Ausgang
- `np.nan_to_num(result)` vor Return
- PhaseResult-dict mit mindestens: `phase_id`, `applied`, `strength`, `metadata`

### §2.29a Inference-Caching bei ML-Phasen

ML-deterministische Phasen: Erster Aufruf mit `strength=1.0` → Cache `audio_full`. Retries nur Wet/Dry-Blend:
`audio_retry = dry + strength × (audio_full − dry)`

**ML-deterministische Phasen** (gecachte Inferenz):
`phase_03`, `phase_06`, `phase_09`, `phase_12`, `phase_18`, `phase_20`,
`phase_23`, `phase_24`, `phase_29`, `phase_42`, `phase_55`, `phase_56`

**Strength-abhängige DSP-Phasen** (bei Retry neu ausführen):
`phase_01`, `phase_02`, `phase_04`, `phase_10`, `phase_14`, `phase_17`, `phase_19`,
`phase_22`, `phase_25`–`phase_28`, `phase_31`–`phase_41`, `phase_43`–`phase_54`

### §2.43 Phase-Preserved Wet/Dry-Blend

STFT-Bereich: `M_blend = (1−α)·M_dry + α·M_wet`, Phase vom Wet-Signal.
Verhindert Phase-Cancellation bei Kopfhörer.

### PHASE_GOAL_EXCLUSIONS — kanonische Tabelle (v10.0.0)

| Phase | Ausgeschlossene Goals | Begründung |
| --- | --- | --- |
| `phase_02` | bass_kraft, authentizitaet, natuerlichkeit, transparenz, groove, timbre_authentizitaet | Kammfilter Hum-Removal |
| `phase_03` | natuerlichkeit, artikulation, authentizitaet, tonal_center, timbre_authentizitaet | CREPE-Load-State + shaped NR |
| `phase_04` | transparenz, brillanz, waerme, authentizitaet, natuerlichkeit, timbre_authentizitaet | EQ |
| `phase_08` | micro_dynamics, artikulation | TDP/HPSS |
| `phase_12` | tonal_center, timbre_authentizitaet | K-S volatile nach Pitch-Korrektur |
| `phase_18` | micro_dynamics, authentizitaet, emotionalitaet, groove | Noise Gate |
| `phase_20` | authentizitaet, natuerlichkeit | SGMSE+ Reverb-Reduction |
| `phase_23` | natuerlichkeit, brillanz, authentizitaet, artikulation, timbre_authentizitaet | AudioSR synthetisiert |
| `phase_24` | natuerlichkeit, brillanz, authentizitaet, artikulation, timbre_authentizitaet | Dropout |
| `phase_29` | artikulation, authentizitaet, natuerlichkeit, tonal_center, timbre_authentizitaet | DeepFilterNet Tape-Hiss |
| `phase_49` | authentizitaet | Dereverb |

**Material-adaptive Relaxation**: `cd_digital`/`dat` → phase_03/phase_29 reduziert auf `{"natuerlichkeit", "artikulation"}`.

### §2.31b Song-Kalibrierungs-Integration (7 PMGG-Schnittstellen)

1. **Threshold**: `global_scalar < 0.85` → ×0.85; `> 1.20` → ×1.15. Begrenzt [0.015, 0.070]
2. **Retry-Leiter**: `initial_strength < 0.90` → Ankerpunkte `[0.80, 0.65, 0.50, 0.35, 0.20]`
3. **Stagnation**: `max(0.002, threshold × 0.15)`
4. **P3-Budget**: tier="good" → 3 Retries; tier="poor" → 1
5. **FeedbackChain target**: Base 0.72/0.78 ±0.035 nach restorability. Begrenzt [0.60, 0.85]
6. **Catastrophic**: `max(0.08, 4.0 × adaptive_threshold)`
7. **Material-adaptive Exclusions**: cd_digital/dat → reduzierter Satz

### PANNs Instrument-Aktivierungsmatrix

| PANNs-Kategorie | Phase | Schwellwert |
| --- | --- | --- |
| Vocals / Singing | phase_19 + phase_42 + phase_43 | ≥ 0.40 / ≥ 0.35 |
| Guitar | phase_44 | ≥ 0.50 |
| Brass / Saxophone | phase_45 | ≥ 0.50 |
| Drum / Percussion | phase_51 | ≥ 0.50 |
| Piano / Keyboard | phase_52 | ≥ 0.50 |

### Vocal-Restaurierungskette (§2.8) — API-Falle

`enhanced, report = self.breath_intelligence.process(audio, sr)` — **KEIN `events`-Argument!**

### §2.36a Phonem-DSP-Klassen

| Klasse | Algorithmus |
| --- | --- |
| fricative | Ramp-Gain 4–8 kHz, KEIN Wiener |
| plosive | TransientShapeGuard (onset gain=1.0), Burst ×1.40 |
| vowel_stressed | LPC Burg → F1–F4 Shelving |
| silence | OMLSA G_floor=0.05 |

### Checkliste neue Phase

```
□ backend/core/phases/phase_XX_<name>.py
□ execute(audio, sr, strength=1.0, **kwargs) → (ndarray, dict)
□ assert sr == 48000; NaN/Inf-Guard + Clip [-1, 1]
□ PMGG-Exclusions festlegen + begründen (§2.55 bidirektional mit CIG!)
□ ML-deterministisch oder strength-abhängig? → Caching-Strategie
□ DSP-Fallback für optionale ML-Imports
□ ml_memory_budget.try_allocate() VOR Modell-Laden
□ defect_locations kwargs opt-in nutzen (§9.1)
□ §2.51 Stereo: M/S-Domain oder Linked-Stereo (kein unabhängiges L/R)
□ ≥ 35 Unit-Tests (Shape, NaN, Bounds, Edge, Mono, Stereo, MG, Groove-DTW, SOFT_SAT, Pass-Through, quality_est)
□ OQS ≥ 80 nachweisbar
□ CHANGELOG.md Eintrag
□ CAUSES + CAUSE_TO_PHASES bidirektional ergänzen (V12 §2.59) — §0a-verbotene Phasen niemals eintragen
□ ADDITIVE Phase: hallucination_guard.py aufrufen (§2.46e) + _MATERIAL_BW_CEILING_HZ einhalten (§6.2c)
□ SUBTRAKTIVE ML-NR: energy_bias=−6 dB (Vokal) / −9 dB (Instrumental) + G_floor ≥ 0.10 via masking_threshold (§2.62)
□ DYNAMICS-Expansion: _MATERIAL_DR_CEILING_DB respektieren (§6.2b) — keine Expansion über Material-Limit
```
