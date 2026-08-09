# §ML-SCOUT — Finale Revision: Musik & Gesang

> **Revision 2:** 2026-07-30 · HF + GitHub + Paper-Referenzen · Musik-Fokus
> **Erkenntnis:** Open-Source-ML für Musik-Restaurierung ist extrem dünn besiedelt.
> Aurik ist bereits eines der umfassendsten Systeme überhaupt.

---

## 1. Gefunden: Musik-taugliche Modelle (jenseits des HF-Speech-Mainstreams)

### 🥇 Stable Audio 3 — SAME Encoder/Decoder (ONNX!)

| Feld | Wert |
|---|---|
| **Quelle** | `stabilityai/stable-audio-3-optimized` |
| **Komponente** | SAME (Stable Audio Music Encoder) — Large + Small |
| **Format** | **ONNX bereits vorhanden!** (`same-s/enc_dynamic_bf16.onnx`, `same-s/dec_dynamic_bf16.onnx`) |
| **Training** | Musik — Stability AI's proprietärer Musik-Datensatz |
| **Lizenz** | Stable Audio Community License (privat + nichtkommerziell, kommerziell auf Anfrage) |
| **Größe** | Small: ~200 MB, Large: ~800 MB |

**Warum Aurik:** Ein professionell trainierter Musik-VAE mit ONNX-Export. Der Encoder komprimiert Musik verlustarm, der Decoder rekonstruiert sie. Als Phase-0-Preprocessor oder Denoiser-Backbone einsetzbar — besser als EAR_VAE weil explizit auf Musik trainiert.

**Risiko:** Stable-Audio-Community-Lizenz — kommerzielle Nutzung nur nach Freigabe durch Stability AI.

---

### 🥈 ACE-Step VAE (MIT-Lizenz!)

| Feld | Wert |
|---|---|
| **Quelle** | `ACE-Step/ace-step-v1.5-1d-vae-stable-audio-format` |
| **Typ** | Musik-VAE, 1D-Convolution, Stable-Audio-kompatibel |
| **Lizenz** | **MIT** ✅ — uneingeschränkt kommerziell nutzbar! |
| **Größe** | ~675 MB (.ckpt) |
| **Paper** | arxiv:2602.00744 |

**Warum Aurik:** MIT-lizenziert, Musik-trainiert, Stable-Audio-Format-kompatibel. Kann EAR_VAE ersetzen mit besserer Musik-Rekonstruktion. Kein Lizenz-Risiko.

**Aufwand:** PyTorch → ONNX exportieren (analog zu EAR_VAE, ~2 Stunden).

---

### 🥉 NovaSR (Apache 2.0, 52 KB)

| Feld | Wert |
|---|---|
| **Quelle** | `YatharthS/NovaSR` |
| **Musik?** | ✅ Explizit Musik-Beispiele im README |

Bleibt als FlashSR-Alternative — aber nur relevant wenn der Qualitätsverlust akzeptabel ist. Für Auriks Weltspitze-Anspruch eher als Fallback für RAM-kritische Umgebungen.

---

## 2. Was es NICHT gibt (und warum)

| Kategorie | Status | Begründung |
|---|---|---|
| **Musik-Denoiser (Open Source)** | ❌ Existiert nicht | Alle Denoiser sind Speech-trainiert |
| **Musik-Declipper (Neural)** | ❌ Existiert nicht | DSP dominiert (Cuesta et al. kein HF-Modell) |
| **Musik-Dereverberation (Neural)** | ❌ Existiert nicht | WPE ist State-of-the-Art |
| **Musik-Bandbreiten-Erweiterung** | ⚠️ Nur NovaSR, FlashSR | FlashSR ist bereits sehr gut |
| **Musik-Inpainting** | ⚠️ AudioLDM2 (via Mel-Latent) | Musik-spezifisches Inpainting nicht verfügbar |

**Grund:** Die ML-Community fokussiert auf Speech (klarer Benchmark: DNS-Challenge) und Generation (MusicGen, Stable Audio). Musik-Restaurierung hat keinen standardisierten Benchmark und kein großes öffentliches Trainingsdatenset.

---

## 3. Auriks Position im ML-Ökosystem

Aurik hat bereits:

- **3 Musik-Source-Separation-Modelle** (Demucs, MDX23C, BS-RoFormer)
- **2 Musik-Encoder** (LAION-CLAP, MERT)
- **2 Vocoder** (BigVGAN, HiFi-GAN)
- **1 Diffusionsmodell** (AudioLDM2 UNet + VAE)
- **1 Musik-VAE** (EAR_VAE auf LAION-DISCO-12M)
- **1 Bandbreiten-Erweiterung** (FlashSR/NVSR)
- **2 Denoiser** (DeepFilterNetV3, SGMSE+)
- **1 Musik-Tagger** (PANNs)

**Das ist ungewöhnlich umfassend.** Die meisten Projekte haben 1–2 Modelle. Aurik integriert 12+ Modelle in einer kohärenten Pipeline.

---

## 4. Empfehlung: Was Aurik WIRKLICH voranbringt (Qualität first)

| Rang | Maßnahme | Hörgewinn | Aufwand |
|---|---|---|---|
| **1** | **ACE-Step VAE integrieren** (MIT, Musik-trainiert) | ★★★★ | 2–3 Tage |
| **2** | **EAR_VAE als Phase-0 aktivieren** (bereits integriert, LAION-DISCO-12M) | ★★★☆ | 0,5 Tage |
| **3** | **Bestehende Modelle besser abstimmen** (Phase-Strengths, Thresholds) | ★★★★ | Laufend |
| **4** | **Eigenes Fine-Tuning** (Aurik-Datenset auf DeepFilterNet/FlashSR) | ★★★★★ | 2–4 Wochen |
| **5** | Stable Audio SAME testen | ★★★ | 1–2 Tage (Lizenz prüfen!) |

---

## 5. Konkreter nächster Schritt

**ACE-Step VAE evaluieren:**

```bash
# Download
huggingface-cli download ACE-Step/ace-step-v1.5-1d-vae-stable-audio-format

# Prüfen: Architektur (config.json), Gewichte (.ckpt)
# → ONNX exportieren (analog zu EAR_VAE)
# → A/B-Test: EAR_VAE vs ACE-Step auf Musik-Material
# → Bei besserer Qualität: EAR_VAE ersetzen
```

**Warum ACE-Step vor Stable Audio:** MIT-Lizenz — kein rechtliches Risiko. Gleiche Architektur-Familie wie Stable Audio (stable-audio-tools). Wenn die Qualität stimmt, ist das der Musik-VAE, den Aurik verdient.
