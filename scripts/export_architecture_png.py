"""
Aurik 10.0.0.x.x — Systemarchitektur-Übersicht als hochauflösende PNG.
Ausgabe: docs/aurik_architecture.png  (300 dpi, 7680×4320 px bei 25.6"×14.4")
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
from matplotlib.patches import FancyBboxPatch  # pylint: disable=wrong-import-position

# Standalone rendering script; helper signatures intentionally use short drawing names.
# pylint: disable=redefined-outer-name
# pylint: disable=too-many-positional-arguments

# ---------------------------------------------------------------------------
# Layout-Konstanten
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 25.6, 18.0  # Zoll
DPI = 300
OUT = "docs/aurik_architecture.png"

FONT_MAIN = "DejaVu Sans"
C_BG = "#0f0f1a"  # Hintergrund
C_TITLE = "#e2e8f0"
C_ARROW = "#64748b"

# Farb-Palette je Schicht
COLORS = {
    "fe": ("#7c3aed", "#4c1d95"),  # Lila — Frontend
    "cli": ("#0284c7", "#0c4a6e"),  # Blau — CLI
    "api": ("#0369a1", "#082f49"),  # Dunkelblau — API
    "pre": ("#059669", "#064e3b"),  # Grün — Vor-Analyse
    "reason": ("#b45309", "#451a03"),  # Amber — Defekt-Inferenz
    "guard": ("#374151", "#111827"),  # Grau — Harmonik-Schutz
    "thresh": ("#6d28d9", "#2e1065"),  # Violett — Schwellwerte
    "phases": ("#1d4ed8", "#1e3a8a"),  # Indigo — Phasen
    "stem": ("#0f766e", "#042f2e"),  # Teal — Stem
    "post": ("#7e22ce", "#3b0764"),  # Dunkel-Violett — Post
    "plugin": ("#0e7490", "#083344"),  # Cyan — Plugins
    "mem": ("#374151", "#1f2937"),  # Dunkelgrau — Persistenz
    "out": ("#065f46", "#022c22"),  # Smaragd — Output
}


def rgba(hex_str, alpha=1.0):
    """Wandelt einen Hex-Farbstring in ein RGBA-Tupel um."""
    h = hex_str.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return (r, g, b, alpha)


def draw_box(ax, x, y, w, h, label, sublabel="", color_key="fe", fontsize=7.5, corner_radius=0.012, alpha_bg=0.88):
    """Zeichnet eine abgerundete Box mit optionalem Unter-Label auf die Achse."""
    fc, ec = COLORS[color_key]
    fancy = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.005,rounding_size={corner_radius}",
        linewidth=0.8,
        edgecolor=rgba(ec, 1.0),
        facecolor=rgba(fc, alpha_bg),
        zorder=3,
    )
    ax.add_patch(fancy)
    cy = y + h / 2
    if sublabel:
        ax.text(
            x + w / 2,
            cy + h * 0.12,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color="white",
            zorder=4,
            fontfamily=FONT_MAIN,
            clip_on=True,
        )
        ax.text(
            x + w / 2,
            cy - h * 0.15,
            sublabel,
            ha="center",
            va="center",
            fontsize=fontsize - 1.5,
            color=rgba("#cbd5e1"),
            zorder=4,
            fontfamily=FONT_MAIN,
            clip_on=True,
        )
    else:
        ax.text(
            x + w / 2,
            cy,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color="white",
            zorder=4,
            fontfamily=FONT_MAIN,
            clip_on=True,
        )


def draw_section_label(ax, x, y, w, h, label, color_key="fe", fontsize=8):
    """Zeichnet einen farbigen Header-Streifen mit Beschriftung für einen Abschnitt."""
    _fc, ec = COLORS[color_key]
    # Header-Streifen
    header = FancyBboxPatch(
        (x, y + h - 0.022),
        w,
        0.022,
        boxstyle="round,pad=0.002,rounding_size=0.008",
        linewidth=0,
        edgecolor="none",
        facecolor=rgba(ec, 0.95),
        zorder=5,
    )
    ax.add_patch(header)
    ax.text(
        x + w / 2,
        y + h - 0.011,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color="white",
        zorder=6,
        fontfamily=FONT_MAIN,
    )


def draw_section(ax, x, y, w, h, label, color_key="fe"):
    """Zeichnet einen Abschnitts-Rahmen mit Hintergrund und Header-Label."""
    fc, ec = COLORS[color_key]
    bg = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.005,rounding_size=0.015",
        linewidth=1.2,
        edgecolor=rgba(ec, 0.8),
        facecolor=rgba(fc, 0.12),
        zorder=2,
    )
    ax.add_patch(bg)
    draw_section_label(ax, x, y, w, h, label, color_key, fontsize=7.5)


def arrow(ax, x0, y0, x1, y1, color="#64748b", lw=0.7, style="->"):
    """Zeichnet einen geraden Pfeil zwischen zwei Koordinaten."""
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops={
            "arrowstyle": style,
            "color": color,
            "lw": lw,
            "connectionstyle": "arc3,rad=0.0",
        },
        zorder=5,
    )


def curved_arrow(ax, x0, y0, x1, y1, color="#64748b", lw=0.6, rad=0.2):
    """Zeichnet einen gebogenen Pfeil zwischen zwei Koordinaten."""
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "lw": lw,
            "connectionstyle": f"arc3,rad={rad}",
        },
        zorder=5,
    )


# ---------------------------------------------------------------------------
# Figur erstellen
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
fig.patch.set_facecolor(C_BG)
ax = fig.add_axes([0, 0, 1, 1])  # type: ignore[call-overload]
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("auto")
ax.axis("off")
ax.set_facecolor(C_BG)

# ==================== TITEL ====================
ax.text(
    0.5,
    0.975,
    "AURIK 9.x.x — SYSTEMARCHITEKTUR",
    ha="center",
    va="top",
    fontsize=16,
    fontweight="bold",
    color=rgba("#a78bfa"),
    fontfamily=FONT_MAIN,
    zorder=10,
)
ax.text(
    0.5,
    0.960,
    "Desktop Audio-Restaurierung · Psychoakustische DSP + ML · 15 Musical Goals · 56 Phasen",
    ha="center",
    va="top",
    fontsize=8,
    color=rgba("#94a3b8"),
    fontfamily=FONT_MAIN,
    zorder=10,
)

# ==================== TRENNLINIEN ====================
for yy in [0.945]:
    ax.axhline(yy, color=rgba("#334155", 0.6), lw=0.5, zorder=1)

# ==================== SCHICHT 1: FRONTEND + CLI ====================
draw_section(ax, 0.01, 0.880, 0.60, 0.058, "① FRONTEND — PyQt5 (frameless, Dark Theme)", "fe")

boxes_fe = [
    (0.015, 0.888, 0.13, 0.038, "Magic Button\nRESTORATION", "Modus: restoration", "fe"),
    (0.155, 0.888, 0.13, 0.038, "Magic Button\nSTUDIO 2026", "Modus: studio2026", "fe"),
    (0.295, 0.888, 0.10, 0.038, "WaveformWidget", "Wellenform-Overlay", "fe"),
    (0.405, 0.888, 0.10, 0.038, "AudioPlayer", "QMediaPlayer", "fe"),
    (0.515, 0.888, 0.09, 0.038, "Musical Goals\nRadar", "14 Ziele visualisiert", "fe"),
]
for bx, by, bw, bh, bl, bs, bk in boxes_fe:
    draw_box(ax, bx, by, bw, bh, bl, bs, bk, fontsize=6.5)

draw_section(ax, 0.62, 0.880, 0.37, 0.058, "② CLI — aurik_cli.py", "cli")
draw_box(
    ax,
    0.625,
    0.888,
    0.175,
    0.038,
    "aurik_cli.py",
    "--input --output --mode\n--material --pre-assess --ensemble",
    "cli",
    fontsize=6.2,
)
draw_box(
    ax,
    0.810,
    0.888,
    0.175,
    0.038,
    "ProgressiveQualityMode",
    "Stage-1: 5s Vorschau ≤8s\nStage-2: volle Pipeline",
    "cli",
    fontsize=6.2,
)

# ==================== SCHICHT 2: API ====================
draw_section(ax, 0.01, 0.820, 0.98, 0.052, "③ API-SCHICHT — FastAPI", "api")
api_boxes = [
    (0.015, 0.827, 0.15, 0.033, "REST /health\n/stream /feedback", "/download_report\n/download_log", "api"),
    (0.175, 0.827, 0.15, 0.033, "Batch-API /start\n/status /result", "/cancel /audit\n/list", "api"),
    (
        0.335,
        0.827,
        0.15,
        0.033,
        "progress_callback()\n(percent, phase, eta)",
        "QThread → Qt-Signal\nkein Polling",
        "api",
    ),
    (0.495, 0.827, 0.15, 0.033, "WebSocket\n(Echtzeit-Fortschritt)", "reserviert für\nv10.x", "api"),
    (0.655, 0.827, 0.15, 0.033, "clean_nans()\nJSON-Serialisierung", "NaN/Inf → None\nbei jeder Response", "api"),
    (
        0.815,
        0.827,
        0.17,
        0.033,
        "BatchSessionLearner\nGP-Warm-Start",
        "SHA256[:8] Session-ID\n~/.aurik/batch_sessions/",
        "api",
    ),
]
for bx, by, bw, bh, bl, bs, bk in api_boxes:
    draw_box(ax, bx, by, bw, bh, bl, bs, bk, fontsize=6.0)

# ==================== SCHICHT 3: VOR-ANALYSE ====================
draw_section(ax, 0.01, 0.748, 0.98, 0.063, "④ VOR-ANALYSE & KLASSIFIKATION", "pre")

pre_boxes = [
    (0.015, 0.756, 0.13, 0.040, "TransientDecoupled\nProcessing (TDP)", "HPSS Kernel 31\nPercussive ≠ NR", "pre"),
    (0.155, 0.756, 0.13, 0.040, "RestorabilityEstimator", "<5s · Score 0–100\nPredicted MOS ± CI", "pre"),
    (0.295, 0.756, 0.16, 0.040, "EraClassifier", "1890–2025\nCLAP+DSP-Rolloff+Mikrofon-Typ", "pre"),
    (
        0.465,
        0.756,
        0.16,
        0.040,
        "GermanSchlager\nClassifier",
        "6-Schicht Zero-Shot\nAkkordeon·HSI·Rhythmus·Vokal",
        "pre",
    ),
    (0.635, 0.756, 0.155, 0.040, "MediumClassifier", "12 MaterialTypes\nCLAP+DSP-Fingerprint", "pre"),
    (
        0.800,
        0.756,
        0.185,
        0.040,
        "ArtistSignature\nStore",
        "Formant·Vibrato·Breathiness\n~/.aurik/artist_signatures/",
        "pre",
    ),
]
for bx, by, bw, bh, bl, bs, bk in pre_boxes:
    draw_box(ax, bx, by, bw, bh, bl, bs, bk, fontsize=6.0)

# ==================== SCHICHT 4: DEFEKT-INFERENZ ====================
draw_section(ax, 0.01, 0.678, 0.98, 0.062, "⑤ DEFEKT-INFERENZ & GP-OPTIMIERUNG", "reason")

reason_boxes = [
    (
        0.015,
        0.686,
        0.175,
        0.040,
        "DefectScanner",
        "23 DefectTypes · 12 MaterialTypes\nSNR·Click·Hum·Wow·Dropout·Codec…",
        "reason",
    ),
    (
        0.200,
        0.686,
        0.175,
        0.040,
        "CausalDefectReasoner",
        "Bayesianische Kausalinferenz\n11 Ursachen · RestorationPlan",
        "reason",
    ),
    (
        0.385,
        0.686,
        0.155,
        0.040,
        "UncertaintyQuantifier",
        "high≥0.80 / medium / low<0.50\nGP-Bounds konservativer",
        "reason",
    ),
    (
        0.550,
        0.686,
        0.205,
        0.040,
        "GPParameterOptimizer",
        "RBF-GP + UCB κ=2.0 · MOO Pareto-Front\n14 Objectives · ~/.aurik/gp_memory/",
        "reason",
    ),
    (
        0.765,
        0.686,
        0.22,
        0.040,
        "AdaptiveGoalThresholds\n+ GoalApplicabilityFilter",
        "Material·Ära·Restorability-Skalierung\nPhysicalCeilingEstimator · GoalPriorityProtocol",
        "reason",
    ),
]
for bx, by, bw, bh, bl, bs, bk in reason_boxes:
    draw_box(ax, bx, by, bw, bh, bl, bs, bk, fontsize=6.0)

# ==================== SCHICHT 5: HARMONIK-SCHUTZ ====================
draw_section(ax, 0.01, 0.618, 0.98, 0.052, "⑥ HARMONIK-SCHUTZ & AUFMERKSAMKEIT", "guard")

guard_boxes = [
    (
        0.015,
        0.626,
        0.22,
        0.032,
        "HarmonicPreservationGuard (HPG)",
        "CREPE/pYIN f₀ · Harm. Gitter fₙ=n·f₀·√(1+Bn²) · G_floor=0.85 · PGHI",
        "guard",
    ),
    (
        0.245,
        0.626,
        0.205,
        0.032,
        "PerceptualAttentionModel (PAM)",
        "Salienz-Karte [n_frames×24 Bark] · Vocal×1.8 · Stille×0.3 · Transienten×1.2",
        "guard",
    ),
    (
        0.460,
        0.626,
        0.19,
        0.032,
        "PerceptualEmbedder",
        "256-dim L2-norm · 5 psychoak. Kanäle\nA:STFT B:Bark C:CQT D:AM/FM E:HPSS",
        "guard",
    ),
    (
        0.660,
        0.626,
        0.175,
        0.032,
        "PsychoacousticMaskingModel",
        "ISO 11172-3 · Simultan+Temporal\nGain-Modifier nach OMLSA",
        "guard",
    ),
    (0.845, 0.626, 0.14, 0.032, "StereoAuthenticity\nInvariant", "Mono-Ära M/S≥0.97\nDecca·Abbey-Road", "guard"),
]
for bx, by, bw, bh, bl, bs, bk in guard_boxes:
    draw_box(ax, bx, by, bw, bh, bl, bs, bk, fontsize=5.8)

# ==================== SCHICHT 6: PHASEN 01–56 ====================
draw_section(ax, 0.01, 0.475, 0.65, 0.135, "⑦ PHASE-PIPELINE 01–56 (core/phases/)", "phases")

phase_boxes = [
    (
        0.015,
        0.548,
        0.148,
        0.052,
        "RAUSCHUNTER-\nDRÜCKUNG",
        "phase_03/29\nOMLSA+IMCRA\nDeepFilterNet v3\nHPG G_floor=0.85",
        "phases",
    ),
    (
        0.172,
        0.548,
        0.148,
        0.052,
        "CLICK / CRACKLE\nENTFERNUNG",
        "phase_01/09/27\nRBME+Sparse Bayes\nConsistent Wiener\nPGHI-konsistent",
        "phases",
    ),
    (
        0.329,
        0.548,
        0.148,
        0.052,
        "DROPOUT\nINPAINTING",
        "phase_24/55\nNMF-β+DiffWave\nCQTdiff+ / Flow Matching\nPhraseContext±30s",
        "phases",
    ),
    (
        0.486,
        0.548,
        0.148,
        0.052,
        "CODEC-ARTEFAKTE\nREPARATUR",
        "phase_23/50\nApollo (Mamba)\nResemble-Enhance\nSpectral Repair+PGHI",
        "phases",
    ),
    (0.643, 0.548, 0.0, 0.0, "", "", "phases"),  # placeholder
]

# Erste Reihe Phasen
phase_row1 = [
    (0.015, 0.548, 0.148, 0.055, "NR\nphase_03/29", "OMLSA+DeepFilterNet\nHPG G_floor=0.85", "phases"),
    (0.172, 0.548, 0.148, 0.055, "Click/Crackle\nphase_01/09/27", "RBME+Sparse Bayes\nPGHI-konsistent", "phases"),
    (0.329, 0.548, 0.148, 0.055, "Inpainting\nphase_24/55", "NMF-β+CQTdiff+\nFlow Matching", "phases"),
    (0.486, 0.548, 0.148, 0.055, "Codec\nphase_23/50", "Apollo (Mamba)\nSpectral Repair", "phases"),
]
for bx, by, bw, bh, bl, bs, bk in phase_row1:
    draw_box(ax, bx, by, bw, bh, bl, bs, bk, fontsize=6.0)

# Zweite Reihe Phasen
phase_row2 = [
    (0.015, 0.484, 0.148, 0.055, "Gesang\nphase_42/43", "VocalAIEnhancement\nDe-Esser+Formant", "phases"),
    (0.172, 0.484, 0.148, 0.055, "Instrument\nphase_44–52", "PANNs+DDSP\nGitarre·Bläser·Piano", "phases"),
    (0.329, 0.484, 0.148, 0.055, "Mastering\nphase_35–48", "Multiband·LUFS\nTruePeak·Stereo", "phases"),
    (0.486, 0.484, 0.148, 0.055, "SpectralBandGap\nphase_56", "HEAD_WEAR-Defekt\nconf≥0.55", "phases"),
]
for bx, by, bw, bh, bl, bs, bk in phase_row2:
    draw_box(ax, bx, by, bw, bh, bl, bs, bk, fontsize=6.0)

# PMGG-Box rechts auf Phasen-Sektion
draw_box(ax, 0.643, 0.484, 0.012, 0.119, "P\nM\nG\nG", "", "guard", fontsize=5.0)
ax.text(
    0.657,
    0.543,
    "PerPhaseMusicalGoalsGate\n6 Schnell-Ziele · 5s Sample\nmax. 5 Retries · Rollback\nRegression < −threshold",
    ha="left",
    va="center",
    fontsize=5.5,
    color="white",
    zorder=6,
    fontfamily=FONT_MAIN,
)

# ==================== SCHICHT 6b: STEM-VERARBEITUNG ====================
draw_section(ax, 0.67, 0.475, 0.32, 0.135, "⑧ STEM-VERARBEITUNG", "stem")

draw_box(
    ax,
    0.675,
    0.536,
    0.145,
    0.062,
    "MDX23C\nKim_Vocal_2 + Kim_Inst",
    "2× 64MB ONNX · lokal\nChunk 485100 Samples\nSOTA-Upgrade: BS-RoFormer\nMBR 860MB · +2–3dB SDR",
    "stem",
    fontsize=5.8,
)
draw_box(
    ax,
    0.830,
    0.536,
    0.150,
    0.062,
    "StemRemixBalancer",
    "LUFS-korrekter Re-Mix\ng_voc=10^((L_orig−L')/20)\n|LUFS(mix)−L_orig|≤0.3LU\nPANNs vocal_weight",
    "stem",
    fontsize=5.8,
)
draw_box(
    ax,
    0.675,
    0.484,
    0.305,
    0.044,
    "VocalAIEnhancement + ConsonantEnhancement",
    "VoiceGender MALE/FEMALE/CHILD/ANDROGYNOUS · Formant-Pearson≥0.95 · Breathiness±0.05 · Sibilant SNR≥+3dB",
    "stem",
    fontsize=5.6,
)

# ==================== SCHICHT 7: POST-VERARBEITUNG ====================
draw_section(ax, 0.01, 0.352, 0.98, 0.116, "⑨ POST-VERARBEITUNG & QUALITÄTSSICHERUNG", "post")

post_row1 = [
    (
        0.015,
        0.408,
        0.155,
        0.050,
        "EraAuthentic\nPerceptualCompletion",
        "BW<10kHz+Brillanz applicable\nDDSP+PGHI · ERA_CEILING/dek.",
        "post",
    ),
    (
        0.180,
        0.408,
        0.155,
        0.050,
        "Introduced\nArtifactDetector",
        "ML_HALLUC·NMF_CLICK\nPHASE_SMEAR·MUSICAL_NOISE\nRetry×0.5 · max 2×",
        "post",
    ),
    (
        0.345,
        0.408,
        0.155,
        0.050,
        "FeedbackChain",
        "5 Iterationen max\nΔMOS<0.02 = Konvergenz\nRegression→best_result",
        "post",
    ),
    (
        0.510,
        0.408,
        0.155,
        0.050,
        "TemporalQuality\nCoherenceMetric",
        "10s-Segmente · Spanne≤0.30\nσ(MOS)≤0.15 · ≥25s Dateien",
        "post",
    ),
    (
        0.675,
        0.408,
        0.155,
        0.050,
        "PerceptualQuality\nScorer (PQS)",
        "Gammatone 25 Bänder\nNSIM+MCD+LUFS+MOS\nMOS∈[1,5]",
        "post",
    ),
    (
        0.840,
        0.408,
        0.145,
        0.050,
        "Excellence\nOptimizer (GP)",
        "GP-Params·MOO-Pareto\n10 Parameter\nola·harmonic·modulation",
        "post",
    ),
]
for bx, by, bw, bh, bl, bs, bk in post_row1:
    draw_box(ax, bx, by, bw, bh, bl, bs, bk, fontsize=5.8)

post_row2 = [
    (
        0.015,
        0.360,
        0.19,
        0.040,
        "MusicalGoalsChecker",
        "14 Ziele · measure_all(audio,sr) · Brillanz≥0.85 · Natürlichkeit≥0.90 · TonalCenter≥0.95",
        "post",
    ),
    (
        0.215,
        0.360,
        0.19,
        0.040,
        "EmotionalArcPreservation",
        "Arousal-Pearson≥0.85 · Valence≥0.80 · Klimax-Peak±2 Segmente · ≥30s",
        "post",
    ),
    (
        0.415,
        0.360,
        0.19,
        0.040,
        "MicroDynamicsEnvelope\nMorphing (MDEM)",
        "400ms LUFS-Profil · Savitzky-Golay · ±3.0LU · True-Peak nach Morphing",
        "post",
    ),
    (
        0.615,
        0.360,
        0.175,
        0.040,
        "Vocos 48kHz ONNX",
        "nativ 48kHz · wenn PQS-MOS<4.3\nHiFi-GAN 3.6MB → PGHI-iSTFT",
        "post",
    ),
    (
        0.800,
        0.360,
        0.185,
        0.040,
        "RestorationGenealogy\n+ Audit-Trail",
        "SampleOperation · SHA256 · JSON\n~/.aurik/genealogy/ · --archive-mode",
        "post",
    ),
]
for bx, by, bw, bh, bl, bs, bk in post_row2:
    draw_box(ax, bx, by, bw, bh, bl, bs, bk, fontsize=5.6)

# ==================== SCHICHT 8: PLUGINS ====================
draw_section(ax, 0.01, 0.242, 0.68, 0.102, "⑩ PLUGINS (plugins/ — ONNX · CPU-only)", "plugin")

plugin_boxes = [
    (0.015, 0.286, 0.115, 0.045, "DeepFilterNet v3", "37MB · 3 ONNX\nPRIMÄR NR", "plugin"),
    (0.140, 0.286, 0.115, 0.045, "MDX23C", "Kim_Vocal_2\n2×64MB ONNX", "plugin"),
    (0.265, 0.286, 0.115, 0.045, "Apollo", "65MB ONNX\nCodec PRIMÄR", "plugin"),
    (0.390, 0.286, 0.115, 0.045, "CREPE full", "85MB ONNX\nPitch f₀", "plugin"),
    (0.515, 0.286, 0.115, 0.045, "DiffWave", "552KB ONNX\nDropout-Inpaint", "plugin"),
    (0.640, 0.286, 0.0, 0.0, "", "", "plugin"),  # placeholder
]
plugin_row1 = [
    (0.015, 0.287, 0.120, 0.044, "DeepFilterNet v3", "37MB · 3 ONNX · PRIMÄR NR", "plugin"),
    (0.145, 0.287, 0.120, 0.044, "MDX23C", "Kim_Vocal_2+Inst\n2×64MB ONNX", "plugin"),
    (0.275, 0.287, 0.120, 0.044, "Apollo", "65MB · Mamba\nCodec PRIMÄR", "plugin"),
    (0.405, 0.287, 0.120, 0.044, "CREPE full", "85MB ONNX\nPitch-Tracking f₀", "plugin"),
    (0.535, 0.287, 0.137, 0.044, "BS-RoFormer", "SOTA-Upgrade\n+2–3dB SDR", "plugin"),
]
for bx, by, bw, bh, bl, bs, bk in plugin_row1:
    draw_box(ax, bx, by, bw, bh, bl, bs, bk, fontsize=6.0)

plugin_row2 = [
    (0.015, 0.250, 0.120, 0.030, "DiffWave", "552KB · Dropout-Inpaint", "plugin"),
    (0.145, 0.250, 0.120, 0.030, "PANNs CNN14", "81KB · Audio-Tagging", "plugin"),
    (0.275, 0.250, 0.120, 0.030, "WpePlugin", "3-Tier WPE · DSP only", "plugin"),
    (0.405, 0.250, 0.120, 0.030, "Vocos 48kHz", "nativ 48kHz · Vocoder", "plugin"),
    (0.535, 0.250, 0.137, 0.030, "MERT", "Music Understanding\n3.9GB lazy-load", "plugin"),
]
for bx, by, bw, bh, bl, bs, bk in plugin_row2:
    draw_box(ax, bx, by, bw, bh, bl, bs, bk, fontsize=5.8)

# ==================== SCHICHT 8b: MANIFEST ====================
draw_section(ax, 0.70, 0.242, 0.29, 0.102, "(11) MANIFEST & SOTA-UPGRADES", "mem")
draw_box(
    ax,
    0.705,
    0.250,
    0.275,
    0.082,
    "models/manifest.json  (v2 · 19 Einträge)",
    "bundled_path + sha256 · required · fallback\n"
    "sota_upgrade: {url · license · reference}\n"
    "Alle Primärmodelle: bundled=true · kein Download\n"
    "ModelDownloader · SHA256-Verifikation\n"
    "ONNX: CPUExecutionProvider zwingend\n"
    "PyTorch: device=cpu · set_num_threads(os.cpu_count())",
    "mem",
    fontsize=6.0,
)

# ==================== SCHICHT 9: PERSISTENZ ====================
draw_section(ax, 0.01, 0.160, 0.50, 0.075, "(12) PERSISTENZ  ~/.aurik/", "mem")

mem_boxes = [
    (0.015, 0.168, 0.085, 0.055, "gp_memory/", "<material>.json\nGP-Observ.+Best", "mem"),
    (0.110, 0.168, 0.085, 0.055, "artist_signatures/", "<id>.json\nFormant·Vibrato", "mem"),
    (0.205, 0.168, 0.085, 0.055, "batch_sessions/", "<session>.json\nGP-Warm-Start", "mem"),
    (0.300, 0.168, 0.085, 0.055, "era_cache/", "<sha>.json\nDekaden-Prior", "mem"),
    (0.395, 0.168, 0.095, 0.055, "genealogy/", "<sha>_<ts>.json\nAudit-Trail", "mem"),
]
for bx, by, bw, bh, bl, bs, bk in mem_boxes:
    draw_box(ax, bx, by, bw, bh, bl, bs, bk, fontsize=6.0)

# ==================== SCHICHT 9b: OUTPUT ====================
draw_section(ax, 0.52, 0.160, 0.47, 0.075, "(13) OUTPUT & EXPORT", "out")

draw_box(
    ax,
    0.525,
    0.168,
    0.200,
    0.055,
    "RestorationResult",
    "defect_analysis · pqs_result · musical_goals\n"
    "excellence · temporal_coherence · emotional_arc\n"
    "genealogy · harmonic_fingerprint · phase_gate_log\n"
    "adaptive_thresholds · physical_ceiling · era_decade",
    "out",
    fontsize=5.6,
)
draw_box(
    ax,
    0.735,
    0.168,
    0.245,
    0.055,
    "Export",
    "FLAC 24-bit · WAV 24-bit · MP3 CBR/VBR (LAME)\n"
    "AIFF 24-bit · OGG Vorbis · EBU R128 −14 LUFS\n"
    "True-Peak −1.0 dBTP · POW-r Dithering 24→16bit\n"
    "ID3v2.4 / Vorbis Comments · Restaurierungs-Meta",
    "out",
    fontsize=5.6,
)

# ==================== LEGENDE ====================
legend_items = [
    ("fe", "Frontend (PyQt5)"),
    ("cli", "CLI / Adapter"),
    ("api", "API-Schicht"),
    ("pre", "Vor-Analyse"),
    ("reason", "Defekt-Inferenz"),
    ("guard", "Harmonik/Schutz"),
    ("phases", "Phase-Pipeline"),
    ("stem", "Stem-Verarbeitung"),
    ("post", "Post-Verarbeitung"),
    ("plugin", "Plugins (ONNX)"),
    ("mem", "Manifest/Persistenz"),
    ("out", "Output/Export"),
]
lx = 0.013
ly = 0.140
for i, (key, label) in enumerate(legend_items):
    fc, ec = COLORS[key]
    rect = FancyBboxPatch(
        (lx + i * 0.081, ly),
        0.016,
        0.010,
        boxstyle="round,pad=0.001",
        facecolor=rgba(fc, 0.85),
        edgecolor=rgba(ec),
        linewidth=0.5,
        zorder=5,
    )
    ax.add_patch(rect)
    ax.text(
        lx + i * 0.081 + 0.019,
        ly + 0.005,
        label,
        va="center",
        ha="left",
        fontsize=5.2,
        color=rgba("#e2e8f0"),
        fontfamily=FONT_MAIN,
        zorder=6,
    )

# ==================== VERTIKALE FLOW-PFEILE ====================
flow_y = [
    (0.877, 0.872),
    (0.819, 0.811),
    (0.748, 0.740),
    (0.678, 0.670),
    (0.617, 0.609),
    (0.474, 0.468),
    (0.352, 0.344),
    (0.242, 0.235),
    (0.160, 0.152),
]
for y0, y1 in flow_y:
    arrow(ax, 0.5, y0, 0.5, y1, color="#4b5563", lw=0.9)

# Haupt-Flow-Linie links
ax.plot([0.008, 0.008], [0.152, 0.940], color=rgba("#334155", 0.4), lw=1.5, zorder=1)

# ==================== FUSSZEILE ====================
ax.text(
    0.5,
    0.012,
    "Aurik 10.0.0 · 6312 Tests · Python 3.10 · PyQt5 · FastAPI · ONNX Runtime (CPU-only) · "
    "EBU R128 · ITU-R BS.1770-4 · ISO 532-1:2017 · ISO 226:2003 · Stand: Februar 2026",
    ha="center",
    va="bottom",
    fontsize=5.8,
    color=rgba("#475569"),
    fontfamily=FONT_MAIN,
    zorder=10,
)


# ==================== SPEICHERN ====================
out_path = pathlib.Path("/media/michael/Software 4TB/Aurik_Standalone/docs/aurik_architecture.png")
out_path.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(str(out_path), dpi=DPI, bbox_inches="tight", facecolor=C_BG, edgecolor="none", format="png")
plt.close(fig)
print(f"✅ Gespeichert: {out_path}")
print(f"   Größe: {out_path.stat().st_size / 1_048_576:.1f} MB")
