"""Phase result summary — human-readable Fazit-Meldung for every Aurik phase.

Usage inside a phase's process():
    from backend.core.phase_fazit import log_phase_fazit
    log_phase_fazit(
        phase="03",
        name="Entrauschen",
        score=8.5,
        summary="Rauschen um 12.3 dB reduziert, Sprachverständlichkeit erhalten",
        details={"snr_before_db": 18.2, "snr_after_db": 30.5},
    )

Output in logs:
    ┌─ Phase 03 (Entrauschen) ──────────────────────────────────────────┐
    │ ✅ Rauschen um 12.3 dB reduziert, Sprachverständlichkeit erhalten  │
    │ 📊 Score: 8.5 / 10.0  (SNR: 18.2 → 30.5 dB)                      │
    └────────────────────────────────────────────────────────────────────┘

Score meaning:
    10.0 = Phase zu 100% wie geplant umgesetzt, alle Defekte unhörbar
     7.0 = Deutliche Verbesserung, leichte Restdefekte hörbar
     5.0 = Moderate Verbesserung, Defekte teilweise reduziert
     3.0 = Geringe Verbesserung, Defekte noch deutlich hörbar
     0.0 = Keine Verbesserung / Phase wirkungslos
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Width of the box
_WIDTH = 72


def log_phase_fazit(
    phase: str,
    name: str,
    score: float,
    summary: str,
    details: dict[str, Any] | None = None,
) -> None:
    """Log a human-readable phase summary with score.

    Args:
        phase: Phase number (e.g. "03", "09", "24")
        name:  Human-readable phase name (e.g. "Entrauschen", "Knackser-Entfernung")
        score: 0.0–10.0 rating of phase success
        summary: One-sentence summary of what was achieved
        details: Optional dict of key metrics (e.g. {"SNR": "18→30 dB"})
    """
    score = float(max(0.0, min(10.0, score)))
    score_emoji = _score_emoji(score)

    # Build header
    header = f" Phase {phase} ({name}) "
    pad_total = _WIDTH - len(header) - 2  # 2 for ┌─ and ─┐
    if pad_total > 0:
        header = "┌─" + header + "─" * pad_total + "┐"
    else:
        header = "┌─" + header[: _WIDTH - 4] + "─┐"

    # Build score line
    score_text = f" {score_emoji} Score: {score:.1f} / 10.0"
    if details:
        detail_parts = []
        for k, v in list(details.items())[:3]:
            detail_parts.append(f"{k}: {v}")
        score_text += "  (" + ", ".join(detail_parts) + ")"
    score_text = score_text.ljust(_WIDTH - 2)[: _WIDTH - 2] + " │"

    # Build summary line (may wrap)
    summary_lines = _wrap_text(f" {summary}", _WIDTH - 4)
    summary_formatted = []
    for sl in summary_lines:
        summary_formatted.append("│ " + sl.ljust(_WIDTH - 4) + " │")

    # Build footer
    footer = "└" + "─" * (_WIDTH - 2) + "┘"

    # Log as single multi-line INFO message
    lines = [header] + summary_formatted + [score_text, footer]
    logger.info("\n".join(lines))


def _score_emoji(score: float) -> str:
    """Return an emoji for the score bracket."""
    if score >= 9.0:
        return "🏆"
    elif score >= 7.5:
        return "✅"
    elif score >= 5.0:
        return "👍"
    elif score >= 2.5:
        return "⚠️"
    else:
        return "❌"


def _wrap_text(text: str, width: int) -> list[str]:
    """Wrap text to width, returning list of lines."""
    if len(text) <= width:
        return [text]
    lines = []
    while len(text) > width:
        # Find last space within width
        split = text.rfind(" ", 0, width)
        if split < 0:
            split = width
        lines.append(text[:split])
        text = " " + text[split:].lstrip()
    if text.strip():
        lines.append(text)
    return lines


def log_restoration_summary(
    total_time_s: float,
    rt_factor: float,
    quality_pct: float,
    chain: list[str] | None = None,
    genre: str = "",
    era_decade: int = 0,
    phases_count: int = 0,
    mushra_score: float = 0.0,
    hpi_score: float = 0.0,
    degradation_status: str = "ok",
    fail_reason: str = "",
    oqs_threshold: float | None = None,
    residual_audible_defects: int = 0,
    no_effect_count: int = 0,
) -> None:
    """Post-restoration summary in plain German for non-expert users.

    Produces a comprehensive, human-readable box explaining what was done,
    what was found, and how good the result is.
    """
    W = 72
    quality_label = _quality_label_de(quality_pct)
    chain_story = _chain_story_de(chain) if chain else ""
    _degraded = str(degradation_status or "ok").lower() in {"degraded", "eingeschraenkt", "eingeschränkt"}

    lines = []
    lines.append("┌" + "─" * (W - 2) + "┐")
    if _degraded:
        lines.append(_center("✨ AURIK RESTAURATION MIT EINSCHRÄNKUNG ✨", W))
    else:
        lines.append(_center("✨ AURIK RESTAURATION ABGESCHLOSSEN ✨", W))
    lines.append("│" + " " * (W - 2) + "│")

    # ── Degradations-Status (§v10.201/§v10.202): Einschraenkung + Ursache sichtbar machen ──
    if _degraded:
        if fail_reason:
            lines.append(_center(f"⚠️  Grund: {fail_reason}", W))
        if oqs_threshold is not None:
            lines.append(_center(f"📏 OQS-Floor: {oqs_threshold:.0f}", W))
        if residual_audible_defects > 0:
            lines.append(_center(f"👂 {residual_audible_defects} hörbare Restdefekte verblieben", W))
        if no_effect_count > 0:
            lines.append(_center(f"🛑 {no_effect_count} ohne Wirkung", W))
        lines.append("│" + " " * (W - 2) + "│")

    # ── Technische Qualität (objektive Signalanalyse) ──
    _qa_explanation = _quality_explanation_de(quality_pct)
    lines.append(_center(f"📊 Signalqualität: {quality_pct:.0f}% — {quality_label}", W))
    for _qel in _wrap_text(_qa_explanation, W - 4):
        lines.append("│ " + _qel.ljust(W - 4) + " │")

    # ── Hörerlebnis (perzeptuell, MUSHRA) ──
    if mushra_score > 0:
        _mushra_label, _mushra_expl = _mushra_label_de(mushra_score)
        lines.append("│" + " " * (W - 2) + "│")
        lines.append(_center(f"🎧 Hörerlebnis: {mushra_score:.0f}/100 — {_mushra_label}", W))
        for _mel in _wrap_text(_mushra_expl, W - 4):
            lines.append("│ " + _mel.ljust(W - 4) + " │")

    # ── Restaurations-Index (HPI) ──
    if hpi_score > 0:
        _hpi_label, _hpi_expl = _hpi_label_de(hpi_score)
        lines.append("│" + " " * (W - 2) + "│")
        lines.append(_center(f"🧠 Restaurations-Index: {hpi_score:.3f} — {_hpi_label}", W))
        for _hel in _wrap_text(_hpi_expl, W - 4):
            lines.append("│ " + _hel.ljust(W - 4) + " │")

    # Chain story
    if chain_story:
        lines.append("│" + " " * (W - 2) + "│")
        lines.append(_center("📀 Tonträgerkette", W))
        for cl in _wrap_text(chain_story, W - 4):
            lines.append("│ " + cl.ljust(W - 4) + " │")

    # Genre + Era
    if genre or era_decade:
        parts = []
        if genre:
            parts.append(f"Genre: {genre}")
        if era_decade:
            parts.append(f"Aufnahme: {era_decade}er Jahre")
        lines.append("│" + " " * (W - 2) + "│")
        lines.append(_center(" ⋂ ".join(parts), W))

    # Stats
    lines.append("│" + " " * (W - 2) + "│")
    lines.append(
        _center(
            f"⏱ {total_time_s:.0f}s  ⋂  {rt_factor:.1f}× Echtzeit  ⋂  {phases_count} Phasen",
            W,
        )
    )

    lines.append("│" + " " * (W - 2) + "│")
    lines.append(_center("ℹ️  Details in den Phasen-Fazits oberhalb", W))
    lines.append("└" + "─" * (W - 2) + "┘")

    logger.info("\n".join(lines))


def _quality_label_de(pct: float) -> str:
    if pct >= 95:
        return "🏆 Weltklasse — wie Studio-Aufnahme"
    elif pct >= 85:
        return "✅ Ausgezeichnet — kaum vom Original zu unterscheiden"
    elif pct >= 70:
        return "👍 Sehr gut — deutliche Verbesserung"
    elif pct >= 50:
        return "⚡ Gut — hörbare Verbesserung, leichte Restdefekte"
    elif pct >= 30:
        return "⚠️  Ausreichend — das Material limitiert die Qualität"
    else:
        return "❌ Schwieriges Material — starke Degradation"


def _quality_explanation_de(pct: float) -> str:
    """Erklärt die technische Signalqualität in verständlichem Deutsch."""
    if pct >= 85:
        return (
            "Technische Analyse von Rauschabstand, Dynamik und Verzerrung. "
            "Die Quelle ist in sehr gutem technischem Zustand."
        )
    elif pct >= 70:
        return (
            "Technische Analyse von Rauschabstand, Dynamik und Verzerrung. "
            "Die Quelle hat leichte technische Limits (z.B. MP3-Kompression, "
            "Bandrauschen), die das maximale technische Niveau begrenzen."
        )
    elif pct >= 50:
        return (
            "Technische Analyse von Rauschabstand, Dynamik und Verzerrung. "
            "Die Quelle ist deutlich beschädigt — das technische Limit ist "
            "durch das Ausgangsmaterial vorgegeben, nicht durch die Restauration."
        )
    else:
        return (
            "Technische Analyse von Rauschabstand, Dynamik und Verzerrung. "
            "Stark beschädigtes Material — Verbesserungen sind hörbar, "
            "aber die Quelle setzt enge technische Grenzen."
        )


def _mushra_label_de(score: float) -> tuple[str, str]:
    """Liefert (Label, Erklärung) für den MUSHRA-Hörerlebnisscore."""
    if score >= 90:
        return (
            "Hervorragend",
            "Simuliert ein geschultes Hörerpanel. 90–100 = Die Aufnahme klingt "
            "klar, natürlich und frei von störenden Artefakten — als wäre sie "
            "nie beschädigt gewesen.",
        )
    elif score >= 80:
        return (
            "Ausgezeichnet",
            "Simuliert ein geschultes Hörerpanel. 80–89 = Sehr gute Klangqualität, "
            "minimale Restspuren der ursprünglichen Schäden noch erkennbar.",
        )
    elif score >= 60:
        return (
            "Gut",
            "Simuliert ein geschultes Hörerpanel. 60–79 = Hörbare Verbesserung, "
            "aber einige Artefakte oder Klangverluste bleiben bestehen.",
        )
    else:
        return (
            "Ausreichend",
            "Simuliert ein geschultes Hörerpanel. < 60 = Die Restauration konnte "
            "nur begrenzt helfen — das Material ist sehr stark beschädigt.",
        )


def _hpi_label_de(score: float) -> tuple[str, str]:
    """Liefert (Label, Erklärung) für den Holistic Perceptual Index."""
    if score >= 0.85:
        return (
            "Exzellente Restauration",
            "Der HPI bewertet die Gesamt-Restauration: Klangtreue zum Original, "
            "Artefaktfreiheit, emotionaler Erhalt und Natürlichkeit. "
            "> 0.85 = Die Pipeline hat den Klang verbessert, ohne zu schaden.",
        )
    elif score >= 0.70:
        return (
            "Gute Restauration",
            "Der HPI bewertet die Gesamt-Restauration: Klangtreue zum Original, "
            "Artefaktfreiheit, emotionaler Erhalt und Natürlichkeit. "
            "0.70–0.85 = Solide Verbesserung mit minimalen Kompromissen.",
        )
    elif score >= 0.50:
        return (
            "Akzeptable Restauration",
            "Der HPI bewertet die Gesamt-Restauration: Klangtreue zum Original, "
            "Artefaktfreiheit, emotionaler Erhalt und Natürlichkeit. "
            "0.50–0.70 = Teilweise Verbesserung, aber auch leichte Verluste.",
        )
    else:
        return (
            "Schwieriges Material",
            "Der HPI bewertet die Gesamt-Restauration: Klangtreue zum Original, "
            "Artefaktfreiheit, emotionaler Erhalt und Natürlichkeit. "
            "< 0.50 = Das Material konnte nur begrenzt restauriert werden.",
        )


def _chain_story_de(chain: list[str]) -> str:
    """Convert a technical chain into a human-readable German story."""
    names = {
        "reel_tape": "Tonband (Studio-Aufnahme)",
        "vinyl": "Vinyl-Schallplatte (Veröffentlichung)",
        "shellac": "Schellackplatte (78rpm)",
        "wax_cylinder": "Wachswalze",
        "lacquer_disc": "Lackplatte (Mitschnitt)",
        "wire_recording": "Stahldraht-Aufnahme",
        "cassette": "Compact Cassette (Überspielung)",
        "cartridge_8track": "8-Spur-Cartridge",
        "cd_digital": "Compact Disc (CD)",
        "dat": "Digital Audio Tape (DAT)",
        "dcc": "Digital Compact Cassette (DCC)",
        "minidisc": "MiniDisc",
        "mp3_low": "MP3 (niedrige Bitrate, Digitalisierung)",
        "mp3_high": "MP3 (hohe Bitrate)",
        "aac": "AAC/M4A",
        "streaming": "Streaming-Dienst",
    }
    parts = []
    for i, m in enumerate(chain):
        name = names.get(m, m)
        if i == 0:
            parts.append(f"📀 {name}")
        elif i == len(chain) - 1 and m.startswith("mp3"):
            parts.append(f"💾 {name}")
        else:
            parts.append(f"→ {name}")
    return " ".join(parts)


def _center(text: str, width: int) -> str:
    """Center text in a box line."""
    visible = len(text)
    pad = width - 2 - visible
    if pad < 0:
        return "│ " + text[: width - 4] + " │"
    left = pad // 2
    right = pad - left
    return "│" + " " * left + text + " " * right + "│"
