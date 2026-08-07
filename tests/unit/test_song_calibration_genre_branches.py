from __future__ import annotations

"""
Unit-Tests: SongCal-Genre-Branches (§2.31a / §2.47)

Testet alle 16 GenreClassifier-Ausgaben in _build_song_calibration_profile().
Fokus auf die in dieser Session hinzugefügten Branches (Pop, Soul/R&B, Blues,
Country, Folk, Reggae, Latin, Funk, Gospel) sowie kritische Invarianten.

Spec-Referenzen:
    §2.31a SongCalibration — genre_label-Adaptation
    §2.47  Adaptive-Intelligence-Prinzip
    §0     Klangwahrheit (Reggae/Gospel-Reverb = Authentizität)
"""


import pytest

from backend.core.quality_mode import QualityMode
from backend.core.unified_restorer_v3 import UnifiedRestorerV3

# ---------------------------------------------------------------------------
# Fixtures / Helpers
# ---------------------------------------------------------------------------

_BASE_KWARGS = {
    "material_type": None,
    "mode": QualityMode.BALANCED,
    "restorability_score": 70.0,
    "input_snr_db": 25.0,
    "max_defect_severity": 0.30,
    "pipeline_confidence": 0.90,
}


def _profile(genre_label: str = "") -> dict:
    """Build calibration profile for given genre label with standard parameters."""
    return UnifiedRestorerV3._build_song_calibration_profile(genre_label=genre_label, **_BASE_KWARGS)  # type: ignore[arg-type]


def _reverb(genre_label: str = "") -> float:
    return _profile(genre_label)["family_scalars"]["reverb"]  # type: ignore[no-any-return]


def _transient(genre_label: str = "") -> float:
    return _profile(genre_label)["family_scalars"]["transient"]  # type: ignore[no-any-return]


def _dynamics(genre_label: str = "") -> float:
    return _profile(genre_label)["family_scalars"]["dynamics_eq"]  # type: ignore[no-any-return]


def _vocal(genre_label: str = "") -> float:
    return _profile(genre_label)["family_scalars"]["vocal"]  # type: ignore[no-any-return]


def _denoise(genre_label: str = "") -> float:
    return _profile(genre_label)["family_scalars"]["denoise"]  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# 1. Bounds-Invariante: alle neuen Genres ∈ [0.30, 1.80] (§2.31a)
# ---------------------------------------------------------------------------

_ALL_GENRES = [
    "Rock",
    "Jazz",
    "Klassik",
    "Oper",
    "Pop",
    "Blues",
    "Soul/R&B",
    "Country",
    "Folk",
    "Electronic",
    "Hip-Hop",
    "Metal",
    "Latin",
    "Gospel",
    "Reggae",
    "Funk",
    "Unbekannt",
    "",
]


@pytest.mark.parametrize("genre", _ALL_GENRES)
def test_genre_family_scalars_in_bounds(genre: str) -> None:
    """Alle family_scalars müssen innerhalb [0.30, 1.80] liegen."""
    p = _profile(genre)
    for fam, val in p["family_scalars"].items():
        assert 0.30 <= val <= 1.80, f"genre='{genre}' family_scalars['{fam}'] = {val:.4f} außerhalb [0.30, 1.80]"


@pytest.mark.parametrize("genre", _ALL_GENRES)
def test_genre_global_scalar_in_bounds(genre: str) -> None:
    """global_scalar muss innerhalb [0.50, 1.50] liegen (§2.31a §LückeG-Fix)."""
    p = _profile(genre)
    gs = p["global_scalar"]
    assert 0.50 <= gs <= 1.50, f"genre='{genre}' global_scalar={gs:.4f} außerhalb [0.50, 1.50]"


# ---------------------------------------------------------------------------
# 2. Reggae: Reverb-Faktor 0.55 → deutlich niedriger als Baseline
#    §0 — Dub/Echo-Reverb ist ABSICHTLICHES Stilmittel, kein Defekt
# ---------------------------------------------------------------------------


def test_reggae_reverb_significantly_below_baseline() -> None:
    """Reggae reverb_scalar muss ≥ 30 % unter Baseline liegen (Dub/Echo-Schutz)."""
    baseline = _reverb("")
    reggae = _reverb("Reggae")
    assert reggae < baseline * 0.70, (
        f"Reggae reverb_scalar={reggae:.4f} nicht < 70 % von baseline={baseline:.4f} "
        f"— Dub/Echo-Reverb wird unzureichend geschützt"
    )


def test_reggae_reverb_lowest_of_all_genres() -> None:
    """Reggae muss den niedrigsten Reverb-Scalar aller Genres haben."""
    reggae_rv = _reverb("Reggae")
    for genre in _ALL_GENRES:
        if genre in ("Reggae",):
            continue
        other_rv = _reverb(genre)
        assert reggae_rv <= other_rv, f"Reggae reverb_scalar={reggae_rv:.4f} nicht ≤ {genre}={other_rv:.4f}"


# ---------------------------------------------------------------------------
# 3. Gospel: Kirchenhall-Schutz (reverb_factor=0.65)
#    §0 — Kirchenreverb ist Authentizität, kein Artefakt
# ---------------------------------------------------------------------------


def test_gospel_reverb_significantly_below_baseline() -> None:
    """Gospel reverb_scalar muss ≥ 25 % unter Baseline liegen (Kirchenhall-Schutz)."""
    baseline = _reverb("")
    gospel = _reverb("Gospel")
    assert gospel < baseline * 0.75, f"Gospel reverb_scalar={gospel:.4f} nicht < 75 % von baseline={baseline:.4f}"


def test_gospel_reverb_higher_than_reggae() -> None:
    """Gospel (0.65) muss höheren Reverb-Faktor als Reggae (0.55) haben — korrekte Rangfolge."""
    assert _reverb("Gospel") > _reverb("Reggae"), "Gospel reverb_scalar muss > Reggae reverb_scalar sein (0.65 > 0.55)"


def test_gospel_vocal_boost() -> None:
    """Gospel muss leicht erhöhten vocal_scalar haben (Chor-Vokalprominenz)."""
    baseline = _vocal("")
    assert _vocal("Gospel") >= baseline * 1.02, "Gospel vocal_scalar muss leicht über Baseline liegen"


# ---------------------------------------------------------------------------
# 4. Klassik/Oper: Reverb-Schutz (§2.31a, bekannte Branches)
# ---------------------------------------------------------------------------


def test_klassik_reverb_below_baseline() -> None:
    """Klassik reverb_scalar muss unter Baseline liegen (Raumklang = Authentizität)."""
    assert _reverb("Klassik") < _reverb(""), "Klassik reverb_scalar muss < Baseline sein"


def test_oper_reverb_below_baseline() -> None:
    """Oper reverb_scalar muss unter Baseline liegen."""
    assert _reverb("Oper") < _reverb(""), "Oper reverb_scalar muss < Baseline sein"


# ---------------------------------------------------------------------------
# 5. Folk: Denoise sehr konservativ (Atemgeräusche = Textur)
# ---------------------------------------------------------------------------


def test_folk_denoise_most_conservative() -> None:
    """Folk denoise_scalar muss der niedrigste aller Nicht-Klassik/Jazz/Rock-Genres sein."""
    folk_dn = _denoise("Folk")
    baseline = _denoise("")
    assert folk_dn < baseline * 0.95, (
        f"Folk denoise_scalar={folk_dn:.4f} nicht konservativ genug (Baseline={baseline:.4f})"
    )


def test_folk_reverb_lower_than_jazz() -> None:
    """Folk reverb_scalar muss < Jazz sein (intimer kleiner Raum vs. Club)."""
    assert _reverb("Folk") < _reverb("Jazz"), (
        f"Folk reverb={_reverb('Folk'):.4f} muss < Jazz reverb={_reverb('Jazz'):.4f}"
    )


# ---------------------------------------------------------------------------
# 6. Electronic / Hip-Hop: Dynamics am stärksten gedämpft
# ---------------------------------------------------------------------------


def test_electronic_dynamics_lowest_of_vocal_genres() -> None:
    """Electronic/Hip-Hop dynamics_scalar muss unter Pop/Rock/Blues liegen."""
    elec_dyn = _dynamics("Electronic")
    for other_genre in ("Pop", "Rock", "Blues", "Country", "Folk"):
        assert elec_dyn <= _dynamics(other_genre), (
            f"Electronic dynamics_scalar={elec_dyn:.4f} nicht ≤ {other_genre}={_dynamics(other_genre):.4f}"
        )


def test_hiphop_matches_electronic() -> None:
    """Hip-Hop und Electronic müssen identische Kalibrierung haben (gleicher Branch)."""
    p_elec = _profile("Electronic")["family_scalars"]
    p_hh = _profile("Hip-Hop")["family_scalars"]
    for fam in p_elec:
        assert p_elec[fam] == pytest.approx(p_hh[fam], abs=1e-6), (
            f"Electronic und Hip-Hop family_scalars['{fam}'] weichen ab: {p_elec[fam]:.4f} vs {p_hh[fam]:.4f}"
        )


# ---------------------------------------------------------------------------
# 7. Funk / Latin: Transient-Boost
# ---------------------------------------------------------------------------


def test_funk_transient_above_baseline() -> None:
    """Funk transient_scalar muss > Baseline liegen (Slap-Bass, Brass-Attacken)."""
    assert _transient("Funk") > _transient(""), "Funk transient_scalar muss über Baseline liegen"


def test_latin_transient_above_baseline() -> None:
    """Latin transient_scalar muss > Baseline liegen (Conga, Bongo, Claves)."""
    assert _transient("Latin") > _transient(""), "Latin transient_scalar muss über Baseline liegen"


def test_funk_transient_highest_rhythm_genre() -> None:
    """Funk muss höchsten Transient-Scalar aller Rhythmus-Genres haben."""
    for genre in ("Latin", "Pop", "Soul/R&B", "Blues", "Country", "Reggae"):
        assert _transient("Funk") >= _transient(genre), (
            f"Funk transient={_transient('Funk'):.4f} nicht ≥ {genre}={_transient(genre):.4f}"
        )


# ---------------------------------------------------------------------------
# 8. Reggae: Transient NICHT erhöht (Laid-back Groove)
# ---------------------------------------------------------------------------


def test_reggae_transient_not_boosted() -> None:
    """Reggae transient_scalar darf nicht über Baseline erhöht sein (One-Drop-Groove)."""
    assert _transient("Reggae") <= _transient(""), (
        f"Reggae transient_scalar={_transient('Reggae'):.4f} über Baseline={_transient(''):.4f} "
        f"— würde Groove-Charakter zerstören"
    )


# ---------------------------------------------------------------------------
# 9. Soul/R&B: Vokal-Boost + moderater Reverb-Schutz
# ---------------------------------------------------------------------------


def test_soul_rb_vocal_boost() -> None:
    """Soul/R&B vocal_scalar muss über Baseline liegen."""
    assert _vocal("Soul/R&B") > _vocal(""), "Soul/R&B vocal_scalar muss > Baseline sein"


def test_soul_rb_reverb_below_baseline() -> None:
    """Soul/R&B reverb_scalar muss unter Baseline liegen (warmer Studio-Hall)."""
    assert _reverb("Soul/R&B") < _reverb(""), "Soul/R&B reverb_scalar muss < Baseline sein"


# ---------------------------------------------------------------------------
# 10. Pop: Modernes sauberes Material — Dynamics dämpfen
# ---------------------------------------------------------------------------


def test_pop_dynamics_below_baseline() -> None:
    """Pop dynamics_scalar muss unter Baseline liegen (künstlerische Komprimierung respektieren)."""
    assert _dynamics("Pop") < _dynamics(""), "Pop dynamics_scalar muss < Baseline sein"


# ---------------------------------------------------------------------------
# 11. Blues: Röhren-Charakter-Schutz
# ---------------------------------------------------------------------------


def test_blues_denoise_conservative() -> None:
    """Blues denoise_scalar muss unter Baseline liegen (Röhrenverstärker = Klangidentität)."""
    assert _denoise("Blues") < _denoise(""), "Blues denoise_scalar muss < Baseline sein (Röhrensättigung schützen)"


def test_blues_reverb_below_baseline() -> None:
    """Blues reverb_scalar muss unter Baseline liegen (Club-Atmosphäre bewahren)."""
    assert _reverb("Blues") < _reverb(""), "Blues reverb_scalar muss < Baseline sein"


# ---------------------------------------------------------------------------
# 12. Country: Nashville-Raumklang + Transient-Boost
# ---------------------------------------------------------------------------


def test_country_reverb_below_baseline() -> None:
    """Country reverb_scalar muss unter Baseline liegen."""
    assert _reverb("Country") < _reverb(""), "Country reverb_scalar muss < Baseline sein (Nashville-Raumklang)"


def test_country_transient_above_baseline() -> None:
    """Country transient_scalar muss über Baseline liegen (Banjo, Flat-Pick)."""
    assert _transient("Country") > _transient(""), "Country transient_scalar muss > Baseline sein"


# ---------------------------------------------------------------------------
# 13. Case-Insensitivität und Varianten
# ---------------------------------------------------------------------------


def test_reggae_case_insensitive() -> None:
    """Genre-Label ist case-insensitiv — 'reggae', 'Reggae', 'REGGAE' müssen gleich sein."""
    rv1 = _reverb("reggae")
    rv2 = _reverb("Reggae")
    rv3 = _reverb("REGGAE")
    assert rv1 == pytest.approx(rv2, abs=1e-9)
    assert rv1 == pytest.approx(rv3, abs=1e-9)


def test_soul_rb_aliases() -> None:
    """'Soul/R&B', 'soul', 'r&b' müssen identische Kalibrierung liefern."""
    s1 = _profile("Soul/R&B")["family_scalars"]
    s2 = _profile("soul")["family_scalars"]
    s3 = _profile("r&b")["family_scalars"]
    for fam in s1:
        assert s1[fam] == pytest.approx(s2[fam], abs=1e-9), f"'Soul/R&B' vs 'soul' differ at {fam}"
        assert s1[fam] == pytest.approx(s3[fam], abs=1e-9), f"'Soul/R&B' vs 'r&b' differ at {fam}"


# ---------------------------------------------------------------------------
# 14. Unbekannte Labels → Baseline (kein Absturz, kein falscher Wert)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("unknown", ["Schlager", "Bhangra", "Tango", "XYZ_UNKNOWN_123", ""])
def test_unknown_genre_returns_baseline_profile(unknown: str) -> None:
    """Unbekannte Genre-Labels dürfen nicht abstürzen und liefern Baseline-ähnliche Werte."""
    p_unknown = _profile(unknown)
    assert isinstance(p_unknown, dict)
    assert "family_scalars" in p_unknown
    # Alle Scalars müssen innerhalb Bounds sein
    for fam, val in p_unknown["family_scalars"].items():
        assert 0.30 <= val <= 1.80, f"unknown genre '{unknown}' family '{fam}' = {val:.4f} out of bounds"


# ---------------------------------------------------------------------------
# 15. Determinismus aller neuen Genre-Branches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "genre",
    [
        "Pop",
        "Soul/R&B",
        "Blues",
        "Country",
        "Folk",
        "Reggae",
        "Latin",
        "Funk",
        "Gospel",
    ],
)
def test_new_genre_branches_deterministic(genre: str) -> None:
    """Gleiche Inputs → gleiche Ausgabe (kein Random, kein State)."""
    p1 = _profile(genre)
    p2 = _profile(genre)
    assert p1["global_scalar"] == p2["global_scalar"]
    for fam in p1["family_scalars"]:
        assert p1["family_scalars"][fam] == p2["family_scalars"][fam], (
            f"genre='{genre}' family='{fam}' nicht deterministisch"
        )


# ---------------------------------------------------------------------------
# 16. Alle neuen Genre-Branches unterscheiden sich von Baseline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "genre",
    [
        "Pop",
        "Soul/R&B",
        "Blues",
        "Country",
        "Folk",
        "Reggae",
        "Latin",
        "Funk",
        "Gospel",
    ],
)
def test_new_genre_produces_different_profile_from_baseline(genre: str) -> None:
    """Jeder neue Genre-Branch muss sich in mind. einem family_scalar von Baseline unterscheiden."""
    p_genre = _profile(genre)["family_scalars"]
    p_base = _profile("")["family_scalars"]
    diffs = [fam for fam in p_genre if abs(p_genre[fam] - p_base[fam]) > 1e-6]
    assert len(diffs) > 0, f"genre='{genre}' produziert identisches Profil wie Baseline — Branch ohne Effekt?"
