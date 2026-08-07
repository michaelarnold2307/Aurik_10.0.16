#!/usr/bin/env python3
"""Discover and download missing core model artifacts from HuggingFace.

This script is best-effort and prefers exact target filenames.
Downloaded files are staged into local drop-in folders, then can be ingested via:
  scripts/auto_ingest_core_models.py

Safety constraints:
- No overwrite by default.
- Exact filename is preferred; alias/extension/size filters still apply.
- Minimum file size checks prevent obvious wrong artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DROPIN = ROOT / "models" / ".dropin"


@dataclass(frozen=True)
class Target:
    name: str
    target_rel: str
    filename: str
    query: str
    min_size_bytes: int
    aliases: tuple[str, ...]
    allowed_ext: tuple[str, ...]


TARGETS = [
    Target(
        "rmvpe",
        "models/rmvpe/rmvpe.onnx",
        "rmvpe.onnx",
        "rmvpe",
        1_000_000,
        aliases=("rmvpe",),
        allowed_ext=(".onnx",),
    ),
    Target(
        "beats",
        "models/beats/beats_iter3.onnx",
        "beats_iter3.onnx",
        "beats",
        1_000_000,
        aliases=("beats_iter3", "beats_iter", "beats"),
        allowed_ext=(".onnx",),
    ),
    Target(
        "sgmse_plus",
        "models/sgmse_plus/sgmse_plus.ts",
        "sgmse_plus.ts",
        "sgmse",
        1_000_000,
        aliases=("sgmse_plus", "sgmse", "sgmseplus", "sgmse_plus.ts"),
        allowed_ext=(".ts", ".pt", ".pth", ".ckpt"),
    ),
    Target(
        "versa",
        "models/versa/hub_cache/checkpoints/ft_wav2vec2_large_ll60k_mdf_p1_200epochs_all_192epochs.pth",
        "ft_wav2vec2_large_ll60k_mdf_p1_200epochs_all_192epochs.pth",
        "singmos",
        1_000_000,
        aliases=("ft_wav2vec2_large_ll60k", "singmos_pro", "singmos", "versa"),
        allowed_ext=(".pth", ".pt", ".ckpt"),
    ),
    Target(
        "flow_matching",
        "models/flow_matching/flow_matching.onnx",
        "flow_matching.onnx",
        "flow matching",
        1_000_000,
        aliases=("flow_matching", "flowmatching", "flow-matching", "flow", "matching", "score_network"),
        allowed_ext=(".onnx", ".pt", ".pth", ".ckpt"),
    ),
    Target(
        "mp_senet",
        "models/mp_senet/mp_senet.onnx",
        "mp_senet.onnx",
        "mp-senet",
        1_000_000,
        aliases=("mp_senet", "mpsenet", "mp-senet"),
        allowed_ext=(".onnx",),
    ),
    Target(
        "gacela",
        "models/gacela/model/01_400000.pt",
        "01_400000.pt",
        "gacela",
        5_000_000,
        aliases=("01_400000", "gacela"),
        allowed_ext=(".pt", ".pth"),
    ),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch missing core model artifacts from HuggingFace")
    p.add_argument("--overwrite", action="store_true", help="Overwrite staged files in drop-in dir")
    p.add_argument("--apply", action="store_true", help="Actually download files (default is discovery-only)")
    p.add_argument("--limit", type=int, default=50, help="HF search result limit per target")
    p.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="Optional subset of targets (e.g. --only rmvpe beats)",
    )
    return p.parse_args()


def _request_headers() -> dict[str, str]:
    """Build request headers, optionally including HF auth token.

    Token lookup order:
    1) HF_TOKEN
    2) HUGGINGFACE_TOKEN
    """
    headers = {"User-Agent": "Aurik-CoreFetcher/1.0"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token.strip()}"
    return headers


def _has_hf_token() -> bool:
    return bool((os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or "").strip())


def _http_json(url: str) -> list | dict:
    req = urllib.request.Request(url, headers=_request_headers())
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.load(r)  # type: ignore[no-any-return]


def _search_models(query: str, limit: int) -> list[str]:
    q = urllib.parse.quote(query)
    url = f"https://huggingface.co/api/models?search={q}&limit={limit}"
    data = _http_json(url)
    ids: list[str] = []
    if isinstance(data, list):
        for item in data:
            mid = item.get("id") if isinstance(item, dict) else None
            if isinstance(mid, str) and mid:
                ids.append(mid)
    return ids


def _model_files(model_id: str) -> list[dict]:
    url = f"https://huggingface.co/api/models/{urllib.parse.quote(model_id, safe='/')}"
    data = _http_json(url)
    siblings = data.get("siblings", []) if isinstance(data, dict) else []
    return [s for s in siblings if isinstance(s, dict)]


def _pick_candidate(target: Target, model_ids: list[str]) -> tuple[str, str, int] | None:
    expected = target.filename.lower()
    best: tuple[int, int, str, str, int] | None = None

    for mid in model_ids:
        try:
            files = _model_files(mid)
        except Exception:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
            continue
        for f in files:
            name = str(f.get("rfilename", ""))
            if not name:
                continue
            base = Path(name).name.lower()
            ext = Path(base).suffix
            if ext not in target.allowed_ext:
                continue

            alias_hit = any(a in base for a in target.aliases)
            if not alias_hit and base != expected:
                continue

            size = int(f.get("size") or 0)
            if size < target.min_size_bytes:
                continue

            # Score: exact filename > startswith alias > contains alias.
            score = 0
            if base == expected:
                score += 100
            for a in target.aliases:
                if base.startswith(a):
                    score += 30
                elif a in base:
                    score += 10

            cand = (score, size, mid, name, size)
            if best is None or cand[:2] > best[:2]:
                best = cand

    if best is None:
        return None
    return best[2], best[3], best[4]


def _download(model_id: str, repo_file: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{model_id}/resolve/main/{urllib.parse.quote(repo_file, safe='/')}"
    req = urllib.request.Request(url, headers=_request_headers())
    with urllib.request.urlopen(req, timeout=60) as resp, out_path.open("wb") as fh:
        shutil.copyfileobj(resp, fh)


def main() -> int:
    args = _parse_args()
    only = {x.strip().lower() for x in args.only if x.strip()}
    targets = [t for t in TARGETS if not only or t.name in only]

    DROPIN.mkdir(parents=True, exist_ok=True)

    staged = 0
    missing = 0

    print("=" * 100)
    print("HF Core Model Discovery")
    print("=" * 100)
    if not _has_hf_token():
        print(
            "INFO HF token nicht gesetzt. Für private/gated Modelle können 401/403 auftreten. "
            "Setze HF_TOKEN oder HUGGINGFACE_TOKEN in der Shell."
        )

    for t in targets:
        canonical = ROOT / t.target_rel
        staged_path = DROPIN / t.filename

        if canonical.exists():
            print(f"OK      {t.name}: canonical exists ({t.target_rel})")
            continue
        if staged_path.exists() and not args.overwrite:
            print(f"OK      {t.name}: staged exists ({staged_path.relative_to(ROOT)})")
            continue

        try:
            model_ids = _search_models(t.query, args.limit)
        except Exception as exc:
            print(f"MISSING {t.name}: HF search failed ({exc})")
            missing += 1
            continue

        cand = _pick_candidate(t, model_ids)
        if cand is None:
            print(f"MISSING {t.name}: no suitable candidate (filename/alias/ext/size) in first {len(model_ids)} repos")
            missing += 1
            continue

        model_id, repo_file, size = cand
        print(f"FOUND   {t.name}: {model_id}/{repo_file} ({size / (1024 * 1024):.1f} MB)")

        if not args.apply:
            print("        discovery-only mode (use --apply to download)")
            staged += 1
            continue

        # Keep original suffix from remote file to avoid misleading staged extensions.
        staged_name = Path(repo_file).name
        staged_path = DROPIN / staged_name
        try:
            _download(model_id, repo_file, staged_path)
            print(f"STAGED  {t.name}: {staged_path.relative_to(ROOT)}")
            staged += 1
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                reason = (
                    "401: fehlender Token (HF_TOKEN/HUGGINGFACE_TOKEN)"
                    if not _has_hf_token()
                    else "401: Token ungültig oder kein Repo-Zugriff"
                )
                print(f"ERROR   {t.name}: download failed ({reason})")
            elif exc.code == 403:
                reason = (
                    "403: Zugriff verweigert (Token/Terms prüfen)"
                    if _has_hf_token()
                    else "403: fehlender Token oder Terms nicht akzeptiert"
                )
                print(f"ERROR   {t.name}: download failed ({reason})")
            else:
                print(f"ERROR   {t.name}: download failed (http_{exc.code})")
            missing += 1
        except (urllib.error.URLError, OSError) as exc:
            print(f"ERROR   {t.name}: download failed ({exc})")
            missing += 1

    print("\nSummary:")
    print(f"  staged_or_found={staged}")
    print(f"  unresolved={missing}")
    if missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
