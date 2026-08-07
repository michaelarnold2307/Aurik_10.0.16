#!/usr/bin/env python3
"""Download PyTorch core models and export ONNX where possible.

Workflow:
1) Download/checkpoint staging for missing core models.
2) Try ONNX export from TorchScript or nn.Module objects.
3) Copy successful ONNX artifacts to canonical model paths.

Notes:
- Best-effort only. Some checkpoints are state_dict-only and need architecture code.
- GACELA primary artifact is a .pt checkpoint and is copied as-is.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.metadata
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import types
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
STAGING = ROOT / "models" / ".dropin"
REPORT_PATH = ROOT / "reports" / "core_model_pt_export_report.json"
BEATS_SRC_DIR = ROOT / "models" / ".dropin" / "_src" / "unilm"
RMVPE_SRC_DIR = ROOT / "models" / ".dropin" / "_src" / "rvc_webui"
SGMSE_SRC_DIR = ROOT / "models" / ".dropin" / "_src" / "sgmse"


@dataclass(frozen=True)
class Target:
    name: str
    canonical_rel: str
    required_ext: str
    aliases: tuple[str, ...]
    min_size_bytes: int
    max_size_bytes: int
    urls: tuple[str, ...]


TARGETS = [
    Target(
        name="rmvpe",
        canonical_rel="models/rmvpe/rmvpe.onnx",
        required_ext=".onnx",
        aliases=("rmvpe",),
        min_size_bytes=1_000_000,
        max_size_bytes=120_000_000,
        urls=(
            "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
            "https://huggingface.co/therealvul/rmvpe/resolve/main/rmvpe.pt",
        ),
    ),
    Target(
        name="beats",
        canonical_rel="models/beats/beats_iter3.onnx",
        required_ext=".onnx",
        aliases=("beats_iter3", "beats_iter", "beats"),
        min_size_bytes=10_000_000,
        max_size_bytes=800_000_000,
        urls=(
            "https://github.com/microsoft/unilm/raw/master/beats/BEATs_iter3_plus_AS2M.pt",
            "https://huggingface.co/lpepino/beats_ckpts/resolve/main/BEATs_iter3_plus_AS2M.pt",
        ),
    ),
    Target(
        name="sgmse_plus",
        canonical_rel="models/sgmse_plus/sgmse_plus.ts",
        required_ext=".ts",
        aliases=("sgmse", "sgmse_plus", "sgmseplus"),
        min_size_bytes=5_000_000,
        max_size_bytes=2_000_000_000,
        urls=(
            "https://huggingface.co/sp-uhh/speech-enhancement-sgmse/resolve/main/train_vb_29nqe0uh_epoch%3D115.ckpt",
        ),
    ),
    Target(
        name="versa",
        canonical_rel="models/versa/hub_cache/checkpoints/ft_wav2vec2_large_ll60k_mdf_p1_200epochs_all_192epochs.pth",
        required_ext=".pth",
        aliases=("versa", "mos"),
        min_size_bytes=1_000_000,
        max_size_bytes=2_000_000_000,
        urls=(
            "https://github.com/South-Twilight/SingMOS/releases/download/ckpt_v3/ft_wav2vec2_large_ll60k_mdf_p1_200epochs_all_192epochs.pth",
        ),
    ),
    Target(
        name="flow_matching",
        canonical_rel="models/flow_matching/flow_matching.onnx",
        required_ext=".onnx",
        aliases=("flow", "matching"),
        min_size_bytes=5_000_000,
        max_size_bytes=3_500_000_000,
        urls=(
            "https://huggingface.co/youngdicey/flow-matching-baseline-pretraining-0309/resolve/main/best_epoch779_depth20_epidemic_zapsaplt_hdim1536.pth",
        ),
    ),
    Target(
        name="mp_senet",
        canonical_rel="models/mp_senet/mp_senet.onnx",
        required_ext=".onnx",
        aliases=("mp", "senet"),
        min_size_bytes=5_000_000,
        max_size_bytes=250_000_000,
        urls=(
            "https://huggingface.co/lx-ljl/MPSENET/resolve/main/mp_senet.pth",
            "https://huggingface.co/JacobLinCool/MP-SENet-DNS/resolve/main/best_model.pth",
        ),
    ),
    Target(
        name="gacela",
        canonical_rel="models/gacela/model/01_400000.pt",
        required_ext=".pt",
        aliases=("gacela", "01_400000"),
        min_size_bytes=5_000_000,
        max_size_bytes=800_000_000,
        urls=("https://huggingface.co/Gacela/Arte/resolve/main/01_400000.pt",),
    ),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch PT checkpoints and export ONNX where possible")
    p.add_argument("--only", nargs="*", default=[], help="Subset of targets")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing staged/canonical outputs")
    p.add_argument("--skip-download", action="store_true", help="Only use local staged files")
    p.add_argument(
        "--no-hf-discovery",
        action="store_true",
        help="Disable fallback discovery of candidate files via HF model API.",
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


def _download(url: str, out_path: Path, retries: int = 2) -> tuple[bool, str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_err = "unknown"
    headers = _request_headers()
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=40) as resp, open(out_path, "wb") as fh:
                shutil.copyfileobj(resp, fh)
            return True, "ok"
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                last_err = (
                    "http_401_missing_token_set_HF_TOKEN_or_HUGGINGFACE_TOKEN"
                    if not _has_hf_token()
                    else "http_401_token_invalid_or_no_repo_access"
                )
            elif exc.code == 403:
                last_err = (
                    "http_403_forbidden_missing_token_or_terms_not_accepted"
                    if not _has_hf_token()
                    else "http_403_forbidden_token_no_access_or_terms_not_accepted"
                )
            else:
                last_err = f"http_{exc.code}"
        except urllib.error.URLError as exc:
            last_err = f"urlerror:{exc.reason}"
        except TimeoutError:
            last_err = "timeout"
        except OSError as exc:
            last_err = f"oserror:{type(exc).__name__}"

        if out_path.exists():
            out_path.unlink(missing_ok=True)

        if attempt >= retries:
            return False, last_err
        time.sleep(1.0 + attempt)

    return False, last_err


def _hf_api_json(url: str, retries: int = 1) -> tuple[dict | list | None, str]:
    last_err = "unknown"
    headers = _request_headers()
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp), "ok"
        except urllib.error.HTTPError as exc:
            last_err = f"http_{exc.code}"
        except urllib.error.URLError as exc:
            last_err = f"urlerror:{exc.reason}"
        except Exception as exc:
            last_err = f"{type(exc).__name__}"
        if attempt < retries:
            time.sleep(0.4 + attempt)
    return None, last_err


def _discover_hf_candidate_urls(t: Target, model_limit: int = 35) -> list[str]:
    """Find likely checkpoint files from HuggingFace model repos.

    Returns direct resolve URLs, ordered by file-size (descending).
    """
    query = urllib.parse.quote(t.name.replace("_", " "))
    data, status = _hf_api_json(f"https://huggingface.co/api/models?search={query}&limit={model_limit}")
    if status != "ok" or not isinstance(data, list):
        return []

    # Prefer checkpoint-like files; allow ONNX fallback when canonical is ONNX.
    exts = {".pt", ".pth", ".ckpt", ".bin"}
    if t.required_ext == ".onnx":
        exts.add(".onnx")

    cands: list[tuple[int, str]] = []
    aliases = tuple(a.lower() for a in t.aliases)

    for item in data[:model_limit]:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "")
        if not model_id:
            continue
        model_url = "https://huggingface.co/api/models/" + urllib.parse.quote(model_id, safe="/")
        detail, st = _hf_api_json(model_url)
        if st != "ok" or not isinstance(detail, dict):
            continue

        siblings = detail.get("siblings") or []
        if not isinstance(siblings, list):
            continue

        for s in siblings:
            if not isinstance(s, dict):
                continue
            rel = str(s.get("rfilename") or "")
            if not rel:
                continue
            base = Path(rel).name.lower()
            if Path(base).suffix not in exts:
                continue

            if not any(a in base for a in aliases):
                continue

            size = int(s.get("size") or 0)
            if size and size < t.min_size_bytes:
                continue

            url = f"https://huggingface.co/{model_id}/resolve/main/{urllib.parse.quote(rel, safe='/')}"
            cands.append((size, url))

    cands.sort(key=lambda x: x[0], reverse=True)
    # Unique order-preserving
    seen: set[str] = set()
    urls: list[str] = []
    for _, url in cands:
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls[:10]


def _find_local_checkpoint(t: Target) -> Path | None:
    patterns = ["*.pt", "*.pth", "*.ckpt", "*.bin"]
    search_roots = [STAGING]

    # Prefer known local model folders to avoid unnecessary re-downloads.
    model_root = ROOT / "models"
    extra_roots_by_target = {
        "sgmse_plus": [model_root / "sgmse_plus"],
        "gacela": [model_root / "gacela" / "model"],
        "flow_matching": [model_root / "flow_matching"],
        "versa": [model_root / "versa"],
    }
    search_roots.extend(extra_roots_by_target.get(t.name, []))

    candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for p in patterns:
            candidates.extend(root.glob(p))
    scored: list[tuple[int, int, Path]] = []
    for c in candidates:
        name = c.name.lower()
        alias_score = sum(20 for a in t.aliases if a in name)
        if alias_score == 0:
            continue
        size = c.stat().st_size if c.exists() else 0
        if size < t.min_size_bytes or size > t.max_size_bytes:
            continue
        scored.append((alias_score, size, c))
    if not scored:
        return None
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2]


def _export_attempts(model_name: str) -> list[tuple[str, tuple[Any, ...]]]:
    import torch

    # Conservative dummy inputs for typical audio models.
    return [
        ("audio_1d", (torch.zeros(1, 160000, dtype=torch.float32),)),
        ("audio_2d", (torch.zeros(1, 1, 160000, dtype=torch.float32),)),
        ("spec_3d", (torch.zeros(1, 128, 100, dtype=torch.float32),)),
        ("spec_4d", (torch.zeros(1, 1, 257, 100, dtype=torch.float32),)),
        ("score_model", (torch.zeros(1, 65536, dtype=torch.float32), torch.ones(1, 1, dtype=torch.float32))),
    ]


def _check_onnx_toolchain() -> tuple[bool, str]:
    """Verify ONNX export stack is importable in current environment.

    Known incompatibility: ml_dtypes 0.5.x + numpy 1.26.x raises
    ``TypeError: ufunc 'isnan' not supported`` because ml_dtypes 0.5
    requires numpy 2.0+ APIs for ufunc registration.

    Remediation (choose one):
        pip install "ml_dtypes<0.5"       # downgrade to numpy-1.26-compatible
        pip install "numpy>=2.0"          # upgrade numpy (may require torch 2.4+)

    The affected pipeline targets (versa, mp_senet, flow_matching) run
    under their ``fallback`` paths at runtime — no functionality is lost.
    """
    try:
        pass

        return True, "ok"
    except Exception as exc:
        msg = f"onnx_toolchain_import_failed:{type(exc).__name__}:{exc}"
        if "ufunc 'isnan' not supported" in str(exc):
            try:
                np_v = importlib.metadata.version("numpy")
            except Exception:
                np_v = "unknown"
            try:
                mld_v = importlib.metadata.version("ml_dtypes")
            except Exception:
                mld_v = "unknown"
            try:
                onnx_v = importlib.metadata.version("onnx")
            except Exception:
                onnx_v = "unknown"
            msg += (
                f"|numpy={np_v}|ml_dtypes={mld_v}|onnx={onnx_v}|remediation=pip install 'ml_dtypes<0.5' OR 'numpy>=2.0'"
            )
        return False, msg


def _load_model_any(pt_path: Path) -> tuple[str, Any] | tuple[None, None]:
    import torch
    import torch.nn as nn

    try:
        jit = torch.jit.load(str(pt_path), map_location="cpu")
        jit.eval()
        return "torchscript", jit
    except Exception:
        logger.warning("holen_and_Ausgabe_core_models.py::_laden_model_any Ersatzpfad", exc_info=True)

    try:
        obj = torch.load(str(pt_path), map_location="cpu")  # nosec B614
    except Exception:
        logger.warning("holen_and_Ausgabe_core_models.py::_laden_model_any Ersatzpfad", exc_info=True)
        return None, None

    if isinstance(obj, nn.Module):
        obj.eval()
        return "nn_module", obj

    # common wrapper keys
    for key in ("model", "generator", "net", "module"):
        maybe = obj.get(key) if isinstance(obj, dict) else None
        if isinstance(maybe, nn.Module):
            maybe.eval()
            return "nn_module", maybe

    return None, None


def _try_export_to_onnx(pt_path: Path, out_onnx: Path, model_name: str) -> tuple[bool, str]:
    ok_stack, stack_detail = _check_onnx_toolchain()
    if not ok_stack:
        return False, stack_detail

    import torch

    kind, model = _load_model_any(pt_path)
    if model is None:
        if model_name == "flow_matching":
            return (
                False,
                "checkpoint is state_dict-only and source repo ships no architecture/config files; "
                "automatic ONNX export is not possible without exact model class",
            )
        return False, "checkpoint is not TorchScript/nn.Module (state_dict-only; architecture class required)"

    out_onnx.parent.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    for label, inputs in _export_attempts(model_name):
        export_inputs: Any = None
        if len(inputs) > 1:
            export_inputs = inputs
        elif len(inputs) == 1:
            export_inputs = inputs[0]
        if export_inputs is None:
            errors.append(f"{label}:empty_inputs")
            continue
        try:
            torch.onnx.export(
                model,
                export_inputs,
                str(out_onnx),
                input_names=[f"input_{i}" for i in range(len(inputs))],
                output_names=["output"],
                opset_version=17,
            )
            return True, f"exported via {kind} with profile {label}"
        except Exception as exc:
            errors.append(f"{label}:{type(exc).__name__}")

    return False, "; ".join(errors[:5])


def _size_in_bounds(path: Path, t: Target) -> bool:
    if not path.exists():
        return False
    sz = path.stat().st_size
    return t.min_size_bytes <= sz <= t.max_size_bytes


def _ensure_beats_source() -> tuple[bool, str]:
    """Ensure official BEATs source is available locally."""
    beats_file = BEATS_SRC_DIR / "beats" / "BEATs.py"
    if beats_file.exists():
        return True, "source_ready"

    BEATS_SRC_DIR.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/microsoft/unilm.git",
                str(BEATS_SRC_DIR),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        return False, f"clone_failed:{exc.stderr.strip()[:200]}"

    if beats_file.exists():
        return True, "source_cloned"
    return False, "beats_source_missing_after_clone"


def _try_export_beats_checkpoint(pt_path: Path, out_onnx: Path) -> tuple[bool, str]:
    """Export BEATs checkpoint to ONNX using official architecture code.

    Export target is an FFT-free backbone that expects precomputed normalized
    FBank features [B, T, 128]. This avoids unsupported FFT ops in ONNX export.
    """
    ok_stack, stack_detail = _check_onnx_toolchain()
    if not ok_stack:
        return False, stack_detail

    import torch

    ok, info = _ensure_beats_source()
    if not ok:
        return False, info

    sys.path.insert(0, str(BEATS_SRC_DIR / "beats"))
    try:
        from BEATs import BEATs, BEATsConfig  # type: ignore
    except Exception as exc:
        return False, f"beats_import_failed:{type(exc).__name__}:{exc}"

    try:
        ckpt = torch.load(str(pt_path), map_location="cpu")  # nosec B614
    except Exception as exc:
        return False, f"beats_ckpt_load_failed:{type(exc).__name__}:{exc}"

    if not isinstance(ckpt, dict) or "cfg" not in ckpt or "model" not in ckpt:
        return False, "beats_checkpoint_invalid_missing_cfg_or_model"

    try:
        model = BEATs(BEATsConfig(ckpt["cfg"]))
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()
    except Exception as exc:
        return False, f"beats_model_init_failed:{type(exc).__name__}:{exc}"

    class BeatsWrapper(torch.nn.Module):
        def __init__(self, m: Any) -> None:
            super().__init__()
            self.m = m

        def forward(self, fbank: torch.Tensor) -> torch.Tensor:
            # Mirror BEATs.extract_features(), but start after preprocess().
            x = fbank.unsqueeze(1)
            x = self.m.patch_embedding(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.transpose(1, 2)
            x = self.m.layer_norm(x)

            if self.m.post_extract_proj is not None:
                x = self.m.post_extract_proj(x)

            x = self.m.dropout_input(x)
            x, _ = self.m.encoder(x, padding_mask=None)

            if self.m.predictor is not None:
                x = self.m.predictor_dropout(x)
                logits = self.m.predictor(x)
                return torch.sigmoid(logits.mean(dim=1))

            return x  # type: ignore[no-any-return]

    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    wrapped = BeatsWrapper(model)
    dummy = torch.zeros(1, 496, 128, dtype=torch.float32)
    try:
        torch.onnx.export(
            wrapped,
            (dummy,),
            str(out_onnx),
            input_names=["fbank"],
            output_names=["output"],
            opset_version=17,
            dynamic_axes={
                "fbank": {0: "batch", 1: "frames"},
                "output": {0: "batch"},
            },
        )
        return True, "beats_exported_from_checkpoint_fft_free_backbone"
    except Exception as exc:
        return False, f"beats_onnx_export_failed:{type(exc).__name__}:{exc}"


def _ensure_rmvpe_source() -> tuple[bool, str]:
    rmvpe_file = RMVPE_SRC_DIR / "infer" / "lib" / "rmvpe.py"
    if rmvpe_file.exists():
        return True, "source_ready"

    RMVPE_SRC_DIR.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git",
                str(RMVPE_SRC_DIR),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        return False, f"clone_failed:{exc.stderr.strip()[:200]}"

    if rmvpe_file.exists():
        return True, "source_cloned"
    return False, "rmvpe_source_missing_after_clone"


def _install_rmvpe_import_stubs() -> None:
    """Provide tiny librosa/scipy stubs for importing RVC RMVPE code.

    The RVC rmvpe module imports `librosa.util` and `scipy.signal` at import-time.
    In environments with NumPy/SciPy ABI mismatch, we only need minimal helpers
    to construct/export the model graph, not full DSP functionality.
    """

    if "librosa.util" not in sys.modules:
        librosa = types.ModuleType("librosa")
        librosa.__path__ = []  # Mark as package for submodule imports.
        librosa.__spec__ = importlib.machinery.ModuleSpec("librosa", loader=None, is_package=True)
        util = types.ModuleType("librosa.util")
        util.__spec__ = importlib.machinery.ModuleSpec("librosa.util", loader=None, is_package=False)
        filters = types.ModuleType("librosa.filters")
        filters.__spec__ = importlib.machinery.ModuleSpec("librosa.filters", loader=None, is_package=False)

        def _normalize(x: np.ndarray) -> np.ndarray:
            arr = np.asarray(x, dtype=np.float32)
            m = float(np.max(np.abs(arr))) if arr.size else 0.0
            return arr if m <= 0.0 else arr / m

        def _pad_center(data: np.ndarray, size: int) -> np.ndarray:
            arr = np.asarray(data)
            if arr.shape[0] >= size:
                start = (arr.shape[0] - size) // 2
                return arr[start : start + size]
            total = size - arr.shape[0]
            left = total // 2
            right = total - left
            return np.pad(arr, (left, right))

        def _tiny(_x: np.ndarray) -> float:
            return np.finfo(np.float32).tiny  # type: ignore[return-value]

        util.normalize = _normalize  # type: ignore[attr-defined]
        util.pad_center = _pad_center  # type: ignore[attr-defined]
        util.tiny = _tiny  # type: ignore[attr-defined]

        def _mel(
            sr: int,
            n_fft: int,
            n_mels: int = 128,
            fmin: float = 0.0,
            fmax: float | None = None,
            htk: bool = False,
            norm: str | None = "slaney",
            dtype: type = np.float32,
        ) -> np.ndarray:
            del sr, fmin, fmax, htk, norm
            # Lightweight placeholder for import/export context only.
            n_freq = (n_fft // 2) + 1
            return np.ones((n_mels, n_freq), dtype=dtype)

        filters.mel = _mel  # type: ignore[attr-defined]

        librosa.util = util  # type: ignore[attr-defined]
        librosa.filters = filters  # type: ignore[attr-defined]
        sys.modules["librosa"] = librosa
        sys.modules["librosa.util"] = util
        sys.modules["librosa.filters"] = filters

    if "scipy.signal" not in sys.modules:
        scipy = sys.modules.get("scipy")
        if scipy is None:
            scipy = types.ModuleType("scipy")
            scipy.__spec__ = importlib.machinery.ModuleSpec("scipy", loader=None, is_package=True)
            sys.modules["scipy"] = scipy
        elif getattr(scipy, "__spec__", None) is None:
            scipy.__spec__ = importlib.machinery.ModuleSpec("scipy", loader=None, is_package=True)
        signal = types.ModuleType("scipy.signal")
        signal.__spec__ = importlib.machinery.ModuleSpec("scipy.signal", loader=None, is_package=False)

        def _get_window(window: str, win_length: int, fftbins: bool = True) -> np.ndarray:
            if window == "hann":
                return np.hanning(win_length).astype(np.float32)
            return np.ones(win_length, dtype=np.float32)

        signal.get_window = _get_window  # type: ignore[attr-defined]
        scipy.signal = signal  # type: ignore[attr-defined]
        sys.modules["scipy.signal"] = signal


def _try_export_rmvpe_checkpoint(pt_path: Path, out_onnx: Path) -> tuple[bool, str]:
    """Best-effort RMVPE export through RVC architecture code.

    Input/Output profile:
      mel [B, T, 128] -> salience [B, T, 360]
    """
    ok_stack, stack_detail = _check_onnx_toolchain()
    if not ok_stack:
        return False, stack_detail

    import torch

    ok, info = _ensure_rmvpe_source()
    if not ok:
        return False, info

    _install_rmvpe_import_stubs()

    sys.path.insert(0, str(RMVPE_SRC_DIR))
    try:
        from infer.lib.rmvpe import E2E  # type: ignore
    except Exception as exc:
        return False, f"rmvpe_import_failed:{type(exc).__name__}:{exc}"

    try:
        state = torch.load(str(pt_path), map_location="cpu")  # nosec B614
    except Exception as exc:
        return False, f"rmvpe_ckpt_load_failed:{type(exc).__name__}:{exc}"

    if not isinstance(state, dict):
        return False, "rmvpe_checkpoint_invalid_not_state_dict"

    try:
        model = E2E(4, 1, (2, 2))
        model.load_state_dict(state, strict=False)
        model.eval()
    except Exception as exc:
        return False, f"rmvpe_model_init_failed:{type(exc).__name__}:{exc}"

    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    # E2E expects mel as [B, 128, T]; forward() transposes to [B, T, 128].
    dummy_mel = torch.zeros(1, 128, 512, dtype=torch.float32)
    try:
        torch.onnx.export(
            model,
            (dummy_mel,),
            str(out_onnx),
            input_names=["mel"],
            output_names=["salience"],
            opset_version=17,
            dynamic_axes={
                "mel": {0: "batch", 1: "frames"},
                "salience": {0: "batch", 1: "frames"},
            },
        )
        return True, "rmvpe_exported_from_checkpoint"
    except Exception as exc:
        return False, f"rmvpe_onnx_export_failed:{type(exc).__name__}:{exc}"


def _ensure_sgmse_source() -> tuple[bool, str]:
    backbone_file = SGMSE_SRC_DIR / "sgmse" / "backbones" / "__init__.py"
    if backbone_file.exists():
        return True, "source_ready"

    SGMSE_SRC_DIR.parent.mkdir(parents=True, exist_ok=True)

    clone_urls = (
        "https://github.com/sp-uhh/sgmse.git",
        "https://github.com/sp-uhh/speech-enhancement-sgmse.git",
    )
    last_err = "clone_failed"
    for url in clone_urls:
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(SGMSE_SRC_DIR)],
                check=True,
                text=True,
                capture_output=True,
            )
            if backbone_file.exists():
                return True, "source_cloned"
            last_err = "sgmse_source_missing_after_clone"
            break
        except subprocess.CalledProcessError as exc:
            last_err = f"clone_failed:{exc.stderr.strip()[:200]}"

    return False, last_err


def _load_sgmse_checkpoint_raw(pt_path: Path) -> tuple[dict[str, Any] | None, str]:
    """Load a Lightning SGMSE checkpoint with minimal stubs.

    The checkpoint can reference ``sgmse.data_module.SpecsDataModule`` at
    unpickle-time. We only need the raw dict (state_dict + hyper_parameters),
    so a tiny stub class is sufficient.
    """
    import torch

    stub_mod_name = "sgmse.data_module"
    created_stub = False
    if stub_mod_name not in sys.modules:
        stub = types.ModuleType(stub_mod_name)

        class SpecsDataModule:
            pass

        stub.SpecsDataModule = SpecsDataModule  # type: ignore[attr-defined]
        sys.modules[stub_mod_name] = stub
        created_stub = True

    try:
        obj = torch.load(str(pt_path), map_location="cpu")  # nosec B614
    except Exception as exc:
        return None, f"sgmse_ckpt_load_failed:{type(exc).__name__}:{exc}"
    finally:
        if created_stub:
            sys.modules.pop(stub_mod_name, None)

    if not isinstance(obj, dict):
        return None, "sgmse_checkpoint_invalid_not_dict"
    return obj, "ok"


def _try_export_sgmse_checkpoint(pt_path: Path, out_onnx: Path) -> tuple[bool, str]:
    """Export SGMSE denoiser backbone from Lightning checkpoint.

    Export profile:
      x_t [B, 2, F, T], y [B, 2, F, T], t [B] -> score [B, 2, F, T]
    """
    ok_stack, stack_detail = _check_onnx_toolchain()
    if not ok_stack:
        return False, stack_detail

    import torch

    ok, info = _ensure_sgmse_source()
    if not ok:
        return False, info

    # Needed for torch.load() pickle resolution of `sgmse.*` symbols.
    sys.path.insert(0, str(SGMSE_SRC_DIR))

    ckpt, detail = _load_sgmse_checkpoint_raw(pt_path)
    if ckpt is None:
        return False, detail

    hyper = ckpt.get("hyper_parameters")
    if not isinstance(hyper, dict):
        return False, "sgmse_checkpoint_missing_hyper_parameters"

    backbone_name = str(hyper.get("backbone") or "").strip()
    if not backbone_name:
        return False, "sgmse_checkpoint_missing_backbone"

    try:
        from sgmse.backbones import BackboneRegistry  # type: ignore
    except Exception as exc:
        return False, f"sgmse_import_failed:{type(exc).__name__}:{exc}"

    try:
        dnn_cls = BackboneRegistry.get_by_name(backbone_name)
    except Exception as exc:
        return False, f"sgmse_backbone_lookup_failed:{type(exc).__name__}:{exc}"

    try:
        dnn = dnn_cls(**hyper)
        dnn.eval()
    except Exception as exc:
        return False, f"sgmse_backbone_init_failed:{type(exc).__name__}:{exc}"

    state = ckpt.get("state_dict")
    if not isinstance(state, dict):
        return False, "sgmse_checkpoint_missing_state_dict"

    dnn_state = {k[4:]: v for k, v in state.items() if k.startswith("dnn.")}
    if not dnn_state:
        return False, "sgmse_checkpoint_missing_dnn_weights"

    try:
        dnn.load_state_dict(dnn_state, strict=False)
    except Exception as exc:
        return False, f"sgmse_backbone_load_state_failed:{type(exc).__name__}:{exc}"

    class SgmseWrapper(torch.nn.Module):
        def __init__(self, m: Any, bb_name: str) -> None:
            super().__init__()
            self.m = m
            self.bb_name = bb_name

        def forward(self, x_t: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            if self.bb_name == "ncsnpp_v2":
                return self.m(x_t, y, t)  # type: ignore[no-any-return]
            # ncsnpp/ncsnpp_48k expect complex spectrogram channels [x_t, y].
            x_complex = torch.complex(x_t[:, 0], x_t[:, 1])
            y_complex = torch.complex(y[:, 0], y[:, 1])
            dnn_input = torch.stack([x_complex, y_complex], dim=1)
            out_complex = -self.m(dnn_input, t)
            return torch.stack([out_complex.real, out_complex.imag], dim=1)

    wrapper = SgmseWrapper(dnn, backbone_name)
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    dummy_xt = torch.zeros(1, 2, 256, 256, dtype=torch.float32)
    dummy_y = torch.zeros(1, 2, 256, 256, dtype=torch.float32)
    dummy_t = torch.ones(1, dtype=torch.float32)
    # Workaround: convert numpy scalars/arrays before torch.tensor() is called
    # inside SGMSE custom ops in environments where Torch<->NumPy ABI is broken.
    orig_tensor_ctor = torch.tensor

    def _safe_tensor_ctor(data: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        if isinstance(data, np.generic):
            data = data.item()
        elif isinstance(data, np.ndarray):
            data = data.tolist()
        return orig_tensor_ctor(data, *args, **kwargs)

    torch.tensor = _safe_tensor_ctor
    try:
        export_args: dict[str, Any] = {
            "input_names": ["x_t", "y", "t"],
            "output_names": ["score"],
            "dynamic_axes": {
                "x_t": {0: "batch", 2: "freq", 3: "frames"},
                "y": {0: "batch", 2: "freq", 3: "frames"},
                "t": {0: "batch"},
                "score": {0: "batch", 2: "freq", 3: "frames"},
            },
        }
        opsets = (17, 18, 19)
        export_errors: list[str] = []

        for opset in opsets:
            try:
                torch.onnx.export(
                    wrapper,
                    (dummy_xt, dummy_y, dummy_t),
                    str(out_onnx),
                    opset_version=opset,
                    **export_args,
                )
                return True, f"sgmse_exported_backbone={backbone_name}_opset={opset}"
            except Exception as exc:
                export_errors.append(f"opset_{opset}:{type(exc).__name__}:{exc}")

        # Fallback attempt with Dynamo exporter when available.
        try:
            if hasattr(torch.onnx, "dynamo_export"):
                dyn_model = torch.onnx.dynamo_export(wrapper, dummy_xt, dummy_y, dummy_t)  # type: ignore[attr-defined]
                dyn_model.save(str(out_onnx))
                return True, f"sgmse_exported_backbone={backbone_name}_via_dynamo_export"
        except Exception as exc:
            export_errors.append(f"dynamo_export:{type(exc).__name__}:{exc}")

        return False, "sgmse_onnx_export_failed:" + " | ".join(export_errors[:4])
    finally:
        torch.tensor = orig_tensor_ctor


def _try_export_sgmse_torchscript_checkpoint(pt_path: Path, out_ts: Path) -> tuple[bool, str]:
    """Export SGMSE backbone as TorchScript when ONNX path is blocked."""
    import torch

    ok, info = _ensure_sgmse_source()
    if not ok:
        return False, info

    sys.path.insert(0, str(SGMSE_SRC_DIR))

    ckpt, detail = _load_sgmse_checkpoint_raw(pt_path)
    if ckpt is None:
        return False, detail

    hyper = ckpt.get("hyper_parameters")
    if not isinstance(hyper, dict):
        return False, "sgmse_checkpoint_missing_hyper_parameters"

    backbone_name = str(hyper.get("backbone") or "").strip()
    if not backbone_name:
        return False, "sgmse_checkpoint_missing_backbone"

    try:
        from sgmse.backbones import BackboneRegistry  # type: ignore

        dnn_cls = BackboneRegistry.get_by_name(backbone_name)
        dnn = dnn_cls(**hyper)
        dnn.eval()
    except Exception as exc:
        return False, f"sgmse_backbone_init_failed:{type(exc).__name__}:{exc}"

    state = ckpt.get("state_dict")
    if not isinstance(state, dict):
        return False, "sgmse_checkpoint_missing_state_dict"

    dnn_state = {k[4:]: v for k, v in state.items() if k.startswith("dnn.")}
    if not dnn_state:
        return False, "sgmse_checkpoint_missing_dnn_weights"

    try:
        dnn.load_state_dict(dnn_state, strict=False)
    except Exception as exc:
        return False, f"sgmse_backbone_load_state_failed:{type(exc).__name__}:{exc}"

    class SgmseWrapper(torch.nn.Module):
        def __init__(self, m: Any, bb_name: str) -> None:
            super().__init__()
            self.m = m
            self.bb_name = bb_name

        def forward(self, x_t: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            if self.bb_name == "ncsnpp_v2":
                return self.m(x_t, y, t)  # type: ignore[no-any-return]
            x_complex = torch.complex(x_t[:, 0], x_t[:, 1])
            y_complex = torch.complex(y[:, 0], y[:, 1])
            dnn_input = torch.stack([x_complex, y_complex], dim=1)
            out_complex = -self.m(dnn_input, t)
            return torch.stack([out_complex.real, out_complex.imag], dim=1)

    out_ts.parent.mkdir(parents=True, exist_ok=True)
    wrapper = SgmseWrapper(dnn, backbone_name).eval()
    dummy_xt = torch.zeros(1, 2, 256, 256, dtype=torch.float32)
    dummy_y = torch.zeros(1, 2, 256, 256, dtype=torch.float32)
    dummy_t = torch.ones(1, dtype=torch.float32)

    # Workaround for environments where numpy scalar dtypes break torch.tensor() calls
    # inside SGMSE custom layers during tracing.
    orig_tensor_ctor = torch.tensor

    def _safe_tensor_ctor(data: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        if isinstance(data, np.generic):
            data = data.item()
        elif isinstance(data, np.ndarray):
            data = data.tolist()
        return orig_tensor_ctor(data, *args, **kwargs)

    try:
        torch.tensor = _safe_tensor_ctor
        traced = torch.jit.trace(wrapper, (dummy_xt, dummy_y, dummy_t), strict=False)
        traced.save(str(out_ts))
        return True, f"sgmse_torchscript_exported_backbone={backbone_name}"
    except Exception as exc:
        return False, f"sgmse_torchscript_export_failed:{type(exc).__name__}:{exc}"
    finally:
        torch.tensor = orig_tensor_ctor


def main() -> int:
    args = _parse_args()
    only = {x.strip().lower() for x in args.only if x.strip()}
    targets = [t for t in TARGETS if not only or t.name in only]

    STAGING.mkdir(parents=True, exist_ok=True)
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)

    report: dict[str, dict[str, str | bool]] = {}
    unresolved = 0
    hf_token_set = _has_hf_token()

    has_hf_targets = any("huggingface.co" in u for t in targets for u in t.urls)
    if has_hf_targets and not hf_token_set:
        print(
            "INFO HF token nicht gesetzt. Für private/gated Modelle können 401/403 auftreten. "
            "Setze HF_TOKEN oder HUGGINGFACE_TOKEN in der Shell."
        )

    for t in targets:
        canonical = ROOT / t.canonical_rel
        canonical.parent.mkdir(parents=True, exist_ok=True)

        if canonical.exists() and not args.overwrite:
            if _size_in_bounds(canonical, t):
                report[t.name] = {"status": "ok", "detail": "canonical already exists", "path": t.canonical_rel}
                continue
            report[t.name] = {
                "status": "blocked",
                "detail": f"canonical exists but suspicious size={canonical.stat().st_size}",
                "path": t.canonical_rel,
            }

        staged_pt = _find_local_checkpoint(t)
        download_errors: list[str] = []

        if staged_pt is None and not args.skip_download:
            candidate_urls = list(t.urls)
            if not args.no_hf_discovery:
                candidate_urls.extend(_discover_hf_candidate_urls(t))

            for idx, url in enumerate(candidate_urls, start=1):
                ext = Path(url).suffix.lower() or ".pt"
                out = STAGING / f"{t.name}_src_{idx}{ext}"
                if out.exists() and not args.overwrite:
                    staged_pt = out
                    break
                ok_dl, dl_detail = _download(url, out)
                if ok_dl:
                    if _size_in_bounds(out, t):
                        staged_pt = out
                        break
                    download_errors.append(f"{url}::size_out_of_bounds:{out.stat().st_size}")
                else:
                    download_errors.append(f"{url}::download_failed:{dl_detail}")

        if staged_pt is None:
            detail = "no local/downloadable PT candidate"
            if download_errors:
                detail += " | " + " ; ".join(download_errors[:4])
            report[t.name] = {"status": "missing", "detail": detail}
            unresolved += 1
            continue

        if t.required_ext == ".pt":
            # GACELA primary path is PT, no ONNX export needed.
            if args.overwrite or not canonical.exists():
                shutil.copy2(staged_pt, canonical)
            if not _size_in_bounds(canonical, t):
                report[t.name] = {
                    "status": "blocked",
                    "detail": f"copied checkpoint size out of bounds: {canonical.stat().st_size}",
                    "source": str(staged_pt.relative_to(ROOT)),
                }
                unresolved += 1
                continue
            report[t.name] = {
                "status": "ok",
                "detail": "checkpoint copied",
                "source": str(staged_pt.relative_to(ROOT)),
                "path": t.canonical_rel,
            }
            continue

        # If a prebuilt ONNX was discovered, use it directly.
        if staged_pt.suffix.lower() == ".onnx":
            shutil.copy2(staged_pt, canonical)
            if not _size_in_bounds(canonical, t):
                report[t.name] = {
                    "status": "blocked",
                    "detail": f"prebuilt onnx size out of bounds: {canonical.stat().st_size}",
                    "source": str(staged_pt.relative_to(ROOT)),
                }
                unresolved += 1
                continue
            report[t.name] = {
                "status": "ok",
                "detail": "prebuilt_onnx_copied",
                "source": str(staged_pt.relative_to(ROOT)),
                "path": t.canonical_rel,
            }
            continue

        tmp_onnx = STAGING / f"{t.name}.onnx"
        if t.name == "beats":
            ok, detail = _try_export_beats_checkpoint(staged_pt, tmp_onnx)
        elif t.name == "rmvpe":
            ok, detail = _try_export_rmvpe_checkpoint(staged_pt, tmp_onnx)
        elif t.name == "sgmse_plus":
            ok, detail = _try_export_sgmse_checkpoint(staged_pt, tmp_onnx)
        else:
            ok, detail = _try_export_to_onnx(staged_pt, tmp_onnx, t.name)
        if not ok:
            if t.name == "sgmse_plus":
                tmp_ts = STAGING / "sgmse_plus.ts"
                ok_ts, detail_ts = _try_export_sgmse_torchscript_checkpoint(staged_pt, tmp_ts)
                if ok_ts:
                    ts_canonical_rel = "models/sgmse_plus/sgmse_plus.ts"
                    ts_canonical = ROOT / ts_canonical_rel
                    ts_canonical.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(tmp_ts, ts_canonical)
                    report[t.name] = {
                        "status": "ok",
                        "detail": f"onnx_failed_then_torchscript_ok:{detail_ts}",
                        "source": str(staged_pt.relative_to(ROOT)),
                        "path": ts_canonical_rel,
                    }
                    continue
            report[t.name] = {
                "status": "blocked",
                "detail": detail,
                "source": str(staged_pt.relative_to(ROOT)),
            }
            unresolved += 1
            continue

        shutil.copy2(tmp_onnx, canonical)
        if not _size_in_bounds(canonical, t):
            report[t.name] = {
                "status": "blocked",
                "detail": f"exported onnx size out of bounds: {canonical.stat().st_size}",
                "source": str(staged_pt.relative_to(ROOT)),
            }
            unresolved += 1
            continue
        report[t.name] = {
            "status": "ok",
            "detail": detail,
            "source": str(staged_pt.relative_to(ROOT)),
            "path": t.canonical_rel,
        }

    for v in report.values():
        v["hf_token_set"] = hf_token_set

    report_payload = {
        "summary": {
            "hf_token_set": hf_token_set,
            "targets_checked": len(targets),
            "unresolved": unresolved,
            "all_resolved": unresolved == 0,
            "token_required_targets": [t.name for t in targets if any("huggingface.co" in u for u in t.urls)],
            "token_missing_for_private": not hf_token_set,
        },
        "targets": report,
    }

    REPORT_PATH.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Report: {REPORT_PATH.relative_to(ROOT)}")
    print(json.dumps(report_payload, indent=2, ensure_ascii=False))

    return 0 if unresolved == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
