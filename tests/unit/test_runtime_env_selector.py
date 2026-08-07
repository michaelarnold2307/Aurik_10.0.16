from __future__ import annotations

from pathlib import Path

import pytest

from backend.core.runtime_env_selector import RuntimeProbe, select_runtime_python


def _probe(path: Path, *, has_rocm: bool) -> RuntimeProbe:
    return RuntimeProbe(
        python_path=path,
        torch_import=True,
        torch_cuda_available=has_rocm,
        torch_hip="6.2.0" if has_rocm else None,
        torch_cuda_version=None,
        onnxruntime_import=True,
        ort_providers=("ROCMExecutionProvider", "CPUExecutionProvider") if has_rocm else ("CPUExecutionProvider",),
    )


@pytest.mark.unit
def test_select_runtime_python_prefers_rocm(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path
    rocm_python = repo_root / ".venv_gpu" / "bin" / "python"
    aurik_python = repo_root / ".venv_aurik" / "bin" / "python"
    rocm_python.parent.mkdir(parents=True)
    aurik_python.parent.mkdir(parents=True)
    rocm_python.write_text("", encoding="utf-8")
    aurik_python.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "backend.core.runtime_env_selector.probe_python_runtime",
        lambda path: _probe(path, has_rocm=path == rocm_python),
    )

    selected = select_runtime_python(repo_root)
    assert selected == rocm_python


def test_select_runtime_python_falls_back_to_cpu_venv(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path
    rocm_python = repo_root / ".venv_gpu" / "bin" / "python"
    aurik_python = repo_root / ".venv_aurik" / "bin" / "python"
    rocm_python.parent.mkdir(parents=True)
    aurik_python.parent.mkdir(parents=True)
    rocm_python.write_text("", encoding="utf-8")
    aurik_python.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "backend.core.runtime_env_selector.probe_python_runtime",
        lambda path: _probe(path, has_rocm=False),
    )

    selected = select_runtime_python(repo_root)
    assert selected == aurik_python


def test_select_runtime_python_uses_current_interpreter_if_no_venv(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("backend.core.runtime_env_selector.probe_python_runtime", lambda path: None)

    selected = select_runtime_python(tmp_path)
    assert selected.exists()


def test_select_runtime_python_prefers_rocm_windows_venv(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path
    rocm_python = repo_root / ".venv_gpu" / "Scripts" / "python.exe"
    aurik_python = repo_root / ".venv_aurik" / "Scripts" / "python.exe"
    rocm_python.parent.mkdir(parents=True)
    aurik_python.parent.mkdir(parents=True)
    rocm_python.write_text("", encoding="utf-8")
    aurik_python.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "backend.core.runtime_env_selector.probe_python_runtime",
        lambda path: _probe(path, has_rocm=path == rocm_python) if path.exists() else None,
    )

    selected = select_runtime_python(repo_root)
    assert selected == rocm_python


def test_select_runtime_python_falls_back_to_aurik_windows_venv(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path
    rocm_python = repo_root / ".venv_gpu" / "Scripts" / "python.exe"
    aurik_python = repo_root / ".venv_aurik" / "Scripts" / "python.exe"
    rocm_python.parent.mkdir(parents=True)
    aurik_python.parent.mkdir(parents=True)
    rocm_python.write_text("", encoding="utf-8")
    aurik_python.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "backend.core.runtime_env_selector.probe_python_runtime",
        lambda path: _probe(path, has_rocm=False) if path.exists() else None,
    )

    selected = select_runtime_python(repo_root)
    assert selected == aurik_python
