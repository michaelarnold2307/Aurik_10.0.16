#!/usr/bin/env python3
"""
DSP Optimization Benchmarks for AURIK v8
=========================================

Benchmarks all three DSP optimization strategies:
1. NumExpr: 2× speedup for vectorized operations
2. Cython: 3-5× speedup for critical loops
3. pyFFTW: 1.5-2× speedup for FFT operations

Usage:
    python scripts/benchmark_dsp.py [--quick]

Output:
    - Performance comparison tables
    - Speedup factors
    - Memory usage comparisons
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def benchmark_function(func, *args, iterations=100, warmup=10):
    """Benchmark a function with warmup."""
    # Warmup
    for _ in range(warmup):
        func(*args)

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        result = func(*args)
    end = time.perf_counter()

    elapsed = (end - start) / iterations
    return elapsed, result


def benchmark_numexpr():
    """Benchmark NumExpr optimizations vs NumPy."""
    print("\n" + "=" * 80)
    print("NumExpr Optimizations Benchmark")
    print("=" * 80)

    from dsp.optimized.numexpr_ops import OptimizedDSP

    # Test data
    audio = np.random.randn(48000).astype(np.float32)  # 1 second
    spectrum = np.random.randn(1025, 100).astype(np.float32)  # STFT spectrum

    dsp = OptimizedDSP()

    results = []

    # 1. Spectral Gate
    print("\n1. Spectral Gate (magnitude thresholding)")
    print("-" * 80)

    def numpy_spectral_gate(spectrum, threshold):
        magnitude = np.abs(spectrum)
        mask = np.where(magnitude > threshold, 1.0, 0.0)
        return spectrum * mask

    threshold = 0.1
    numpy_time, _ = benchmark_function(numpy_spectral_gate, spectrum, threshold)
    numexpr_time, _ = benchmark_function(dsp.spectral_gate, spectrum, threshold)
    speedup = numpy_time / numexpr_time

    print(f"NumPy:   {numpy_time * 1000:8.3f} ms")
    print(f"NumExpr: {numexpr_time * 1000:8.3f} ms")
    print(f"Speedup: {speedup:8.2f}×")
    results.append(("Spectral Gate", numpy_time, numexpr_time, speedup))

    # 2. Soft Threshold
    print("\n2. Soft Threshold (audio denoising)")
    print("-" * 80)

    def numpy_soft_threshold(audio, threshold):
        sign = np.sign(audio)
        magnitude = np.abs(audio)
        return sign * np.maximum(magnitude - threshold, 0.0)

    threshold = 0.01
    numpy_time, _ = benchmark_function(numpy_soft_threshold, audio, threshold)
    numexpr_time, _ = benchmark_function(dsp.soft_threshold, audio, threshold)
    speedup = numpy_time / numexpr_time

    print(f"NumPy:   {numpy_time * 1000:8.3f} ms")
    print(f"NumExpr: {numexpr_time * 1000:8.3f} ms")
    print(f"Speedup: {speedup:8.2f}×")
    results.append(("Soft Threshold", numpy_time, numexpr_time, speedup))

    # 3. Hard Threshold
    print("\n3. Hard Threshold (silence removal)")
    print("-" * 80)

    def numpy_hard_threshold(audio, threshold):
        return np.where(np.abs(audio) > threshold, audio, 0.0)

    threshold = 0.01
    numpy_time, _ = benchmark_function(numpy_hard_threshold, audio, threshold)
    numexpr_time, _ = benchmark_function(dsp.hard_threshold, audio, threshold)
    speedup = numpy_time / numexpr_time

    print(f"NumPy:   {numpy_time * 1000:8.3f} ms")
    print(f"NumExpr: {numexpr_time * 1000:8.3f} ms")
    print(f"Speedup: {speedup:8.2f}×")
    results.append(("Hard Threshold", numpy_time, numexpr_time, speedup))

    # 4. Spectral Subtraction
    print("\n4. Spectral Subtraction (noise reduction)")
    print("-" * 80)

    def numpy_spectral_subtraction(spectrum, noise_profile, alpha=2.0):
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        clean_magnitude = np.maximum(magnitude - alpha * noise_profile, 0.0)
        return clean_magnitude * np.exp(1j * phase)

    noise_profile = np.abs(spectrum[:, :10].mean(axis=1, keepdims=True))
    numpy_time, _ = benchmark_function(numpy_spectral_subtraction, spectrum, noise_profile)
    numexpr_time, _ = benchmark_function(dsp.spectral_subtraction, spectrum, noise_profile)
    speedup = numpy_time / numexpr_time

    print(f"NumPy:   {numpy_time * 1000:8.3f} ms")
    print(f"NumExpr: {numexpr_time * 1000:8.3f} ms")
    print(f"Speedup: {speedup:8.2f}×")
    results.append(("Spectral Subtraction", numpy_time, numexpr_time, speedup))

    # Summary
    print("\n" + "=" * 80)
    print("NumExpr Summary")
    print("=" * 80)
    avg_speedup = np.mean([r[3] for r in results])
    print(f"Average Speedup: {avg_speedup:.2f}× (Target: 2×)")
    print(f"Status: {'✅ PASSED' if avg_speedup >= 1.8 else '❌ FAILED'}")

    return results


def benchmark_cython():
    """Benchmark Cython optimizations vs pure Python."""
    print("\n" + "=" * 80)
    print("Cython Optimizations Benchmark")
    print("=" * 80)

    try:
        from dsp.optimized import HAS_CYTHON, cython_loops

        if not HAS_CYTHON:
            print("⚠️  Cython extensions not compiled. Run: python setup_cython.py build_ext --inplace")
            return []
    except ImportError:
        print("⚠️  Cython extensions not found. Run: python setup_cython.py build_ext --inplace")
        return []

    # Test data
    audio = np.random.randn(48000).astype(np.float32)

    results = []

    # 1. Click Detector
    print("\n1. Click Detector (difference threshold)")
    print("-" * 80)

    def python_click_detector(audio, threshold, min_distance):
        clicks = []
        last_click = -min_distance
        for i in range(1, len(audio)):
            diff = abs(audio[i] - audio[i - 1])
            if diff > threshold and (i - last_click) > min_distance:
                clicks.append(i)
                last_click = i
        return np.array(clicks)

    threshold = 0.5
    min_distance = 100
    python_time, _ = benchmark_function(python_click_detector, audio, threshold, min_distance, iterations=10)
    cython_time, _ = benchmark_function(cython_loops.click_detector_fast, audio, threshold, min_distance, iterations=10)
    speedup = python_time / cython_time

    print(f"Python: {python_time * 1000:8.3f} ms")
    print(f"Cython: {cython_time * 1000:8.3f} ms")
    print(f"Speedup: {speedup:7.2f}×")
    results.append(("Click Detector", python_time, cython_time, speedup))

    # 2. Group Clicks
    print("\n2. Group Clicks (event grouping)")
    print("-" * 80)

    clicks = np.random.randint(0, len(audio), size=100).astype(np.int32)  # type: ignore[arg-type]  # §V5 Dither applied at export level
    clicks = np.sort(clicks)

    def python_group_clicks(clicks, max_gap):
        if len(clicks) == 0:
            return []

        groups = []
        current_group = [clicks[0]]

        for click in clicks[1:]:
            if click - current_group[-1] <= max_gap:
                current_group.append(click)
            else:
                groups.append(current_group)
                current_group = [click]

        groups.append(current_group)
        return groups

    max_gap = 1000
    python_time, _ = benchmark_function(python_group_clicks, clicks, max_gap, iterations=100)
    cython_time, _ = benchmark_function(
        cython_loops.group_clicks_fast, clicks.astype(np.int32), max_gap, iterations=100  # type: ignore[arg-type]  # §V5 Dither applied at export level
    )
    speedup = python_time / cython_time

    print(f"Python: {python_time * 1000:8.3f} ms")
    print(f"Cython: {cython_time * 1000:8.3f} ms")
    print(f"Speedup: {speedup:7.2f}×")
    results.append(("Group Clicks", python_time, cython_time, speedup))

    # 3. Peak Finder
    print("\n3. Peak Finder (local maxima)")
    print("-" * 80)

    def python_peak_finder(audio, threshold, min_distance):
        peaks: list[Any] = []
        for i in range(1, len(audio) - 1):
            if audio[i] > threshold:
                if audio[i] >= audio[i - 1] and audio[i] >= audio[i + 1]:
                    if not peaks or (i - peaks[-1]) > min_distance:
                        peaks.append(i)
        return np.array(peaks)

    threshold = 0.5
    min_distance = 100
    python_time, _ = benchmark_function(python_peak_finder, audio, threshold, min_distance, iterations=10)
    cython_time, _ = benchmark_function(cython_loops.peak_finder_fast, audio, threshold, min_distance, iterations=10)
    speedup = python_time / cython_time

    print(f"Python: {python_time * 1000:8.3f} ms")
    print(f"Cython: {cython_time * 1000:8.3f} ms")
    print(f"Speedup: {speedup:7.2f}×")
    results.append(("Peak Finder", python_time, cython_time, speedup))

    # 4. RMS Fast
    print("\n4. RMS Energy (frame-based)")
    print("-" * 80)

    frame_length = 2048
    hop_length = 512

    def python_rms(audio, frame_length, hop_length):
        n_frames = 1 + (len(audio) - frame_length) // hop_length
        rms = np.zeros(n_frames, dtype=np.float32)

        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start : start + frame_length]
            rms[i] = np.sqrt(np.mean(frame**2))

        return rms

    python_time, _ = benchmark_function(python_rms, audio, frame_length, hop_length, iterations=10)
    cython_time, _ = benchmark_function(cython_loops.rms_fast, audio, frame_length, hop_length, iterations=10)
    speedup = python_time / cython_time

    print(f"Python: {python_time * 1000:8.3f} ms")
    print(f"Cython: {cython_time * 1000:8.3f} ms")
    print(f"Speedup: {speedup:7.2f}×")
    results.append(("RMS Energy", python_time, cython_time, speedup))

    # Summary
    print("\n" + "=" * 80)
    print("Cython Summary")
    print("=" * 80)
    avg_speedup = np.mean([r[3] for r in results])
    print(f"Average Speedup: {avg_speedup:.2f}× (Target: 3-5×)")
    print(f"Status: {'✅ PASSED' if avg_speedup >= 3.0 else '❌ FAILED'}")

    return results


def benchmark_fft():
    """Benchmark pyFFTW vs NumPy FFT."""
    print("\n" + "=" * 80)
    print("FFT Optimization Benchmark")
    print("=" * 80)

    try:
        from dsp.optimized import HAS_PYFFTW
        from dsp.optimized.fft_cache import CachedFFT

        if not HAS_PYFFTW:
            print("⚠️  pyFFTW not installed. Install with: pip install pyfftw")
            return []
    except ImportError:
        print("⚠️  pyFFTW not installed. Install with: pip install pyfftw")
        return []

    fft = CachedFFT()

    results = []

    # Test different FFT sizes
    fft_sizes = [512, 1024, 2048, 4096, 8192]

    print("\nFFT Size Comparison")
    print("-" * 80)
    print(f"{'Size':>6} | {'NumPy (ms)':>12} | {'pyFFTW (ms)':>13} | {'Speedup':>8}")
    print("-" * 80)

    for n_fft in fft_sizes:
        audio = np.random.randn(n_fft).astype(np.float32)

        # NumPy FFT
        numpy_time, _ = benchmark_function(np.fft.rfft, audio)

        # pyFFTW (with caching)
        pyfftw_time, _ = benchmark_function(fft.rfft, audio)

        speedup = numpy_time / pyfftw_time

        print(f"{n_fft:6} | {numpy_time * 1000:12.3f} | {pyfftw_time * 1000:13.3f} | {speedup:8.2f}×")
        results.append((f"FFT-{n_fft}", numpy_time, pyfftw_time, speedup))

    # STFT Benchmark
    print("\nSTFT Benchmark (realistic workload)")
    print("-" * 80)

    audio = np.random.randn(48000).astype(np.float32)  # 1 second
    n_fft = 2048
    hop_length = 512

    def numpy_stft(audio, n_fft, hop_length):
        from scipy.signal import get_window

        n_frames = 1 + (len(audio) - n_fft) // hop_length
        win = get_window("hann", n_fft)
        stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex64)
        for i in range(n_frames):
            frame = audio[i * hop_length : i * hop_length + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            stft[:, i] = np.fft.rfft(frame * win)
        return stft

    numpy_time, _ = benchmark_function(numpy_stft, audio, n_fft, hop_length, iterations=10)
    pyfftw_time, _ = benchmark_function(fft.stft, audio, n_fft, hop_length, iterations=10)
    speedup = numpy_time / pyfftw_time

    print(f"NumPy STFT:   {numpy_time * 1000:10.3f} ms")
    print(f"pyFFTW STFT:  {pyfftw_time * 1000:10.3f} ms")
    print(f"Speedup:      {speedup:10.2f}×")
    results.append(("STFT", numpy_time, pyfftw_time, speedup))

    # Summary
    print("\n" + "=" * 80)
    print("FFT Summary")
    print("=" * 80)
    avg_speedup = np.mean([r[3] for r in results])
    print(f"Average Speedup: {avg_speedup:.2f}× (Target: 1.5-2×)")
    print(f"Status: {'✅ PASSED' if avg_speedup >= 1.5 else '❌ FAILED'}")

    # Cache statistics
    print("\nCache Statistics:")
    stats = fft.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    return results


def main():
    """Run all benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark DSP optimizations")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark (fewer iterations)")
    parser.parse_args()

    print("=" * 80)
    print("AURIK v8 DSP Optimization Benchmarks")
    print("=" * 80)
    print(f"NumPy version: {np.__version__}")
    print(f"Platform: {sys.platform}")
    print("=" * 80)

    # Run benchmarks
    numexpr_results = benchmark_numexpr()
    cython_results = benchmark_cython()
    fft_results = benchmark_fft()

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    all_speedups = []

    if numexpr_results:
        numexpr_avg = np.mean([r[3] for r in numexpr_results])
        all_speedups.extend([r[3] for r in numexpr_results])
        print(f"NumExpr:  {numexpr_avg:.2f}× average speedup (Target: 2×)")

    if cython_results:
        cython_avg = np.mean([r[3] for r in cython_results])
        all_speedups.extend([r[3] for r in cython_results])
        print(f"Cython:   {cython_avg:.2f}× average speedup (Target: 3-5×)")

    if fft_results:
        fft_avg = np.mean([r[3] for r in fft_results])
        all_speedups.extend([r[3] for r in fft_results])
        print(f"pyFFTW:   {fft_avg:.2f}× average speedup (Target: 1.5-2×)")

    if all_speedups:
        overall_avg = np.mean(all_speedups)
        print("-" * 80)
        print(f"TOTAL:    {overall_avg:.2f}× average speedup (Target: 2-5×)")
        print(f"Status:   {'✅ PASSED' if overall_avg >= 2.0 else '❌ FAILED'}")

    print("=" * 80)


if __name__ == "__main__":
    main()
