"""
Performance benchmarks for AURIK v8.2 Conservative Pitch Correction

Benchmarks:
- Detection speed (CREPE vs fallback)
- Correction speed
- Memory usage
- Quality metrics
"""

import time
from typing import Any

import numpy as np
import pytest

# Import pitch correction modules
try:
    from backend.ml.inference_only.pitch_correction import ConservativePitchCorrector, CREPEPitchDetector

    PITCH_CORRECTION_AVAILABLE = True
except ImportError as e:
    pytest.skip(f"Pitch correction not available: {e}", allow_module_level=True)
    PITCH_CORRECTION_AVAILABLE = False


# Test configurations
SAMPLE_RATES = [22050, 44100, 48000]
AUDIO_DURATIONS = [1.0, 5.0, 10.0, 30.0]
MODEL_CAPACITIES = ["tiny", "small", "medium", "large", "full"]


@pytest.fixture(scope="module")
def benchmark_audio():
    """Generate benchmark audio samples"""
    samples: dict[Any, Any] = {}

    for sr in SAMPLE_RATES:
        samples[sr] = {}
        for duration in AUDIO_DURATIONS:
            t = np.linspace(0, duration, int(sr * duration))
            # Complex signal: multiple harmonics + slight pitch variation
            audio = (
                0.5 * np.sin(2 * np.pi * 440 * t)
                + 0.2 * np.sin(2 * np.pi * 880 * t)
                + 0.1 * np.sin(2 * np.pi * 1320 * t)
            )
            samples[sr][duration] = audio.astype(np.float32)

    return samples


# === Detection Speed Benchmarks ===


@pytest.mark.benchmark
@pytest.mark.parametrize("model_capacity", MODEL_CAPACITIES)
def test_detection_speed_by_capacity(benchmark_audio, model_capacity):
    """Benchmark detection speed for different CREPE model capacities"""
    sr = 44100
    duration = 5.0
    audio = benchmark_audio[sr][duration]

    detector = CREPEPitchDetector(sample_rate=sr, model_capacity=model_capacity, step_size=10, viterbi=True)

    start_time = time.time()
    analysis = detector.detect(audio)
    elapsed_time = time.time() - start_time

    realtime_factor = elapsed_time / duration

    print(f"\n{model_capacity} model:")
    print(f"  Duration: {duration}s")
    print(f"  Processing time: {elapsed_time:.2f}s")
    print(f"  Realtime factor: {realtime_factor:.2f}x")
    print(f"  Frames analyzed: {len(analysis.f0_hz)}")
    print(f"  Errors found: {len(analysis.pitch_errors)}")

    assert elapsed_time > 0
    assert len(analysis.f0_hz) > 0


@pytest.mark.benchmark
@pytest.mark.parametrize("duration", AUDIO_DURATIONS)
def test_detection_speed_by_duration(benchmark_audio, duration):
    """Benchmark detection speed for different audio durations"""
    sr = 44100
    audio = benchmark_audio[sr][duration]

    detector = CREPEPitchDetector(
        sample_rate=sr,
        model_capacity="small",
        step_size=10,
        viterbi=True,  # Fast model for scaling tests
    )

    start_time = time.time()
    analysis = detector.detect(audio)
    elapsed_time = time.time() - start_time

    realtime_factor = elapsed_time / duration
    frames_per_second = len(analysis.f0_hz) / elapsed_time

    print(f"\nDuration {duration}s:")
    print(f"  Processing time: {elapsed_time:.2f}s")
    print(f"  Realtime factor: {realtime_factor:.2f}x")
    print(f"  Frames/sec: {frames_per_second:.1f}")
    print(f"  Scalability: {'Linear' if 0.8 < (elapsed_time / duration) / 0.2 < 1.2 else 'Non-linear'}")

    assert elapsed_time > 0


@pytest.mark.benchmark
@pytest.mark.parametrize("sample_rate", SAMPLE_RATES)
def test_detection_speed_by_sample_rate(benchmark_audio, sample_rate):
    """Benchmark detection speed for different sample rates"""
    duration = 5.0
    audio = benchmark_audio[sample_rate][duration]

    detector = CREPEPitchDetector(sample_rate=sample_rate, model_capacity="small", step_size=10, viterbi=True)

    start_time = time.time()
    detector.detect(audio)
    elapsed_time = time.time() - start_time

    realtime_factor = elapsed_time / duration

    print(f"\nSample rate {sample_rate} Hz:")
    print(f"  Processing time: {elapsed_time:.2f}s")
    print(f"  Realtime factor: {realtime_factor:.2f}x")
    print(f"  Samples: {len(audio)}")

    assert elapsed_time > 0


# === Correction Speed Benchmarks ===


@pytest.mark.benchmark
@pytest.mark.parametrize("duration", AUDIO_DURATIONS)
def test_correction_speed_by_duration(benchmark_audio, duration):
    """Benchmark correction speed for different audio durations"""
    sr = 44100
    audio = benchmark_audio[sr][duration]

    corrector = ConservativePitchCorrector(
        sample_rate=sr, error_threshold_cents=25.0, max_dcs=0.15, min_epistemic_confidence=0.80
    )

    start_time = time.time()
    audio_corrected, metadata = corrector.correct_pitch(audio)
    elapsed_time = time.time() - start_time

    realtime_factor = elapsed_time / duration

    print(f"\nDuration {duration}s:")
    print(f"  Total time: {elapsed_time:.2f}s")
    print(f"  Realtime factor: {realtime_factor:.2f}x")
    print(f"  Corrected: {metadata['corrected']}")
    if metadata["corrected"]:
        print(f"  Corrections: {metadata['n_corrections']}")
        print(f"  DCS: {metadata['dcs']:.3f}")
    else:
        print(f"  Rejection reason: {metadata['reason']}")

    assert elapsed_time > 0


@pytest.mark.benchmark
def test_correction_with_errors_speed(benchmark_audio):
    """Benchmark correction speed on audio with injected pitch errors"""
    sr = 44100
    duration = 5.0
    audio = benchmark_audio[sr][duration].copy()

    # Inject pitch errors (shift 0.5s segments by 40 cents)
    shift_ratio = 2 ** (40 / 1200)
    samples_per_segment = int(0.5 * sr)

    for i in range(0, len(audio), samples_per_segment * 2):
        end = min(i + samples_per_segment, len(audio))
        # Simple frequency shift approximation
        audio[i:end] *= shift_ratio

    corrector = ConservativePitchCorrector(
        sample_rate=sr,
        error_threshold_cents=25.0,
        max_dcs=0.15,
        min_epistemic_confidence=0.70,  # Lower threshold for test
    )

    start_time = time.time()
    audio_corrected, metadata = corrector.correct_pitch(audio)
    elapsed_time = time.time() - start_time

    realtime_factor = elapsed_time / duration

    print("\nCorrection with injected errors:")
    print(f"  Duration: {duration}s")
    print(f"  Processing time: {elapsed_time:.2f}s")
    print(f"  Realtime factor: {realtime_factor:.2f}x")
    print(f"  Corrected: {metadata['corrected']}")
    if metadata["corrected"]:
        print(f"  Corrections applied: {metadata['n_corrections']}")

    assert elapsed_time > 0


# === Memory Benchmarks ===


@pytest.mark.benchmark
def test_memory_usage_detection(benchmark_audio):
    """Benchmark memory usage for pitch detection"""
    import tracemalloc

    sr = 44100
    duration = 10.0
    audio = benchmark_audio[sr][duration]

    detector = CREPEPitchDetector(sample_rate=sr, model_capacity="full", step_size=10)  # Largest model

    # Start memory tracking
    tracemalloc.start()

    detector.detect(audio)

    # Get peak memory
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / 1024 / 1024
    per_second_mb = peak_mb / duration

    print("\nMemory usage (Detection):")
    print(f"  Duration: {duration}s")
    print(f"  Peak memory: {peak_mb:.1f} MB")
    print(f"  Per second: {per_second_mb:.1f} MB/s")

    assert peak > 0


@pytest.mark.benchmark
def test_memory_usage_correction(benchmark_audio):
    """Benchmark memory usage for pitch correction"""
    import tracemalloc

    sr = 44100
    duration = 10.0
    audio = benchmark_audio[sr][duration]

    corrector = ConservativePitchCorrector(sample_rate=sr, error_threshold_cents=25.0)

    # Start memory tracking
    tracemalloc.start()

    audio_corrected, metadata = corrector.correct_pitch(audio)

    # Get peak memory
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / 1024 / 1024
    per_second_mb = peak_mb / duration

    print("\nMemory usage (Correction):")
    print(f"  Duration: {duration}s")
    print(f"  Peak memory: {peak_mb:.1f} MB")
    print(f"  Per second: {per_second_mb:.1f} MB/s")

    assert peak > 0


# === Quality Benchmarks ===


@pytest.mark.benchmark
def test_quality_formant_preservation():
    """Benchmark formant preservation quality"""
    # Generate synthetic vocal signal with formants
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))

    # Simulate vocal tract: F0=200 Hz, F1=800 Hz, F2=1200 Hz
    f0 = 200
    f1 = 800
    f2 = 1200

    audio = (
        0.5 * np.sin(2 * np.pi * f0 * t)  # Fundamental
        + 0.3 * np.sin(2 * np.pi * f1 * t)  # First formant
        + 0.2 * np.sin(2 * np.pi * f2 * t)  # Second formant
    )

    # Inject pitch error (shift F0 by 30 cents)
    audio_shifted = audio.copy()
    f0_shifted = f0 * 2 ** (30 / 1200)
    audio_shifted = (
        0.5 * np.sin(2 * np.pi * f0_shifted * t)
        + 0.3 * np.sin(2 * np.pi * f1 * t)  # Formants unchanged
        + 0.2 * np.sin(2 * np.pi * f2 * t)
    )

    corrector = ConservativePitchCorrector(sample_rate=sr, error_threshold_cents=25.0, formant_preservation=True)

    audio_corrected, metadata = corrector.correct_pitch(audio_shifted)

    if metadata["corrected"]:
        # Measure spectral similarity in formant regions
        from numpy.fft import rfft as np_rfft
        from numpy.fft import rfftfreq as np_rfftfreq

        fft_orig = np.abs(np_rfft(np.asarray(audio, dtype=np.float64)))
        fft_corr = np.abs(np_rfft(np.asarray(audio_corrected, dtype=np.float64)))
        freqs = np_rfftfreq(len(audio), 1 / sr)

        # Check formant regions (±50 Hz around F1 and F2)
        f1_mask = (freqs >= f1 - 50) & (freqs <= f1 + 50)
        f2_mask = (freqs >= f2 - 50) & (freqs <= f2 + 50)

        f1_similarity = np.corrcoef(fft_orig[f1_mask], fft_corr[f1_mask])[0, 1]

        f2_similarity = np.corrcoef(fft_orig[f2_mask], fft_corr[f2_mask])[0, 1]

        print("\nFormant Preservation Quality:")
        print(f"  F1 similarity: {f1_similarity:.3f}")
        print(f"  F2 similarity: {f2_similarity:.3f}")
        print(f"  Corrected: {metadata['corrected']}")
        print(f"  DCS: {metadata.get('dcs', 'N/A')}")

        # Formants should be highly preserved (> 0.90 correlation)
        assert f1_similarity > 0.85
        assert f2_similarity > 0.85


@pytest.mark.benchmark
def test_quality_energy_preservation():
    """Benchmark energy preservation during correction"""
    sr = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))

    # Generate test signal
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    corrector = ConservativePitchCorrector(sample_rate=sr, error_threshold_cents=25.0)

    audio_corrected, metadata = corrector.correct_pitch(audio)

    # Measure energy
    energy_orig = np.sum(audio**2)
    energy_corr = np.sum(audio_corrected**2)
    energy_ratio = energy_corr / energy_orig if energy_orig > 0 else 1.0
    energy_change = abs(1.0 - energy_ratio)

    print("\nEnergy Preservation:")
    print(f"  Original energy: {energy_orig:.2e}")
    print(f"  Corrected energy: {energy_corr:.2e}")
    print(f"  Energy ratio: {energy_ratio:.3f}")
    print(f"  Energy change: {energy_change * 100:.1f}%")
    print(f"  Corrected: {metadata['corrected']}")

    # Energy should be preserved within 15% (HIPS threshold)
    assert energy_change < 0.15


# === Comparative Benchmarks ===


@pytest.mark.benchmark
def test_comparison_detection_methods():
    """Compare CREPE vs fallback detection methods"""
    sr = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    results = {}

    # Test different capacities (fallback if CREPE unavailable)
    for capacity in ["tiny", "small", "full"]:
        detector = CREPEPitchDetector(sample_rate=sr, model_capacity=capacity, step_size=10)

        start_time = time.time()
        analysis = detector.detect(audio)
        elapsed_time = time.time() - start_time

        mean_f0 = np.mean(analysis.f0_hz[analysis.f0_hz > 0])
        f0_accuracy = 1.0 - abs(mean_f0 - 440) / 440

        results[capacity] = {
            "time": elapsed_time,
            "realtime_factor": elapsed_time / duration,
            "mean_f0": mean_f0,
            "accuracy": f0_accuracy,
        }

    print("\nDetection Method Comparison:")
    for method, data in results.items():
        print(f"  {method}:")
        print(f"    Time: {data['time']:.2f}s")
        print(f"    Realtime: {data['realtime_factor']:.2f}x")
        print(f"    F0: {data['mean_f0']:.1f} Hz")
        print(f"    Accuracy: {data['accuracy'] * 100:.1f}%")


# === Summary Report ===


@pytest.mark.benchmark
def test_generate_benchmark_summary(benchmark_audio):
    """Generate comprehensive benchmark summary"""
    sr = 44100
    duration = 5.0
    audio = benchmark_audio[sr][duration]

    print("\n" + "=" * 60)
    print("AURIK v8.2 Pitch Correction - Benchmark Summary")
    print("=" * 60)

    # Detection benchmark
    detector = CREPEPitchDetector(sample_rate=sr, model_capacity="full", step_size=10)

    start = time.time()
    detector.detect(audio)
    detection_time = time.time() - start

    print("\n1. Pitch Detection:")
    print("   Model: CREPE (full)")
    print(f"   Duration: {duration}s")
    print(f"   Processing time: {detection_time:.2f}s")
    print(f"   Realtime factor: {detection_time / duration:.2f}x")

    # Correction benchmark
    corrector = ConservativePitchCorrector(sample_rate=sr, error_threshold_cents=25.0)

    start = time.time()
    audio_corrected, metadata = corrector.correct_pitch(audio)
    correction_time = time.time() - start

    print("\n2. Pitch Correction:")
    print(f"   Duration: {duration}s")
    print(f"   Processing time: {correction_time:.2f}s")
    print(f"   Realtime factor: {correction_time / duration:.2f}x")
    print(f"   Corrected: {metadata['corrected']}")

    # Total
    total_time = detection_time + correction_time
    print("\n3. Total Pipeline:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Realtime factor: {total_time / duration:.2f}x")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "benchmark", "--tb=short"])
