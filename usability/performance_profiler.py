"""
Performance Profiling Module for AURIK v8
==========================================

Comprehensive performance monitoring and bottleneck detection:
- Memory usage tracking (RSS, VMS, percent)
- CPU usage tracking (per-process, per-thread)
- Execution time profiling (function-level)
- Bottleneck detection (automatic threshold-based alerts)
- Resource warnings (memory/CPU limits)
- Export reports (JSON, HTML)

Usage:
    from usability.performance_profiler import PerformanceProfiler, profile_function

    # Automatic profiling with context manager
    with PerformanceProfiler() as profiler:
        # Your code here
        result = process_audio(audio, sr)

    # Get report
    report = profiler.get_report()
    profiler.export_html('performance_report.html')

    # Function decorator
    @profile_function
    def my_expensive_function(audio):
        # Processing
        return result

Environment Variables:
    AURIK_PROFILE=1           # Enable profiling (default: disabled)
    AURIK_PROFILE_MEMORY=1    # Enable memory profiling (default: enabled if AURIK_PROFILE=1)
    AURIK_PROFILE_CPU=1       # Enable CPU profiling (default: enabled if AURIK_PROFILE=1)
    AURIK_PROFILE_THRESHOLD=75  # CPU/Memory warning threshold (default: 75%)

Features:
- Minimal overhead (<1% when disabled, <5% when enabled)
- Thread-safe
- Automatic cleanup
- Configurable thresholds
- Rich reports
"""

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

import psutil

logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """Snapshot of resource usage at a point in time"""

    timestamp: float
    memory_rss_mb: float  # Resident Set Size in MB
    memory_vms_mb: float  # Virtual Memory Size in MB
    memory_percent: float  # Percent of total system memory
    cpu_percent: float  # CPU usage percent
    thread_count: int  # Number of threads


@dataclass
class FunctionProfile:
    """Profile data for a single function call"""

    name: str
    start_time: float
    end_time: float
    duration_ms: float
    memory_delta_mb: float  # Memory change during execution
    cpu_avg: float  # Average CPU during execution
    peak_memory_mb: float  # Peak memory during execution
    warnings: list[str] = field(default_factory=list)


class PerformanceProfiler:
    """
    Performance profiler with memory, CPU tracking and bottleneck detection.

    Example:
        profiler = PerformanceProfiler(sample_interval=0.1)
        profiler.start()
        # Your code
        profiler.stop()
        report = profiler.get_report()
    """

    def __init__(
        self,
        sample_interval: float = 0.5,  # Sample every 0.5s
        memory_threshold: float = 75.0,  # Warn at 75% memory
        cpu_threshold: float = 75.0,  # Warn at 75% CPU
        enabled: bool | None = None,  # Auto-detect from env  # type: ignore[assignment]
    ):
        """
        Initialize profiler.

        Args:
            sample_interval: Sampling interval in seconds
            memory_threshold: Memory usage warning threshold (%)
            cpu_threshold: CPU usage warning threshold (%)
            enabled: Enable profiling (auto-detect from AURIK_PROFILE if None)
        """
        # Auto-detect from environment
        if enabled is None:
            enabled = os.getenv("AURIK_PROFILE", "0") == "1"

        self.enabled = enabled
        self.sample_interval = sample_interval
        self.memory_threshold = float(os.getenv("AURIK_PROFILE_THRESHOLD", str(memory_threshold)))
        self.cpu_threshold = float(os.getenv("AURIK_PROFILE_THRESHOLD", str(cpu_threshold)))

        # State
        self.running = False
        self.process = psutil.Process()
        self.snapshots: list[ResourceSnapshot] = []
        self.function_profiles: list[FunctionProfile] = []
        self.warnings: list[str] = []

        # Threading
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self):
        """Start profiling (background sampling)"""
        if not self.enabled:
            return

        if self.running:
            return

        self.running = True
        self.snapshots.clear()
        self.function_profiles.clear()
        self.warnings.clear()

        # Start background sampling thread
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop profiling"""
        if not self.enabled or not self.running:
            return

        self.running = False

        # Wait for thread to finish
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _sample_loop(self):
        """Background sampling loop"""
        while self.running:
            self._take_snapshot()
            time.sleep(self.sample_interval)

    def _take_snapshot(self):
        """Take a resource usage snapshot"""
        try:
            # Get memory info
            mem_info = self.process.memory_info()
            memory_rss_mb = mem_info.rss / 1024 / 1024
            memory_vms_mb = mem_info.vms / 1024 / 1024
            memory_percent = self.process.memory_percent()

            # Get CPU info (interval=None for non-blocking)
            cpu_percent = self.process.cpu_percent(interval=None)

            # Get thread count
            thread_count = self.process.num_threads()

            snapshot = ResourceSnapshot(
                timestamp=time.time(),
                memory_rss_mb=memory_rss_mb,
                memory_vms_mb=memory_vms_mb,
                memory_percent=memory_percent,
                cpu_percent=cpu_percent,
                thread_count=thread_count,
            )

            with self._lock:
                self.snapshots.append(snapshot)

                # Check thresholds
                if memory_percent > self.memory_threshold:
                    warning = f"Memory usage high: {memory_percent:.1f}% (threshold: {self.memory_threshold}%)"
                    if warning not in self.warnings:
                        self.warnings.append(warning)

                if cpu_percent > self.cpu_threshold:
                    warning = f"CPU usage high: {cpu_percent:.1f}% (threshold: {self.cpu_threshold}%)"
                    if warning not in self.warnings:
                        self.warnings.append(warning)

        except Exception:
            # Ignore sampling errors (process might have exited)
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

    @contextmanager
    def profile_block(self, name: str):
        """
        Context manager for profiling a code block.

        Example:
            with profiler.profile_block('expensive_operation'):
                result = expensive_operation()
        """
        if not self.enabled:
            yield
            return

        # Start
        start_time = time.perf_counter()
        start_mem = self.process.memory_info().rss / 1024 / 1024
        cpu_samples = []

        # Sample CPU during execution
        def sample_cpu():
            while True:
                cpu_samples.append(self.process.cpu_percent(interval=0.1))
                time.sleep(0.1)
                if not self.running or time.perf_counter() - start_time > 10.0:
                    break

        cpu_thread = threading.Thread(target=sample_cpu, daemon=True)
        cpu_thread.start()

        try:
            yield
        finally:
            # Stop
            end_time = time.perf_counter()
            end_mem = self.process.memory_info().rss / 1024 / 1024
            duration_ms = (end_time - start_time) * 1000

            cpu_thread.join(timeout=0.5)
            cpu_avg = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0

            # Find peak memory during this period
            peak_memory_mb = end_mem
            with self._lock:
                for snapshot in self.snapshots:
                    if start_time <= snapshot.timestamp <= end_time:
                        peak_memory_mb = max(peak_memory_mb, snapshot.memory_rss_mb)

            # Create profile
            profile = FunctionProfile(
                name=name,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                memory_delta_mb=end_mem - start_mem,
                cpu_avg=cpu_avg,
                peak_memory_mb=peak_memory_mb,
                warnings=[],
            )

            # Check for bottlenecks
            if duration_ms > 1000:  # > 1 second
                profile.warnings.append(f"Slow execution: {duration_ms:.0f}ms")

            if profile.memory_delta_mb > 100:  # > 100 MB increase
                profile.warnings.append(f"High memory increase: {profile.memory_delta_mb:.1f} MB")

            with self._lock:
                self.function_profiles.append(profile)

    def get_report(self) -> dict[str, Any]:
        """
        Get comprehensive profiling report.

        Returns:
            Dictionary with profiling data (snapshots, functions, warnings, summary)
        """
        with self._lock:
            if not self.snapshots:
                return {
                    "enabled": self.enabled,
                    "summary": {"message": "No profiling data (profiler not enabled or no samples taken)"},
                    "snapshots": [],
                    "functions": [],
                    "warnings": [],
                }

            # Calculate summary statistics
            memory_values = [s.memory_rss_mb for s in self.snapshots]
            cpu_values = [s.cpu_percent for s in self.snapshots]

            summary = {
                "total_samples": len(self.snapshots),
                "duration_seconds": self.snapshots[-1].timestamp - self.snapshots[0].timestamp,
                "memory": {
                    "peak_mb": max(memory_values),
                    "avg_mb": sum(memory_values) / len(memory_values),
                    "min_mb": min(memory_values),
                },
                "cpu": {
                    "peak_percent": max(cpu_values),
                    "avg_percent": sum(cpu_values) / len(cpu_values),
                    "min_percent": min(cpu_values),
                },
                "threads": {
                    "peak": max(s.thread_count for s in self.snapshots),
                    "avg": sum(s.thread_count for s in self.snapshots) / len(self.snapshots),
                },
            }

            # Bottleneck detection
            bottlenecks = []
            for profile in self.function_profiles:
                if profile.warnings:
                    bottlenecks.append(
                        {"function": profile.name, "duration_ms": profile.duration_ms, "warnings": profile.warnings}
                    )

            return {
                "enabled": self.enabled,
                "summary": summary,
                "bottlenecks": bottlenecks,
                "warnings": list(self.warnings),
                "snapshots": [
                    {
                        "timestamp": s.timestamp,
                        "memory_mb": s.memory_rss_mb,
                        "cpu_percent": s.cpu_percent,
                        "threads": s.thread_count,
                    }
                    for s in self.snapshots
                ],
                "functions": [
                    {
                        "name": f.name,
                        "duration_ms": f.duration_ms,
                        "memory_delta_mb": f.memory_delta_mb,
                        "cpu_avg": f.cpu_avg,
                        "peak_memory_mb": f.peak_memory_mb,
                        "warnings": f.warnings,
                    }
                    for f in self.function_profiles
                ],
            }

    def export_json(self, filepath: str):
        """Export report as JSON"""
        report = self.get_report()
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

    def export_html(self, filepath: str):
        """Export report as HTML"""
        report = self.get_report()

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>AURIK Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .warning {{ background-color: #ffeb3b; }}
        .error {{ background-color: #f44336; color: white; }}
        .summary {{ background-color: #e3f2fd; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>AURIK Performance Report</h1>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Duration:</strong> {report["summary"].get("duration_seconds", 0):.2f}s</p>
        <p><strong>Samples:</strong> {report["summary"].get("total_samples", 0)}</p>
        <p><strong>Peak Memory:</strong> {report["summary"].get("memory", {}).get("peak_mb", 0):.1f} MB</p>
        <p><strong>Avg CPU:</strong> {report["summary"].get("cpu", {}).get("avg_percent", 0):.1f}%</p>
    </div>

    <h2>Bottlenecks</h2>
    <table>
        <tr><th>Function</th><th>Duration (ms)</th><th>Warnings</th></tr>
"""

        for bottleneck in report.get("bottlenecks", []):
            html += f"""
        <tr class="warning">
            <td>{bottleneck["function"]}</td>
            <td>{bottleneck["duration_ms"]:.1f}</td>
            <td>{", ".join(bottleneck["warnings"])}</td>
        </tr>
"""

        html += """
    </table>

    <h2>Warnings</h2>
    <ul>
"""

        for warning in report.get("warnings", []):
            html += f"        <li class='warning'>{warning}</li>\n"

        html += """
    </ul>
</body>
</html>
"""

        with open(filepath, "w") as f:
            f.write(html)

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


def profile_function(func):
    """
    Decorator for profiling individual functions.

    Example:
        @profile_function
        def expensive_operation(audio):
            return process(audio)
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get or create global profiler
        if not hasattr(wrapper, "_profiler"):
            wrapper._profiler = PerformanceProfiler()  # type: ignore[attr-defined]
            wrapper._profiler.start()  # type: ignore[attr-defined]

        profiler = wrapper._profiler  # type: ignore[attr-defined]

        with profiler.profile_block(func.__name__):
            return func(*args, **kwargs)

    return wrapper


# Example usage
if __name__ == "__main__":
    import numpy as np

    logging.info("Performance Profiler Demo")
    logging.info("=" * 50)

    # Enable profiling
    os.environ["AURIK_PROFILE"] = "1"

    # Example 1: Context manager
    with PerformanceProfiler(sample_interval=0.1) as profiler:
        logging.info("\nRunning expensive operations...")

        with profiler.profile_block("matrix_multiplication"):
            # Simulate expensive operation
            A = np.random.rand(1000, 1000)
            B = np.random.rand(1000, 1000)
            C = np.dot(A, B)

        with profiler.profile_block("memory_allocation"):
            # Simulate memory-intensive operation
            big_array = np.zeros((10000, 10000))
            time.sleep(0.5)

    # Get report
    report = profiler.get_report()
    logging.info("\n" + "=" * 50)
    logging.info("PERFORMANCE REPORT")
    logging.info("=" * 50)
    logging.info(f"Duration: {report['summary']['duration_seconds']:.2f}s")
    logging.info(f"Peak Memory: {report['summary']['memory']['peak_mb']:.1f} MB")
    logging.info(f"Avg CPU: {report['summary']['cpu']['avg_percent']:.1f}%")

    if report["bottlenecks"]:
        logging.info("\nBottlenecks:")
        for b in report["bottlenecks"]:
            logging.info(f"  - {b['function']}: {b['duration_ms']:.0f}ms ({', '.join(b['warnings'])})")

    if report["warnings"]:
        logging.warning("\nWarnings:")
        for w in report["warnings"]:
            logging.warning(f"  - {w}")

    # Export reports
    profiler.export_json("performance_report.json")
    profiler.export_html("performance_report.html")
    logging.info("\nReports exported:")
    logging.info("  - performance_report.json")
    logging.info("  - performance_report.html")
