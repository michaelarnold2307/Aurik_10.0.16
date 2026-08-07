# Typing-Imports für Typannotationen

# Test für Hybrid Restoration Plugin (Aurik 2.0)


def test_hybrid_restoration_basic():
    # Echte Testdaten und Plugin-Ausführung
    import numpy as np

    from plugins.hybrid_restoration import HybridRestorationPlugin

    audio = np.zeros(44100)  # Dummy-Audio, 1 Sekunde
    sr = 44100
    plugin = HybridRestorationPlugin()
    restored = plugin.process(audio, sr)  # type: ignore[arg-type]
    assert restored is not None
    assert restored.shape == audio.shape
