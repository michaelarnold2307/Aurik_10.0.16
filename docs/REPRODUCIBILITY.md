# Aurik Reproduzierbarkeits-Garantie — §v10.700 G4

Aurik garantiert deterministische Restaurierung: gleicher Input + gleicher Seed = gleicher Output.

## Garantie-Matrix

| Konfiguration | Determinismus | Toleranz |
|---|---|---|
| CPU → CPU (gleicher Seed) | ✅ Bit-identisch | SHA-256 match |
| CPU → CPU (unterschiedlicher Seed) | ❌ Unterschiedlich | — |
| GPU → GPU (gleicher Seed) | ⚠️ Plattform-abhängig | ±1e-6 pro Sample |
| GPU → CPU (gleicher Seed) | ❌ Unterschiedlich | ONNX-Operationen variieren |

## Seed-Parameter

```python
from backend.core.unified_restorer_v3 import UnifiedRestorerV3

restorer = UnifiedRestorerV3()

# Deterministische Restaurierung
result = restorer.restore(audio, sr=48000, seed=42)

# Zweiter Lauf — identisches Ergebnis
result2 = restorer.restore(audio, sr=48000, seed=42)
assert (result.audio == result2.audio).all()  # ✅
```

## CI-Gate

```bash
# Reproduzierbarkeits-Test (Teil des Solo-Release-Gates)
make test-golden-gate

# Explizit:
pytest tests/normative/test_reproducibility.py -m reproducibility
```

## Einschränkungen

- **GPU-ONNX**: Float-Operationen auf GPU können von Lauf zu Lauf minimal variieren.
  CPU-Modus ist vollständig deterministisch.
- **ML-Modelle**: Manche Modelle (FlashSR, Demucs) können non-deterministische
  Operationen enthalten. In diesem Fall wird der nächstbeste deterministische
  DSP-Fallback verwendet.
- **Seed-Garantie**: Nur wenn `seed` explizit gesetzt wird. Ohne Seed wird
  `np.random` ohne festen Startzustand verwendet → nicht deterministisch.
