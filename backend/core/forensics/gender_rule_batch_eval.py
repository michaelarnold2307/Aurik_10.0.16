import csv
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from gender_rule_based import RuleBasedGenderDetector

logger = logging.getLogger(__name__)


def batch_evaluate(audio_dir, label_csv, out_csv, sr=16000) -> None:
    detector = RuleBasedGenderDetector(sr=sr)
    # Lade Labels: CSV mit Spalten 'file','label'
    labels = {}
    with open(label_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["file"]] = row["label"]
    # Batch-Auswertung
    results = []
    for fname in os.listdir(audio_dir):
        if not fname.lower().endswith((".wav", ".flac", ".mp3")):
            continue
        path = os.path.join(audio_dir, fname)
        # Erweiterung: Features extrahieren
        audio, sr2 = detector.sr, sr
        f0, voiced_ratio = 0, 0
        f1, f2 = 0, 0
        try:
            audio, sr2 = detector._load_audio(path)
            f0, voiced_ratio = detector._estimate_pitch(audio, sr2)
            f1, f2 = detector._estimate_formants(audio, sr2)
        except Exception as _exc:
            logger.debug("Operation fehlgeschlagen (unkritisch): %s", _exc)
        gender = detector.detect_gender(path)
        true_label = labels.get(fname, "unknown")
        results.append(
            {
                "file": fname,
                "predicted": gender,
                "true": true_label,
                "f0": f0,
                "f1": f1,
                "f2": f2,
                "voiced_ratio": voiced_ratio,
            }
        )
    # Schreibe Ergebnisse
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "predicted", "true", "f0", "f1", "f2", "voiced_ratio"],
        )
        writer.writeheader()
        writer.writerows(results)
    logger.debug("Batch-Auswertung abgeschlossen. Ergebnisse in %s", out_csv)

    # Metriken berechnen
    y_true = [r["true"] for r in results]
    y_pred = [r["predicted"] for r in results]
    acc = np.mean([yt == yp for yt, yp in zip(y_true, y_pred)])
    logger.debug("Accuracy: %.3f", acc)
    # Confusion Matrix
    classes = sorted(set(y_true) | set(y_pred))
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    class_idx = {c: i for i, c in enumerate(classes)}
    for yt, yp in zip(y_true, y_pred):
        matrix[class_idx[yt], class_idx[yp]] += 1
    logger.debug("Confusion Matrix:")
    logger.debug("\t" + "\t".join(classes))
    for i, c in enumerate(classes):
        logger.debug("%s\t%s", c, "\t".join(str(matrix[i, j]) for j in range(len(classes))))

    # Scatterplot f0 vs. f1, farbcodiert nach true label
    try:
        colors = {"male": "blue", "female": "red", "child": "green", "unknown": "gray"}
        plt.figure(figsize=(7, 5))
        for c in classes:
            xs = [r["f0"] for r in results if r["true"] == c]
            ys = [r["f1"] for r in results if r["true"] == c]
            plt.scatter(xs, ys, label=c, alpha=0.7, color=colors.get(c, "black"))
        plt.xlabel("f0 (Hz)")
        plt.ylabel("f1 (Hz)")
        plt.title("f0 vs. f1 nach true label")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_csv.replace(".csv", "_f0f1.png"))
        logger.debug("Scatterplot gespeichert: %s", out_csv.replace(".csv", "_f0f1.png"))
    except Exception as e:
        logger.debug("[Warnung] Scatterplot nicht erzeugt: %s", e)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        logger.debug("Nutzung: python gender_rule_batch_eval.py <audio_dir> <label_csv> <out_csv>")
        exit(1)
    batch_evaluate(sys.argv[1], sys.argv[2], sys.argv[3])
