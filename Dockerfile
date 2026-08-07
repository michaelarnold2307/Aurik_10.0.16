# Dockerfile — §v10.700 K3: Aurik Docker-Container.
#
# Bauen:  docker build -t aurik/aurik:10.14.0 .
# Nutzen: docker run -v $PWD/audio:/audio aurik/aurik restore /audio/file.wav
#
# Alle Abhängigkeiten vorinstalliert. Kein lokales Python nötig.

FROM python:3.10-slim

LABEL org.opencontainers.image.title="Aurik"
LABEL org.opencontainers.image.version="10.14.0"
LABEL org.opencontainers.image.description="Weltklasse-Audio-Restaurierung"

# System-Abhängigkeiten
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python-Abhängigkeiten
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    soundfile \
    pyyaml \
    pytest

# Optional: ONNX Runtime (CPU-only für Docker)
RUN pip install --no-cache-dir onnxruntime || echo "ONNX Runtime optional"

# Aurik-Code kopieren
WORKDIR /aurik
COPY . .

# Einstiegspunkt
ENTRYPOINT ["python", "-m", "cli.aurik_cli"]
CMD ["--help"]
