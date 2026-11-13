# MetaDrive sim-to-sim transfer test: ONNX policy + safety shield, no torch / no training.
#   docker build -f docker/metadrive.Dockerfile -t safe-rl-drive/metadrive .
#   docker run --rm -v $PWD/models:/app/models -v $PWD/results:/app/results safe-rl-drive/metadrive \
#       python sim2sim_demo/metadrive_demo.py --onnx models/rppo_highway.onnx --episodes 20
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# MetaDrive pins panda3d 1.10.13; highway-env is only needed because safe_rl_drive.envs imports it.
RUN pip install --no-cache-dir "metadrive-simulator==0.4.3" onnxruntime highway-env gymnasium pyyaml \
        imageio imageio-ffmpeg pillow
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps . && python -c "import metadrive, safe_rl_drive.safety"
COPY sim2sim_demo ./sim2sim_demo
ENV SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1
CMD ["python", "sim2sim_demo/metadrive_demo.py", "--help"]
