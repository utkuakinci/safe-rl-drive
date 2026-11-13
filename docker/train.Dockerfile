# Training / evaluation / explainability image (CPU; small MLP/LSTM policies train fastest on CPU).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV PYTHONUNBUFFERED=1 WANDB_DIR=/app/wandb

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e ".[tracking,deploy,active-inference,dev]"
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests

# docker run --rm -v $PWD/runs:/app/runs -e WANDB_API_KEY srd-train \
#     safe_rl_drive.training.train --config configs/train/highway_rppo.yaml --all-seeds
ENTRYPOINT ["python", "-m"]
CMD ["safe_rl_drive.training.train", "--config", "configs/train/highway_ppo.yaml", "--wandb-mode", "offline"]
