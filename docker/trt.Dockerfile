# TensorRT engine build + PyTorch / ONNX Runtime / TensorRT benchmark (needs an NVIDIA GPU).
# TRT 10.x, CUDA 12.x, Python 3.12
FROM nvcr.io/nvidia/tensorrt:24.12-py3

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir onnxruntime-gpu \
    && pip install --no-cache-dir -e ".[deploy]" \
    && pip uninstall -y onnxruntime || true
COPY configs ./configs
COPY scripts ./scripts

# docker run --rm --gpus all -v $PWD:/work -w /work srd-trt bash scripts/deploy_pipeline.sh runs/highway/rppo/seed_0 highway
CMD ["bash"]
