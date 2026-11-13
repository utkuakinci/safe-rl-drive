#!/usr/bin/env bash
# Model -> ONNX -> TensorRT (FP32/FP16/INT8) -> benchmark table. Skips TensorRT if it is not installed.
#   bash scripts/deploy_pipeline.sh runs/highway/rppo/seed_0 highway
set -euo pipefail
RUN=${1:?run dir}; SCENARIO=${2:-highway}
NAME=$(python -c "import json,sys;m=json.load(open('$RUN/meta.json'));print(f\"{m['name']}_{m['scenario']}\")")
mkdir -p models data results

python -m safe_rl_drive.deploy.export_onnx --model "$RUN" --out "models/$NAME.onnx"
python -m safe_rl_drive.deploy.collect_obs --model "$RUN" --scenario "$SCENARIO" --episodes 30 --out "data/obs_$NAME.npz"

ENGINES=()
if python -c "import tensorrt" 2>/dev/null; then
  for P in fp32 fp16 int8; do
    python -m safe_rl_drive.deploy.build_trt --onnx "models/$NAME.onnx" --precision $P \
      --calib "data/obs_$NAME.npz" --out "models/${NAME}_$P.engine"
    ENGINES+=(--engine "$P=models/${NAME}_$P.engine")
  done
fi

python -m safe_rl_drive.deploy.benchmark --model "$RUN" --onnx "models/$NAME.onnx" --data "data/obs_$NAME.npz" \
  "${ENGINES[@]}" --batch-sizes 1 32 --closed-loop-episodes 20 --out "results/benchmark_$NAME"
