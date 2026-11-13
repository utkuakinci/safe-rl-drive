"""Build TensorRT engines (FP32 / FP16 / INT8) from the exported ONNX policy.

INT8 uses post-training entropy calibration on *on-policy* observations recorded with
``collect_obs`` (random inputs would give meaningless activation ranges).

    python -m safe_rl_drive.deploy.build_trt --onnx models/rppo_highway.onnx --precision fp16 \
        --out models/rppo_highway_fp16.engine
    python -m safe_rl_drive.deploy.build_trt --onnx models/rppo_highway.onnx --precision int8 \
        --calib data/obs_highway_rppo.npz --out models/rppo_highway_int8.engine

Equivalent ``trtexec`` command for FP16 (useful on Jetson)::

    trtexec --onnx=models/ppo_highway.onnx --fp16 --saveEngine=models/ppo_highway_fp16.engine \
        --minShapes=obs:1x8x7 --optShapes=obs:1x8x7 --maxShapes=obs:64x8x7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _calib_feeds(npz_path: Path, recurrent: bool) -> dict[str, np.ndarray]:
    d = np.load(npz_path)
    feeds = {"obs": d["obs"].astype(np.float32)}
    if recurrent:
        feeds["h_in"] = d["h"].astype(np.float32)  # [N, L, H]; transposed to [L, B, H] per batch
        feeds["c_in"] = d["c"].astype(np.float32)
    return feeds


def make_calibrator(trt, feeds: dict[str, np.ndarray], batch: int, cache: Path):
    import torch

    class NpzCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.n = len(feeds["obs"])
            self.idx = 0
            self.dev: dict[str, torch.Tensor] = {}

        def get_batch_size(self):
            return batch

        def get_batch(self, names):
            if self.idx + batch > self.n:
                return None
            ptrs = []
            for name in names:
                x = feeds[name][self.idx : self.idx + batch]
                if name in ("h_in", "c_in"):
                    x = np.ascontiguousarray(np.transpose(x, (1, 0, 2)))
                self.dev[name] = torch.from_numpy(np.ascontiguousarray(x)).cuda()
                ptrs.append(int(self.dev[name].data_ptr()))
            self.idx += batch
            return ptrs

        def read_calibration_cache(self):
            return cache.read_bytes() if cache.exists() else None

        def write_calibration_cache(self, data):
            cache.write_bytes(bytes(data))

    return NpzCalibrator()


def build(
    onnx_path: Path,
    out: Path,
    precision: str = "fp16",
    calib: Path | None = None,
    max_batch: int = 64,
    opt_batch: int = 1,
    workspace_mb: int = 512,
):
    import tensorrt as trt

    meta = json.loads(onnx_path.with_suffix(".json").read_text())
    recurrent = meta["recurrent"]
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):  # TRT 8.x
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed:\n{errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    profile = builder.create_optimization_profile()
    V, F = meta["obs_shape"]
    profile.set_shape("obs", (1, V, F), (opt_batch, V, F), (max_batch, V, F))
    if recurrent:
        L, H = meta["lstm_layers"], meta["lstm_hidden"]
        for n in ("h_in", "c_in"):
            profile.set_shape(n, (L, 1, H), (L, opt_batch, H), (L, max_batch, H))
    config.add_optimization_profile(profile)

    if precision in ("fp16", "int8"):
        config.set_flag(trt.BuilderFlag.FP16)  # INT8 engines fall back to FP16 where INT8 is unsupported
    if precision == "int8":
        if calib is None:
            raise ValueError("INT8 needs --calib (npz from collect_obs)")
        config.set_flag(trt.BuilderFlag.INT8)
        feeds = _calib_feeds(calib, recurrent)
        config.int8_calibrator = make_calibrator(
            trt, feeds, batch=min(32, max_batch), cache=out.with_suffix(".calib")
        )
        config.set_calibration_profile(profile)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT engine build failed")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(bytes(engine))
    print(f"built {precision} engine -> {out} ({out.stat().st_size / 1e6:.2f} MB)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    p.add_argument("--calib", type=Path)
    p.add_argument("--max-batch", type=int, default=64)
    a = p.parse_args(argv)
    build(a.onnx, a.out, a.precision, a.calib, a.max_batch)


if __name__ == "__main__":
    main()
