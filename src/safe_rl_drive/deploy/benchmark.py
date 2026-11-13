"""PyTorch vs ONNX Runtime vs TensorRT: latency, memory, accuracy loss and closed-loop driving KPIs.

    python -m safe_rl_drive.deploy.benchmark --model runs/highway/rppo/seed_0 \
        --onnx models/rppo_highway.onnx --data data/obs_highway_rppo.npz \
        --engine fp16=models/rppo_highway_fp16.engine --engine int8=models/rppo_highway_int8.engine \
        --batch-sizes 1 32 --closed-loop-episodes 20 --out results/benchmark_rppo_highway

Every backend that is unavailable on the machine (no CUDA, no TensorRT) is skipped, not faked.
Accuracy is measured against PyTorch FP32 on CPU over recorded on-policy observations.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
from pathlib import Path

import numpy as np


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 2**20
    except ImportError:  # pragma: no cover
        return float("nan")


def _gpu_free_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.mem_get_info()[0] / 2**20
    except Exception:  # noqa: S110
        pass
    return float("nan")


def _feeds(data: dict, idx: np.ndarray, meta: dict) -> dict[str, np.ndarray]:
    f = {"obs": np.ascontiguousarray(data["obs"][idx])}
    if meta["recurrent"]:
        f["h_in"] = np.ascontiguousarray(np.transpose(data["h"][idx], (1, 0, 2)))
        f["c_in"] = np.ascontiguousarray(np.transpose(data["c"][idx], (1, 0, 2)))
    return f


def candidate_backends(args, meta):
    """Yield (label, factory) for every backend this machine can run."""
    from safe_rl_drive.deploy import backends as B

    yield "pytorch-cpu-fp32", lambda: B.TorchBackend(args.model, "cpu", threads=args.threads)
    yield (
        "onnxruntime-cpu-fp32",
        lambda: B.OrtBackend(args.onnx, "CPUExecutionProvider", threads=args.threads),
    )
    try:
        import torch

        cuda = torch.cuda.is_available()
    except ImportError:
        cuda = False
    if cuda:
        yield "pytorch-cuda-fp32", lambda: B.TorchBackend(args.model, "cuda")
        yield "pytorch-cuda-fp16", lambda: B.TorchBackend(args.model, "cuda", fp16=True)
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            yield "onnxruntime-cuda-fp32", lambda: B.OrtBackend(args.onnx, "CUDAExecutionProvider")
    except ImportError:
        pass
    for spec in args.engine:
        prec, _, path = spec.partition("=")
        yield f"tensorrt-cuda-{prec}", (lambda p=path, q=prec: B.TensorRTBackend(p, q))


def latency(backend, feeds, warmup: int, iters: int) -> np.ndarray:
    for _ in range(warmup):
        backend.infer(feeds)
    backend.sync()
    t = np.empty(iters)
    for i in range(iters):
        t0 = time.perf_counter_ns()
        backend.infer(feeds)  # host -> host, includes H2D/D2H copies
        t[i] = time.perf_counter_ns() - t0
    return t / 1e3  # microseconds


def accuracy(backend, ref_backend, data, meta, n: int, chunk: int = 64) -> dict:
    n = min(n, len(data["obs"]))
    diffs, agree = [], []
    for s in range(0, n, chunk):
        idx = np.arange(s, min(s + chunk, n))
        f = _feeds(data, idx, meta)
        a, b = backend.infer(f)[0], ref_backend.infer(f)[0]
        diffs.append(np.abs(a - b).max(axis=1))
        agree.append(a.argmax(1) == b.argmax(1))
    d = np.concatenate(diffs)
    return {
        "max_abs_logit_diff": float(d.max()),
        "mean_abs_logit_diff": float(d.mean()),
        "action_agreement": float(np.concatenate(agree).mean()),
    }


def closed_loop(backend, meta, scenario: str, episodes: int, seed: int = 30_000, mask_features=()) -> dict:
    from safe_rl_drive.envs.factory import make_env

    env = make_env(scenario, seed=seed, mask_features=mask_features)
    succ, coll = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        if meta["recurrent"]:
            h = np.zeros((meta["lstm_layers"], 1, meta["lstm_hidden"]), np.float32)
            c = h.copy()
        done = False
        while not done:
            feeds = {"obs": obs[None].astype(np.float32)}
            if meta["recurrent"]:
                feeds.update(h_in=h, c_in=c)
            out = backend.infer(feeds)
            if meta["recurrent"]:
                h, c = out[1], out[2]
            obs, _, term, trunc, info = env.step(int(out[0][0].argmax()))
            done = term or trunc
        succ.append(info["episode_metrics"]["success"])
        coll.append(info["episode_metrics"]["collision"])
    return {"success": float(np.mean(succ)), "collision": float(np.mean(coll))}


def to_markdown(rows: list[dict]) -> str:
    cols = [
        ("backend", "Backend", "{}"),
        ("batch", "Batch", "{}"),
        ("p50_us", "p50 [µs]", "{:.1f}"),
        ("p99_us", "p99 [µs]", "{:.1f}"),
        ("throughput", "Throughput [inf/s]", "{:,.0f}"),
        ("host_mem_mb", "Host RAM Δ [MB]", "{:.1f}"),
        ("gpu_mem_mb", "GPU mem Δ [MB]", "{:.1f}"),
        ("model_mb", "Artifact [MB]", "{:.2f}"),
        ("max_abs_logit_diff", "max abs Δlogit", "{:.1e}"),
        ("action_agreement", "Action agreement", "{:.2%}"),
        ("cl_success", "Closed-loop success", "{:.0%}"),
        ("cl_collision", "Closed-loop collision", "{:.0%}"),
    ]
    lines = ["| " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for key, _, fmt in cols:
            v = r.get(key)
            cells.append("-" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="SB3 run dir (PyTorch reference)")
    p.add_argument("--onnx", required=True)
    p.add_argument("--engine", action="append", default=[], help="precision=path.engine (repeatable)")
    p.add_argument("--data", required=True, help="npz from collect_obs")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32])
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--threads", type=int, default=1, help="CPU threads (1 = deterministic real-time setting)")
    p.add_argument("--accuracy-samples", type=int, default=5000)
    p.add_argument("--closed-loop-episodes", type=int, default=0)
    p.add_argument("--scenario", default=None, help="defaults to the scenario in the run's meta.json")
    p.add_argument("--out", type=Path, default=Path("results/benchmark"))
    args = p.parse_args(argv)

    from safe_rl_drive.deploy.backends import TorchBackend

    meta = json.loads(Path(args.onnx).with_suffix(".json").read_text())
    run_meta_file = Path(args.model) / "meta.json"
    run_meta = json.loads(run_meta_file.read_text()) if run_meta_file.exists() else {}
    scenario = args.scenario or run_meta.get("scenario", "highway")
    mask = tuple(run_meta.get("mask_features", ()))
    data = dict(np.load(args.data))
    ref = TorchBackend(args.model, "cpu", threads=args.threads)
    rng = np.random.default_rng(0)

    rows = []
    for label, factory in candidate_backends(args, meta):
        gc.collect()
        rss0, gpu0 = _rss_mb(), _gpu_free_mb()
        try:
            backend = factory()
            backend.infer(_feeds(data, np.arange(1), meta))
        except Exception as e:  # report and continue: a missing provider must not kill the benchmark
            print(f"[skip] {label}: {type(e).__name__}: {e}")
            continue
        host_mem, gpu_mem = _rss_mb() - rss0, gpu0 - _gpu_free_mb()
        acc = accuracy(backend, ref, data, meta, args.accuracy_samples)
        cl = (
            closed_loop(backend, meta, scenario, args.closed_loop_episodes, mask_features=mask)
            if args.closed_loop_episodes
            else {}
        )
        for bs in args.batch_sizes:
            feeds = _feeds(data, rng.integers(0, len(data["obs"]), size=bs), meta)
            t = latency(backend, feeds, args.warmup, args.iters)
            row = {
                "backend": label,
                "batch": bs,
                "p50_us": float(np.percentile(t, 50)),
                "p99_us": float(np.percentile(t, 99)),
                "mean_us": float(t.mean()),
                "throughput": bs * 1e6 / float(t.mean()),
                "host_mem_mb": host_mem,
                "gpu_mem_mb": gpu_mem,
                "model_mb": backend.model_file.stat().st_size / 1e6 if backend.model_file else float("nan"),
                **acc,
                "cl_success": cl.get("success"),
                "cl_collision": cl.get("collision"),
            }
            rows.append(row)
            print(
                f"{label:24s} bs={bs:<3d} p50={row['p50_us']:8.1f}us p99={row['p99_us']:8.1f}us agree={acc['action_agreement']:.4f}"
            )
        del backend

    args.out.mkdir(parents=True, exist_ok=True)
    env_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "threads": args.threads,
    }
    try:
        import torch

        if torch.cuda.is_available():
            env_info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    (args.out / "benchmark.json").write_text(
        json.dumps({"env": env_info, "model": meta, "rows": rows}, indent=2)
    )
    md = f"Hardware: `{env_info}`\n\n" + to_markdown(rows)
    (args.out / "benchmark.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
