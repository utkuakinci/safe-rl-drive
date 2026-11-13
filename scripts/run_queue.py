"""Run many (config, seed) trainings in parallel on one machine.

Each job is a separate process (``python -m safe_rl_drive.training.train``) with its own vectorised
envs; finished runs (``meta.json`` present) are skipped, so the queue can be stopped and resumed.

    python scripts/run_queue.py --configs configs/train/highway_ppo.yaml configs/train/highway_rppo.yaml \
        --seeds 0 1 2 --parallel 4 --n-envs 4 --wandb-mode disabled
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from safe_rl_drive.training.train import load_config  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--out", default="runs")
    p.add_argument("--wandb-mode", default="disabled")
    p.add_argument("--logs", type=Path, default=Path("runs/_logs"))
    p.add_argument("--set", nargs="*", default=[], help="extra overrides passed to every job")
    a = p.parse_args(argv)

    a.logs.mkdir(parents=True, exist_ok=True)
    jobs = []
    for cfg_path in a.configs:
        cfg = load_config(Path(cfg_path))
        for seed in a.seeds:
            run_dir = Path(a.out) / cfg["scenario"] / cfg["name"] / f"seed_{seed}"
            if (run_dir / "meta.json").exists():
                print(f"skip (done): {run_dir}")
                continue
            log = a.logs / f"{cfg['scenario']}_{cfg['name']}_s{seed}.log"
            cmd = [sys.executable, "-W", "ignore", "-m", "safe_rl_drive.training.train", "--config", cfg_path,
                   "--seed", str(seed), "--out", a.out, "--wandb-mode", a.wandb_mode,
                   "--set", f"n_envs={a.n_envs}", "verbose=0", *a.set]  # fmt: skip
            jobs.append((cmd, log, run_dir))

    print(f"{len(jobs)} jobs, {a.parallel} in parallel")
    running: list[tuple[subprocess.Popen, Path, float]] = []
    t_start = time.time()
    while jobs or running:
        while jobs and len(running) < a.parallel:
            cmd, log, run_dir = jobs.pop(0)
            fh = open(log, "w")  # noqa: SIM115 - kept open for the child's lifetime
            running.append((subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT), run_dir, time.time()))
            print(f"[{time.strftime('%H:%M:%S')}] start {run_dir}", flush=True)
        for proc, run_dir, t0 in list(running):
            if proc.poll() is not None:
                running.remove((proc, run_dir, t0))
                status = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
                print(
                    f"[{time.strftime('%H:%M:%S')}] {status} {run_dir} in {(time.time() - t0) / 60:.1f} min",
                    flush=True,
                )
        time.sleep(5)
    print(f"all done in {(time.time() - t_start) / 3600:.2f} h")


if __name__ == "__main__":
    main()
