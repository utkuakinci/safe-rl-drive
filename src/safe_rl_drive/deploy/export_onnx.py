"""Export the actor of a PPO / RecurrentPPO policy to ONNX and verify numerical parity.

Graph interface (also what the C++ ROS2 node expects):

* feed-forward:  ``obs [B, V, F] -> logits [B, A]``
* recurrent:     ``obs [B, V, F], h_in [L, B, H], c_in [L, B, H] -> logits [B, A], h_out, c_out``

The interface contract (feature list, normalisation ranges, action names, recurrent sizes) is
embedded as ONNX ``metadata_props`` and mirrored into a ``<model>.json`` sidecar.

    python -m safe_rl_drive.deploy.export_onnx --model runs/highway/rppo/seed_0 --out models/rppo_highway.onnx
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from safe_rl_drive.envs.spec import ACTIONS, FEATURES, FEATURES_RANGE, OBS_SHAPE
from safe_rl_drive.policy import is_recurrent, load_model


class ActorMLP(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.features = policy.pi_features_extractor
        self.mlp = policy.mlp_extractor.policy_net
        self.head = policy.action_net

    def forward(self, obs):
        return self.head(self.mlp(self.features(obs)))


class ActorLSTM(nn.Module):
    """Single-step recurrent actor with explicit state I/O (episode resets are done by the caller)."""

    def __init__(self, policy):
        super().__init__()
        self.features = policy.pi_features_extractor
        self.lstm = policy.lstm_actor
        self.mlp = policy.mlp_extractor.policy_net
        self.head = policy.action_net

    def forward(self, obs, h, c):
        feat = self.features(obs).unsqueeze(0)  # [1, B, D]
        out, (h1, c1) = self.lstm(feat, (h, c))
        return self.head(self.mlp(out.squeeze(0))), h1, c1


def build_actor(model) -> tuple[nn.Module, dict]:
    policy = model.policy.to("cpu").eval()
    if is_recurrent(model):
        layers, _, hidden = policy.lstm_hidden_state_shape
        return ActorLSTM(policy).eval(), {"recurrent": True, "lstm_layers": layers, "lstm_hidden": hidden}
    return ActorMLP(policy).eval(), {"recurrent": False}


def interface_metadata(info: dict, source: str) -> dict:
    return {
        "obs_shape": list(OBS_SHAPE),
        "features": list(FEATURES),
        "features_range": {k: list(v) for k, v in FEATURES_RANGE.items()},
        "actions": [ACTIONS[i] for i in range(len(ACTIONS))],
        "source": source,
        **info,
    }


def dummy_inputs(info: dict, batch: int = 1):
    obs = torch.zeros((batch, *OBS_SHAPE), dtype=torch.float32)
    if not info["recurrent"]:
        return (obs,)
    h = torch.zeros((info["lstm_layers"], batch, info["lstm_hidden"]), dtype=torch.float32)
    return obs, h, h.clone()


def export(model_path: str | Path, out: str | Path, opset: int = 17, verify: bool = True) -> dict:
    import onnx

    model = load_model(model_path)
    actor, info = build_actor(model)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = info["recurrent"]
    inputs = ["obs", "h_in", "c_in"] if rec else ["obs"]
    outputs = ["logits", "h_out", "c_out"] if rec else ["logits"]
    axes = {"obs": {0: "batch"}, "logits": {0: "batch"}}
    if rec:
        axes.update({n: {1: "batch"} for n in ("h_in", "c_in", "h_out", "c_out")})
    kwargs = dict(
        input_names=inputs,
        output_names=outputs,
        dynamic_axes=axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False  # TorchScript exporter: stable LSTM op export, TensorRT friendly
    torch.onnx.export(actor, dummy_inputs(info), str(out), **kwargs)

    meta = interface_metadata(info, str(model_path))
    proto = onnx.load(str(out))
    for k, v in meta.items():
        entry = proto.metadata_props.add()
        entry.key, entry.value = k, json.dumps(v)
    onnx.checker.check_model(proto)
    onnx.save(proto, str(out))
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    if verify:
        meta["parity"] = verify_parity(actor, out, info)
        print(f"parity vs PyTorch: {meta['parity']}")
        out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"exported {out}")
    return meta


def verify_parity(actor: nn.Module, onnx_path: Path, info: dict, n: int = 256, seed: int = 0) -> dict:
    """Compare ONNX Runtime vs PyTorch on random in-range observations (and random LSTM states)."""
    import onnxruntime as ort

    rng = np.random.default_rng(seed)
    obs = rng.uniform(-1, 1, size=(n, *OBS_SHAPE)).astype(np.float32)
    obs[:, :, 0] = (obs[:, :, 0] > 0).astype(np.float32)  # presence is binary
    feeds = {"obs": obs}
    torch_in = [torch.from_numpy(obs)]
    if info["recurrent"]:
        shape = (info["lstm_layers"], n, info["lstm_hidden"])
        h = rng.normal(0, 0.5, size=shape).astype(np.float32)
        c = rng.normal(0, 0.5, size=shape).astype(np.float32)
        feeds.update(h_in=h, c_in=c)
        torch_in += [torch.from_numpy(h), torch.from_numpy(c)]
    with torch.no_grad():
        ref = actor(*torch_in)
    ref_logits = (ref[0] if isinstance(ref, tuple) else ref).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, feeds)[0]
    max_abs = float(np.abs(got - ref_logits).max())
    agree = float((got.argmax(1) == ref_logits.argmax(1)).mean())
    if max_abs > 1e-3 or agree < 1.0:
        raise AssertionError(f"ONNX parity failed: max|dlogit|={max_abs:.2e}, action agreement={agree:.3f}")
    return {"max_abs_logit_diff": max_abs, "action_agreement": agree, "n": n}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="run dir or model.zip")
    p.add_argument("--out", required=True)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--no-verify", action="store_true")
    a = p.parse_args(argv)
    export(a.model, a.out, a.opset, verify=not a.no_verify)


if __name__ == "__main__":
    main()
