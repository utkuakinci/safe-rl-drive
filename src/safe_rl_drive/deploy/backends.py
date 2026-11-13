"""Inference backends with a common numpy interface: ``infer(feeds) -> [logits, (h_out, c_out)]``.

``feeds`` is ``{"obs": [B,V,F]}`` plus ``{"h_in", "c_in": [L,B,H]}`` for recurrent policies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class Backend:
    name = "base"
    precision = "fp32"
    device = "cpu"
    model_file: Path | None = None

    def infer(self, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        raise NotImplementedError

    def sync(self):
        pass


class TorchBackend(Backend):
    name = "PyTorch"

    def __init__(self, model_path, device: str = "cpu", fp16: bool = False, threads: int = 1):
        import torch

        from safe_rl_drive.deploy.export_onnx import build_actor
        from safe_rl_drive.policy import load_model

        torch.set_num_threads(threads)
        self.torch = torch
        self.actor, self.info = build_actor(load_model(model_path))
        self.dtype = torch.float16 if fp16 else torch.float32
        self.actor = self.actor.to(device=device, dtype=self.dtype)
        self.device, self.precision = device, "fp16" if fp16 else "fp32"
        p = Path(model_path)
        self.model_file = p / "model.zip" if p.is_dir() else p

    def infer(self, feeds):
        t = self.torch
        with t.inference_mode():
            args = [
                t.from_numpy(feeds[k]).to(self.device, self.dtype)
                for k in ("obs", "h_in", "c_in")
                if k in feeds
            ]
            out = self.actor(*args)
            out = out if isinstance(out, tuple) else (out,)
            return [o.float().cpu().numpy() for o in out]

    def sync(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()


class OrtBackend(Backend):
    name = "ONNX Runtime"

    def __init__(self, onnx_path, provider: str = "CPUExecutionProvider", threads: int = 1):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=[provider])
        self.inputs = [i.name for i in self.sess.get_inputs()]
        self.device = "cuda" if "CUDA" in provider else "cpu"
        self.model_file = Path(onnx_path)

    def infer(self, feeds):
        return self.sess.run(None, {k: feeds[k] for k in self.inputs})


class TensorRTBackend(Backend):
    """TensorRT 10 engine runner. Device buffers are torch CUDA tensors (no pycuda dependency)."""

    name = "TensorRT"

    def __init__(self, engine_path, precision: str = "fp16"):
        import tensorrt as trt
        import torch

        self.trt, self.torch = trt, torch
        self.precision, self.device = precision, "cuda"
        self.model_file = Path(engine_path)
        logger = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.stream = torch.cuda.Stream()
        self._bufs: dict[str, torch.Tensor] = {}

    def _buffer(self, name, shape, dtype):
        buf = self._bufs.get(name)
        if buf is None or tuple(buf.shape) != tuple(shape):
            buf = self.torch.empty(tuple(shape), dtype=dtype, device="cuda")
            self._bufs[name] = buf
        return buf

    def infer(self, feeds):
        t = self.torch
        with t.cuda.stream(self.stream):
            for n in self.inputs:
                host = t.from_numpy(np.ascontiguousarray(feeds[n]))
                dev = self._buffer(n, host.shape, t.float32)
                dev.copy_(host, non_blocking=True)
                self.ctx.set_input_shape(n, tuple(host.shape))
                self.ctx.set_tensor_address(n, dev.data_ptr())
            for n in self.outputs:
                dev = self._buffer(n, tuple(self.ctx.get_tensor_shape(n)), t.float32)
                self.ctx.set_tensor_address(n, dev.data_ptr())
            self.ctx.execute_async_v3(self.stream.cuda_stream)
            outs = [self._bufs[n].cpu() for n in self.outputs]
        self.stream.synchronize()
        return [o.numpy() for o in outs]
