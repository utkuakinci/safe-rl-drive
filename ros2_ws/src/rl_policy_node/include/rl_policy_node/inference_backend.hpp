#pragma once

#include <memory>
#include <string>
#include <vector>

namespace rl_policy_node
{

// Single-sample policy inference. Recurrent backends keep (h, c) internally; reset_state() is called
// on episode start. All buffers are pre-allocated at construction: infer() does no heap allocation
// on the hot path for the TensorRT backend and only ORT-internal ones for ONNX Runtime.
class InferenceBackend
{
public:
  virtual ~InferenceBackend() = default;
  virtual const std::vector<float> & infer(const std::vector<float> & obs) = 0;  // -> logits
  virtual void reset_state() = 0;
  virtual size_t obs_size() const = 0;
  virtual size_t num_actions() const = 0;
  virtual bool recurrent() const = 0;
  virtual std::string name() const = 0;
};

// kind: "onnxruntime" | "onnxruntime_cuda" | "tensorrt"
std::unique_ptr<InferenceBackend> make_backend(const std::string & kind, const std::string & model_path);

}  // namespace rl_policy_node
