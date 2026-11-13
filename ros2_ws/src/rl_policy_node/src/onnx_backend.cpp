#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "rl_policy_node/inference_backend.hpp"

namespace rl_policy_node
{

class OnnxRuntimeBackend : public InferenceBackend
{
public:
  OnnxRuntimeBackend(const std::string & path, bool use_cuda)
  : env_(ORT_LOGGING_LEVEL_WARNING, "rl_policy_node"), use_cuda_(use_cuda)
  {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);  // deterministic latency on a shared real-time CPU
    opts.SetInterOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (use_cuda) {
      OrtCUDAProviderOptions cuda{};
      opts.AppendExecutionProvider_CUDA(cuda);
    }
    session_ = std::make_unique<Ort::Session>(env_, path.c_str(), opts);

    Ort::AllocatorWithDefaultOptions alloc;
    for (size_t i = 0; i < session_->GetInputCount(); ++i) {
      in_names_.emplace_back(session_->GetInputNameAllocated(i, alloc).get());
    }
    for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
      out_names_.emplace_back(session_->GetOutputNameAllocated(i, alloc).get());
    }
    for (auto & n : in_names_) in_ptrs_.push_back(n.c_str());
    for (auto & n : out_names_) out_ptrs_.push_back(n.c_str());

    // obs: [B, V, F] with dynamic batch -> fix B = 1
    obs_shape_ = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    obs_shape_[0] = 1;
    obs_.assign(obs_shape_[1] * obs_shape_[2], 0.f);
    auto logit_shape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    logits_.assign(logit_shape.back(), 0.f);

    recurrent_ = in_names_.size() == 3;
    if (recurrent_) {
      state_shape_ = session_->GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();  // [L, B, H]
      state_shape_[1] = 1;
      h_.assign(state_shape_[0] * state_shape_[2], 0.f);
      c_ = h_;
    }
  }

  const std::vector<float> & infer(const std::vector<float> & obs) override
  {
    if (obs.size() != obs_.size()) throw std::invalid_argument("observation size mismatch");
    std::copy(obs.begin(), obs.end(), obs_.begin());
    std::vector<Ort::Value> inputs;
    inputs.reserve(3);
    inputs.push_back(Ort::Value::CreateTensor<float>(mem_, obs_.data(), obs_.size(), obs_shape_.data(), obs_shape_.size()));
    if (recurrent_) {
      inputs.push_back(Ort::Value::CreateTensor<float>(mem_, h_.data(), h_.size(), state_shape_.data(), state_shape_.size()));
      inputs.push_back(Ort::Value::CreateTensor<float>(mem_, c_.data(), c_.size(), state_shape_.data(), state_shape_.size()));
    }
    auto out = session_->Run(Ort::RunOptions{nullptr}, in_ptrs_.data(), inputs.data(), inputs.size(),
                             out_ptrs_.data(), out_ptrs_.size());
    const float * l = out[0].GetTensorData<float>();
    std::copy(l, l + logits_.size(), logits_.begin());
    if (recurrent_) {
      const float * h = out[1].GetTensorData<float>();
      const float * c = out[2].GetTensorData<float>();
      std::copy(h, h + h_.size(), h_.begin());
      std::copy(c, c + c_.size(), c_.begin());
    }
    return logits_;
  }

  void reset_state() override
  {
    std::fill(h_.begin(), h_.end(), 0.f);
    std::fill(c_.begin(), c_.end(), 0.f);
  }
  size_t obs_size() const override { return obs_.size(); }
  size_t num_actions() const override { return logits_.size(); }
  bool recurrent() const override { return recurrent_; }
  std::string name() const override { return use_cuda_ ? "onnxruntime-cuda" : "onnxruntime-cpu"; }

private:
  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
  Ort::MemoryInfo mem_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<std::string> in_names_, out_names_;
  std::vector<const char *> in_ptrs_, out_ptrs_;
  std::vector<int64_t> obs_shape_, state_shape_;
  std::vector<float> obs_, h_, c_, logits_;
  bool recurrent_ = false;
  bool use_cuda_ = false;
};

#ifdef WITH_TENSORRT
std::unique_ptr<InferenceBackend> make_tensorrt_backend(const std::string & path);
#endif

std::unique_ptr<InferenceBackend> make_backend(const std::string & kind, const std::string & path)
{
  if (kind == "onnxruntime") return std::make_unique<OnnxRuntimeBackend>(path, false);
  if (kind == "onnxruntime_cuda") return std::make_unique<OnnxRuntimeBackend>(path, true);
#ifdef WITH_TENSORRT
  if (kind == "tensorrt") return make_tensorrt_backend(path);
#endif
  throw std::invalid_argument("unknown / not compiled backend: " + kind);
}

}  // namespace rl_policy_node
