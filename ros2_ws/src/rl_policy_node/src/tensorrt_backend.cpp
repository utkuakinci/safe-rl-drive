// TensorRT 10 backend (compiled only with -DWITH_TENSORRT=ON).
// LSTM state stays on the GPU between steps: h_out/c_out are copied device-to-device into h_in/c_in.
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>

#include "rl_policy_node/inference_backend.hpp"

#define CUDA_CHECK(x)                                                                    \
  do {                                                                                   \
    cudaError_t err = (x);                                                               \
    if (err != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(err)); \
  } while (0)

namespace rl_policy_node
{

namespace
{
class TrtLogger : public nvinfer1::ILogger
{
  void log(Severity severity, const char * msg) noexcept override
  {
    if (severity <= Severity::kWARNING) std::cerr << "[TensorRT] " << msg << std::endl;
  }
};

size_t volume(const nvinfer1::Dims & d)
{
  size_t v = 1;
  for (int i = 0; i < d.nbDims; ++i) v *= static_cast<size_t>(d.d[i]);
  return v;
}
}  // namespace

class TensorRTBackend : public InferenceBackend
{
public:
  explicit TensorRTBackend(const std::string & path)
  {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open engine " + path);
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine_) throw std::runtime_error("engine deserialization failed");
    context_.reset(engine_->createExecutionContext());
    CUDA_CHECK(cudaStreamCreate(&stream_));

    // Fix the dynamic batch dimension to 1.
    auto obs_dims = engine_->getTensorShape("obs");
    obs_dims.d[0] = 1;
    context_->setInputShape("obs", obs_dims);
    recurrent_ = engine_->getNbIOTensors() == 6;
    if (recurrent_) {
      auto s = engine_->getTensorShape("h_in");
      s.d[1] = 1;
      context_->setInputShape("h_in", s);
      context_->setInputShape("c_in", s);
    }
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
      const char * name = engine_->getIOTensorName(i);
      size_t n = volume(context_->getTensorShape(name));
      void * ptr = nullptr;
      CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(float)));
      CUDA_CHECK(cudaMemset(ptr, 0, n * sizeof(float)));
      bufs_[name] = {ptr, n};
      context_->setTensorAddress(name, ptr);
    }
    obs_size_ = bufs_["obs"].second;
    logits_.assign(bufs_["logits"].second, 0.f);
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&pinned_obs_), obs_size_ * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&pinned_logits_), logits_.size() * sizeof(float)));
  }

  ~TensorRTBackend() override
  {
    for (auto & [_, b] : bufs_) cudaFree(b.first);
    cudaFreeHost(pinned_obs_);
    cudaFreeHost(pinned_logits_);
    cudaStreamDestroy(stream_);
  }

  const std::vector<float> & infer(const std::vector<float> & obs) override
  {
    if (obs.size() != obs_size_) throw std::invalid_argument("observation size mismatch");
    std::copy(obs.begin(), obs.end(), pinned_obs_);
    CUDA_CHECK(cudaMemcpyAsync(bufs_["obs"].first, pinned_obs_, obs_size_ * sizeof(float), cudaMemcpyHostToDevice, stream_));
    if (!context_->enqueueV3(stream_)) throw std::runtime_error("enqueueV3 failed");
    CUDA_CHECK(cudaMemcpyAsync(pinned_logits_, bufs_["logits"].first, logits_.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    if (recurrent_) {
      for (auto [src, dst] : {std::pair{"h_out", "h_in"}, std::pair{"c_out", "c_in"}}) {
        CUDA_CHECK(cudaMemcpyAsync(bufs_[dst].first, bufs_[src].first, bufs_[dst].second * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
      }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    std::copy(pinned_logits_, pinned_logits_ + logits_.size(), logits_.begin());
    return logits_;
  }

  void reset_state() override
  {
    if (!recurrent_) return;
    for (const char * n : {"h_in", "c_in"}) {
      CUDA_CHECK(cudaMemsetAsync(bufs_[n].first, 0, bufs_[n].second * sizeof(float), stream_));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

  size_t obs_size() const override { return obs_size_; }
  size_t num_actions() const override { return logits_.size(); }
  bool recurrent() const override { return recurrent_; }
  std::string name() const override { return "tensorrt"; }

private:
  TrtLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  cudaStream_t stream_{};
  std::map<std::string, std::pair<void *, size_t>> bufs_;
  float * pinned_obs_ = nullptr;
  float * pinned_logits_ = nullptr;
  size_t obs_size_ = 0;
  std::vector<float> logits_;
  bool recurrent_ = false;
};

std::unique_ptr<InferenceBackend> make_tensorrt_backend(const std::string & path)
{
  return std::make_unique<TensorRTBackend>(path);
}

}  // namespace rl_policy_node
