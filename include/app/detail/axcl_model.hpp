#pragma once

#include "axcl_context.hpp"
#include <cstdint>
#include <string>
#include <vector>
#include <memory>

namespace app {
namespace detail {

// Tensor descriptor
struct TensorInfo {
    int index;
    std::string name;
    std::vector<int> shape;  // e.g. {1, 640, 640, 3}
    size_t size;             // bytes
    void* vir_addr;         // host virtual address for H2D/D2H
    uint64_t phy_addr;      // device physical address
};

// RAII wrapper for AXCL model: load, inference, cleanup.
// Each instance is tied to one engine handle. For multi-threaded use,
// create one AxclModel + one AxclContext per thread.
//
// Usage:
//   AxclModel model;
//   model.load("model.file");
//   model.prepare_io();
//
//   // In inference thread:
//   AxclContext ctx(model.engine_handle());  // creates engine context
//   model.inference(data, size, outputs);
//
class AxclModel {
public:
    AxclModel();
    ~AxclModel();

    // Load model from file (reads into memory, copies to device).
    int load(const std::string& model_path);

    // Pre-allocate IO buffers (input/output device memory).
    int prepare_io();

    // Synchronous inference: H2D copy + execute + D2H copy.
    // Caller must create AxclContext in the current thread first.
    // Fills output_tensors with output TensorInfo (vir_addr populated).
    // Pass ctx.context_id() as the context_id parameter.
    int inference(uint64_t context_id, const uint8_t* input_data, size_t input_size,
                  std::vector<TensorInfo>& output_tensors);

    int num_inputs() const;
    int num_outputs() const;
    TensorInfo get_input(int idx);
    TensorInfo get_output(int idx);

    uint64_t engine_handle() const { return engine_handle_; }
    bool isValid() const;

    // Release all resources.
    void release();

    // non-copyable
    AxclModel(const AxclModel&) = delete;
    AxclModel& operator=(const AxclModel&) = delete;

private:
    uint64_t engine_handle_ = 0;
    axclrtEngineIOInfo io_info_ = nullptr;  // void* = axclrtEngineIOInfo
    axclrtEngineIO io_handle_ = nullptr;   // void* = axclrtEngineIO
    std::vector<TensorInfo> inputs_;
    std::vector<TensorInfo> outputs_;
    std::vector<void*> dev_buffers_;       // track device allocations for cleanup
    std::vector<uint8_t> model_buffer_;    // kept alive for duration
    int32_t group_count_ = 0;
};

} // namespace detail
} // namespace app
