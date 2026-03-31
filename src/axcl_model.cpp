#include "app/detail/axcl_model.hpp"
#include <cstdio>
#include <cstring>
#include <fstream>

namespace app {
namespace detail {

namespace {

bool read_file(const char* path, std::vector<uint8_t>& data) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    size_t len = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    data.resize(len);
    return f.read(reinterpret_cast<char*>(data.data()), len) && f.gcount() == static_cast<std::streamsize>(len);
}

} // anonymous namespace

AxclModel::AxclModel() = default;

AxclModel::~AxclModel() {
    release();
}

int AxclModel::load(const std::string& model_path) {
    if (!read_file(model_path.c_str(), model_buffer_)) {
        fprintf(stderr, "[AxclModel] read_file failed: %s\n", model_path.c_str());
        return -1;
    }

    void* dev_mem = nullptr;
    axclError ret = axclrtMalloc(&dev_mem, model_buffer_.size(), AXCL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != 0) {
        fprintf(stderr, "[AxclModel] axclrtMalloc failed: 0x%x\n", ret);
        return -1;
    }

    ret = axclrtMemcpy(dev_mem, model_buffer_.data(), model_buffer_.size(), AXCL_MEMCPY_HOST_TO_DEVICE);
    if (ret != 0) {
        fprintf(stderr, "[AxclModel] axclrtMemcpy failed: 0x%x\n", ret);
        axclrtFree(dev_mem);
        return -1;
    }

    ret = axclrtEngineLoadFromMem(dev_mem, model_buffer_.size(), &engine_handle_);
    axclrtFree(dev_mem); // safe to free after load

    if (ret != 0) {
        fprintf(stderr, "[AxclModel] axclrtEngineLoadFromMem failed: 0x%x\n", ret);
        engine_handle_ = 0;
        return -1;
    }

    return 0;
}

int AxclModel::prepare_io() {
    axclError ret = axclrtEngineGetIOInfo(engine_handle_, &io_info_);
    if (ret != 0) {
        fprintf(stderr, "[AxclModel] axclrtEngineGetIOInfo failed: 0x%x\n", ret);
        return -1;
    }

    int32_t group_count = 0;
    ret = axclrtEngineGetShapeGroupsCount(io_info_, &group_count);
    if (ret != 0 || group_count <= 0) {
        fprintf(stderr, "[AxclModel] axclrtEngineGetShapeGroupsCount failed or returned 0 groups\n");
        return -1;
    }
    group_count_ = group_count;

    // Create IO handle for group 0
    ret = axclrtEngineCreateIO(io_info_, &io_handle_);
    if (ret != 0) {
        fprintf(stderr, "[AxclModel] axclrtEngineCreateIO failed: 0x%x\n", ret);
        return -1;
    }

    auto input_num = axclrtEngineGetNumInputs(io_info_);
    auto output_num = axclrtEngineGetNumOutputs(io_info_);

    // Pre-allocate input buffers
    for (uint32_t i = 0; i < input_num; ++i) {
        TensorInfo info;
        info.index = static_cast<int>(i);
        info.size = axclrtEngineGetInputSizeByIndex(io_info_, 0, i);
        info.name = axclrtEngineGetInputNameByIndex(io_info_, i);

        axclrtEngineIODims dims;
        axclrtEngineGetInputDims(io_info_, 0, i, &dims);
        info.shape.resize(dims.dimCount);
        for (int32_t j = 0; j < dims.dimCount; ++j) info.shape[j] = dims.dims[j];

        void* dev_ptr = nullptr;
        ret = axclrtMalloc(&dev_ptr, info.size, AXCL_MEM_MALLOC_HUGE_FIRST);
        if (ret != 0) {
            fprintf(stderr, "[AxclModel] input malloc failed (index %d, size %zu): 0x%x\n", i, info.size, ret);
            return -1;
        }

        info.phy_addr = reinterpret_cast<uint64_t>(dev_ptr);
        info.vir_addr = malloc(info.size);
        memset(info.vir_addr, 0, info.size);

        ret = axclrtEngineSetInputBufferByIndex(io_handle_, i, dev_ptr, info.size);
        if (ret != 0) {
            fprintf(stderr, "[AxclModel] set input buffer failed: 0x%x\n", ret);
            return -1;
        }

        inputs_.push_back(info);
        dev_buffers_.push_back(dev_ptr);
    }

    // Pre-allocate output buffers
    for (uint32_t i = 0; i < output_num; ++i) {
        TensorInfo info;
        info.index = static_cast<int>(i);
        info.size = axclrtEngineGetOutputSizeByIndex(io_info_, 0, i);
        info.name = axclrtEngineGetOutputNameByIndex(io_info_, i);

        axclrtEngineIODims dims;
        axclrtEngineGetOutputDims(io_info_, 0, i, &dims);
        info.shape.resize(dims.dimCount);
        for (int32_t j = 0; j < dims.dimCount; ++j) info.shape[j] = dims.dims[j];

        void* dev_ptr = nullptr;
        ret = axclrtMalloc(&dev_ptr, info.size, AXCL_MEM_MALLOC_HUGE_FIRST);
        if (ret != 0) {
            fprintf(stderr, "[AxclModel] output malloc failed: 0x%x\n", ret);
            return -1;
        }

        info.phy_addr = reinterpret_cast<uint64_t>(dev_ptr);
        info.vir_addr = malloc(info.size);
        memset(info.vir_addr, 0, info.size);

        ret = axclrtEngineSetOutputBufferByIndex(io_handle_, i, dev_ptr, info.size);
        if (ret != 0) {
            fprintf(stderr, "[AxclModel] set output buffer failed: 0x%x\n", ret);
            return -1;
        }

        outputs_.push_back(info);
        dev_buffers_.push_back(dev_ptr);
    }

    return 0;
}

int AxclModel::inference(uint64_t context_id, const uint8_t* input_data, size_t input_size, std::vector<TensorInfo>& output_tensors) {
    if (!isValid()) return -1;
    if (inputs_.empty()) return -1;

    // Copy input: host → device
    // If input_data is non-null, use it; otherwise use the pre-filled vir_addr buffer.
    const uint8_t* src = input_data ? input_data : reinterpret_cast<const uint8_t*>(inputs_[0].vir_addr);
    size_t copy_size = input_data ? std::min(static_cast<size_t>(inputs_[0].size), input_size)
                                   : inputs_[0].size;
    axclrtMemcpy(reinterpret_cast<void*>(inputs_[0].phy_addr), src, copy_size,
                 AXCL_MEMCPY_HOST_TO_DEVICE);

    // Execute with the caller's context (bound to the current thread by AxclContext)
    axclError ret = axclrtEngineExecute(engine_handle_, context_id, 0, io_handle_);
    if (ret != 0) {
        fprintf(stderr, "[AxclModel] axclrtEngineExecute failed: 0x%x\n", ret);
        return -1;
    }

    // Copy output: device → host
    for (auto& out : outputs_) {
        axclrtMemcpy(out.vir_addr, reinterpret_cast<const void*>(out.phy_addr), out.size,
                     AXCL_MEMCPY_DEVICE_TO_HOST);
    }

    output_tensors = outputs_;
    return 0;
}

int AxclModel::num_inputs() const { return static_cast<int>(inputs_.size()); }
int AxclModel::num_outputs() const { return static_cast<int>(outputs_.size()); }
TensorInfo AxclModel::get_input(int idx) { return inputs_.at(idx); }
TensorInfo AxclModel::get_output(int idx) { return outputs_.at(idx); }

void AxclModel::release() {
    for (auto& in : inputs_) {
        if (in.vir_addr) free(in.vir_addr);
    }
    for (auto& out : outputs_) {
        if (out.vir_addr) free(out.vir_addr);
    }
    inputs_.clear();
    outputs_.clear();

    for (void* ptr : dev_buffers_) {
        if (ptr) axclrtFree(ptr);
    }
    dev_buffers_.clear();

    if (io_handle_) {
        axclrtEngineDestroyIO(io_handle_);
        io_handle_ = nullptr;
    }
    if (io_info_) {
        axclrtEngineDestroyIOInfo(io_info_);
        io_info_ = nullptr;
    }
    if (engine_handle_) {
        axclrtEngineUnload(engine_handle_);
        engine_handle_ = 0;
    }
    model_buffer_.clear();
}

bool AxclModel::isValid() const { return engine_handle_ != 0 && io_handle_ != nullptr; }

} // namespace detail
} // namespace app
