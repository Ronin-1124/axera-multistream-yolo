#pragma once

#include <cstdint>
#include <axcl.h>

namespace app {
namespace detail {

// RAII wrapper for AXCL engine context.
//
// IMPORTANT: There are TWO independent context systems in AXCL:
//
// 1. Device Context API:
//    axclrtCreateContext(deviceId) → axclrtContext (void*)
//    axclrtSetCurrentContext(ctx)  ← binds context to thread
//    axclrtDestroyContext(ctx)
//
// 2. Engine Context API:
//    axclrtEngineCreateContext(modelId) → uint64_t context_id
//    axclrtEngineExecute(modelId, context_id, ...) ← context binding handled internally
//
// This class wraps the ENGINE context system (system 2). Do NOT mix with
// axclrtSetCurrentContext / axclrtCreateContext.
//
// Context is created from engine handle and passed directly to axclrtEngineExecute.
// No explicit thread binding is needed — axclrtEngineExecute handles it.
class AxclContext {
public:
    // Construct: create context from engine handle.
    // Must be called from the thread that will run inference.
    // Returns 0 on success, negative on failure.
    explicit AxclContext(uint64_t engine_handle)
        : engine_handle_(engine_handle)
        , context_id_(0)
    {
        if (engine_handle == 0) return;

        axclError ret = axclrtEngineCreateContext(engine_handle, &context_id_);
        if (ret != 0) {
            context_id_ = 0;
        }
    }

    ~AxclContext() {
        // axclrtEngineCreateContext creates a context tied to the engine.
        // The context is freed when axclrtEngineUnload is called on the engine.
        // There is no separate axclrtEngineDestroyContext API.
        context_id_ = 0;
        engine_handle_ = 0;
    }

    // non-copyable, movable
    AxclContext(const AxclContext&) = delete;
    AxclContext& operator=(const AxclContext&) = delete;
    AxclContext(AxclContext&&) = delete;
    AxclContext& operator=(AxclContext&&) = delete;

    bool isValid() const { return context_id_ != 0; }
    uint64_t context_id() const { return context_id_; }
    uint64_t engine_handle() const { return engine_handle_; }

private:
    uint64_t engine_handle_;
    uint64_t context_id_;
};

} // namespace detail
} // namespace app
