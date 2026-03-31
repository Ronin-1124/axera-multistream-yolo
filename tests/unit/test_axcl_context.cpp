// Unit test for AxclContext and AxclModel
// Run with hardware: ./test_axcl_context <model_file>

#include "app/detail/axcl_context.hpp"
#include "app/detail/axcl_model.hpp"
#include <axcl.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>

using namespace app::detail;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_file>\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];

    // 0. Initialize AXCL (required before any other AXCL call)
    axclError ret = axclInit(nullptr);
    if (ret != 0) {
        fprintf(stderr, "FAIL: axclInit failed: 0x%x\n", ret);
        return 1;
    }
    printf("PASS: axclInit\n");

    // 0b. Get device list and set device (required before EngineInit)
    axclrtDeviceList lst;
    ret = axclrtGetDeviceList(&lst);
    if (ret != 0 || lst.num == 0) {
        fprintf(stderr, "FAIL: axclrtGetDeviceList failed: 0x%x, num=%d\n", ret, lst.num);
        return 1;
    }
    printf("PASS: found %d AXCL device(s)\n", lst.num);

    ret = axclrtSetDevice(lst.devices[0]);
    if (ret != 0) {
        fprintf(stderr, "FAIL: axclrtSetDevice failed: 0x%x\n", ret);
        return 1;
    }
    printf("PASS: set device %d\n", lst.devices[0]);

    // 0b. Initialize AXCL engine
    ret = axclrtEngineInit(AXCL_VNPU_DISABLE);
    if (ret != 0) {
        fprintf(stderr, "FAIL: axclrtEngineInit failed: 0x%x\n", ret);
        return 1;
    }
    printf("PASS: axclrtEngineInit\n");

    // 1. Test invalid construction (null engine handle)
    {
        AxclContext ctx(0);
        if (ctx.isValid()) {
            fprintf(stderr, "FAIL: null engine handle should create invalid context\n");
            axclFinalize();
            return 1;
        }
        printf("PASS: null engine handle → invalid context\n");
    }

    // 2. Load model and create context
    AxclModel model;
    int err = model.load(model_path);
    if (err != 0) {
        fprintf(stderr, "FAIL: model.load failed with code %d\n", err);
        axclFinalize();
        return 1;
    }
    printf("PASS: model loaded (engine handle: 0x%lx)\n", model.engine_handle());

    err = model.prepare_io();
    if (err != 0) {
        fprintf(stderr, "FAIL: model.prepare_io failed with code %d\n", err);
        axclFinalize();
        return 1;
    }
    printf("PASS: IO buffers prepared (inputs: %d, outputs: %d)\n",
           model.num_inputs(), model.num_outputs());

    // 3. Create context in main thread
    AxclContext ctx(model.engine_handle());
    if (!ctx.isValid()) {
        fprintf(stderr, "FAIL: context creation failed\n");
        axclFinalize();
        return 1;
    }
    printf("PASS: context created (id: 0x%lx)\n", ctx.context_id());

    // 4. Create context in a worker thread (mimics per-thread context creation)
    std::atomic<bool> worker_ok{false};
    uint64_t worker_engine_handle = model.engine_handle();
    int worker_device_id = lst.devices[0];
    std::thread worker([&]() {
        // Each thread needs axclInit + axclrtSetDevice before AXCL calls
        axclrtSetDevice(worker_device_id);
        axclrtSetDevice(worker_device_id);

        // Create device context and bind to thread
        axclrtContext dev_ctx = nullptr;
        axclError ctx_ret = axclrtCreateContext(&dev_ctx, worker_device_id);
        printf("  [worker] axclrtCreateContext: ret=0x%x ctx=%p\n", ctx_ret, dev_ctx);
        if (ctx_ret != 0 || dev_ctx == nullptr) {
            fprintf(stderr, "FAIL: axclrtCreateContext failed: 0x%x\n", ctx_ret);
            return;
        }
        ctx_ret = axclrtSetCurrentContext(dev_ctx);
        printf("  [worker] axclrtSetCurrentContext: ret=0x%x\n", ctx_ret);
        if (ctx_ret != 0) {
            fprintf(stderr, "FAIL: axclrtSetCurrentContext failed: 0x%x\n", ctx_ret);
            return;
        }

        // Now create engine context
        uint64_t worker_ctx_id = 0;
        ctx_ret = axclrtEngineCreateContext(worker_engine_handle, &worker_ctx_id);
        printf("  [worker] axclrtEngineCreateContext: ret=0x%x ctx_id=0x%lx\n", ctx_ret, worker_ctx_id);
        if (ctx_ret != 0) {
            fprintf(stderr, "FAIL: worker axclrtEngineCreateContext failed: 0x%x\n", ctx_ret);
            return;
        }
        if (worker_ctx_id != 0) {
            printf("PASS: worker thread context created (id: 0x%lx)\n", worker_ctx_id);
            worker_ok = true;
        } else {
            fprintf(stderr, "FAIL: worker thread context id is 0\n");
        }
    });
    worker.join();

    if (!worker_ok) {
        axclFinalize();
        return 1;
    }

    // 5. Single inference test (main thread context)
    {
        auto input_info = model.get_input(0);

        std::vector<uint8_t> input_data(input_info.size, 128);
        std::vector<TensorInfo> outputs;

        auto start = std::chrono::high_resolution_clock::now();
        ret = model.inference(ctx.context_id(), input_data.data(), input_data.size(), outputs);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        if (ret != 0) {
            fprintf(stderr, "FAIL: inference failed with code %d\n", ret);
            axclFinalize();
            return 1;
        }
        printf("PASS: single inference in %.2f ms (context 0x%lx)\n", ms, ctx.context_id());
    }

    printf("\nAll tests passed!\n");
    axclFinalize();
    return 0;
}
