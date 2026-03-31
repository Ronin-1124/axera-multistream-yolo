# axera-multistream-yolo

基于 Radxa AX-M1 (Rock5b+ Host)的多路视频识别项目

## 硬件环境

- **DEVICE**：Radxa AX-M1（Rock 5B+）
- **NPU**：AX650N M.2 AI 加速卡（AXCL SDK V3.6.5）
- **解码**：FFmpeg 7.1（`h264_axdec` / `hevc_axdec`）

## 项目结构

```
axera-multistream-yolo/
├── src/                     # 核心实现
│   ├── inference_engine.cpp # NPU 推理封装（letterbox + AXCL + NMS）
│   ├── pipeline.cpp         # StreamWorker × N 编排
│   ├── stream_source.cpp    # FFmpeg 硬件解码（RTSP / 本地文件）
│   └── ...
├── include/app/             # 头文件
├── tools/npu_bench.cpp      # NPU 性能基准测试工具
├── configs/                 # JSON 配置文件
└── yolo26_demo/             # AXCL SDK 参考实现（仅供查阅）
```

## 快速开始

### 编译

```bash
mkdir build && cd build
cmake .. && make -j$(nproc)
```

### 配置

编辑 `configs/app_config.json`：

```json
{
  "streams": [
    { "id": 0, "url": "./data/test_videos_360P/test1.mp4", "enabled": true },
    { "id": 1, "url": "rtsp://user:pass@192.168.1.100:554/stream", "enabled": true }
  ],
  "model": { "path": "./models/yolo26n.axmodel", "input_h": 640, "input_w": 640 },
  "output": { "display": true, "mosaic_cols": 2, "queue_depth": 4 }
}
```

### 运行

```bash
LD_LIBRARY_PATH="/usr/lib/axcl/ffmpeg:/usr/lib/axcl:$LD_LIBRARY_PATH" \
./build/axera_multistream_yolo --config ./configs/app_config.json
```

`display=false` 时仅输出帧率监控日志：

```
[Collector] S0:14.0 S1:13.0 S2:14.0 S3:13.0 fps (1s)
[Collector] S0:15.0 S1:14.0 S2:14.0 S3:14.0 fps (1s)
```

## NPU 性能测试工具

```bash
# 阶段时间分解（H2D / NPU kernel / D2H / Post）
./build/tools/npu_bench <model> <image> --stages [N]

# 线程扩展性扫描（1..N 线程）
./build/tools/npu_bench <model> <image> --sweep <max_threads> <iters>

# 多线程吞吐测试
./build/tools/npu_bench <model> <image> --mt <threads> <iters>
```

示例（SDK 层阶段分解）：

```
  Stage breakdown (SDK)  (100 iterations, 640x640 input)
               avg       min       max
  Pre          5.69      3.82     16.86  ms
  H2D          3.27      2.94      3.83  ms
  NPU          1.97      1.75      2.20  ms
  D2H          6.47      5.38      7.51  ms
  Post         5.31      2.20     12.31  ms
  Total       22.71  ms
```

## 架构

**1:1 固定配比**：每路 = 1 读取线程 + 1 推理线程，完全独立。

```
StreamWorker × N（每路独立）
  ├─ StreamSource（FFmpeg 解码）→ ThreadSafeQueue → InferenceThread
  └─────────────────────────────────────────────────────► ResultQueue
                                                         │
                                                   ResultCollector
                                                         └─ MosaicRenderer → imshow
```

## 性能数据（YOLO26n，640×640）

| 配置 | 帧率 |
|------|------|
| 单路端到端 | ~25 fps |
| 8 路并发（mosaic 显示） | 53-57 fps mosaic |
| AXCL 多线程扩展上限 | ~2.6x（2 线程后停滞） |
