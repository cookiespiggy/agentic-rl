# 量化与 ONNX 推理基准 v1

- 任务：`intent`　max_length=192
- 单请求重复 30 次（预热 3 次不计入），批吞吐重复 5 次
- 硬件：macOS-26.6.2-arm64-arm-64bit / arm64
- torch `2.14.0`　onnxruntime `1.30.0`

> ⚠️ 量化与 ONNX 都是 CPU 侧优化，不能直接与 MPS 基线比；报告分「设备差异」与「同设备形态差异」两组

## 单请求延迟（毫秒，p50 长度档）

| 变体 | P50 | P95 | P99 | 体积 MB |
|---|---|---|---|---|
| `torch-mps-fp32` | 6.783 | 7.549 | 7.562 | 390.6 |
| `torch-cpu-fp32` | 11.289 | 11.52 | 11.59 | 390.6 |
| `onnx-cpu-fp32` | 5.153 | 5.294 | 5.324 | 390.3 |
| `onnx-cpu-int8` | 4.043 | 4.418 | 4.495 | 98.3 |

## 批吞吐

| 变体 | 批大小 | 批延迟 ms | 单样本 ms | QPS |
|---|---|---|---|---|
| `torch-mps-fp32` | 1 | 6.952 | 6.952 | 143.8 |
| `torch-mps-fp32` | 8 | 11.616 | 1.452 | 688.7 |
| `torch-mps-fp32` | 32 | 29.589 | 0.925 | 1081.5 |
| `torch-cpu-fp32` | 1 | 11.413 | 11.413 | 87.6 |
| `torch-cpu-fp32` | 8 | 28.987 | 3.623 | 276.0 |
| `torch-cpu-fp32` | 32 | 85.893 | 2.684 | 372.6 |
| `onnx-cpu-fp32` | 1 | 5.208 | 5.208 | 192.0 |
| `onnx-cpu-fp32` | 8 | 18.957 | 2.37 | 422.0 |
| `onnx-cpu-fp32` | 32 | 64.787 | 2.025 | 493.9 |
| `onnx-cpu-int8` | 1 | 3.756 | 3.756 | 266.3 |
| `onnx-cpu-int8` | 8 | 19.985 | 2.498 | 400.3 |
| `onnx-cpu-int8` | 32 | 72.086 | 2.253 | 443.9 |

## 量化一致性校验（int8 vs fp32）

| 变体 | 样本数 | argmax 一致率 | 最大 logit 偏差 |
|---|---|---|---|
| `onnx-cpu-int8` | 96 | 0.9375 | 2.2743 |

> 量化基准只报「更快更小」是不负责任的，必须同时报精度损失。


## 分组对比

### A 组：设备差异（同 fp32）

- MPS `6.783 ms` vs CPU `11.289 ms` → **MPS 快 1.66 倍**

### B 组：同设备（CPU）上的形态差异

| 变体 | P50 ms | 相对 torch-cpu-fp32 | 批32 QPS | 体积 MB |
|---|---|---|---|---|
| `torch-cpu-fp32` | 11.289 | 1.0x | 372.6 | 390.6 |
| `onnx-cpu-fp32` | 5.153 | 2.19x | 493.9 | 390.3 |
| `onnx-cpu-int8` | 4.043 | 2.79x | 443.9 | 98.3 |

## 不可用的变体

（完整原因见 `reports/quantization_benchmark.json`）

- `torch-cpu-int8`：RuntimeError: Didn't find engine for operation quantized::linear_prepack NoQEngine

