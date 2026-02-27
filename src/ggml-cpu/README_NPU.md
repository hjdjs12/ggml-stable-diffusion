# GGML NPU Acceleration for Stable Diffusion

本文档说明如何使用 RK3588 NPU 加速 GGML 矩阵乘法运算。

## 功能概述

本实现为 GGML 的 `ggml_compute_forward_mul_mat` 函数添加了 NPU 加速支持，可以显著提升 Stable Diffusion 模型的推理速度。

### 支持的数据类型

- **GGML_TYPE_Q8_0**: INT8 量化格式（最优性能）
- **GGML_TYPE_F32**: 32位浮点数（动态量化为 INT8）
- **GGML_TYPE_F16**: 16位浮点数（动态量化为 INT8）
- **GGML_TYPE_BF16**: BFloat16（动态量化为 INT8）

### 加速策略

1. **Q8_0 类型**: 
   - 先反量化为 FP32
   - 量化为 NPU INT8 格式（带 scale）
   - 使用 NPU 执行 INT8 矩阵乘法
   - CPU 执行反量化和累加

2. **FP32/FP16/BF16 类型**:
   - 直接量化为 NPU INT8 格式
   - 使用 NPU 执行 INT8 矩阵乘法
   - CPU 执行反量化和累加

3. **CPU & NPU 并行**:
   - 使用双缓冲区机制
   - NPU 计算当前 Block 时，CPU 处理上一个 Block 的结果
   - 通过 NEON 指令加速 CPU 端的反量化

## 编译选项

### 启用 NPU 支持

在 CMakeLists.txt 中添加：

```cmake
option(GGML_USE_NPU "Enable NPU acceleration" ON)

if(GGML_USE_NPU)
    add_definitions(-DGGML_USE_NPU)
    
    # 添加 NPU 源文件
    set(GGML_CPU_NPU_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/ggml/src/ggml-cpu/ggml-cpu-matmul-npu.cpp
    )
    
    # 添加 RK NPU 驱动目录
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ggml/rknpu/user-driver)
    
    # 链接 RK NPU 库
    target_sources(ggml PRIVATE ${GGML_CPU_NPU_SOURCES})
    target_link_libraries(ggml PRIVATE pthread dl)
endif()
```

### 编译命令

```bash
cd build
cmake .. -DGGML_USE_NPU=ON
make -j$(nproc)
```

## 使用方法

### 1. 初始化 NPU

在程序启动时调用：

```c
// 初始化 NPU 子系统
ggml_npu_init();
```

### 2. 正常使用 GGML

无需修改现有代码，GGML 会自动判断是否使用 NPU：

```c
// 创建矩阵乘法计算图
struct ggml_tensor* result = ggml_mul_mat(ctx, weight, input);

// 执行计算（自动使用 NPU 加速）
ggml_graph_compute_with_ctx(ctx, graph, n_threads);
```

### 3. 释放资源

程序退出前调用：

```c
// 释放 NPU 资源
ggml_npu_free();
```

## 性能优化建议

### 1. 矩阵维度

NPU 加速在以下场景效果最佳：
- M（批次大小）≥ 4
- K（共享维度）≥ 128
- N（输出维度）≥ 128

小矩阵会自动回退到 CPU 计算。

### 2. 内存对齐

确保 tensor 数据按页对齐（4KB），以优化 DMA 传输：

```c
// 使用对齐的内存分配
void* aligned_data = aligned_alloc(4096, tensor_size);
```

### 3. Domain 配置

根据模型大小配置 IOMMU 域：

```cpp
// config.hpp 中配置
#define WEIGHT_SIZE (2560UL * 1024 * 1024)  // 每个 domain 2.5GB
#define DOMAIN_SIZE (3072UL * 1024 * 1024)  // 总域大小 3GB
```

### 4. 并行策略

调整批量大小以优化流水线并行：

```cpp
// rk-mem.hpp 中配置
int TASKS_LOCAL_PER_NUM = 2;  // 每次提交 2-4 个任务
const int LOOKAHEAD = 12;      // Cache 预取距离
```

## 调试选项

### 性能分析

启用性能统计（取消注释 printf）：

```cpp
// ggml-cpu-matmul-npu.cpp
// 取消注释以下行
// printf("[Matmul]: M:%d K:%d N:%d tasks_size:%ld all_time:%ld\n", ...);
```

### 日志输出

启用详细日志：

```cpp
// config.hpp
// 取消注释调试输出
// printf("[RKMEM]: ...\n");
```

## 故障排除

### 1. NPU 初始化失败

**问题**: `ggml_npu_init()` 返回错误

**解决方案**:
- 检查 `/dev/rknpu` 设备是否存在
- 检查用户是否有权限访问 NPU 设备
- 确认 RKNPU 驱动已加载：`lsmod | grep rknpu`

### 2. 内存不足

**问题**: NPU 分配内存失败

**解决方案**:
- 减小 `DOMAIN_SIZE` 配置
- 减少同时运行的模型数量
- 使用 `ulimit -l unlimited` 允许锁定更多内存

### 3. 计算结果不正确

**问题**: NPU 结果与 CPU 不一致

**可能原因**:
- 量化精度损失（INT8 vs FP32）
- 内存布局不匹配
- Cache 一致性问题

**解决方案**:
- 检查 `flush_cache()` 和 `invalid_cache()` 是否正确调用
- 验证 `feature_data()` 布局转换是否正确
- 对比 CPU 和 NPU 的中间结果

### 4. 性能不佳

**问题**: NPU 速度慢于 CPU

**可能原因**:
- 矩阵太小（转换开销 > 计算加速）
- DMA 传输瓶颈
- 双缓冲区未充分利用

**解决方案**:
- 调大批次大小（增加 M 维度）
- 减少任务切分（增加 BLOCK_SHARED 和 BLOCK_WEIGHT）
- 调整 `TASKS_LOCAL_PER_NUM` 参数

## 技术原理

### 1. 数据流

```
GGML Tensor (Q8_0/FP32/FP16)
    ↓ [CPU] 反量化/转换
FP32 Matrix
    ↓ [CPU] 动态量化
MatrixQ (INT8 + scales)
    ↓ [DMA] 传输到 NPU
NPU CBUF (INT8)
    ↓ [NPU] 矩阵乘法 (CNA)
NPU Output (INT32)
    ↓ [DMA] 传输回 CPU
CPU Memory (INT32)
    ↓ [CPU+NEON] 反量化
FP32 Result
    ↓ [CPU] 写回 
GGML Tensor (FP32)
```

### 2. 内存布局

#### GGML 格式（Q8_0）
```
block_q8_0[i].delta  → scale (FP16)
block_q8_0[i].qs[j]  → quantized values (INT8 × 32)
```

#### NPU 格式（MatrixQ）
```
data[feature_data(M, 16, k, m)]  → INT8 value
scale[m * K / GS + k / GS]       → FP32 scale
```

#### 布局转换 (feature_data)
```
输入: [m=0, k=0..K-1], [m=1, k=0..K-1], ...
输出: [k=0..15, m=0..M-1], [k=16..31, m=0..M-1], ...
```

### 3. 双缓冲机制

```
时间轴:
t0: NPU 处理 Block 0 → Buffer[0]
t1: CPU 读取 Buffer[0]，NPU 处理 Block 1 → Buffer[1]
t2: CPU 读取 Buffer[1]，NPU 处理 Block 2 → Buffer[0]
...（循环）
```

## 参考资料

- [GGML 文档](https://github.com/ggerganov/ggml)
- [RK3588 NPU 编程指南](./docs/rknpu_programming_guide.md)
- [量化算法说明](./docs/quantization.md)
- [性能优化指南](./docs/performance.md)

## 许可证

本代码遵循 GNU General Public License v3.0 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题，请提交 GitHub Issue。
