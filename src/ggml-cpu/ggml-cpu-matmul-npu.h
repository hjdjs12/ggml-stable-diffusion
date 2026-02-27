/**
 * @file ggml-cpu-matmul-npu.h
 * @brief C API for NPU-accelerated matrix multiplication
 */

#ifndef GGML_CPU_MATMUL_NPU_H
#define GGML_CPU_MATMUL_NPU_H

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_compute_params;

/**
 * @brief NPU 加速的矩阵乘法（C 接口）
 * 
 * @param params 计算参数
 * @param dst 输出 tensor
 */
void ggml_compute_forward_mul_mat_npu(
    const struct ggml_compute_params* params,
    struct ggml_tensor* dst);

/**
 * @brief 初始化 NPU 子系统
 */
void ggml_npu_init(void);

/**
 * @brief 释放 NPU 资源
 */
void ggml_npu_free(void);

/**
 * @brief 检查是否可以使用 NPU 加速
 * 
 * @param src0 输入 tensor
 * @param src1 权重 tensor
 * @return 1 if NPU can be used, 0 otherwise
 */
int ggml_can_use_npu(const struct ggml_tensor* src0, const struct ggml_tensor* src1);

#ifdef __cplusplus
}
#endif

#endif // GGML_CPU_MATMUL_NPU_H
