/**
 * @file ggml-cpu-matmul-npu.cpp
 * @brief GGML Matrix Multiplication with NPU Acceleration (Simplified Implementation)
 * 
 * Architecture:
 * - src0 = weight tensor (any type, converted to Q8_0)
 * - src1 = input tensor (FP32, converted to Q8_0)
 * 
 * Implementation Strategy:
 * - All tensors are converted to Q8_0 format (NPU only supports INT8)
 * - Uses CPU & NPU parallel execution with double buffering
 * - NPU worker thread (npu_work) handles async NPU submission
 * - CPU dequantizes INT32 results to FP32 while NPU computes next batch
 * - NEON SIMD instructions for efficient dequantization
 * - Cache prefetching for improved memory bandwidth
 * 
 * Reference Architecture:
 * - Based on rk3588-npu matrix multiplication implementation
 * - Supports block splitting: BLOCK_WEIGHT (256) × BLOCK_SHARED (4096)
 * - Double buffering: buffer_free[2] for CPU/NPU synchronization
 * - Task batching: TASKS_LOCAL_PER_NUM (2-3) tasks per submission
 * 
 * NOTE: This is a SIMPLIFIED version. Full implementation requires:
 * 1. Complete RKNPU driver integration (rknpu-ioctl.h)
 * 2. DMA memory allocation via Memory/Domain classes
 * 3. gen_matmul_int8() for NPU register command generation
 * 4. ioctl(DRM_IOCTL_RKNPU_SUBMIT) for actual NPU execution
 * 5. Cache coherency management (flush_cache/invalid_cache via cache_manager)
 * 
 * Current Status:
 * - Architecture and data flow are correct
 * - Task splitting and double buffering implemented
 * - NPU submission is PLACEHOLDER (rknpu_matmul returns dummy data)
 * - Requires full driver/hardware support to actually execute on NPU
 */

#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "ggml-quants.h"
#include "ggml-cpu-matmul-npu.h"
#include "../../rknpu/user-driver/config.hpp"
#include "../../rknpu/user-driver/include/rk-mem.hpp"
#include "../../rknpu/user-driver/include/npu_matmul.h"
#include "../../rknpu/user-driver/include/npu_interface.h"
#include <vector>
#include <cstring>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <tuple>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// ============================================================================
// Global State
// ============================================================================

static bool g_npu_initialized = false;
static int g_npu_fd = -1;

// ============================================================================
// Q8_0 Path: CPU & NPU Parallel Computing (Double Buffering)
// ============================================================================

// Cache prefetch constants
constexpr int LOOKAHEAD = 12;
constexpr int PER_TASK_CORE_NUM = 3;
constexpr int BLOCK_WEIGHT = 256;
constexpr int BLOCK_SHARED = 4096;
constexpr uint32_t BATCH_SIZE = 512;

// Task queue
struct matmul_task_t {
    uint64_t input_dma;
    uint64_t weight_dma;
    int M, K, N;
};

struct rknpu_tasks_t {
    matmul_task_t tasks[PER_TASK_CORE_NUM];
};

struct rknpu_tasks_result_t {
    int32_t* output[PER_TASK_CORE_NUM];
};

static std::vector<std::shared_ptr<std::vector<std::tuple<int, int, matmul_task_t>>>> tasks_list;
static std::shared_ptr<std::vector<int32_t*>> npu_tasks_shared;

// Double buffering control
static int TASKS_LOCAL_PER_NUM = 2;
static std::atomic<int> buffer_free[2];
static int npu_domain_id = 0;

// Thread synchronization
static std::mutex npu_worker_mtx;
static std::condition_variable npu_cv;
static std::mutex cpu_worker_mtx;
static std::condition_variable cpu_cv;
static std::atomic<bool> npu_stop_flag{false};
static std::thread npu_worker_thread;

// Forward declarations
static rknpu_tasks_result_t rknpu_matmul(rknpu_tasks_t tasks, int domain_id, int output_index);

// ============================================================================
// Helper Functions
// ============================================================================

static Domain* find_tensor_domain(const void* tensor_data) {
    uint64_t addr = (uint64_t)tensor_data;
    // Iterate through all FileDomains in file_mapping
    for (auto& [file_path, file_domains] : file_mapping) {
        if (!file_domains) continue;
        // Iterate through all Domains in each FileDomains
        for (auto* domain_ptr : file_domains->domains) {
            if (!domain_ptr || !domain_ptr->virtual_addr) continue;
            uint64_t va = (uint64_t)domain_ptr->virtual_addr;
            if (addr >= va && addr < va + DOMAIN_SIZE) {
                return domain_ptr;
            }
        }
    }
    return nullptr;
}

static int get_default_domain_id() {
    // Search through file_mapping to find first available domain
    for (auto& [file_path, file_domains] : file_mapping) {
        if (file_domains && !file_domains->domains.empty()) {
            return file_domains->domains[0]->id;
        }
    }
    return 0;
}

static inline int ceil_int(int x, int y) {
    return (x + y - 1) / y;
}

// ============================================================================
// NPU Worker Thread (Q8_0 Path)
// ============================================================================

/**
 * @brief NPU worker thread for async execution with double buffering
 */
static void npu_work() {
    buffer_free[0].store(0, std::memory_order_release);
    buffer_free[1].store(0, std::memory_order_release);
    
    while (!npu_stop_flag.load(std::memory_order_acquire)) {
        std::shared_ptr<std::vector<std::tuple<int, int, matmul_task_t>>> tasks;
        {
            std::unique_lock<std::mutex> lock(npu_worker_mtx);
            npu_cv.wait(lock, [&] { 
                return !tasks_list.empty() || npu_stop_flag.load();
            });

            if (npu_stop_flag.load()) break;
            if (tasks_list.empty()) continue;

            tasks = tasks_list.back();
            tasks_list.pop_back();
        }

        npu_tasks_shared = std::make_shared<std::vector<int32_t*>>(tasks->size(), nullptr);
        int index = 0;

        for (int t = 0; t < (int)tasks->size(); t += TASKS_LOCAL_PER_NUM) {
            // Prepare task batch
            rknpu_tasks_t _t = {};
            for (int off = 0; off < TASKS_LOCAL_PER_NUM && off + t < (int)tasks->size(); off++) {
                _t.tasks[off] = std::get<2>(tasks->at(off + t));
            }
            
            // Wait for buffer to be free
            {
                std::unique_lock<std::mutex> lock(npu_worker_mtx);
                npu_cv.wait(lock, [&] { 
                    return buffer_free[index].load(std::memory_order_acquire) == 0;
                });
            }
            
            // Submit to NPU
            {
                // Call actual NPU matmul function
                auto outq = rknpu_matmul(_t, npu_domain_id, index * PER_TASK_CORE_NUM);
                
                // Store output pointers
                std::lock_guard<std::mutex> lock(cpu_worker_mtx);
                for (int off = 0; off < TASKS_LOCAL_PER_NUM && off + t < (int)tasks->size(); off++) {
                    npu_tasks_shared->at(t + off) = outq.output[off];
                }
                
                buffer_free[index].store(1, std::memory_order_release);
                cpu_cv.notify_one();
            }
            
            index = (index + 1) & 0x1;
        }
    }
}

// ============================================================================
// Q8_0 Extraction and Layout Conversion
// ============================================================================

/**
 * @brief Extract INT8 data and scales from Q8_0 blocks
 */
static void extract_q8_0(
    const block_q8_0* blocks,
    int nrows, int ncols,  // ncols must be multiple of 32
    int8_t* int8_out,
    float* scales_out) {
    
    const int QK = 32;
    const int nb = ncols / QK;
    
    // Debug: Check memory validity before parallel access
    printf("[DEBUG] extract_q8_0: blocks=%p, nrows=%d, ncols=%d, nb=%d\n", 
           blocks, nrows, ncols, nb);
    printf("[DEBUG] extract_q8_0: int8_out=%p, scales_out=%p\n", int8_out, scales_out);
    printf("[DEBUG] extract_q8_0: total blocks=%d, total bytes=%zu\n", 
           nrows * nb, nrows * nb * sizeof(block_q8_0));
    
    // Test if first block is readable
    printf("[DEBUG] Testing first block access...\n");
    volatile uint8_t test_byte = ((const uint8_t*)blocks)[0];
    printf("[DEBUG] First byte: 0x%02x (readable)\n", test_byte);
    
    // Test if we can read the first block_q8_0 structure
    printf("[DEBUG] Testing first block_q8_0...\n");
    const block_q8_0& first_blk = blocks[0];
    volatile uint16_t d = first_blk.d;
    volatile int8_t q0 = first_blk.qs[0];
    printf("[DEBUG] First block: d=0x%04x, qs[0]=%d\n", d, q0);
    
    // TEMPORARY FIX: Disable OpenMP to isolate the issue
    // If this works, the problem is with parallel memory access
    // #pragma omp parallel for
    for (int i = 0; i < nrows; i++) {
        if (i % 1024 == 0) {
            printf("[DEBUG] Processing row %d/%d\n", i, nrows);
        }
        for (int j = 0; j < nb; j++) {
            // Detailed debug for first few blocks
            if (i == 0 && j < 5) {
                printf("[DEBUG] Block [%d,%d]: blk=%p, blk.d=0x%04x, blk.qs=%p\n", 
                       i, j, &blocks[i * nb + j], blocks[i * nb + j].d, blocks[i * nb + j].qs);
                printf("[DEBUG]   scales_out[%d]=%p, int8_out dest=%p\n",
                       i * nb + j, &scales_out[i * nb + j], int8_out + i * ncols + j * QK);
            }
            
            const block_q8_0& blk = blocks[i * nb + j];
            scales_out[i * nb + j] = GGML_FP16_TO_FP32(blk.d);
            
            // CRITICAL FIX: Use byte-wise copy instead of memcpy
            // blk.qs has +2 offset (not 4-byte aligned), which can cause SIGBUS
            // when memcpy uses NEON/SIMD instructions on mmap memory
            int8_t* dst = int8_out + i * ncols + j * QK;
            const int8_t* src = blk.qs;
            
            // Manual copy to avoid alignment issues
            for (int k = 0; k < QK; k++) {
                dst[k] = src[k];
            }
        }
    }
    printf("[DEBUG] extract_q8_0: completed successfully\n");
}

// NPU layout conversion helper (from reference code)
// Note: These are declared in npu_matmul.h, so remove 'static' to match
inline int feature_data(int H, int C2, int c, int h) {
    int plane = c / C2;
    int src = plane * H * C2;
    int offset = c % C2;
    int pos = src + C2 * h + offset;
    return pos;
}

inline int weight_int8(int C, int k, int c) {
    int dst = 0;
    int kpg = (k / 32);
    int cpg = (c / 32);
    dst = ((cpg * 32) * 32) + (kpg * 32 * C);
    dst = dst + (c % 32) + ((k % 32) * 32);
    return dst;
}

static void to_npu_feature_layout(const int8_t* src, int M, int K, int8_t* dst) {
    const int K_aligned = (K + 15) & ~15;
    memset(dst, 0, M * K_aligned);
    
    // TEMPORARY FIX: Disable OpenMP
    // #pragma omp parallel for
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            dst[feature_data(M, 16, k, m)] = src[m * K + k];
        }
    }
}

static void to_npu_weight_layout(const int8_t* src, int N, int K, int8_t* dst) {
    const int K_aligned = (K + 31) & ~31;
    memset(dst, 0, N * K_aligned);
    
    // TEMPORARY FIX: Disable OpenMP
    // #pragma omp parallel for
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            dst[weight_int8(K_aligned, k, n)] = src[n * K + k];
        }
    }
}

// Helper functions from reference code
static void flush_cache(void* addr, size_t len) {
    // Implement cache flush via cache_manager driver
    // For simplicity, using __builtin___clear_cache
    #ifdef __aarch64__
    __builtin___clear_cache((char*)addr, (char*)addr + len);
    #endif
}

static void invalid_cache(void* addr, size_t len) {
    // Invalidate cache to ensure CPU reads NPU-written data
    #ifdef __aarch64__
    asm volatile("dc civac, %0" : : "r"(addr) : "memory");
    #endif
}

// NPU matmul submission function (simplified - requires full driver integration)
static rknpu_tasks_result_t rknpu_matmul(rknpu_tasks_t tasks, int domain_id, int output_index) {
    rknpu_tasks_result_t result = {};
    
    // NOTE: This is a simplified placeholder implementation.
    // Full implementation requires:
    // 1. Calling gen_matmul_int8() to generate NPU register commands
    // 2. Setting up rknpu_task structures with proper DMA addresses
    // 3. Calling ioctl(DRM_IOCTL_RKNPU_SUBMIT) to submit to NPU driver
    // 4. Waiting for NPU completion
    // 5. Returning INT32 output pointers
    
    // For now, allocate output buffers
    for (int t = 0; t < PER_TASK_CORE_NUM && tasks.tasks[t].input_dma; t++) {
        auto& tsk = tasks.tasks[t];
        
        // Allocate output buffer: M x N x INT32
        size_t output_size = tsk.M * tsk.N * sizeof(int32_t);
        
        // This would come from pre-allocated NPU output memory
        // result.output[t] = (int32_t*)get_npu_output_buffer(domain_id, output_index + t);
        
        // Placeholder: zero-filled output
        static std::vector<int32_t> dummy_output(BATCH_SIZE * BLOCK_WEIGHT, 0);
        result.output[t] = dummy_output.data();
        
        fprintf(stderr, "[NPU] Warning: Using placeholder rknpu_matmul - NPU not actually invoked\n");
    }
    
    return result;
}

// ============================================================================
// Q8_0 Path: Parallel matmul with double buffering
// ============================================================================

/**
 * @brief Q8_0 matrix multiplication with CPU & NPU parallel execution
 * 
 * This is the optimized path using double buffering:
 * - NPU writes to buffer 0 while CPU reads from buffer 1
 * - Uses NEON instructions for dequantization
 * - Cache prefetching for better performance
 * 
 * NOTE: Quantization and IOMMU mapping are done earlier in model.cpp
 *       This function works with pre-quantized Q8_0 tensors that have DMA addresses
 */
static void compute_matmul_q8_0_parallel(
    const struct ggml_tensor* src0,  // weight (M x K, Q8_0, pre-quantized with IOMMU)
    const struct ggml_tensor* src1,  // input (N x K, FP32)
    struct ggml_tensor* dst,         // output (N x M, FP32)
    int domain_id) {
    
    const int M = src0->ne[1];
    const int K = src0->ne[0];
    const int N = src1->ne[1];
    const int QK = 32;
    
    // Step 1: Quantize input to Q8_0 (input is still FP32, not pre-quantized)
    printf("[DEBUG] Allocating input buffers: N=%d, K=%d, QK=%d\n", N, K, QK);
    printf("[DEBUG]   input_q8 size: %zu blocks\n", (size_t)(N * K) / QK);
    printf("[DEBUG]   input_int8 size: %zu bytes\n", (size_t)(N * K));
    printf("[DEBUG]   input_scales size: %zu floats\n", (size_t)(N * K) / QK);
    
    std::vector<block_q8_0> input_q8((N * K) / QK);
    quantize_row_q8_0_ref((const float*)src1->data, input_q8.data(), N * K);
    
    std::vector<int8_t> input_int8(N * K);
    std::vector<float> input_scales(N * K / QK);
    extract_q8_0(input_q8.data(), N, K, input_int8.data(), input_scales.data());
    
    // Step 2: Extract weight data (already Q8_0 with IOMMU mapping)
    printf("[DEBUG] Allocating weight buffers: M=%d, K=%d, QK=%d\n", M, K, QK);
    printf("[DEBUG]   weight_int8 size: %zu bytes (%.2f MB)\n", (size_t)(M * K), (M * K) / 1024.0 / 1024.0);
    printf("[DEBUG]   weight_scales size: %zu floats (%.2f MB)\n", (size_t)(M * K) / QK, ((M * K) / QK * 4) / 1024.0 / 1024.0);
    
    // CRITICAL: Check if allocation is feasible
    size_t required_bytes = (size_t)M * K + ((size_t)M * K / QK) * sizeof(float);
    printf("[DEBUG] Total memory required: %.2f MB\n", required_bytes / 1024.0 / 1024.0);
    
    std::vector<int8_t> weight_int8;
    std::vector<float> weight_scales;
    
    try {
        printf("[DEBUG] Allocating weight_int8...\n");
        weight_int8.resize(M * K);
        printf("[DEBUG]   weight_int8.data() = %p, size() = %zu, capacity() = %zu\n", 
               weight_int8.data(), weight_int8.size(), weight_int8.capacity());
        
        printf("[DEBUG] Allocating weight_scales...\n");
        weight_scales.resize(M * K / QK);
        printf("[DEBUG]   weight_scales.data() = %p, size() = %zu, capacity() = %zu\n", 
               weight_scales.data(), weight_scales.size(), weight_scales.capacity());
        
        // Test write to first element
        printf("[DEBUG] Testing write access to weight_int8[0]...\n");
        weight_int8[0] = 123;
        printf("[DEBUG]   Write successful, value = %d\n", weight_int8[0]);
        
        printf("[DEBUG] Testing write access to weight_int8[last]...\n");
        weight_int8[M * K - 1] = 45;
        printf("[DEBUG]   Write successful, value = %d\n", weight_int8[M * K - 1]);
        
    } catch (const std::exception& e) {
        fprintf(stderr, "[NPU] Failed to allocate weight buffers: %s\n", e.what());
        throw;
    }
    
    // Weight tensor should already be Q8_0 format with IOMMU mapping
    if (src0->type != GGML_TYPE_Q8_0) {
        fprintf(stderr, "[NPU] Error: Weight tensor should be pre-quantized to Q8_0, got type %d\n", src0->type);
        throw std::runtime_error("Weight tensor not pre-quantized");
    }
    
    // Extract INT8 data and scales from pre-quantized weight
    printf("[DEBUG] Extracting weight data from src0->data=%p\n", src0->data);
    extract_q8_0((const block_q8_0*)src0->data, M, K, weight_int8.data(), weight_scales.data());
    
    // Step 3: Convert to NPU layout
    const int K_in = (K + 15) & ~15;
    const int K_w = (K + 31) & ~31;
    
    std::vector<int8_t> input_npu(N * K_in);
    to_npu_feature_layout(input_int8.data(), N, K, input_npu.data());
    
    std::vector<int8_t> weight_npu(M * K_w);
    to_npu_weight_layout(weight_int8.data(), M, K, weight_npu.data());
    
    // Step 4: Get DMA addresses from existing IOMMU mappings
    // Weight tensor should already have IOMMU mapping created in model.cpp
    Domain* weight_domain = find_tensor_domain(src0->data);
    uint64_t weight_dma_base = 0;
    
    if (weight_domain) {
        // Find the tensor in the domain's tensor map to get its IommuConfig
        for (auto& [name, tensor_tuple] : weight_domain->tensors) {
            auto* tensor = std::get<0>(tensor_tuple);
            if (tensor && tensor->data == src0->data) {
                IommuConfig* iommu_config = std::get<2>(tensor_tuple);
                if (iommu_config && iommu_config->iommu_addr) {
                    weight_dma_base = (uint64_t)iommu_config->iommu_addr;
                    fprintf(stderr, "[NPU] Found weight DMA address: 0x%lx\n", weight_dma_base);
                    break;
                }
            }
        }
    }
    
    // For input, we need to allocate temporary DMA buffer (input is runtime data)
    std::vector<int8_t> input_dma_buf(N * K_in);
    memcpy(input_dma_buf.data(), input_npu.data(), N * K_in);
    flush_cache(input_dma_buf.data(), N * K_in);
    uint64_t input_dma_base = (uint64_t)input_dma_buf.data();
    
    // If weight doesn't have DMA address yet, fall back to temporary buffer
    // (This shouldn't happen if quantization pipeline is working correctly)
    if (weight_dma_base == 0) {
        fprintf(stderr, "[NPU] Warning: Weight tensor has no IOMMU mapping, using temporary buffer\n");
        std::vector<int8_t> weight_dma_buf(M * K_w);
        memcpy(weight_dma_buf.data(), weight_npu.data(), M * K_w);
        flush_cache(weight_dma_buf.data(), M * K_w);
        weight_dma_base = (uint64_t)weight_dma_buf.data();
    }
    
    // Step 5: Build task list (block splitting)
    auto tasks = std::make_shared<std::vector<std::tuple<int, int, matmul_task_t>>>();
    tasks->reserve(ceil_int(M, BLOCK_WEIGHT) * ceil_int(K, BLOCK_SHARED));
    
    uint64_t weight_dma = weight_dma_base;
    
    // Split matrix into blocks
    // Output matrix: N x M (src1 rows x src0 rows)
    // Weight matrix: M x K (src0)
    // Input matrix: N x K (src1)
    for (int j = 0; j < M;) {
        auto _n = std::min(M - j, BLOCK_WEIGHT);
        auto input_dma = input_dma_base;
        
        for (int k = 0; k < K;) {
            auto _k = std::min(K - k, BLOCK_SHARED);
            
            tasks->emplace_back(j, k, matmul_task_t{
                input_dma, weight_dma, N, _k, _n
            });
            
            input_dma += N * _k;
            weight_dma += _k * _n;
            k += _k;
        }
        j += _n;
    }
    
    // Step 6: Adjust batch size based on task count
    if (tasks->size() <= PER_TASK_CORE_NUM) {
        TASKS_LOCAL_PER_NUM = PER_TASK_CORE_NUM;
    } else if (tasks->size() > 2 * PER_TASK_CORE_NUM + 1) {
        TASKS_LOCAL_PER_NUM = PER_TASK_CORE_NUM;
    } else {
        TASKS_LOCAL_PER_NUM = 2;
    }
    
    // Step 7: Initialize output to zero
    std::fill((float*)dst->data, (float*)dst->data + N * M, 0.0f);
    
    // Step 8: Submit tasks to NPU worker thread
    {
        std::lock_guard<std::mutex> lock(npu_worker_mtx);
        tasks_list.push_back(tasks);
        npu_domain_id = domain_id;
        npu_cv.notify_one();
    }
    
    // Step 9: CPU processes NPU output with double buffering
    int index = 0;
    const int scale_per_k = K / QK;
    float* dst_data = (float*)dst->data;
    
    for (int t = 0; t < (int)tasks->size(); t += TASKS_LOCAL_PER_NUM) {
        {
            std::unique_lock<std::mutex> lock(cpu_worker_mtx);
            cpu_cv.wait(lock, [&] { 
                return npu_tasks_shared.use_count() != 0 && 
                       npu_tasks_shared->at(t) != nullptr && 
                       buffer_free[index].load(std::memory_order_acquire) == 1;
            });
        }
        
        for (int toff = 0; toff < TASKS_LOCAL_PER_NUM && t + toff < (int)tasks->size(); toff++) {
            const auto [j, k, task] = tasks->at(t + toff);
            int32_t* task_output = npu_tasks_shared->at(t + toff);
            
            // Invalidate cache to ensure CPU reads NPU-written data
            invalid_cache(task_output, N * std::min(M - j, BLOCK_WEIGHT) * sizeof(int32_t));
            
            int joff_max = std::min(M - j, BLOCK_WEIGHT);
            
            // Process output with NEON optimization
            #pragma omp parallel for num_threads(4)
            for (int joff = 0; joff < joff_max; joff += 4) {
                const int j_start = j + joff;
                
                // Load 4 weight scales
                int w_scale_off = j_start * scale_per_k + k / QK;
                
                #ifdef __ARM_NEON
                float32x4_t w_scale = {
                    weight_scales[w_scale_off], 
                    weight_scales[w_scale_off + scale_per_k * 1],
                    weight_scales[w_scale_off + scale_per_k * 2],
                    weight_scales[w_scale_off + scale_per_k * 3]
                };
                #endif
                
                int feature_offset = feature_data(N, 4, joff, 0);
                
                for (int i = 0; i < N; i++) {
                    // Cache prefetch optimization
                    if (i + LOOKAHEAD < N) {
                        __builtin_prefetch(&dst_data[(i + LOOKAHEAD) * M + j_start], 1, 3);
                        if ((i % 4) == 0) {
                            __builtin_prefetch(&task_output[feature_offset + (LOOKAHEAD * 4)], 0, 3);
                        }
                    }
                    
                    // Dequantize: INT32 -> FP32
                    const float input_scale = input_scales[i * scale_per_k + k / QK];
                    
                    #ifdef __ARM_NEON
                    int32x4_t v_int32 = vld1q_s32(&task_output[feature_offset]);
                    float32x4_t v_val = vcvtq_f32_s32(v_int32);
                    float32x4_t v_input_scale = vdupq_n_f32(input_scale);
                    v_val = vmulq_f32(v_val, vmulq_f32(v_input_scale, w_scale));
                    
                    float* out_ptr = &dst_data[i * M + j_start];
                    float32x4_t out_old = vld1q_f32(out_ptr);
                    out_old = vaddq_f32(out_old, v_val);
                    vst1q_f32(out_ptr, out_old);
                    #else
                    for (int vi = 0; vi < 4; vi++) {
                        if (j_start + vi < M) {
                            float val = task_output[feature_offset + vi] * input_scale * weight_scales[w_scale_off + scale_per_k * vi];
                            dst_data[i * M + j_start + vi] += val;
                        }
                    }
                    #endif
                    
                    feature_offset += 4;
                }
            }
        }
        
        {
            std::lock_guard<std::mutex> lock(npu_worker_mtx);
            buffer_free[index].store(0, std::memory_order_release);
            npu_cv.notify_one();
        }
        index = (index + 1) & 0x1;
    }
    
    npu_tasks_shared.reset();
}



// ============================================================================
// Capability Check
// ============================================================================

int ggml_can_use_npu(const struct ggml_tensor* src0, const struct ggml_tensor* src1) {
    if (!g_npu_initialized) return 0;
    
    const int64_t ne00 = src0->ne[0];  // K
    const int64_t ne01 = src0->ne[1];  // M
    const int64_t ne10 = src1->ne[0];  // K
    const int64_t ne11 = src1->ne[1];  // N
    
    if (ne00 != ne10) return 0;  // K must match
    
    const int M = ne01, K = ne00, N = ne11;
    
    // Size constraints (relaxed for testing)
    if (M < 1 || K < 32 || N < 1) return 0;  // Too small
    if (M > 65535 || K > 65535 || N > 65535) return 0;  // Too large
    
    // Input must be FP32
    if (src1->type != GGML_TYPE_F32) return 0;
    
    // Weight must be Q8_0 (pre-quantized in model loading stage)
    // NOTE: With the new pipeline, weights should already be Q8_0
    if (src0->type != GGML_TYPE_Q8_0) {
        fprintf(stderr, "[NPU] Weight type is %d, expected Q8_0 (pre-quantized)\n", src0->type);
        return 0;
    }
    
    // Check memory layout (contiguous)
    const size_t type_size = ggml_type_size(src0->type);
    const size_t blck_size = ggml_blck_size(src0->type);
    if (src0->nb[0] != type_size) return 0;
    if (src1->nb[0] != sizeof(float)) return 0;
    
    return 1;
}

// ============================================================================
// Main Implementation with Type Routing
// ============================================================================

void ggml_compute_forward_mul_mat_npu(
    const struct ggml_compute_params* params,
    struct ggml_tensor* dst) {
    
    // Only process on thread 0 (single-threaded NPU execution)
    // Note: params->ith accesses internal structure
    if (params->ith != 0) return;
    
    const struct ggml_tensor* src0 = dst->src[0];  // weight
    const struct ggml_tensor* src1 = dst->src[1];  // input
    
    Domain* weight_domain = find_tensor_domain(src0->data);
    int domain_id = weight_domain ? weight_domain->id : get_default_domain_id();
    
    try {
        // All types converted to Q8_0 (NPU only supports INT8)
        compute_matmul_q8_0_parallel(src0, src1, dst, domain_id);
    } catch (const std::exception& e) {
        fprintf(stderr, "[NPU] Error: %s, falling back to CPU\n", e.what());
        // Let GGML handle CPU fallback by not writing to dst
        throw;
    }
}

// ============================================================================
// C API: Initialization and Cleanup
// ============================================================================

void ggml_npu_init() {
    if (g_npu_initialized) return;
    
    g_npu_fd = npu_open();
    if (g_npu_fd < 0) {
        fprintf(stderr, "[NPU] Failed to open device\n");
        return;
    }
    
    if (npu_reset(g_npu_fd) < 0) {
        fprintf(stderr, "[NPU] Failed to reset\n");
        npu_close(g_npu_fd);
        g_npu_fd = -1;
        return;
    }
    
    // Start NPU worker thread for Q8_0 parallel path
    npu_stop_flag.store(false, std::memory_order_release);
    npu_worker_thread = std::thread(npu_work);
    
    g_npu_initialized = true;
    printf("[NPU] Initialized (fd=%d, worker thread started)\n", g_npu_fd);
}

void ggml_npu_free() {
    if (!g_npu_initialized) return;
    
    // Stop NPU worker thread
    {
        std::lock_guard<std::mutex> lock(npu_worker_mtx);
        npu_stop_flag.store(true, std::memory_order_release);
        npu_cv.notify_all();
    }
    
    if (npu_worker_thread.joinable()) {
        npu_worker_thread.join();
    }
    
    if (g_npu_fd >= 0) {
        npu_close(g_npu_fd);
        g_npu_fd = -1;
    }
    
    g_npu_initialized = false;
    printf("[NPU] Freed\n");
}
