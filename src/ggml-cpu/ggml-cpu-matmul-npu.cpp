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
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <errno.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// ============================================================================
// Global State
// ============================================================================

static bool g_npu_initialized = false;
static int g_npu_fd = -1;

// Cache manager 全局变量
static int manager_cache_fd = -1;  
static const char* manager_cache_dev = "/dev/cache_manager";

// ============================================================================
// Cache Manager Definitions
// ============================================================================

// Cache operation structures
struct cache_flush_range {
    void* va_start;
    uint64_t len;
};

struct cache_inval_range {
    void* va_start;
    uint64_t len;
};

// Cache manager ioctl commands (from cache-manager kernel driver)
#define CACHE_FLUSH_RANGE _IOW('C', 1, struct cache_flush_range)
#define CACHE_INVAL_RANGE _IOW('C', 2, struct cache_inval_range)

// ============================================================================
// Q8_0 Path: CPU & NPU Parallel Computing (Double Buffering)
// ============================================================================

// Cache prefetch constants
constexpr int LOOKAHEAD = 12;
constexpr int PER_TASK_CORE_NUM = 3;
constexpr int BLOCK_WEIGHT = 256;
constexpr int BLOCK_SHARED = 4096;
constexpr uint32_t BATCH_SIZE = 512;
constexpr uint32_t BLOCK_WHOLE_NR = 9;  // Block 总数

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

/**
 * @brief 从 tensor 名称查找对应的 Domain
 * 
 * @param target_tensor 目标 tensor
 * @return Domain* 找到的 Domain，未找到返回 nullptr
 */
static Domain* find_tensor_domain(const ggml_tensor* target_tensor) {
    if (!target_tensor || !target_tensor->name) return nullptr;
    
    std::string tensor_name(target_tensor->name);
    
    // Iterate through all FileDomains in file_mapping
    for (auto& [file_path, file_domains] : file_mapping) {
        if (!file_domains) continue;
        // Iterate through all Domains in each FileDomains
        for (auto* domain_ptr : file_domains->domains) {
            if (!domain_ptr) continue;
            // Direct lookup in map by tensor name
            auto it = domain_ptr->tensors.find(tensor_name);
            if (it != domain_ptr->tensors.end()) {
                return domain_ptr;
            }
        }
    }
    return nullptr;
}

/**
 * @brief 获取默认 Domain ID（用于后备）
 * 
 * @return int 默认 Domain ID
 */
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
 * @brief 从 Q8_0 块中提取 INT8 数据和 scale
 * 
 * Q8_0 格式: struct block_q8_0 { ggml_fp16_t d; int8_t qs[32]; }
 * - d: FP16 scale（2 字节）
 * - qs: INT8 数据（32 字节）
 * 
 * 注意事项:
 * - qs 数组有 2 字节偏移（非 4 字节对齐）
 * - 在 mmap 内存上使用 memcpy 可能触发 SIGBUS（NEON/SIMD 指令要求对齐）
 * - 使用逐字节拷贝避免对齐问题
 * 
 * @param blocks     Q8_0 块数组
 * @param nrows      行数
 * @param ncols      列数（必须是 32 的倍数）
 * @param int8_out   输出: INT8 数据（nrows × ncols）
 * @param scales_out 输出: scale 数组（nrows × nb，nb = ncols/32）
 */
static void extract_q8_0(
    const block_q8_0* blocks,
    int nrows, int ncols,
    int8_t* int8_out,
    float* scales_out) {
    
    const int QK = 32;  // Q8_0 块大小
    const int nb = ncols / QK;  // 每行的块数
    
    // 串行处理（避免 OpenMP 并行访问 mmap 内存的潜在问题）
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < nb; j++) {
            const block_q8_0& blk = blocks[i * nb + j];
            
            // 提取 scale（FP16 → FP32）
            scales_out[i * nb + j] = GGML_FP16_TO_FP32(blk.d);
            
            // 逐字节拷贝 INT8 数据（避免对齐问题）
            int8_t* dst = int8_out + i * ncols + j * QK;
            const int8_t* src = blk.qs;
            for (int k = 0; k < QK; k++) {
                dst[k] = src[k];
            }
        }
    }
}

// ============================================================================
// NPU Layout Conversion Functions
// ============================================================================

/**
 * @brief 计算 NPU 特征数据布局的线性偏移（NCHW16/4 格式）
 * 
 * feature_data(M, 16, k, m) 用于输入数据（每 16 个通道为一块）
 * feature_data(M, 4, n, m)  用于输出数据（每 4 个通道为一块）
 * 
 * 内存布局: [plane_0][plane_1]...[plane_P]，每个 plane 大小为 H × C2
 * 
 * @param H  高度维度（矩阵行数 M）
 * @param C2 通道分块大小（16 用于输入，4 用于输出）
 * @param c  当前通道索引
 * @param h  当前高度索引
 * @return   线性偏移量
 */
inline int feature_data(int H, int C2, int c, int h) {
    int plane = c / C2;            // 计算平面索引（第几个 C2 通道块）
    int src = plane * H * C2;      // 该平面的起始偏移量
    int offset = c % C2;           // 元素在块内的相对通道偏移
    int pos = src + C2 * h + offset; // 最终偏移 = 平面起始 + 行偏移 + 通道偏移
    return pos;
}

/**
 * @brief 计算 NPU INT8 权重布局的线性偏移（32×32 分块存储）
 * 
 * 权重矩阵逻辑形状: K×N，NPU 物理布局: 按 32×32 分块存储
 * 每个块内按列优先存储: 先存储 32 个输入通道，再跳到下一个输出通道
 * 
 * @param C 权重矩阵的列数（输入通道数 K）
 * @param k 输出通道索引（0 到 N-1）
 * @param c 输入通道索引（0 到 K-1）
 * @return  线性偏移量
 */
inline int weight_int8(int C, int k, int c) {
    int kpg = (k / 32);          // 输出通道块索引（每 32 个输出通道为一块）
    int cpg = (c / 32);          // 输入通道块索引（每 32 个输入通道为一块）
    // 计算块起始偏移
    int dst = ((cpg * 32) * 32) + (kpg * 32 * C);
    // 计算块内偏移（列优先存储）
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

/**
 * @brief 刷新 CPU Cache（确保数据从 CPU Cache 写回内存）
 * 
 * 使用场景: CPU 写入数据后，NPU DMA 需要读取该数据
 * 实现方式: 通过 /dev/cache_manager 驱动的 ioctl 接口
 * 
 * @param va_start 起始虚拟地址
 * @param len  长度
 */
static void flush_cache(void* va_start, uint64_t len) {
    int ret = 0;
    struct cache_flush_range range;
    range.va_start = va_start;  // 起始地址
    range.len = len;            // 长度

    // 延迟打开 cache_manager 设备
    if(manager_cache_fd < 0) {
        manager_cache_fd = open(manager_cache_dev, O_RDWR);
        if (manager_cache_fd < 0) {
            fprintf(stderr, "[NPU] Warning: Failed to open cache_manager device\n");
            return;
        }
    }

    // 调用 ioctl 刷新 Cache
    ret = ioctl(manager_cache_fd, CACHE_FLUSH_RANGE, &range);
    if (ret != 0) {
        fprintf(stderr, "[NPU] Warning: ioctl flush_cache failed: %d\n", errno);
    }
}


/**
 * @brief 无效化 CPU Cache（确保 CPU 读取到 NPU 写入的最新数据）
 * 
 * 使用场景: NPU 写入数据后，CPU 需要读取该数据
 * 实现方式: 通过 /dev/cache_manager 驱动的 ioctl 接口
 * 
 * @param va_start 起始虚拟地址
 * @param len  长度
 */
static void invalid_cache(void* va_start, uint64_t len) {
    int ret = 0;
    struct cache_inval_range range;
    range.va_start = va_start;  // 起始地址
    range.len = len;            // 长度

    // 延迟打开 cache_manager 设备（第一次调用时）
    if(manager_cache_fd < 0) {
        manager_cache_fd = open(manager_cache_dev, O_RDWR);
        if (manager_cache_fd < 0) {
            fprintf(stderr, "[NPU] Warning: Failed to open cache_manager device\n");
            return;
        }
    }
    
    // 调用 ioctl 无效化 Cache
    ret = ioctl(manager_cache_fd, CACHE_INVAL_RANGE, &range);
    if (ret != 0) {
        fprintf(stderr, "[NPU] Warning: ioctl invalid_cache failed: %d\n", errno);
    }
}

/**
 * @brief NPU 矩阵乘法提交函数（完整实现 - 使用 LeftMemory）
 * 
 * 功能流程:
 * 1. 从 find_domain_by_id() 获取 Domain 对象
 * 2. 直接操作 LeftMemory 生成 NPU 寄存器命令
 * 3. 填充任务描述符（rknpu_task）
 * 4. 构造 rknpu_submit 结构
 * 5. 通过 ioctl 提交给 RKNPU 驱动
 * 6. 无效化输出缓冲区的 Cache
 * 7. 返回 NPU 输出的指针数组
 * 
 * @param tasks        任务批次（最多 PER_TASK_CORE_NUM=3 个任务）
 * @param domain_id    IOMMU 域 ID
 * @param output_index 输出缓冲区索引偏移
 * @return rknpu_tasks_result_t 返回 NPU 输出的指针数组
 */
static rknpu_tasks_result_t rknpu_matmul(rknpu_tasks_t tasks, int domain_id = 0, int output_index = 0) {
    rknpu_tasks_result_t result = {};  // 初始化结果结构

    // ===== 第0步：获取 Domain 对象 =====
    Domain* domain = find_domain_by_id(domain_id);
    if (!domain) {
        fprintf(stderr, "[rknpu_matmul] Error: Domain %d not found\n", domain_id);
        return result;
    }
    
    // 检查必要的缓冲区是否已分配
    if (!domain->regcmd || !domain->tasks_mem || !domain->output) {
        fprintf(stderr, "[rknpu_matmul] Error: Domain %d buffers not initialized\n", domain_id);
        return result;
    }

    // 获取 LeftMemory 的虚拟地址和 DMA 地址
    uint8_t* regcmd_va = static_cast<uint8_t*>(domain->regcmd->virtual_addr);
    uint64_t regcmd_dma = reinterpret_cast<uint64_t>(domain->regcmd->iommu_addr);
    size_t regcmd_size = domain->regcmd->size;
    
    rknpu_task* tasks_va = static_cast<rknpu_task*>(domain->tasks_mem->virtual_addr);
    uint64_t tasks_obj = *domain->tasks_mem->mem_obj_handle;  // 解引用获取句柄值
    
    int32_t* output_va = static_cast<int32_t*>(domain->output->virtual_addr);
    uint64_t output_dma = reinterpret_cast<uint64_t>(domain->output->iommu_addr);

    uint64_t off = 0;  // 寄存器命令偏移量
    int task_num = 0;  // 实际任务数量
    
    // ===== 第1步：遍历所有任务，生成寄存器命令 =====
    for (int t = 0; t < PER_TASK_CORE_NUM && tasks.tasks[t].input_dma; t++) {
        auto &tsk = tasks.tasks[t];  // 当前任务
        auto m = tsk.M;  // 行数
        auto k = tsk.K;  // 共享维度
        auto n = tsk.N;  // 输出维度
        
        // 检查偏移量是否超出缓冲区
        if (off + NPU_REGS_SIZE * sizeof(uint64_t) > regcmd_size) {
            fprintf(stderr, "[rknpu_matmul] Error: regcmd buffer overflow\n");
            break;
        }
        
        // 获取当前任务的寄存器配置地址
        uint64_t* reg_va = reinterpret_cast<uint64_t*>(regcmd_va + off);
        uint64_t reg_dma = regcmd_dma + off;
        
        // ✅ 生成 NPU 寄存器命令（替代 RegCmd 构造函数）
        matmul_params_t params = {
            .m = static_cast<uint16_t>(m),
            .k = static_cast<uint16_t>(k),
            .n = static_cast<uint16_t>(n),
            .tasks = reg_va
        };
        
        if (gen_matmul_int8(&params) != 0) {
            fprintf(stderr, "[rknpu_matmul] Error: gen_matmul_int8 failed\n");
            break;
        }
        
        // ✅ 设置输入/权重/输出的 DMA 地址（替代 RegCmd::setupAddr）
        // 计算每个任务的输出偏移：假设每个任务最多输出 m*n 个 int32_t
        size_t per_task_output_size = m * n;  // 每个任务最大输出元素数
        uint64_t task_output_dma = output_dma + (t + output_index) * per_task_output_size * sizeof(int32_t);
        
        update_matmul_addr(reg_va, tsk.input_dma, tsk.weight_dma, task_output_dma);
        
        // ✅ 填充任务描述符（rknpu_task 结构）
        tasks_va[t].flags = 0;
        tasks_va[t].op_idx = 0;
        tasks_va[t].enable_mask = 0xd;      // 使能 NPU 的三个核心（CNA + CORE + DPU）
        tasks_va[t].int_mask = 0x300;       // 中断掩码：等待 DPU 完成
        tasks_va[t].int_clear = 0x1ffff;    // 清除所有中断标志
        tasks_va[t].int_status = 0;
        tasks_va[t].regcfg_amount = TASK_REG_AMOUNT;  // 寄存器配置数量（108）
        tasks_va[t].regcfg_offset = 0;
        tasks_va[t].regcmd_addr = reg_dma;  // 寄存器命令的 DMA 地址
        
        off += NPU_REGS_SIZE * sizeof(uint64_t);
        task_num++;
    }
    
    if (task_num == 0) {
        fprintf(stderr, "[rknpu_matmul] Warning: No valid tasks\n");
        return result;
    }
    
    // ✅ 刷新 Cache 确保数据写回内存
    flush_cache((void*)tasks_va, sizeof(rknpu_task) * PER_TASK_CORE_NUM);
    flush_cache((void*)regcmd_va, off);  // 刷新所有寄存器命令

    // ===== 第2步：构造 rknpu_submit 结构并提交给驱动 =====
    // 计算 core_mask：指示使用哪些 NPU 核心
    const auto core_mask = static_cast<uint32_t>((task_num >= 1) | ((task_num >= 2) << 1) | ((task_num >= 3) << 2));
    
    struct rknpu_submit submit = {
        .flags = RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG,  // 任务标志
        .timeout = 6000,          // 超时时间（毫秒）
        .task_start = 0,
        .task_number = 1,
        .task_counter = 0,
        .priority = 0,
        .task_obj_addr = tasks_obj,  // 任务描述符对象句柄
        .iommu_domain_id = static_cast<uint32_t>(domain_id),  // IOMMU 域 ID
        .reserved = 0,
        .task_base_addr = 0,
        .hw_elapse_time = 0,
        .core_mask = core_mask,  // NPU 核心掩码
        .fence_fd = -1,
        .subcore_task =  // 子核心任务分配
            {
                {0, task_num >= 1},  // 核心 0：如果有至少 1 个任务
                {1, task_num >= 2},  // 核心 1：如果有至少 2 个任务
                {2, task_num >= 3},  // 核心 2：如果有 3 个任务
                {0, 0},
                {0, 0},
            },
    };

    // 调用 ioctl 提交给 RKNPU 驱动（阻塞等待 NPU 完成）
    rknpu_ioctl(DRM_IOCTL_RKNPU_SUBMIT, &submit, domain_id);
    
    // ===== 第3步：返回 NPU 输出指针 =====
    for (int t = 0; t < task_num; t++) {
        auto &tsk = tasks.tasks[t];
        size_t per_task_output_size = tsk.M * tsk.N;  // 每个任务输出元素数
        
        // 计算输出指针：基地址 + 偏移
        result.output[t] = output_va + (t + output_index) * per_task_output_size;
        
        // 无效化输出缓冲区的 Cache，确保读取到 NPU 写入的最新数据
        // size_t buffer_size = per_task_output_size * sizeof(int32_t);
        // invalid_cache(result.output[t], buffer_size);
    }
    return result;  // 返回结果
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
    
    // Step 1: Quantize input to Q8_0 (input is FP32, runtime data)
    std::vector<block_q8_0> input_q8((N * K) / QK);
    quantize_row_q8_0_ref((const float*)src1->data, input_q8.data(), N * K);
    
    std::vector<int8_t> input_int8(N * K);
    std::vector<float> input_scales(N * K / QK);
    extract_q8_0(input_q8.data(), N, K, input_int8.data(), input_scales.data());
    
    // Step 2: Extract weight data (already Q8_0 with IOMMU mapping)
    std::vector<int8_t> weight_int8(M * K);
    std::vector<float> weight_scales(M * K / QK);
    
    // Weight tensor should already be Q8_0 format with IOMMU mapping
    if (src0->type != GGML_TYPE_Q8_0) {
        fprintf(stderr, "[NPU] Error: Weight tensor should be pre-quantized to Q8_0, got type %d\n", src0->type);
        throw std::runtime_error("Weight tensor not pre-quantized");
    }
    
    // Extract INT8 data and scales from pre-quantized weight
    extract_q8_0((const block_q8_0*)src0->data, M, K, weight_int8.data(), weight_scales.data());
    
    // Step 3: Convert to NPU layout
    const int K_in = (K + 15) & ~15;
    const int K_w = (K + 31) & ~31;
    
    std::vector<int8_t> input_npu(N * K_in);
    to_npu_feature_layout(input_int8.data(), N, K, input_npu.data());
    
    // NOTE: weight_npu will be created in-place at src0->data later (Step 4)
    
    // Step 4: Get DMA addresses from existing IOMMU mappings
    // STRATEGY: Convert Q8_0 block format to NPU layout IN-PLACE at tensor->data
    // This allows reusing the existing IOMMU mapping without creating a new one
    Domain* weight_domain = find_tensor_domain(src0);
    uint64_t weight_dma_base = 0;
    
    if (weight_domain && src0->name) {
        // Direct lookup in map by tensor name
        auto it = weight_domain->tensors.find(std::string(src0->name));
        if (it != weight_domain->tensors.end()) {
            IommuConfig* iommu_config = std::get<2>(it->second);
            if (iommu_config && iommu_config->iommu_addr) {
                // Step 4.1: Check if we have enough space for NPU layout
                size_t q8_size = ggml_nbytes(src0);  // Size of Q8_0 blocks
                size_t npu_layout_size = M * K_w;     // Size needed for NPU layout (int8 only)
                
                fprintf(stderr, "[NPU] In-place conversion check: Q8_0=%zu bytes, NPU layout needs=%zu bytes\n", 
                        q8_size, npu_layout_size);
                
                if (npu_layout_size > q8_size) {
                    fprintf(stderr, "[NPU] ERROR: Not enough space for in-place conversion\n");
                    throw std::runtime_error("Insufficient space for NPU layout conversion");
                }
                
                // Step 4.2: Convert Q8_0 blocks → NPU layout IN-PLACE
                // Since we already extracted weight_int8 from Q8_0 blocks,
                // we can directly convert it to NPU layout and write to src0->data
                fprintf(stderr, "[NPU] Converting tensor %s to NPU layout in-place...\n", src0->name);
                
                // Write NPU layout directly to tensor->data (overwrites Q8_0 blocks)
                to_npu_weight_layout(weight_int8.data(), M, K, (int8_t*)src0->data);
                
                // Use the existing IOMMU DMA address
                weight_dma_base = (uint64_t)iommu_config->iommu_addr;
                fprintf(stderr, "[NPU] Reusing IOMMU DMA address: 0x%lx (tensor->data=%p)\n", 
                        weight_dma_base, src0->data);
                
                // Flush cache to ensure NPU sees the converted data
                flush_cache(src0->data, npu_layout_size);
                
                // NOTE: After this conversion, src0->data no longer contains Q8_0 blocks!
                // It now contains NPU layout (int8 array with special tiling)
                // TODO: If CPU also needs this tensor, we should either:
                //   1. Keep a backup of Q8_0 blocks
                //   2. Convert back after NPU inference
                //   3. Mark tensor as "NPU layout only"
            }
        }
    }
    
    // For input, we need to allocate temporary DMA buffer (input is runtime data)
    std::vector<int8_t> input_dma_buf(N * K_in);
    memcpy(input_dma_buf.data(), input_npu.data(), N * K_in);
    flush_cache(input_dma_buf.data(), N * K_in);
    uint64_t input_dma_base = (uint64_t)input_dma_buf.data();
    
    // Fallback: if no IOMMU mapping found, create temporary buffer
    if (weight_dma_base == 0) {
        fprintf(stderr, "[NPU] Warning: No IOMMU mapping found, using temporary buffer\n");
        std::vector<int8_t> weight_dma_buf(M * K_w);
        to_npu_weight_layout(weight_int8.data(), M, K, weight_dma_buf.data());
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
    // const size_t blck_size = ggml_blck_size(src0->type);  // 未使用，注释掉
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
    
    Domain* weight_domain = find_tensor_domain(src0);
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

/**
 * @brief NPU 初始化函数
 * 
 * 功能：
 * 1. 打开 RKNPU 设备
 * 2. 根据 config.hpp 中的 file_mapping 初始化所有 Domain 的 RkCtx
 * 3. 启动 NPU 工作线程
 */
void ggml_npu_init() {
    if (g_npu_initialized) {
        return;
    }
    
    // Open RKNPU device
    g_npu_fd = npu_open();
    if (g_npu_fd < 0) {
        fprintf(stderr, "[NPU] Failed to open RKNPU device\n");
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
