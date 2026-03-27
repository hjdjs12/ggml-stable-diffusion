#include <fstream>
#include <iomanip>
#include <inttypes.h>
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
#include <arm_neon.h>
#include <vec.h>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// ============================================================================
// Global State
// ============================================================================

static bool g_npu_initialized = false;
int g_npu_fd = -1;  // Global NPU fd (used by config.hpp)

// Cache manager 全局变量
static int manager_cache_fd = -1;  
static const char* manager_cache_dev = "/dev/cache_manager";
inline uint64_t total_tasks_num = 0;
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
constexpr int BLOCK_WEIGHT = 2048;
constexpr int BLOCK_SHARED = 512;  // Match Q8_0 block size for exact dequantization (one scale per task)
constexpr int BLOCK_WEIGHT_FP16 = 2048; // FP16 版本保持与 BLOCK_SHARED 一致，确保每个任务一个 scale
constexpr int BLOCK_SHARED_FP16 = 512; // FP16 版本可以使用更大的共享块
constexpr int BLOCK_N_FP16 = 348; // FP16 版本的 N 块大小
constexpr uint32_t BATCH_SIZE = 512;
constexpr uint32_t BLOCK_WHOLE_NR = 9;  // Block 总数



float inline get_scale(uint16_t d_as_uint) {
    // 实际实现中请替换为你框架的 fp16_to_fp32
    return (float)d_as_uint; 
}


template <int N>
struct block_q8_0_x {
    uint16_t d;       // 缩放因子 (fp16)
    int8_t  qs[N];    // 连续存储的 N 个 int8 权重
};
using block_q8_0_256 = block_q8_0_x<256>;
using block_q8_0_512 = block_q8_0_x<512>;
using block_q8_0_1024 = block_q8_0_x<1024>;

// void convert_q8_32_to_256(const block_q8_0* src, block_q8_0_256* dst, int64_t n_elements) {
//     const int group_ratio = 8; // 256 / 32
//     int64_t n_new_blocks = n_elements / 256;

//     for (int64_t i = 0; i < n_new_blocks; ++i) {
//         float max_s = 0.0f;
//         const block_q8_0* src_sub_group = &src[i * group_ratio];

//         // 1. 寻找 8 个旧 block 中最大的 scale
//         for (int j = 0; j < group_ratio; ++j) {
//             float s = get_scale(src_sub_group[j].d);
//             if (s > max_s) max_s = s;
//         }

//         dst[i].d = (uint16_t)max_s; // 设置新的共享 scale

//         // 2. 重新量化 int8 权重
//         for (int j = 0; j < group_ratio; ++j) {
//             float old_s = get_scale(src_sub_group[j].d);
//             float rescale_factor = (max_s > 0) ? (old_s / max_s) : 0.0f;

//             for (int k = 0; k < 32; ++k) {
//                 // 计算新的 int8 值：new_q = old_q * (old_scale / new_scale)
//                 float rescaled_val = (float)src_sub_group[j].qs[k] * rescale_factor;
                
//                 // 四舍五入并截断到 int8 范围
//                 int32_t q_new = (int32_t)std::round(rescaled_val);
//                 dst[i].qs[j * 32 + k] = (int8_t)std::max(-128, std::min(127, q_new));
//             }
//         }
//     }
// }
inline uint16_t float_to_half(float f) {
    // 使用 ARM NEON 指令将单个 float32 转换为 float16
    // vcvt_f16_f32 是 AArch64 标准指令
    __fp16 f16 = (__fp16)f;
    uint16_t res;
    std::memcpy(&res, &f16, 2);
    return res;
}

template <int NEW_G>
void convert_q8_to_large_block(
    const block_q8_0* src,
    block_q8_0_x<NEW_G>* dst,
    int64_t n_elements
) {
    static_assert(NEW_G % 32 == 0, "NEW_G must be a multiple of 32");

    constexpr int group_ratio  = NEW_G / 32;
    const int64_t n_new_blocks = n_elements / NEW_G;

    for (int64_t i = 0; i < n_new_blocks; ++i) {
        const block_q8_0* src_group = src + i * group_ratio;

        // 1. 找最大 scale
        float max_s = 0.0f;
        for (int j = 0; j < group_ratio; ++j) {
            float s = ggml_fp16_to_fp32(src_group[j].d);
            if (s > max_s) max_s = s;
        }

        dst[i].d = float_to_half(max_s);

        // 2. 重新量化
        for (int j = 0; j < group_ratio; ++j) {
            float old_s          = ggml_fp16_to_fp32(src_group[j].d);
            float rescale_factor = (max_s > 0.0f) ? (old_s / max_s) : 0.0f;

            for (int k = 0; k < 32; ++k) {
                float   val   = (float)src_group[j].qs[k] * rescale_factor;
                int32_t q_new = (int32_t)std::round(val);
                dst[i].qs[j * 32 + k] = (int8_t)std::max(-128, std::min(127, q_new));
            }
        }
    }
}

template <int N>
void quantize_row_q8_0_custom(const float* src, block_q8_0_x<N>* dst, int64_t n) {
    const int num_blocks = n / N;

    for (int i = 0; i < num_blocks; ++i) {
        float amax = 0.0f;
        const float* x = src + i * N;

        for (int j = 0; j < N; ++j) {
            amax = std::max(amax, std::abs(x[j]));
        }

        const float d = amax / 127.0f;
        dst[i].d = float_to_half(d);
        const float id = d > 0.0f ? 1.0f / d : 0.0f;

        for (int j = 0; j < N; ++j) {
            int rounded = (int)std::lround(x[j] * id);
            dst[i].qs[j] = (int8_t)std::max(-128, std::min(127, rounded));
        }
    }
}

template <int N>
void extract_q8_0_custom(
    const block_q8_0_x<N>* src,
    int64_t n,
    int8_t* dst_int8,
    float* dst_scales
) {
    const int num_blocks = n / N;

    for (int i = 0; i < num_blocks; ++i) {
        dst_scales[i] = ggml_fp16_to_fp32(src[i].d);
        // std::memcpy(dst_int8 + i * N, src[i].qs, N);
        for (int j = 0; j < N; j++) {
            dst_int8[i * N + j] = src[i].qs[j];
        }
    }
}
template void quantize_row_q8_0_custom<256> (const float*, block_q8_0_x<256>*,  int64_t);
template void quantize_row_q8_0_custom<512> (const float*, block_q8_0_x<512>*,  int64_t);
template void quantize_row_q8_0_custom<1024>(const float*, block_q8_0_x<1024>*, int64_t);
template void extract_q8_0_custom<256> (const block_q8_0_x<256>*, int64_t, int8_t*, float*);
template void extract_q8_0_custom<512> (const block_q8_0_x<512>*, int64_t, int8_t*, float*);
template void extract_q8_0_custom<1024>(const block_q8_0_x<1024>*, int64_t, int8_t*, float*);
// 别名
inline void convert_q8_to_256(const block_q8_0* src, block_q8_0_x<256>* dst, int64_t n)
{ convert_q8_to_large_block<256>(src, dst, n); }

inline void convert_q8_to_512(const block_q8_0* src, block_q8_0_x<512>* dst, int64_t n)
{ convert_q8_to_large_block<512>(src, dst, n); }

inline void convert_q8_to_1024(const block_q8_0* src, block_q8_0_x<1024>* dst, int64_t n)
{ convert_q8_to_large_block<1024>(src, dst, n); }

// Task queue
struct matmul_task_t {
    uint64_t input_dma;
    uint64_t weight_dma;
    int M, K, N, I;//这里的M是输入的，和后面不同，N是权重的
};

struct rknpu_tasks_t {
    matmul_task_t tasks[PER_TASK_CORE_NUM];
};

struct rknpu_tasks_result_t {
    union{
        int32_t* output[PER_TASK_CORE_NUM];
        float* output_fp16[PER_TASK_CORE_NUM];
    };
};

static std::vector<std::shared_ptr<std::vector<std::tuple<int, int, matmul_task_t>>>> tasks_list;
static std::shared_ptr<std::vector<int32_t*>> npu_tasks_shared;
static std::shared_ptr<std::vector<float*>> npu_tasks_shared_fp16;

// Double buffering control
static int TASKS_LOCAL_PER_NUM = 3;
static std::atomic<int> buffer_free[2];
static int npu_domain_id = 0;

// Thread synchronization
static std::mutex npu_worker_mtx;
static std::condition_variable npu_cv;
static std::mutex cpu_worker_mtx;
static std::condition_variable cpu_cv;
static std::atomic<bool> npu_stop_flag{false};
static std::thread npu_worker_thread;
std::vector<std::string> quanted_tensors;
static std::atomic<int> input_type = 0;
// Forward declarations
static rknpu_tasks_result_t rknpu_matmul(rknpu_tasks_t tasks, int domain_id, int output_index);
static rknpu_tasks_result_t rknpu_matmul_fp16(rknpu_tasks_t tasks, int domain_id, int output_index);
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

inline int fp_16_index = 0;
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
        int current_type = input_type.load(std::memory_order_acquire);
        int index = 0;
        if(current_type){
            npu_tasks_shared = std::make_shared<std::vector<int32_t*>>(tasks->size(), nullptr);
        }else{
            npu_tasks_shared_fp16 = std::make_shared<std::vector<float*>>(tasks->size(), nullptr);
            // index = fp_16_index;
        }
        

        std::cout << " tasks to submit to NPU, total tasks: " << TASKS_LOCAL_PER_NUM  << std::endl;
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
                if(current_type){
                    auto outq = rknpu_matmul(_t, npu_domain_id, index * PER_TASK_CORE_NUM);
                
                    // Store output pointers
                    std::lock_guard<std::mutex> lock(cpu_worker_mtx);
                    for (int off = 0; off < TASKS_LOCAL_PER_NUM && off + t < (int)tasks->size(); off++) {
                        npu_tasks_shared->at(t + off) = outq.output[off];
                            
                    }
                }else{
                    auto outq = rknpu_matmul_fp16(_t, npu_domain_id, index * PER_TASK_CORE_NUM);
                
                    // Store output pointers
                    std::lock_guard<std::mutex> lock(cpu_worker_mtx);
                    for (int off = 0; off < TASKS_LOCAL_PER_NUM && off + t < (int)tasks->size(); off++) {
                        npu_tasks_shared_fp16->at(t + off) = outq.output_fp16[off];
                    }
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
// static void extract_q8_0(
//     const block_q8_0* blocks,
//     int nrows, int ncols,
//     int8_t* int8_out,
//     float* scales_out) {
    
//     const int QK = 32;  // Q8_0 块大小
//     const int nb = ncols / QK;  // 每行的块数
    
//     // 串行处理（避免 OpenMP 并行访问 mmap 内存的潜在问题）
//     for (int i = 0; i < nrows; i++) {
//         for (int j = 0; j < nb; j++) {
//             const block_q8_0& blk = blocks[i * nb + j];
            
//             // 提取 scale（FP16 → FP32）
//             scales_out[i * nb + j] = GGML_FP16_TO_FP32(blk.d);
            
//             // 逐字节拷贝 INT8 数据（避免对齐问题）
//             int8_t* dst = int8_out + i * ncols + j * QK;
//             const int8_t* src = blk.qs;
//             for (int k = 0; k < QK; k++) {
//                 dst[k] = src[k];
//             }
//         }
//     }
// }

// void extract_q8_0_custom_256(
//     const block_q8_0_256* src, 
//     int64_t n,                // 总元素个数 (N * K)
//     int8_t* dst_int8,         // 存放 int8 权重的 buffer
//     float* dst_scales         // 存放 fp32 scale 的 buffer
// ) {
//     const int group_size = 256;
//     const int num_blocks = n / group_size;

//     for (int i = 0; i < num_blocks; ++i) {
//         // 1. 提取并转换 Scale (fp16 -> fp32)
//         // 注意：这里需要你环境中的 fp16 转换函数，或者使用 ggml_fp16_to_fp32
//         dst_scales[i] = ggml_fp16_to_fp32(src[i].d);

//         // 2. 拷贝 int8 数据
//         // 使用 memcpy 通常比逐个赋值快得多
//         std::memcpy(dst_int8 + i * group_size, src[i].qs, group_size);
//     }
// }

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

inline int weight_fp16(int C, int k, int c) {
    int dst = 0;
    int kpg = (k / 16);
    int cpg = (c / 32);
    dst = ((cpg * 32) * 16) + (kpg * 16 * C);
    dst = dst + (c % 32) + ((k % 16) * 32);
    return dst;
}

/**
 * @brief 计算 weight 在 NPU 布局中从 (row=0, col=0) 到 (row, col) 的字节偏移
 * 
 * @param C 权重矩阵对齐后的列数 (K_w, 32-aligned)
 * @param row 输出通道索引（权重矩阵行）
 * @param col 输入通道索引（权重矩阵列）
 * @return 字节偏移量
 */
inline uint64_t weight_dma_offset(int C, int row, int col) {
    // weight_int8(C, k=row, c=col) 给出 (row,col) 在 NPU layout 中的线性索引
    return static_cast<uint64_t>(weight_int8(C, row, col));
}

/**
 * @brief 计算 input 在 NPU feature 布局中从 (row=0, col=0) 到 (row, col) 的字节偏移
 * 
 * @param H 输入矩阵行数
 * @param col 输入通道索引（列）
 * @return 字节偏移量（假设 row=0，因为我们批量处理所有行）
 */
inline uint64_t input_dma_offset(int H, int col) {
    // feature_data(H, 16, c=col, h=0) 给出 (0, col) 处的线性索引
    // 必须完整计算：plane 起始 + col%16 的偏移
    return static_cast<uint64_t>(feature_data(H, 16, col, 0));
}

// static void to_npu_feature_layout_fp16(const ggml_fp16_t* src, int M, int K, ggml_fp16_t* dst) {
//     const int K_aligned = (K + 15) & ~15;
//     // Use simple loop instead of memset to avoid NEON SIMD alignment requirements
//     // (IOMMU memory may not satisfy alignment needed by optimized memset)
//     for (size_t i = 0; i < M * K_aligned; i++) {
//         dst[i] = 0;
//     }
    
//     // TEMPORARY FIX: Disable OpenMP
//     // #pragma omp parallel for
//     for (int m = 0; m < M; m++) {
//         for (int k = 0; k < K; k++) {
//             auto target = feature_data(M, 8, k, m);
//             dst[target] = src[m * K + k];
//         }
//     }
// }

static void to_npu_feature_layout_fp16(const ggml_fp16_t* src, int M, int K, ggml_fp16_t* dst) {
    const int K_aligned = (K + 15) & ~15;
    // Use simple loop instead of memset to avoid NEON SIMD alignment requirements
    // (IOMMU memory may not satisfy alignment needed by optimized memset)
    for (size_t i = 0; i < M * K_aligned; i++) {
        dst[i] = 0;
    }
    
    // TEMPORARY FIX: Disable OpenMP
    // #pragma omp parallel for
    auto cur_dst = dst;
    for (int j = 0; j < M;) {
        int _m   = std::min((int)BLOCK_N_FP16, M - j);
        for (int k = 0; k < K;) {
            auto start_m = j;
            auto start_k = k;
            int _k = std::min((int)BLOCK_SHARED_FP16, K - k);
            for (int joff = 0; joff < _m; joff++){
                for (int koff = 0; koff < _k; koff++){
                    auto target = feature_data(_m, 8, koff, joff);
                    cur_dst[target] = src[joff * K + koff + start_m * K + start_k];
                }
            }
            cur_dst += _m * _k;
            k += _k;
        }
        j += _m;
    }
}


static void to_npu_feature_layout(const int8_t* src, int M, int K, int8_t* dst) {
    const int K_aligned = (K + 15) & ~15;
    // Use simple loop instead of memset to avoid NEON SIMD alignment requirements
    // (IOMMU memory may not satisfy alignment needed by optimized memset)
    for (size_t i = 0; i < M * K_aligned; i++) {
        dst[i] = 0;
    }
    
    // TEMPORARY FIX: Disable OpenMP
    // #pragma omp parallel for
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            auto target = feature_data(M, 16, k, m);
            dst[target] = src[m * K + k];
        }
    }
}
static void to_npu_weight_layout_fp16(const ggml_fp16_t* src, int M, int K, ggml_fp16_t* dst) {
    const int K_aligned = (K + 31) & ~31;
    // Use simple loop instead of memset to avoid NEON SIMD alignment requirements
    // (IOMMU memory may not satisfy alignment needed by optimized memset)
    for (size_t i = 0; i < M * K_aligned; i++) {
        dst[i] = 0;
    }
    
    // TEMPORARY FIX: Disable OpenMP
    // #pragma omp parallel for
    // for (int m = 0; m < M; m++) {
    //     for (int k = 0; k < K; k++) {
    //         // ✅ FIXED: weight_int8(C, k=row_index, c=col_index)
    //         // n is row index (output channel), k is col index (input channel)
    //         auto target = weight_int8(K, m, k);
    //         dst[target] = src[m * K + k];
    //     }
    // }
    auto cur_dst = dst;
    for (int j = 0; j < M;) {
        int _m   = std::min((int)BLOCK_WEIGHT_FP16, M - j);
        for (int k = 0; k < K;) {
            auto start_m = j;
            auto start_k = k;
            int _k = std::min((int)BLOCK_SHARED_FP16, K - k);
            for (int joff = 0; joff < _m; joff++){
                for (int koff = 0; koff < _k; koff++){
                    auto target = weight_fp16(_k, joff, koff);
                    // std::cout << "target: " << target << "   src_offset : " << (joff * K + koff + start_m * K + start_k) << std::endl;
                    cur_dst[target] = src[joff * K + koff + start_m * K + start_k];
                }
            }
            cur_dst += _m * _k;
            k += _k;
        }
        j += _m;
    }
}

static void to_npu_weight_layout(const int8_t* src, int M, int K, int8_t* dst) {
    const int K_aligned = (K + 31) & ~31;
    // Use simple loop instead of memset to avoid NEON SIMD alignment requirements
    // (IOMMU memory may not satisfy alignment needed by optimized memset)
    for (size_t i = 0; i < M * K_aligned; i++) {
        dst[i] = 0;
    }
    
    auto cur_dst = dst;
    for (int j = 0; j < M;) {
        int _m   = std::min((int)BLOCK_WEIGHT, M - j);
        for (int k = 0; k < K;) {
            auto start_m = j;
            auto start_k = k;
            int _k = std::min((int)BLOCK_SHARED, K - k);
            for (int joff = 0; joff < _m; joff++){
                for (int koff = 0; koff < _k; koff++){
                    auto target = weight_int8(_k, joff, koff);
                    // std::cout << "target: " << target << "   src_offset : " << (joff * K + koff + start_m * K + start_k) << std::endl;
                    cur_dst[target] = src[joff * K + koff + start_m * K + start_k];
                }
            }
            cur_dst += _m * _k;
            k += _k;
        }
        j += _m;
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
    uint64_t regcmd_dma = domain->regcmd->iommu_addr;
    size_t regcmd_size = domain->regcmd->size;
    
    rknpu_task* tasks_va = static_cast<rknpu_task*>(domain->tasks_mem->virtual_addr);
    uint64_t tasks_obj = domain->tasks_mem->obj_addr;  // mem_obj_handle 现在是 uint32_t
    
    std::cout << "[rknpu_matmul] Submitting tasks to NPU, Domain ID: " << domain_id 
              << ", regcmd VA: " << static_cast<void*>(regcmd_va) 
              << ", regcmd DMA: " << std::hex << regcmd_dma << std::dec 
              << ", tasks VA: " << static_cast<void*>(tasks_va) 
              << ", tasks OBJ: " << tasks_obj << std::endl;

    int32_t* output_va = static_cast<int32_t*>(domain->output->virtual_addr);
    uint64_t output_dma = domain->output->iommu_addr;

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
        size_t per_task_output_size = m * ((n < 4) ? 4 : n);  // 每个任务最大输出元素数
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
        .task_number = 1,  // ✅ 修复：使用实际任务数
        .task_counter = 0,
        .priority = 0,
        .task_obj_addr = tasks_obj,  // 任务描述符对象句柄
        .iommu_domain_id = static_cast<uint32_t>(domain_id),  // IOMMU 域 ID
        .reserved = 0,
        .task_base_addr = 0,
        .hw_elapse_time = 0,
        .core_mask = core_mask,  // NPU 核心掩码
        .fence_fd = -1,
        .subcore_task =  // 子核心任务分配：每个核心分配的任务数
            {
                {0, task_num >= 1},  // 核心 0：如果有至少 1 个任务
                {1, task_num >= 2},  // 核心 1：如果有至少 2 个任务
                {2, task_num >= 3},  // 核心 2：如果有 3 个任务
                {0, 0},
                {0, 0},
            },
    };
    auto start_time = ggml_time_us();
    // 调用 ioctl 提交给 RKNPU 驱动（阻塞等待 NPU 完成）
    rknpu_ioctl(DRM_IOCTL_RKNPU_SUBMIT, &submit, domain_id);
    auto end_time = ggml_time_us();
    std::cout << "[rknpu_matmul] NPU execution completed, time: " << (end_time - start_time) / 1000.0 << " ms" << std::endl;
    // ===== 第3步：返回 NPU 输出指针 =====
    for (int t = 0; t < task_num; t++) {
        auto &tsk = tasks.tasks[t];
        size_t per_task_output_size = tsk.M * ((tsk.N < 4) ? 4 : tsk.N);  // 每个任务输出元素数
        
        // 计算输出指针：基地址 + 偏移
        result.output[t] = output_va + (t + output_index) * per_task_output_size;
        
        // 无效化输出缓冲区的 Cache，确保读取到 NPU 写入的最新数据
        // size_t buffer_size = per_task_output_size * sizeof(int32_t);
        // invalid_cache(result.output[t], buffer_size);
    }
    // ===== 第3步：返回 NPU 输出指针并打印结果 =====
    return result;  // 返回结果
}


static rknpu_tasks_result_t rknpu_matmul_fp16(rknpu_tasks_t tasks, int domain_id = 0, int output_index = 0) {
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
    uint64_t regcmd_dma = domain->regcmd->iommu_addr;
    size_t regcmd_size = domain->regcmd->size;
    
    rknpu_task* tasks_va = static_cast<rknpu_task*>(domain->tasks_mem->virtual_addr);
    uint64_t tasks_obj = domain->tasks_mem->obj_addr;  // mem_obj_handle 现在是 uint32_t
    
    std::cout << "[rknpu_matmul] Submitting tasks to NPU, Domain ID: " << domain_id 
              << ", regcmd VA: " << static_cast<void*>(regcmd_va) 
              << ", regcmd DMA: " << std::hex << regcmd_dma << std::dec 
              << ", tasks VA: " << static_cast<void*>(tasks_va) 
              << ", tasks OBJ: " << tasks_obj << std::endl;

    float* output_va = static_cast<float*>(domain->output->virtual_addr);
    uint64_t output_dma = domain->output->iommu_addr;

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
        
        if (gen_matmul_fp16(&params) != 0) {
            fprintf(stderr, "[rknpu_matmul] Error: gen_matmul_fp16 failed\n");
            break;
        }
        
        // ✅ 设置输入/权重/输出的 DMA 地址（替代 RegCmd::setupAddr）
        // 计算每个任务的输出偏移：假设每个任务最多输出 m*n 个 fp16（占用 m*n*2 字节）
        size_t per_task_output_size = m * ((n < 4) ? 4 : n);  // 每个任务最大输出元素数
        uint64_t task_output_dma = output_dma + (t + output_index) * per_task_output_size * sizeof(float);
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
        .task_number = 1,  // ✅ 修复：使用实际任务数
        .task_counter = 0,
        .priority = 0,
        .task_obj_addr = tasks_obj,  // 任务描述符对象句柄
        .iommu_domain_id = static_cast<uint32_t>(domain_id),  // IOMMU 域 ID
        .reserved = 0,
        .task_base_addr = 0,
        .hw_elapse_time = 0,
        .core_mask = core_mask,  // NPU 核心掩码
        .fence_fd = -1,
        .subcore_task =  // 子核心任务分配：每个核心分配的任务数
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
        size_t per_task_output_size = tsk.M * ((tsk.N < 4) ? 4 : tsk.N);  // 每个任务输出元素数
        
        // 计算输出指针：基地址 + 偏移
        result.output_fp16[t] = output_va + (t + output_index) * per_task_output_size;
        
        // 无效化输出缓冲区的 Cache，确保读取到 NPU 写入的最新数据
        // size_t buffer_size = per_task_output_size * sizeof(int32_t);
        // invalid_cache(result.output[t], buffer_size);
    }
    // ===== 第3步：返回 NPU 输出指针并打印结果 =====
    return result;  // 返回结果
}



// 根据 DMA 地址在当前 Domain 中反查 CPU 虚拟地址
static void* find_va_by_dma(Domain* domain, uint64_t dma_addr) {
    if (!domain) return nullptr;

    // 1. 检查是否在 Input 缓冲区 (LeftMemory)
    if (domain->input && dma_addr >= domain->input->iommu_addr && 
        dma_addr < domain->input->iommu_addr + domain->input->size) {
        return (uint8_t*)domain->input->virtual_addr + (dma_addr - domain->input->iommu_addr);
    }

    // 2. 检查是否在 Output 缓冲区 (LeftMemory)
    if (domain->output && dma_addr >= domain->output->iommu_addr && 
        dma_addr < domain->output->iommu_addr + domain->output->size) {
        return (uint8_t*)domain->output->virtual_addr + (dma_addr - domain->output->iommu_addr);
    }

    // 3. 检查模型权重 (tensors map)
    // 对应 Domain 里的：std::map<std::string, std::tuple<ggml_tensor*, TensorStorage*, IommuConfig*>> tensors;
    for (auto const& [name, info] : domain->tensors) {
        IommuConfig* cfg = std::get<2>(info);
        if (cfg && dma_addr >= cfg->iommu_addr && dma_addr < cfg->iommu_addr + WEIGHT_SIZE) { // WEIGHT_SIZE 作为上限安全检查
            // IommuConfig 本身不存 VA，但我们可以通过绑定的 ggml_tensor 获取
            ggml_tensor* t = std::get<0>(info);
            if (t && t->data) {
                return (uint8_t*)t->data + (dma_addr - cfg->iommu_addr);
            }
        }
    }

    return nullptr;
}

// ============================================================================
// Q8_0 Path: Parallel matmul with double buffering
// ============================================================================
static void compute_matmul_fp16_parallel(
    const struct ggml_tensor* src0,  // weight (M x K, Q8_0, pre-quantized with IOMMU)
    const struct ggml_tensor* src1,  // input (N x K, FP32)
    struct ggml_tensor* dst,         // output (N x M, FP32)
    int domain_id) {
    

    const int M = 1536;  // weight rows (output dimension when transposed)
    const int K = 1536;  // weight cols = input cols (shared dimension)
    const int N = 2400;  // input rows (batch size)
    size_t src0_size = M * K * sizeof(ggml_fp16_t);
    std::vector<ggml_fp16_t> weight_f16(M * K);

    FILE *f_w = fopen("/mnt/nvme/teacache_input.bin", "rb");
    if (f_w) {
        fread(weight_f16.data(), 1, src0_size, f_w);
        fclose(f_w);
        printf(">>> Loaded weight: %zu bytes\n", src0_size);
    } else {
        printf(">>> Error: Could not open teacache_weight.bin\n");
    }

    // --- 2. 加载 Input (src1) ---
    // 假设 src1 原始形状是 [K, N]，类型是 F16 (基于你之前的 dump 代码)
    size_t src1_size = N * K * sizeof(ggml_fp16_t);
    size_t tmp_size = N * K * sizeof(float);  // 如果需要临时存储 FP32 版本
    std::vector<ggml_fp16_t> input_fp16(N * K);
    std::vector<float> input_fp32(N * K);  // 如果需要转换为 FP32
    FILE *f_i = fopen("/mnt/nvme/teacache_weight.bin", "rb");
    if (f_i) {
        fread(input_fp32.data(), 1, tmp_size, f_i);
        fclose(f_i);
        printf(">>> Loaded input: %zu bytes\n", tmp_size);
    } else {
        printf(">>> Error: Could not open teacache_input.bin\n");
    }
    if(src1->type == GGML_TYPE_F16){
        memcpy(input_fp16.data(), src1->data, N * K * sizeof(ggml_fp16_t));
    }else{
        ggml_cpu_fp32_to_fp16((const float*)input_fp32.data(), input_fp16.data(), N * K);
    }
    // --- 1. 打印 Weight (src0) 前 10 个元素 ---
    if (!weight_f16.empty()) {
        printf(">>> Weight (src0) first 10 elements:\n  ");
        for (int i = 0; i < 10 && i < (M * K); i++) {
            // 转换并打印
            float val = ggml_fp16_to_fp32(weight_f16[i]);
            printf("[%d]: %.6f (0x%04X)  ", i, val, weight_f16[i]);
            if ((i + 1) % 5 == 0) printf("\n  ");
        }
        printf("\n");
    }

    // --- 2. 打印 Input (src1) 前 10 个元素 ---
    if (!input_fp16.empty()) {
        printf(">>> Input (src1) first 10 elements:\n  ");
        for (int i = 0; i < 10 && i < (N * K); i++) {
            // 转换并打印
            float val = ggml_fp16_to_fp32(input_fp16[i]);
            printf("[%d]: %.6f (0x%04X)  ", i, val, input_fp16[i]);
            if ((i + 1) % 5 == 0) printf("\n  ");
        }
        printf("\n");
    }
    // GGML dimensions (physical storage)

    // const int M = src0->ne[1];  // weight rows (output dimension when transposed)
    // const int K = src0->ne[0];  // weight cols = input cols (shared dimension)
    // const int N = src1->ne[1];  // input rows (batch size)
    // const int QK = 512;
    
    

    // std::vector<ggml_fp16_t> input_fp16(N * K, 0);
    // if(src1->type == GGML_TYPE_F16){
    //     memcpy(input_fp16.data(), src1->data, N * K * sizeof(ggml_fp16_t));
    // }else{
    //     ggml_cpu_fp32_to_fp16((const float*)src1->data, input_fp16.data(), N * K);
    // }

    
    // std::vector<ggml_fp16_t> weight_f16(M * K, 0);

    // if (src0->type == GGML_TYPE_F16) {
    //     // 如果本来就是 FP16，直接拷贝
    //     memcpy(weight_f16.data(), src0->data, M * K * sizeof(ggml_fp16_t));
    // }
    // else if (src0->type == GGML_TYPE_BF16) {
    //     fprintf(stderr, "[NPU] Converting Weight: BF16 -> FP16\n");
    //     const uint16_t * bf16_ptr = (const uint16_t *)src0->data;

    //     for (int i = 0; i < M * K; ++i) {
    //         // 1. BF16 转 FP32
    //         // BF16 的比特位就是 FP32 的高 16 位
    //         uint32_t f32_bits = (uint32_t)bf16_ptr[i] << 16;
    //         float f32;
    //         memcpy(&f32, &f32_bits, sizeof(float));

    //         // 2. FP32 转 FP16 (交给 NPU)
    //         weight_f16[i] = ggml_fp32_to_fp16(f32);
    //     }
    // }
    // else if (src0->type == GGML_TYPE_F32) {
    //     // 如果是 FP32，转换为 FP16
    //     fprintf(stderr, "[NPU] Converting Weight: FP32 -> FP16\n");
    //     ggml_cpu_fp32_to_fp16((const float*)src0->data, weight_f16.data(), M * K);
    // }
    // else {
    //     fprintf(stderr, "[NPU] Error: Weight tensor should be pre-quantized to FP16, got type %d\n", src0->type);
    //     throw std::runtime_error("Weight tensor not pre-quantized");
    // }
    
    auto convert_s_time = ggml_time_us();
    // Step 3: Convert to NPU layout
    const int K_in = (K + 15) & ~15;
    const int K_w = (K + 31) & ~31;
    
    std::vector<ggml_fp16_t> input_npu(N * K_in, 0);
    to_npu_feature_layout_fp16(input_fp16.data(), N, K, input_npu.data());
    
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
                size_t fp16_size = ggml_nbytes(src0);  // Size of FP16 blocks
                size_t npu_layout_size = M * K_w;     // Size needed for NPU layout (int8 only)
                
                fprintf(stderr, "[NPU] In-place conversion check: FP16=%zu bytes, NPU layout needs=%zu bytes\n", 
                        fp16_size, npu_layout_size);
                
                if (npu_layout_size > fp16_size) {
                    fprintf(stderr, "[NPU] ERROR: Not enough space for in-place conversion\n");
                    throw std::runtime_error("Insufficient space for NPU layout conversion");
                }
                
                // Step 4.2: Convert FP16 blocks → NPU layout IN-PLACE
                // Since we already extracted weight_fp16 from FP16 blocks,
                // we can directly convert it to NPU layout and write to src0->data
                fprintf(stderr, "[NPU] Converting tensor %s to NPU layout in-place...\n", src0->name);
                std::cout << "start conversation" << std::endl;
                // Write NPU layout directly to tensor->data (overwrites FP16 blocks)
                to_npu_weight_layout_fp16(weight_f16.data(), M, K, (ggml_fp16_t*)src0->data);
                
                printf("--- Full NPU Weight Layout Data (Total 128 elements) ---\n");

                ggml_fp16_t* p = (ggml_fp16_t*)src0->data;

                for (int i = 0; i < 128; ++i) {
                    // 将 fp16 转换为 float 以便打印
                    float val = ggml_fp16_to_fp32(p[i]);
                    
                    // 每 8 个元素换一行，方便观察对齐（NPU 通常以 8 或 16 为对齐单位）
                    printf("%8.4f ", val);
                    
                    if ((i + 1) % 8 == 0) {
                        printf(" | Index: %d\n", i);
                    }
                }

                printf("--- End of Data ---\n");
                // Use the existing IOMMU DMA address
                weight_dma_base = iommu_config->iommu_addr;
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
    

    auto convert_e_time = ggml_time_us();
    std::cout << "Layout conversion time: " << (convert_e_time - convert_s_time) / 1000.0 << " ms" << std::endl;
    
    // ✅ Step 4.5: Get Domain for input buffer (use specialized IOMMU-mapped buffer)
    Domain* input_domain = find_domain_by_id(domain_id);
    if (!input_domain || !input_domain->input) {
        fprintf(stderr, "[NPU] ERROR: Domain %d not found or input buffer not allocated\n", domain_id);
        throw std::runtime_error("Input IOMMU buffer not available");
    }
    
    // Check buffer size
    size_t input_size = N * K_in * 2;
    if (input_size > input_domain->input->size) {
        fprintf(stderr, "[NPU] ERROR: Input size %zu exceeds buffer size %zu\n", 
                input_size, input_domain->input->size);
        throw std::runtime_error("Input buffer overflow");
    }
    
    // Copy input data to IOMMU-mapped buffer
    memcpy(input_domain->input->virtual_addr, input_npu.data(), input_size);

    flush_cache(input_domain->input->virtual_addr, input_size);
    uint64_t input_dma_base = input_domain->input->iommu_addr;
    
    fprintf(stderr, "[NPU] Input buffer: VA=%p, DMA=0x%lx, size=%zu\n", 
            input_domain->input->virtual_addr, 
            input_domain->input->iommu_addr, 
            input_size);
    
    
    // Step 5: Build task list (block splitting)
    auto tasks = std::make_shared<std::vector<std::tuple<int, int, matmul_task_t>>>();
    // tasks->reserve(ceil_int(M, BLOCK_WEIGHT_FP16) * ceil_int(K, BLOCK_SHARED_FP16));
    tasks->reserve(ceil_int(M, BLOCK_WEIGHT_FP16) * ceil_int(K, BLOCK_SHARED_FP16) * ceil_int(N, BLOCK_N_FP16));

    // for (int j = 0; j < M; j += BLOCK_WEIGHT_FP16) {
    //     int _n = std::min(M - j, BLOCK_WEIGHT_FP16);
    //     for (int k = 0; k < K; k += BLOCK_SHARED_FP16) {
    //         int _k = std::min(K - k, BLOCK_SHARED_FP16);
        
    //         tasks->emplace_back(j, k, matmul_task_t{
    //             input_dma_base, weight_dma_base, N, _k, _n
    //         });
    //         input_dma_base += N * _k * sizeof(ggml_fp16_t) ;
    //         weight_dma_base += _k * _n * sizeof(ggml_fp16_t);
    //     }
    //     input_dma_base = input_domain->input->iommu_addr;
    // }
    for (int i = 0; i < N; i += BLOCK_N_FP16) {
        int _n_i = std::min(N - i, BLOCK_N_FP16);
        uint64_t cur_input_start = input_dma_base;
        uint64_t cur_weight_start = weight_dma_base;
        // 要修改input块内的to_layout逻辑
        for (int j = 0; j < M; j += BLOCK_WEIGHT_FP16) {
            int _n = std::min(M - j, BLOCK_WEIGHT_FP16);
            for (int k = 0; k < K; k += BLOCK_SHARED_FP16) {
                int _k = std::min(K - k, BLOCK_SHARED_FP16);
            
                tasks->emplace_back(j, k, matmul_task_t{
                    input_dma_base, weight_dma_base, _n_i, _k, _n, i
                });
                input_dma_base += _n_i * _k * sizeof(ggml_fp16_t) ;
                weight_dma_base += _k * _n * sizeof(ggml_fp16_t);
            }
            input_dma_base = cur_input_start;
        }
        weight_dma_base = cur_weight_start;
        input_dma_base += _n_i * K_in * sizeof(ggml_fp16_t);
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
    auto submit_time = ggml_time_us();
    std::cout << "Tasks submitted to NPU worker, time: " << (submit_time - convert_e_time) / 1000.0 << " ms" << std::endl;
    // Step 9: CPU processes NPU output with double buffering
    int index = 0;
    float* dst_data = (float*)dst->data;
    uint64_t wait_time = 0;
    uint64_t cpu_time = 0;
    std::cout << "task_size: " << tasks->size() << std::endl; 
    // for (int t = 0; t < (int)tasks->size(); t += TASKS_LOCAL_PER_NUM) {
    //     {
    //         auto s_time = ggml_time_us();
    //         std::unique_lock<std::mutex> lock(cpu_worker_mtx);
    //         cpu_cv.wait(lock, [&] { 
    //             return npu_tasks_shared_fp16.use_count() != 0 && 
    //                    npu_tasks_shared_fp16->at(t) != nullptr && 
    //                    buffer_free[index].load(std::memory_order_acquire) == 1;
    //         });
    //         auto e_time = ggml_time_us();
    //         wait_time += (e_time - s_time);
    //     }
    //     // char fname[128];
    //     // snprintf(fname, sizeof(fname), "/mnt/nvme/stable-diffusion-npu.cpp/task_log_t%04d.txt", t);
    //     // FILE* flog = fopen(fname, "w");

    //     auto s_time = ggml_time_us(); 
    //     for (int toff = 0; toff < TASKS_LOCAL_PER_NUM && t + toff < (int)tasks->size(); toff++) {
    //         const auto [j, k, task] = tasks->at(t + toff);
    //         auto i = task.I; 
    //         float* task_output = npu_tasks_shared_fp16->at(t + toff);
            
    //         // Invalidate cache to ensure CPU reads NPU-written data
    //         invalid_cache(task_output, N * std::min(M - j, BLOCK_WEIGHT_FP16) * sizeof(float));
            
    //         int current_m_task = std::min(M - j, BLOCK_WEIGHT_FP16);
    //         int current_n_task = std::min(N - i, BLOCK_N_FP16);
         
            
    //         // Process output with NEON optimization
    //         // For each weight row in [j, j+joff_max), get the scale for block k/32
    //         // NOTE: NPU output layout is NCHW4 with H=N (batch size)!
    //         for (int ioff = 0; ioff < current_n_task; ioff++) { // 行遍历 (Height)
    //             for (int joff = 0; joff < current_m_task; joff ++) { 
    //                 auto cur_M = j + joff;
    //                 auto target = feature_data(current_n_task, 4 , joff, ioff);
    //                 auto cur_offset_in_result = (ioff + i) * M + cur_M ;
    //                 // std::cout << "cur_offset_in_result:" << cur_offset_in_result << "    value:"<< (float)task_output[target] << "i:" <<i << " cur_M" << cur_M <<std::endl;
    //                 // std::cout << "weight_scale_offset:" << weight_scale_offset << "    weight_scale:" << weight_scales[weight_scale_offset] << std::endl;
    //                 auto output = (float)task_output[target];
    //                 // if (flog) {
    //                 //     fprintf(flog, "cur_offset_in_result,target,i,ioff,joff,output: %d,%d,%d,%d,%d,%.6f\n",
    //                 //             cur_offset_in_result, target, i, ioff, joff, output);
    //                 // }
    //                 // std::cout << "cur_offset_in_result, target, i,  N , M , output: " << cur_offset_in_result << "," << target << "," << i << "," <<  ioff << "," << joff << "," << output << std::endl; 
    //                 dst_data[cur_offset_in_result] += output;
    //             }
    //         }
    //     }
    //     // fflush(flog);   // 强制刷新缓冲区
    //     // fclose(flog);   // 关闭文件
    //     auto e_time = ggml_time_us();
    //     cpu_time += (e_time - s_time);
    //     {
    //         std::lock_guard<std::mutex> lock(npu_worker_mtx);
    //         buffer_free[index].store(0, std::memory_order_release);
    //         npu_cv.notify_one();
    //     }
    //     index = (index + 1) & 0x1;
    // }
    for (int t = 0; t < (int)tasks->size(); t += TASKS_LOCAL_PER_NUM) {
        {
            auto s_time = ggml_time_us();
            std::unique_lock<std::mutex> lock(cpu_worker_mtx);
            cpu_cv.wait(lock, [&] { 
                return npu_tasks_shared_fp16.use_count() != 0 && 
                    npu_tasks_shared_fp16->at(t) != nullptr && 
                    buffer_free[index].load(std::memory_order_acquire) == 1;
            });
            auto e_time = ggml_time_us();
            wait_time += (e_time - s_time);
        }

        auto s_time = ggml_time_us();

        for (int toff = 0; toff < TASKS_LOCAL_PER_NUM && t + toff < (int)tasks->size(); toff++) {
            const auto [j, k, task] = tasks->at(t + toff);
            auto i = task.I;
            float* task_output = npu_tasks_shared_fp16->at(t + toff);

            invalid_cache(task_output, N * std::min(M - j, BLOCK_WEIGHT_FP16) * sizeof(float));

            int current_m_task = std::min(M - j, BLOCK_WEIGHT_FP16);
            int current_n_task = std::min(N - i, BLOCK_N_FP16);

            // ---------------------------------------------------------------
            // 外层按 joff 步长4并行，对齐 int8 风格
            // NPU 输出布局: NCHW4，H=current_n_task，W_block=4
            // feature_data(current_n_task, 4, joff, ioff) 随 ioff 每次 +4
            // ---------------------------------------------------------------
    #pragma omp parallel for num_threads(4)
            for (int joff = 0; joff < current_m_task; joff += 4) {
                const int cur_M_base = j + joff;
                const int lanes = std::min(4, current_m_task - joff);

                // 初始 feature_offset，ioff=0 时的偏移
                int feature_offset = feature_data(current_n_task, 4, joff, 0);

                for (int ioff = 0; ioff < current_n_task; ioff++) {
                    // Prefetch 下一行 dst 写目标 和 NPU 输出
                    if (ioff + LOOKAHEAD < current_n_task) {
                        __builtin_prefetch(
                            &dst_data[(ioff + i + LOOKAHEAD) * M + cur_M_base],
                            1 /*write*/, 1 /*keep*/);
                        if ((ioff % 4) == 0) {
                            __builtin_prefetch(
                                &task_output[feature_offset + LOOKAHEAD * 4],
                                0 /*read*/, 1 /*keep*/);
                        }
                    }

                    if (lanes == 4) {
                        // --------------------------------------------------
                        // 满4通道路径：纯 NEON，无分支
                        // task_output 已是 float（fp32），直接 vld1q_f32 加载
                        // --------------------------------------------------
                        float32x4_t v_val = vld1q_f32(&task_output[feature_offset]);

                        float* out_ptr = &dst_data[(ioff + i) * M + cur_M_base];
                        float32x4_t out_old = vld1q_f32(out_ptr);
                        out_old = vaddq_f32(out_old, v_val);
                        vst1q_f32(out_ptr, out_old);
                    } else {
                        // --------------------------------------------------
                        // 边界路径：剩余 lanes < 4，逐元素处理
                        // --------------------------------------------------
                        for (int lane = 0; lane < lanes; lane++) {
                            float output = task_output[feature_offset + lane];
                            dst_data[(ioff + i) * M + cur_M_base + lane] += output;
                        }
                    }

                    feature_offset += 4; // NCHW4 布局，ioff 每步 +4
                }
            }
        }

        auto e_time = ggml_time_us();
        cpu_time += (e_time - s_time);

        {
            std::lock_guard<std::mutex> lock(npu_worker_mtx);
            buffer_free[index].store(0, std::memory_order_release);
            npu_cv.notify_one();
        }
        index = (index + 1) & 0x1;
    }
    std::cout << "Total wait time for NPU tasks: " << wait_time / 1000.0 << " ms" << std::endl;
    std::cout << "Total CPU processing time: " << cpu_time / 1000.0 << " ms" << std::endl;
    npu_tasks_shared_fp16.reset();
}



static void compute_matmul_fp16_parallel_dynamic(
    const struct ggml_tensor* src0,  // weight (M x K, Q8_0, pre-quantized with IOMMU)
    const struct ggml_tensor* src1,  // input (N x K, FP32)
    struct ggml_tensor* dst,         // output (N x M, FP32)
    int domain_id) {
    
    
    // const int M = 1536;  // weight rows (output dimension when transposed)
    // const int K = 1536;  // weight cols = input cols (shared dimension)
    // const int N = 2400;  // input rows (batch size)
    // size_t src0_size = M * K * sizeof(ggml_fp16_t);
    // std::vector<ggml_fp16_t> weight_f16(M * K);

    // FILE *f_w = fopen("/mnt/nvme/teacache_input.bin", "rb");
    // if (f_w) {
    //     fread(weight_f16.data(), 1, src0_size, f_w);
    //     fclose(f_w);
    //     printf(">>> Loaded weight: %zu bytes\n", src0_size);
    // } else {
    //     printf(">>> Error: Could not open teacache_weight.bin\n");
    // }

    // // --- 2. 加载 Input (src1) ---
    // // 假设 src1 原始形状是 [K, N]，类型是 F16 (基于你之前的 dump 代码)
    // size_t src1_size = N * K * sizeof(ggml_fp16_t);
    // size_t tmp_size = N * K * sizeof(float);  // 如果需要临时存储 FP32 版本
    // std::vector<ggml_fp16_t> input_fp16(N * K);
    // std::vector<float> input_fp32(N * K);  // 如果需要转换为 FP32
    // FILE *f_i = fopen("/mnt/nvme/teacache_weight.bin", "rb");
    // if (f_i) {
    //     fread(input_fp32.data(), 1, tmp_size, f_i);
    //     fclose(f_i);
    //     printf(">>> Loaded input: %zu bytes\n", tmp_size);
    // } else {
    //     printf(">>> Error: Could not open teacache_input.bin\n");
    // }
    // if(src1->type == GGML_TYPE_F16){
    //     memcpy(input_fp16.data(), src1->data, N * K * sizeof(ggml_fp16_t));
    // }else{
    //     ggml_cpu_fp32_to_fp16((const float*)input_fp32.data(), input_fp16.data(), N * K);
    // }
    // // --- 1. 打印 Weight (src0) 前 10 个元素 ---
    // if (!weight_f16.empty()) {
    //     printf(">>> Weight (src0) first 10 elements:\n  ");
    //     for (int i = 0; i < 10 && i < (M * K); i++) {
    //         // 转换并打印
    //         float val = ggml_fp16_to_fp32(weight_f16[i]);
    //         printf("[%d]: %.6f (0x%04X)  ", i, val, weight_f16[i]);
    //         if ((i + 1) % 5 == 0) printf("\n  ");
    //     }
    //     printf("\n");
    // }

    // // --- 2. 打印 Input (src1) 前 10 个元素 ---
    // if (!input_fp16.empty()) {
    //     printf(">>> Input (src1) first 10 elements:\n  ");
    //     for (int i = 0; i < 10 && i < (N * K); i++) {
    //         // 转换并打印
    //         float val = ggml_fp16_to_fp32(input_fp16[i]);
    //         printf("[%d]: %.6f (0x%04X)  ", i, val, input_fp16[i]);
    //         if ((i + 1) % 5 == 0) printf("\n  ");
    //     }
    //     printf("\n");
    // }
    // GGML dimensions (physical storage)

    const int M = src0->ne[1];  // weight rows (output dimension when transposed)
    const int K = src0->ne[0];  // weight cols = input cols (shared dimension)
    const int N = src1->ne[1];  // input rows (batch size)
    const int QK = 512;
    std::cout << "result-size:" << N * M << std::endl;
    
    

    std::vector<ggml_fp16_t> input_fp16(N * K, 0);
    if(src1->type == GGML_TYPE_F16){
        memcpy(input_fp16.data(), src1->data, N * K * sizeof(ggml_fp16_t));
    }else{
        ggml_cpu_fp32_to_fp16((const float*)src1->data, input_fp16.data(), N * K);
    }

    
    std::vector<ggml_fp16_t> weight_f16(M * K, 0);

    if (src0->type == GGML_TYPE_F16) {
        // 如果本来就是 FP16，直接拷贝
        memcpy(weight_f16.data(), src0->data, M * K * sizeof(ggml_fp16_t));
    }
    else if (src0->type == GGML_TYPE_BF16) {
        fprintf(stderr, "[NPU] Converting Weight: BF16 -> FP16\n");
        const uint16_t * bf16_ptr = (const uint16_t *)src0->data;

        for (int i = 0; i < M * K; ++i) {
            // 1. BF16 转 FP32
            // BF16 的比特位就是 FP32 的高 16 位
            uint32_t f32_bits = (uint32_t)bf16_ptr[i] << 16;
            float f32;
            memcpy(&f32, &f32_bits, sizeof(float));

            // 2. FP32 转 FP16 (交给 NPU)
            weight_f16[i] = ggml_fp32_to_fp16(f32);
        }
    }
    else if (src0->type == GGML_TYPE_F32) {
        // 如果是 FP32，转换为 FP16
        fprintf(stderr, "[NPU] Converting Weight: FP32 -> FP16\n");
        ggml_cpu_fp32_to_fp16((const float*)src0->data, weight_f16.data(), M * K);
    }
    else {
        fprintf(stderr, "[NPU] Error: Weight tensor should be pre-quantized to FP16, got type %d\n", src0->type);
        throw std::runtime_error("Weight tensor not pre-quantized");
    }
    
    auto convert_s_time = ggml_time_us();
    // Step 3: Convert to NPU layout
    const int K_in = (K + 15) & ~15;
    const int K_w = (K + 31) & ~31;
    
    std::vector<ggml_fp16_t> input_npu(N * K_in, 0);
    to_npu_feature_layout_fp16(input_fp16.data(), N, K, input_npu.data());
    
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
                size_t fp16_size = ggml_nbytes(src0);  // Size of FP16 blocks
                size_t npu_layout_size = M * K_w;     // Size needed for NPU layout (int8 only)
                
                fprintf(stderr, "[NPU] In-place conversion check: FP16=%zu bytes, NPU layout needs=%zu bytes\n", 
                        fp16_size, npu_layout_size);
                
                if (npu_layout_size > fp16_size) {
                    fprintf(stderr, "[NPU] ERROR: Not enough space for in-place conversion\n");
                    throw std::runtime_error("Insufficient space for NPU layout conversion");
                }
                
                // Step 4.2: Convert FP16 blocks → NPU layout IN-PLACE
                // Since we already extracted weight_fp16 from FP16 blocks,
                // we can directly convert it to NPU layout and write to src0->data
                fprintf(stderr, "[NPU] Converting tensor %s to NPU layout in-place...\n", src0->name);
                std::cout << "start conversation" << std::endl;
                // Write NPU layout directly to tensor->data (overwrites FP16 blocks)
                to_npu_weight_layout_fp16(weight_f16.data(), M, K, (ggml_fp16_t*)src0->data);
                
                printf("--- Full NPU Weight Layout Data (Total 128 elements) ---\n");

                ggml_fp16_t* p = (ggml_fp16_t*)src0->data;

                for (int i = 0; i < 128; ++i) {
                    // 将 fp16 转换为 float 以便打印
                    float val = ggml_fp16_to_fp32(p[i]);
                    
                    // 每 8 个元素换一行，方便观察对齐（NPU 通常以 8 或 16 为对齐单位）
                    printf("%8.4f ", val);
                    
                    if ((i + 1) % 8 == 0) {
                        printf(" | Index: %d\n", i);
                    }
                }

                printf("--- End of Data ---\n");
                // Use the existing IOMMU DMA address
                weight_dma_base = iommu_config->iommu_addr;
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
    

    auto convert_e_time = ggml_time_us();
    std::cout << "Layout conversion time: " << (convert_e_time - convert_s_time) / 1000.0 << " ms" << std::endl;
    
    // ✅ Step 4.5: Get Domain for input buffer (use specialized IOMMU-mapped buffer)
    Domain* input_domain = find_domain_by_id(domain_id);
    if (!input_domain || !input_domain->input) {
        fprintf(stderr, "[NPU] ERROR: Domain %d not found or input buffer not allocated\n", domain_id);
        throw std::runtime_error("Input IOMMU buffer not available");
    }
    
    // Check buffer size
    size_t input_size = N * K_in * 2;
    if (input_size > input_domain->input->size) {
        fprintf(stderr, "[NPU] ERROR: Input size %zu exceeds buffer size %zu\n", 
                input_size, input_domain->input->size);
        throw std::runtime_error("Input buffer overflow");
    }
    
    // Copy input data to IOMMU-mapped buffer
    memcpy(input_domain->input->virtual_addr, input_npu.data(), input_size);

    flush_cache(input_domain->input->virtual_addr, input_size);
    uint64_t input_dma_base = input_domain->input->iommu_addr;
    
    fprintf(stderr, "[NPU] Input buffer: VA=%p, DMA=0x%lx, size=%zu\n", 
            input_domain->input->virtual_addr, 
            input_domain->input->iommu_addr, 
            input_size);
    
    
    // Step 5: Build task list (block splitting)
    auto tasks = std::make_shared<std::vector<std::tuple<int, int, matmul_task_t>>>();
    // tasks->reserve(ceil_int(M, BLOCK_WEIGHT_FP16) * ceil_int(K, BLOCK_SHARED_FP16));
    tasks->reserve(ceil_int(M, BLOCK_WEIGHT_FP16) * ceil_int(K, BLOCK_SHARED_FP16) * ceil_int(N, BLOCK_N_FP16));

    // for (int j = 0; j < M; j += BLOCK_WEIGHT_FP16) {
    //     int _n = std::min(M - j, BLOCK_WEIGHT_FP16);
    //     for (int k = 0; k < K; k += BLOCK_SHARED_FP16) {
    //         int _k = std::min(K - k, BLOCK_SHARED_FP16);
        
    //         tasks->emplace_back(j, k, matmul_task_t{
    //             input_dma_base, weight_dma_base, N, _k, _n
    //         });
    //         input_dma_base += N * _k * sizeof(ggml_fp16_t) ;
    //         weight_dma_base += _k * _n * sizeof(ggml_fp16_t);
    //     }
    //     input_dma_base = input_domain->input->iommu_addr;
    // }
    for (int i = 0; i < N; i += BLOCK_N_FP16) {
        int _n_i = std::min(N - i, BLOCK_N_FP16);
        uint64_t cur_input_start = input_dma_base;
        uint64_t cur_weight_start = weight_dma_base;
        // 要修改input块内的to_layout逻辑
        for (int j = 0; j < M; j += BLOCK_WEIGHT_FP16) {
            int _n = std::min(M - j, BLOCK_WEIGHT_FP16);
            for (int k = 0; k < K; k += BLOCK_SHARED_FP16) {
                int _k = std::min(K - k, BLOCK_SHARED_FP16);
            
                tasks->emplace_back(j, k, matmul_task_t{
                    input_dma_base, weight_dma_base, _n_i, _k, _n, i
                });
                input_dma_base += _n_i * _k * sizeof(ggml_fp16_t) ;
                weight_dma_base += _k * _n * sizeof(ggml_fp16_t);
            }
            input_dma_base = cur_input_start;
        }
        weight_dma_base = cur_weight_start;
        input_dma_base += _n_i * K_in * sizeof(ggml_fp16_t);
    }
    // Step 7: Initialize output to zero
    std::fill((float*)dst->data, (float*)dst->data + N * M, 0.0f);
    
    // // Step 8: Submit tasks to NPU worker thread
    {
        std::lock_guard<std::mutex> lock(npu_worker_mtx);
        tasks_list.push_back(tasks);
        npu_domain_id = domain_id;
        npu_cv.notify_one();
    }
    auto submit_time = ggml_time_us();
    std::cout << "Tasks submit, time: " << (submit_time - convert_e_time) / 1000.0 << " ms" << std::endl;
    // Step 9: CPU processes NPU output with double buffering
    int index = 0;
    float* dst_data = (float*)dst->data;
    uint64_t npu_wait_time = 0;
    uint64_t npu_cpu_time = 0;
    uint64_t cpu_time = 0;
    std::cout << "task_size: " << tasks->size() << std::endl; 

    std::mutex dst_mtx; // 保护 dst 的累加操作
    std::atomic<int> next_task_idx{0};
    auto total_tasks = tasks->size();

    std::vector<std::atomic<uint64_t>> bitmap(tasks->size());
    auto npu_worker = [&]() {
        for (int t = 0; t < (int)tasks->size(); t += TASKS_LOCAL_PER_NUM) {
            {
                auto s_time = ggml_time_us();
                std::unique_lock<std::mutex> lock(cpu_worker_mtx);
                cpu_cv.wait(lock, [&] { 
                    return npu_tasks_shared_fp16.use_count() != 0 && 
                        npu_tasks_shared_fp16->at(t) != nullptr && 
                        buffer_free[index].load(std::memory_order_acquire) == 1;
                });
                auto e_time = ggml_time_us();
                npu_wait_time += (e_time - s_time);
            }

            auto s_time = ggml_time_us();
            bool flag = true;
            for (int toff = 0; toff < TASKS_LOCAL_PER_NUM && t + toff < (int)tasks->size(); toff++) {
                // if(bitmap[i].load(std::memory_order_relaxed) == 1){
                //     continue;
                // }else{
                //     next_task_idx.fetch_add(TASKS_LOCAL_PER_NUM, std::memory_order_relaxed);
                // }
                if(t + toff != next_task_idx.load(std::memory_order_relaxed) && flag){
                    continue;
                }
                if (flag) {
                    next_task_idx.fetch_add(TASKS_LOCAL_PER_NUM - toff, std::memory_order_relaxed);
                    flag = false;
                }
                std::cout << "npu processing task :" << t + toff << std::endl;

                const auto [j, k, task] = tasks->at(t + toff);
                auto i = task.I;
                float* task_output = npu_tasks_shared_fp16->at(t + toff);

                invalid_cache(task_output, N * std::min(M - j, BLOCK_WEIGHT_FP16) * sizeof(float));

                int current_m_task = std::min(M - j, BLOCK_WEIGHT_FP16);
                int current_n_task = std::min(N - i, BLOCK_N_FP16);

                // ---------------------------------------------------------------
                // 外层按 joff 步长4并行，对齐 int8 风格
                // NPU 输出布局: NCHW4，H=current_n_task，W_block=4
                // feature_data(current_n_task, 4, joff, ioff) 随 ioff 每次 +4
                // ---------------------------------------------------------------
        #pragma omp parallel for num_threads(4)
                for (int joff = 0; joff < current_m_task; joff += 4) {
                    const int cur_M_base = j + joff;
                    const int lanes = std::min(4, current_m_task - joff);

                    // 初始 feature_offset，ioff=0 时的偏移
                    int feature_offset = feature_data(current_n_task, 4, joff, 0);

                    for (int ioff = 0; ioff < current_n_task; ioff++) {
                        // Prefetch 下一行 dst 写目标 和 NPU 输出
                        if (ioff + LOOKAHEAD < current_n_task) {
                            __builtin_prefetch(
                                &dst_data[(ioff + i + LOOKAHEAD) * M + cur_M_base],
                                1 /*write*/, 1 /*keep*/);
                            if ((ioff % 4) == 0) {
                                __builtin_prefetch(
                                    &task_output[feature_offset + LOOKAHEAD * 4],
                                    0 /*read*/, 1 /*keep*/);
                            }
                        }

                        if (lanes == 4) {
                            // --------------------------------------------------
                            // 满4通道路径：纯 NEON，无分支
                            // task_output 已是 float（fp32），直接 vld1q_f32 加载
                            // --------------------------------------------------
                            float32x4_t v_val = vld1q_f32(&task_output[feature_offset]);

                            float* out_ptr = &dst_data[(ioff + i) * M + cur_M_base];
                            float32x4_t out_old = vld1q_f32(out_ptr);
                            out_old = vaddq_f32(out_old, v_val);
                            vst1q_f32(out_ptr, out_old);
                        } else {
                            // --------------------------------------------------
                            // 边界路径：剩余 lanes < 4，逐元素处理
                            // --------------------------------------------------
                            for (int lane = 0; lane < lanes; lane++) {
                                float output = task_output[feature_offset + lane];
                                dst_data[(ioff + i) * M + cur_M_base + lane] += output;
                            }
                        }

                        feature_offset += 4; // NCHW4 布局，ioff 每步 +4
                    }
                }
            }

            auto e_time = ggml_time_us();
            npu_cpu_time += (e_time - s_time);

            {
                std::lock_guard<std::mutex> lock(npu_worker_mtx);
                buffer_free[index].store(0, std::memory_order_release);
                npu_cv.notify_one();
            }
            index = (index + 1) & 0x1;
        }
    };
    // auto npu_worker = [&]() {
    //     while (true) {
    //         auto s_time = ggml_time_us();
    //         int t = next_task_idx.fetch_add(3, std::memory_order_relaxed);
    //         if (t >= total_tasks) break;
    //         auto local_tasks = std::make_shared<std::vector<std::tuple<int, int, matmul_task_t>>>();
    //         for(int toff = 0; toff < 3 && t + toff < total_tasks; toff++) {
    //             const auto& e = tasks->at(t + toff);
    //             local_tasks->push_back(e);
    //         }
            
    //         // 1. 提交到 NPU 硬件 (逻辑同前，此处略)
    //         {
    //             std::lock_guard<std::mutex> lock(npu_worker_mtx);
    //             tasks_list.push_back(local_tasks);
    //             npu_domain_id = domain_id;
    //             npu_cv.notify_one();
    //         }
    //         // 2. 获取 NPU 结果 task_output (FP32)
    //         {
    //             std::unique_lock<std::mutex> lock(cpu_worker_mtx);
    //             cpu_cv.wait(lock, [&] { 
    //                 return npu_tasks_shared_fp16.use_count() != 0 &&
    //                     npu_tasks_shared_fp16->at(0) != nullptr && 
    //                     buffer_free[index].load(std::memory_order_acquire) == 1;
    //             });
    //             std::cout << "NPU worker got task output for tasks "  << std::endl;
 
    //         }
    //         auto e_time = ggml_time_us();
    //         npu_wait_time += (e_time - s_time);
    //         // 3. 累加到 dst (必须加锁或使用 atomic，因为 CPU 可能也在算同一个 (i,j) 的不同 k)
    //         {
    //             for (int toff = 0; toff < TASKS_LOCAL_PER_NUM && t + toff < total_tasks; toff++) {
    //                 const auto [j, k, task] = tasks->at(t + toff);
    //                 auto i = task.I;
    //                 float* task_output = npu_tasks_shared_fp16->at(toff);

    //                 invalid_cache(task_output, N * std::min(M - j, BLOCK_WEIGHT_FP16) * sizeof(float));

    //                 int current_m_task = std::min(M - j, BLOCK_WEIGHT_FP16);
    //                 int current_n_task = std::min(N - i, BLOCK_N_FP16);

    //                 // ---------------------------------------------------------------
    //                 // 外层按 joff 步长4并行，对齐 int8 风格
    //                 // NPU 输出布局: NCHW4，H=current_n_task，W_block=4
    //                 // feature_data(current_n_task, 4, joff, ioff) 随 ioff 每次 +4
    //                 // ---------------------------------------------------------------
    //         #pragma omp parallel for num_threads(4)
    //                 for (int joff = 0; joff < current_m_task; joff += 4) {
    //                     const int cur_M_base = j + joff;
    //                     const int lanes = std::min(4, current_m_task - joff);

    //                     // 初始 feature_offset，ioff=0 时的偏移
    //                     int feature_offset = feature_data(current_n_task, 4, joff, 0);

    //                     for (int ioff = 0; ioff < current_n_task; ioff++) {
    //                         // Prefetch 下一行 dst 写目标 和 NPU 输出
    //                         if (ioff + LOOKAHEAD < current_n_task) {
    //                             __builtin_prefetch(
    //                                 &dst_data[(ioff + i + LOOKAHEAD) * M + cur_M_base],
    //                                 1 /*write*/, 1 /*keep*/);
    //                             if ((ioff % 4) == 0) {
    //                                 __builtin_prefetch(
    //                                     &task_output[feature_offset + LOOKAHEAD * 4],
    //                                     0 /*read*/, 1 /*keep*/);
    //                             }
    //                         }
    //                         {
    //                             std::lock_guard<std::mutex> lk(dst_mtx);
    //                             if (lanes == 4) {
    //                                 // --------------------------------------------------
    //                                 // 满4通道路径：纯 NEON，无分支
    //                                 // task_output 已是 float（fp32），直接 vld1q_f32 加载
    //                                 // --------------------------------------------------
    //                                 float32x4_t v_val = vld1q_f32(&task_output[feature_offset]);

    //                                 float* out_ptr = &dst_data[(ioff + i) * M + cur_M_base];
    //                                 float32x4_t out_old = vld1q_f32(out_ptr);
    //                                 out_old = vaddq_f32(out_old, v_val);
    //                                 vst1q_f32(out_ptr, out_old);
    //                             } else {
    //                                 // --------------------------------------------------
    //                                 // 边界路径：剩余 lanes < 4，逐元素处理
    //                                 // --------------------------------------------------
    //                                 for (int lane = 0; lane < lanes; lane++) {
    //                                     float output = task_output[feature_offset + lane];
    //                                     dst_data[(ioff + i) * M + cur_M_base + lane] += output;
    //                                 }
    //                             }
    //                         }
    //                         feature_offset += 4; // NCHW4 布局，ioff 每步 +4
    //                     }
    //                 }
                    
    //             }
    //         }

    //         {

    //             // auto e_time = ggml_time_us();
    //             // cpu_time += (e_time - s_time);
    //             {
    //                 std::lock_guard<std::mutex> lock(npu_worker_mtx);
    //                 buffer_free[index].store(0, std::memory_order_release);
    //                 npu_cv.notify_one();
    //             }
    //             index = (index + 1) & 0x1;
    //         }
    //         auto e_time_2 = ggml_time_us();
    //         npu_cpu_time += (e_time_2 - s_time);
    //         fp_16_index = index;
    //     }
    // };


    auto input_start = input_fp16.data();
    auto weight_start = weight_f16.data();
    // auto cpu_worker = [&]() {
    //     while (true) {
    //         int t = next_task_idx.fetch_add(1, std::memory_order_relaxed);
    //         if (t >= total_tasks) break;
    //         const auto [j, k, task] = tasks->at(t);//i是N的索引，j是M的索引，k是K的索引
    //         auto i = task.I;
            
    //         // CPU 直接处理这个 (i, j, k) 分块
    //         // 这里的计算必须是累加性质的：dst[i, j] += A[i, k] * B[k, j]
    //         auto cur_n_block_size = std::min(N - i, BLOCK_N_FP16);
    //         auto cur_m_block_size = std::min(M - j, BLOCK_WEIGHT_FP16);
    //         {
    //             std::lock_guard<std::mutex> lk(dst_mtx);
    //             for (int ii = 0; ii < cur_n_block_size; ++ii) {
    //                 for (int jj = 0; jj < cur_m_block_size; ++jj) {
    //                     auto cur_i = i + ii;
    //                     auto cur_j = j + jj;
    //                     auto cur_i_addr = input_start + (cur_i * K) + k; // 输入矩阵 A 的地址
    //                     auto cur_j_addr = weight_start + (cur_j * K) + k; // 权重矩阵 B 的地址
    //                     float *output = new float(0.0f);
    //                     ggml_vec_dot_f16(BLOCK_SHARED_FP16, output, 0, cur_i_addr, 0, cur_j_addr, 0, 1);
    //                     dst_data[cur_i * M + cur_j] += *output;
    //                     delete output;
    //                 }
    //             }
    //         }
    //     }
    // };
    // ── worker 线程：只写自己的 local_dst，完全无锁 ──────────────────────
    auto cpu_worker = [&](std::vector<float>& local_dst) {
        while (true) {
            auto s_time = ggml_time_us();
            int t = next_task_idx.fetch_add(1, std::memory_order_relaxed);
            if (t >= total_tasks) break;

            std::cout << "cpu processing task " << t << std::endl;
            const auto [j, k, task] = tasks->at(t);
            const auto i = task.I;

            const int cur_n_block_size = std::min(N - i, BLOCK_N_FP16);
            const int cur_m_block_size = std::min(M - j, BLOCK_WEIGHT_FP16);

    #pragma omp parallel for num_threads(4) schedule(static)
            for (int jj = 0; jj < cur_m_block_size; ++jj) {
                const int cur_j          = j + jj;
                ggml_fp16_t* B_col = weight_start + cur_j * K + k;

                if (jj + 1 < cur_m_block_size)
                    __builtin_prefetch(weight_start + (cur_j + 1) * K + k, 0, 1);

                for (int ii = 0; ii < cur_n_block_size; ++ii) {
                    const int cur_i          = i + ii;
                    ggml_fp16_t* A_row = input_start + cur_i * K + k;

                    if (ii + 1 < cur_n_block_size) {
                        __builtin_prefetch(input_start + (cur_i + 1) * K + k, 0, 1);
                        __builtin_prefetch(&local_dst[(cur_i + 1) * M + cur_j], 1, 1);
                    }

                    float dot = 0.0f;
                    ggml_vec_dot_f16(BLOCK_SHARED_FP16, &dot, 0, A_row, 0, B_col, 0, 1);
                    local_dst[cur_i * M + cur_j] += dot; // 完全无锁
                }
            }
            auto e_time = ggml_time_us();
            cpu_time += (e_time - s_time);

        }
    };

    std::vector<float> cpu_local_dst(N * M, 0.0f);
    std::thread t_npu(npu_worker);
    // std::thread t_cpu(cpu_worker);
    std::thread t_cpu([&]() { cpu_worker(cpu_local_dst); });

    t_npu.join();
    t_cpu.join();
    auto thread_end_time = ggml_time_us();
    std::cout << "Total thread execution time: " << (thread_end_time - submit_time) / 1000.0 << " ms" << std::endl;

    int total = N * M;
    int idx = 0;

    for (; idx + 4 <= total; idx += 4) {
        float32x4_t g = vld1q_f32(&dst_data[idx]);
        float32x4_t l = vld1q_f32(&cpu_local_dst[idx]);
        vst1q_f32(&dst_data[idx], vaddq_f32(g, l));
    }
    auto final_merge_time = ggml_time_us();
    std::cout << "Final merge time: " << (final_merge_time - thread_end_time) / 1000.0 << " ms" << std::endl;
    std::cout << "Total NPU wait time: " << npu_wait_time / 1000.0 << " ms" << std::endl;
    std::cout << "Total NPU+CPU time: " << npu_cpu_time / 1000.0 << " ms" << std::endl;
    std::cout << "Total CPU time: " << cpu_time / 1000.0 << " ms" << std::endl;
    npu_tasks_shared_fp16.reset();
}



/**
 * @brief Q8_0 matrix multiplication with CPU & NPU parallel execution
 * 
 * This is the optimized path using double buffering:
 * - NPU writes to buffer 0 while CPU reads from buffer 1
 * - Uses NEON instructions for dequantization
 * - Cache prefetching for better performance
 * 
 * GGML matmul convention: dst = src1 @ src0.T (transposed multiplication)
 * - src0 (weight): M×K stored physically, represents K×M logically (transposed)
 * - src1 (input):  N×K
 * - dst (output):  N×M
 * 
 * NPU computation: input @ weight.T
 * - NPU expects weight in row-major format (M×K physical = N output features, each K-dim)
 * - NPU expects input in col-major format (N×K)
 * 
 * NOTE: Quantization and IOMMU mapping are done earlier in model.cpp
 *       This function works with pre-quantized Q8_0 tensors that have DMA addresses
 */




// // 适配 group_size=256 的量化函数
// void quantize_row_q8_0_custom_256(const float* src, block_q8_0_256* dst, int64_t n) {
//     const int group_size = 256;
//     const int num_blocks = n / group_size;

//     for (int i = 0; i < num_blocks; ++i) {
//         float amax = 0.0f; // 绝对值最大值
//         const float* x = src + i * group_size;

//         // 1. 寻找该组的最大绝对值
//         for (int j = 0; j < group_size; ++j) {
//             amax = std::max(amax, std::abs(x[j]));
//         }

//         // 2. 计算缩放因子 (Scale)
//         // Q8_0 对称量化公式：amax / 127
//         const float d = amax / 127.0f;
//         dst[i].d = float_to_half(d); 
        
//         const float id = d > 0.0f ? 1.0f / d : 0.0f;

//         // 3. 执行量化转换
//         for (int j = 0; j < group_size; ++j) {
//             float val = x[j] * id;
//             // 使用 std::lround 并 clamp 到 int8 范围
//             int rounded = (int)std::lround(val);
//             dst[i].qs[j] = (int8_t)std::max(-128, std::min(127, rounded));
//         }
//     }
// }


static void compute_matmul_q8_0_parallel(
    const struct ggml_tensor* src0,  // weight (M x K, Q8_0, pre-quantized with IOMMU)
    const struct ggml_tensor* src1,  // input (N x K, FP32)
    struct ggml_tensor* dst,         // output (N x M, FP32)
    int domain_id) {
    
    // GGML dimensions (physical storage)
    const int M = src0->ne[1];  // weight rows (output dimension when transposed)
    const int K = src0->ne[0];  // weight cols = input cols (shared dimension)
    const int N = src1->ne[1];  // input rows (batch size)
    const int QK = 512;
    


    // Step 1: Quantize input to Q8_0 (input is FP32, runtime data)
    // std::vector<block_q8_0> input_q8((N * K) / QK);
    // quantize_row_q8_0_ref((const float*)src1->data, input_q8.data(), N * K);
    auto start_time = ggml_time_us();

    // Step 1: 使用自定义的 256 步长类型进行量化
    std::vector<block_q8_0_512> input_q8((N * K) / QK);
    if(src1->type == GGML_TYPE_F32){
        quantize_row_q8_0_custom<QK>((const float*)src1->data, input_q8.data(), N * K);
    }else{
        fprintf(stderr, "[NPU] Error: Weight tensor should be pre-quantized to Q8_0, got type %d\n", src1->type);
        throw std::runtime_error("input type not support!");
    }
    
    
    std::vector<int8_t> input_int8(N * K, 0);
    std::vector<float> input_scales(N * K / QK, 0);
    // extract_q8_0(input_q8.data(), N, K, input_int8.data(), input_scales.data());
    extract_q8_0_custom<QK>(input_q8.data(), N * K, input_int8.data(), input_scales.data());
    auto end_input_time = ggml_time_us();
    printf("[Debug] 输入量化完成: int8数据量=%zu, scale数量=%zu, 耗时=%.2f ms\n", 
           input_int8.size(), input_scales.size(), (end_input_time - start_time) / 1000.0);

    // std::cout << "input_scale:" ;
    // for(int i = 0; i < 64; i++) {
    //     if(i % 8 == 0){
    //         std::cout << std::endl;
    //     }
    //     std::cout << input_scales[i] <<" ";
    // }
    // std::cout << std::endl;

    const int K_in = (K + 15) & ~15;
    const int K_w = (K + 31) & ~31;
    
    std::vector<int8_t> input_npu(N * K_in, 0);
    to_npu_feature_layout(input_int8.data(), N, K, input_npu.data());

    auto end_input_layout_time = ggml_time_us();
    printf("[Debug] 输入布局转换完成: 耗时=%.2f ms\n", 
           (end_input_layout_time - end_input_time) / 1000.0);

    // Step 2: Extract weight data (already Q8_0 with IOMMU mapping)
    // std::vector<int8_t> weight_int8(M * K, 0);
    // std::vector<float> weight_scales(M * K / QK, 0);
    
    // Weight tensor should already be Q8_0 format with IOMMU mapping
    // if (src0->type != GGML_TYPE_Q8_0) {
    //     fprintf(stderr, "[NPU] Error: Weight tensor should be pre-quantized to Q8_0, got type %d\n", src0->type);
    //     throw std::runtime_error("Weight tensor not pre-quantized");
    // }

    uint64_t weight_dma_base = 0;
    Domain* weight_domain = find_tensor_domain(src0);
   
    // Extract INT8 data and scales from pre-quantized weight
    std::cout << "not quanted yet , start quanted" << std::endl;
    // struct ggml_tensor src0_backup = *src0;

    // 2. 如果你需要备份【内存里的数据】（这会消耗额外内存和时间）
    // std::vector<uint8_t> data_backup;
    // size_t nbytes = ggml_nbytes(src0);
    // data_backup.resize(nbytes);
    // memcpy(data_backup.data(), src0->data, nbytes);
    const ggml_fp16_t* weight_scales_ptr = nullptr;
    auto end_backup_time = ggml_time_us();
    // printf("[Debug] 权重备份完成: %zu bytes, 耗时=%.2f ms\n", nbytes, (end_backup_time - end_input_layout_time) / 1000.0);

    
    // std::vector<block_q8_0_512> weight_q8((M * K) / QK);
    // convert_q8_to_512((const block_q8_0*)src0->data, weight_q8.data(), M * K);
    // extract_q8_0_custom<QK>((const block_q8_0_512*)weight_q8.data(), M * K, weight_int8.data(), weight_scales.data());
      
    
    
    if (weight_domain && src0->name) {
        // Direct lookup in map by tensor name
        auto it = weight_domain->tensors.find(std::string(src0->name));
        if (it != weight_domain->tensors.end()) {
            IommuConfig* iommu_config = std::get<2>(it->second);
            if (iommu_config && iommu_config->iommu_addr) {
                // Step 4.1: Check if we have enough space for NPU layout
                size_t q8_size = ggml_nbytes(src0);  // Size of Q8_0 blocks
                size_t npu_layout_size = M * K_w;     // Size needed for NPU layout (int8 only)
                
                // fprintf(stderr, "[NPU] In-place conversion check: Q8_0=%zu bytes, NPU layout needs=%zu bytes\n", 
                //         q8_size, npu_layout_size);
                
                // if (npu_layout_size > q8_size) {
                //     fprintf(stderr, "[NPU] ERROR: Not enough space for in-place conversion\n");
                //     throw std::runtime_error("Insufficient space for NPU layout conversion");
                // }
                
                // Step 4.2: Convert Q8_0 blocks → NPU layout IN-PLACE
                // Since we already extracted weight_int8 from Q8_0 blocks,
                // we can directly convert it to NPU layout and write to src0->data
                // fprintf(stderr, "[NPU] Converting tensor %s to NPU layout in-place...\n", src0->name);
                // std::cout << "start conversation" << std::endl;
                // Write NPU layout directly to tensor->data (overwrites Q8_0 blocks)
                // to_npu_weight_layout(weight_int8.data(), M, K, (int8_t*)src0->data);
                
                // Use the existing IOMMU DMA address
                weight_dma_base = iommu_config->iommu_addr;
                weight_scales_ptr = reinterpret_cast<ggml_fp16_t*>(static_cast<uint8_t*>(src0->data) + iommu_config->scale_offset);
                // fprintf(stderr, "[NPU] Reusing IOMMU DMA address: 0x%lx (tensor->data=%p)\n", 
                //         weight_dma_base, src0->data);
                
                // Flush cache to ensure NPU sees the converted data
                // flush_cache(src0->data, npu_layout_size);
                
                // NOTE: After this conversion, src0->data no longer contains Q8_0 blocks!
                // It now contains NPU layout (int8 array with special tiling)
                // TODO: If CPU also needs this tensor, we should either:
                //   1. Keep a backup of Q8_0 blocks
                //   2. Convert back after NPU inference
                //   3. Mark tensor as "NPU layout only"
            }
        }
    }

    std::cout << "weight_int8 data in NPU layout:" << std::endl;
    auto weight_input_ptr = (uint8_t*)src0->data; // 建议统一用 uint8_t 处理字节流
    std::cout << "--- Memory Hex Dump (First 128 Bytes) ---" << std::endl;

    for (int i = 0; i < 128; i++) {
        // 打印换行，每 16 字节一行
        if (i > 0 && i % 16 == 0) {
            std::cout << std::endl;
        }
        
        // 格式化输出：16进制，两位宽度，不足补0
        std::cout << std::hex << std::setw(2) << std::setfill('0') 
                << (int)weight_input_ptr[i] << " ";
    }
    // std::ofstream outfile("npu_weight_dump.txt");

    // if (outfile.is_open()) {
    //     auto weight_input_ptr = (uint8_t*)src0->data;
        
    //     // 假设你想写入前 128 字节，或者根据需要修改长度
    //     for (int i = 0; i < M * K; i++) {
    //         // 格式化：十六进制、两位宽度、补零、转为大写（可选）
    //         outfile << std::hex << std::setw(2) << std::setfill('0') << std::uppercase
    //                 << (int)weight_input_ptr[i] << std::endl; // std::endl 确保一行一个
    //     }
        
    //     outfile.close();
    //     std::cout << "[Debug] 内存数据已成功写入 npu_weight_dump.txt" << std::endl;
    // } else {
    //     std::cerr << "[Error] 无法打开文件进行写入！" << std::endl;
    // }
    // std::cout << std::dec << std::endl << "---------------------------------------" << std::endl;

    std::cout << "weight_scales:" << std::endl;
    for(int i = 0; i < 128; i++) {
        std::cout << ggml_fp16_to_fp32(weight_scales_ptr[i]) << " ";
        if(i % 16 == 15){
            std::cout << std::endl; 
        }
    }

    auto end_time = ggml_time_us();
    printf("[Debug] 权重提取完成: 耗时=%.2f ms\n", 
            (end_time - end_backup_time) / 1000.0);
    auto end_conversion_time = ggml_time_us();


    // ✅ Step 4.5: Get Domain for input buffer (use specialized IOMMU-mapped buffer)
    Domain* input_domain = find_domain_by_id(domain_id);
    if (!input_domain || !input_domain->input) {
        fprintf(stderr, "[NPU] ERROR: Domain %d not found or input buffer not allocated\n", domain_id);
        throw std::runtime_error("Input IOMMU buffer not available");
    }
    
    // Check buffer size
    size_t input_size = N * K_in;
    if (input_size > input_domain->input->size) {
        fprintf(stderr, "[NPU] ERROR: Input size %zu exceeds buffer size %zu\n", 
                input_size, input_domain->input->size);
        throw std::runtime_error("Input buffer overflow");
    }
    
    // Copy input data to IOMMU-mapped buffer
    memcpy(input_domain->input->virtual_addr, input_npu.data(), input_size);


    auto end_input_copy_time = ggml_time_us();
    printf("[Debug] 输入数据复制完成: %zu bytes, 耗时=%.2f ms\n", input_size, (end_input_copy_time - end_conversion_time) / 1000.0);

    flush_cache(input_domain->input->virtual_addr, input_size);
    uint64_t input_dma_base = input_domain->input->iommu_addr;
    
    fprintf(stderr, "[NPU] Input buffer: VA=%p, DMA=0x%lx, size=%zu\n", 
            input_domain->input->virtual_addr, 
            input_domain->input->iommu_addr, 
            input_size);

    // Step 5: Build task list (block splitting)
    auto tasks = std::make_shared<std::vector<std::tuple<int, int, matmul_task_t>>>();
    tasks->reserve(ceil_int(M, BLOCK_WEIGHT) * ceil_int(K, BLOCK_SHARED));
    
    // Split matrix into blocks
    // Output matrix: N x M (src1 rows x src0 rows)
    // Weight matrix (NPU layout): M x K_w (M output dims, K_w=align32(K) input dims)
    // Input matrix (NPU layout): N x K_in (N batch, K_in=align16(K) input dims)
    
    for (int j = 0; j < M; j += BLOCK_WEIGHT) {
        int _n = std::min(M - j, BLOCK_WEIGHT);
        for (int k = 0; k < K; k += BLOCK_SHARED) {
            int _k = std::min(K - k, BLOCK_SHARED);
        
            tasks->emplace_back(j, k, matmul_task_t{
                input_dma_base, weight_dma_base, N, _k, _n, 0
            });
            input_dma_base += N * _k;
            weight_dma_base += _k * _n;
        }
        input_dma_base = input_domain->input->iommu_addr;
    }
    auto end_task_time = ggml_time_us();
    printf("[Debug] 任务列表构建完成: 任务数量=%zu, 耗时=%.2f ms\n", 
           tasks->size(), (end_task_time - end_input_copy_time) / 1000.0);
    // Step 6: Adjust batch size based on task count
    // if (tasks->size() <= PER_TASK_CORE_NUM) {
    //     TASKS_LOCAL_PER_NUM = PER_TASK_CORE_NUM;
    // } else if (tasks->size() > 2 * PER_TASK_CORE_NUM + 1) {
    //     TASKS_LOCAL_PER_NUM = PER_TASK_CORE_NUM;
    // } else {
    //     TASKS_LOCAL_PER_NUM = 2;
    // }
    
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
    auto end_time2 = ggml_time_us();
    printf("[Debug] 任务提交完成: 任务数量=%zu, 耗时=%.2f ms\n", 
           tasks->size(), (end_time2 - end_task_time) / 1000.0);
    int64_t cpu_total_time  = 0;
    int64_t wait_total_time = 0;
    // for (int t = 0; t < (int)tasks->size(); t += TASKS_LOCAL_PER_NUM) {
    //     {
    //         auto s_time = ggml_time_us();
    //         std::unique_lock<std::mutex> lock(cpu_worker_mtx);
    //         cpu_cv.wait(lock, [&] { 
    //             return npu_tasks_shared.use_count() != 0 && 
    //                    npu_tasks_shared->at(t) != nullptr && 
    //                    buffer_free[index].load(std::memory_order_acquire) == 1;
    //         });
    //         auto e_time = ggml_time_us();
    //         wait_total_time += (e_time - s_time);
    //     }
    //     auto s_time = ggml_time_us();
    //     for (int toff = 0; toff < TASKS_LOCAL_PER_NUM && t + toff < (int)tasks->size(); toff++) {
    //         const auto [j, k, task] = tasks->at(t + toff);
    //         int32_t* task_output = npu_tasks_shared->at(t + toff);
            
    //         // Invalidate cache to ensure CPU reads NPU-written data
    //         invalid_cache(task_output, N * std::min(M - j, BLOCK_WEIGHT) * sizeof(int32_t));
            
    //         int current_m_task = std::min(M - j, BLOCK_WEIGHT);
            
         
            
    //         // Process output with NEON optimization
    //         // For each weight row in [j, j+joff_max), get the scale for block k/32
    //         // NOTE: NPU output layout is NCHW4 with H=N (batch size)!
    //         // cur_block_scale = input_scales[j * scale_per_k + k / QK] * weight_scales[j * scale_per_k + k / QK];
    //         for (int i = 0; i < N; i++) { // 行遍历 (Height)
    //             for (int joff = 0; joff < current_m_task; joff ++) { 
    //                 auto cur_M = j + joff;
    //                 auto target = feature_data(N, 4, joff, i);
    //                 auto cur_offset_in_result = i * M + cur_M ;
    //                 // std::cout << "cur_offset_in_result:" << cur_offset_in_result << "    value:"<< (float)task_output[target] << "i:" <<i << " cur_M" << cur_M <<std::endl;
    //                 // auto weight_scale_offset = cur_M * scale_per_k + k / QK;
    //                 // auto input_scale_offset = i * scale_per_k + k / QK;
    //                 // std::cout << "weight_scale_offset:" << weight_scale_offset << "    weight_scale:" << weight_scales[weight_scale_offset] << std::endl;
    //                 float combined_scale = input_scales[i * scale_per_k + k / QK] * ggml_fp16_to_fp32(weight_scales_ptr[cur_M * scale_per_k + k / QK]);
    //                 dst_data[cur_offset_in_result] += (double)task_output[target] * combined_scale;
    //                 // dst_data[cur_offset_in_result] += (double)task_output[target]* input_scales[input_scale_offset] * weight_scales[weight_scale_offset];
    //                 // dst_data[cur_offset_in_result] += (double)task_output[target];
    //             }
    //         }
    //     }
    //     auto e_time = ggml_time_us();
    //     cpu_total_time += (e_time - s_time);
    //     {
    //         std::lock_guard<std::mutex> lock(npu_worker_mtx);
    //         buffer_free[index].store(0, std::memory_order_release);
    //         npu_cv.notify_one();
    //     }
    //     index = (index + 1) & 0x1;
    // }
    for (int t = 0; t < (int)tasks->size(); t += TASKS_LOCAL_PER_NUM) {
        {
            auto s_time = ggml_time_us();
            std::unique_lock<std::mutex> lock(cpu_worker_mtx);
            cpu_cv.wait(lock, [&] { 
                return npu_tasks_shared.use_count() != 0 && 
                    npu_tasks_shared->at(t) != nullptr && 
                    buffer_free[index].load(std::memory_order_acquire) == 1;
            });
            auto e_time = ggml_time_us();
            wait_total_time += (e_time - s_time);
        }

        auto s_time = ggml_time_us();

        for (int toff = 0; toff < TASKS_LOCAL_PER_NUM && t + toff < (int)tasks->size(); toff++) {
            const auto [j, k, task] = tasks->at(t + toff);
            int32_t* task_output = npu_tasks_shared->at(t + toff);

            invalid_cache(task_output, N * std::min(M - j, BLOCK_WEIGHT) * sizeof(int32_t));

            int current_m_task = std::min(M - j, BLOCK_WEIGHT);
            const int k_scale_idx = k / QK;

            // 预先加载 weight_scales 为 float（避免内层循环重复 fp16->fp32 转换）
            // 将 weight_scales 缓存到局部数组，减少重复转换开销
            // weight_scale_row[joff] = fp32(weight_scales_ptr[(j+joff)*scale_per_k + k/QK])
            float w_scale_cache[BLOCK_WEIGHT];
            for (int joff = 0; joff < current_m_task; joff++) {
                w_scale_cache[joff] = ggml_fp16_to_fp32(
                    weight_scales_ptr[(j + joff) * scale_per_k + k_scale_idx]);
            }

            // 外层按 joff 步长4并行（参考第二段风格），内层遍历 N
            // joff 步长4 → 一次 NEON 处理4个权重行的结果
    #pragma omp parallel for num_threads(4)
            for (int joff = 0; joff < current_m_task; joff += 4) {
                const int cur_M_base = j + joff;
                const int lanes = std::min(4, current_m_task - joff); // 处理边界

                // 加载4个权重scale（可能不足4个时用0填充）
                float32x4_t w_scale = {
                    (lanes > 0) ? w_scale_cache[joff + 0] : 0.0f,
                    (lanes > 1) ? w_scale_cache[joff + 1] : 0.0f,
                    (lanes > 2) ? w_scale_cache[joff + 2] : 0.0f,
                    (lanes > 3) ? w_scale_cache[joff + 3] : 0.0f,
                };

                // feature_data(N, 4, joff, i) 中 i=0 时的初始偏移
                // 随 i 递增，每次 +4（参考第二段 feature_offset += 4）
                int feature_offset = feature_data(N, 4, joff, 0);

                for (int i = 0; i < N; i++) {
                    // Prefetch：预取下一行的 dst 写目标 和 NPU 输出数据
                    if (i + LOOKAHEAD < N) {
                        __builtin_prefetch(&dst_data[(i + LOOKAHEAD) * M + cur_M_base],
                                        1 /*write*/, 1 /*keep*/);
                        if ((i % 4) == 0) {
                            __builtin_prefetch(&task_output[feature_offset + (LOOKAHEAD * 4)],
                                            0 /*read*/, 1 /*keep*/);
                        }
                    }

                    // 当前行 input_scale（标量，广播到4个权重行）
                    const float i_scale = input_scales[i * scale_per_k + k_scale_idx];
                    float32x4_t v_i_scale = vdupq_n_f32(i_scale);

                    // combined_scale[joff+0..3] = i_scale * w_scale[0..3]
                    float32x4_t combined = vmulq_f32(v_i_scale, w_scale);

                    // 从 NPU 输出加载4个 int32，转 float
                    int32x4_t v_int32 = vld1q_s32(&task_output[feature_offset]);
                    float32x4_t v_val  = vcvtq_f32_s32(v_int32);

                    // 乘以 combined_scale
                    v_val = vmulq_f32(v_val, combined);

                    // 累加到 dst_data（注意：第一段原始用 double 累加，
                    // 这里改为 float 以支持 NEON；如需 double 精度请保留原始路径）
                    float* out_ptr = &dst_data[i * M + cur_M_base];
                    float32x4_t out_old = vld1q_f32(out_ptr);
                    out_old = vaddq_f32(out_old, v_val);
                    vst1q_f32(out_ptr, out_old);

                    feature_offset += 4;
                }
            }
        }

        auto e_time = ggml_time_us();
        cpu_total_time += (e_time - s_time);

        {
            std::lock_guard<std::mutex> lock(npu_worker_mtx);
            buffer_free[index].store(0, std::memory_order_release);
            npu_cv.notify_one();
        }
        index = (index + 1) & 0x1;
    }
    auto end_time3 = ggml_time_us();
    printf("[Debug] CPU处理完成: 耗时=%.2f ms\n", (end_time3 - end_time2) / 1000.0);
    printf("[Debug] CPU总处理时间: %.2f ms\n", cpu_total_time / 1000.0);
    printf("[Debug] CPU等待时间: %.2f ms\n", wait_total_time / 1000.0);
    // memcpy(src0->data, data_backup.data(), ggml_nbytes(src0));
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
    if (src0->type != GGML_TYPE_Q8_0_512 && src0->type != GGML_TYPE_F16 ) {
        fprintf(stderr, "[NPU] Weight type is %d, expected Q8_0_512 (pre-quantized)\n", src0->type);
        return 0;
    }
    
    // auto ret = src0->type == GGML_TYPE_Q8_0 ? 1 : 2;
    auto ret = 1; // For testing, allow both Q8_0 and F16 (with different compute paths)
    // Check memory layout (contiguous)
    const size_t type_size = ggml_type_size(src0->type);
    // const size_t blck_size = ggml_blck_size(src0->type);  // 未使用，注释掉
    if (src0->nb[0] != type_size) return 0;
    if (src1->nb[0] != sizeof(float)) return 0;
    
    return ret;
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
    auto type = src0->type;
   
    Domain* weight_domain = find_tensor_domain(src0);
    int domain_id = weight_domain ? weight_domain->id : get_default_domain_id();
    auto start_time = std::chrono::high_resolution_clock::now();
    try {
        // if(type == GGML_TYPE_Q8_0_512) {
        //     std::cout << "ggml_compute_forward_mul_mat_npu: Using Q8_0 parallel path" << std::endl;
        //     input_type.store(1, std::memory_order_release);
        //     compute_matmul_q8_0_parallel(src0, src1, dst, domain_id);
        // }else{
        //     std::cout << "ggml_compute_forward_mul_mat_npu: Using FP16 parallel path" << std::endl;
        //     input_type.store(0, std::memory_order_release);
            // compute_matmul_fp16_parallel(src0, src1, dst, domain_id);
        // }
        compute_matmul_fp16_parallel_dynamic(src0, src1, dst, domain_id);
    } catch (const std::exception& e) {
        fprintf(stderr, "[NPU] Error: %s, falling back to CPU\n", e.what());
        // Let GGML handle CPU fallback by not writing to dst
        throw;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    printf("[NPU] Matmul completed in %.2f ms\n", 
           std::chrono::duration<double, std::milli>(end_time - start_time).count());
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
