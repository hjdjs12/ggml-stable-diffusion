#pragma once

// ============================================================================
// 自动资源管理说明：
// ============================================================================
// 1. Domain 对象会在析构时自动释放 mmap 内存和 IOMMU 域
// 2. FileDomains 对象会在析构时自动删除所有 Domain 对象
// 3. 程序退出时会自动调用 cleanup_all_domains() 清理所有全局资源
//
// 使用方法：
// - 在程序初始化时调用 register_cleanup_handler() 注册清理处理器
// - 创建 Domain 对象时使用 new，不需要手动 delete（由 FileDomains 管理）
// - 程序崩溃或正常退出时，所有资源会自动释放
// ============================================================================

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <sys/mman.h>
#include <vector>
#include <map>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <iostream>
#include <fstream>
#include <sys/syscall.h>      /* 包含 SYS_xxx 定义 */
#include <unistd.h>           /* 包含系统调用相关的宏 */
#include <signal.h>           /* 信号处理 */
#include "include/rknpu-ioctl.h"
#include "include/rk-mem.hpp"
#include "ggml.h"
#include "include/common.h"
// Forward declaration - TensorStorage is defined in model.h
struct TensorStorage;

// NPU file descriptor (defined in ggml-cpu-matmul-npu.cpp)
extern int g_npu_fd;

#define WEIGHT_SIZE (3696UL * 1024 * 1024)
#define DOMAIN_SIZE (4096UL * 1024 * 1024)
#define REGCMD_SIZE (64 * 1024)  // 64KB for register commands
#define TASKS_MEM_SIZE (4 * 1024)  // 4KB for task descriptors
#define NPU_INPUT_BUFFER_SIZE (50 * 1024 * 1024)  // 50MB for input/output buffers
#define NPU_OUTPUT_BUFFER_SIZE (50 * 1024 * 1024)  // 50MB for output buffers



inline uint64_t cur_max_domain_index = 0;   
inline uint64_t cur_used_size = 0;

class IommuConfig{

public:
    uint64_t iommu_addr;
    uint64_t mem_obj_handle;
    uint64_t domain_id;
    IommuConfig() : iommu_addr(0), mem_obj_handle(0), domain_id(0) {}
};




struct MapOps {
    std::string fp;         // 文件路径
    uint64_t handle;        // 驱动返回的句柄（代替 localmmap 的 fd）
    uint64_t va;            // 虚拟地址（映射后的地址）
    uint64_t len;           // 映射长度
};
// ============================================================================
// Forward Declarations - 前向声明
// ============================================================================

/**
 * @brief 为已有内存创建 IOMMU 映射（前向声明）
 * 
 * 实现位置：在 Domain 结构体定义之后
 */
inline IommuConfig* iommu_create_domain(void *virtual_addr, uint64_t domain_id, size_t used_size);
inline struct MapOps* mem_recallmem_mmap(size_t file_len);
inline std::tuple<void*, uint64_t, uint64_t, uint64_t> mem_allocate(size_t size, uint32_t flags,
                   uint64_t domain_id) ;
class TensorInfo {
public:
    uint64_t offset;
    ggml_type dtype;
    uint64_t size;
    TensorInfo() : offset(0), dtype(GGML_TYPE_F32), size(0) {}  // Default constructor
    TensorInfo(uint64_t offset, ggml_type dtype, uint64_t size) : offset(offset), dtype(dtype), size(size) {}
};

class LeftMemory {
public:    
    size_t size;             // 内存大小
    void* virtual_addr;      // 虚拟地址（CPU 访问用）
    uint64_t iommu_addr;        // IOMMU DMA 地址（NPU 访问用）
    uint64_t mem_obj_handle; // IOMMU handle（用于释放）
    uint64_t obj_addr;        // 驱动对象地址（如果需要）
    
    LeftMemory() : size(0), virtual_addr(nullptr), iommu_addr(0), mem_obj_handle(0) {}
    
    /**
     * @brief 分配内存并创建 IOMMU 映射
     * @param mem_size 内存大小
     * @param domain_id IOMMU 域 ID
     */
    void allocate_and_map(size_t mem_size, uint64_t domain_id) {
        if (virtual_addr != nullptr) {
            std::cerr << "[LeftMemory] Warning: Memory already allocated, skipping" << std::endl;
            return;
        }
        
        // 页对齐大小
        size = (mem_size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
        
        // 1. 使用 mmap 分配匿名内存
        // virtual_addr = mmap(nullptr, size, 
        //                    PROT_READ | PROT_WRITE,
        //                    MAP_PRIVATE | MAP_ANONYMOUS,
        //                    -1, 0);
        
        // if (virtual_addr == MAP_FAILED) {
        //     std::cerr << "[LeftMemory] mmap failed: " << strerror(errno) << std::endl;
        //     throw std::runtime_error("LeftMemory: mmap allocation failed");
        // }

        auto result = mem_allocate(size, 0, domain_id);
        virtual_addr = std::get<0>(result);
        obj_addr = std::get<1>(result);
        iommu_addr = std::get<2>(result);
        mem_obj_handle = std::get<3>(result);

        // 2. 锁定内存（防止被 swap）
        // 注意：recallmem 驱动分配的内存通常已经锁定，mlock 可能失败但不影响功能
        // if (mlock(virtual_addr, size) != 0) {
        //     std::cerr << "[LeftMemory] Warning: mlock failed (" << strerror(errno) 
        //               << "), but continuing (driver memory may already be locked)" << std::endl;
        //     // 不抛出异常，继续执行
        // } else {
        //     std::cout << "[LeftMemory] Memory locked successfully" << std::endl;
        // }
        
        // 3. 创建 IOMMU 映射
        // try {
            // IommuConfig* config = iommu_create_domain(virtual_addr, domain_id, size);
            // iommu_addr = config->iommu_addr;
            // mem_obj_handle = config->mem_obj_handle;
            // 注意：不要 delete config，因为它的成员已经被我们保存了
            // delete config;  // 只删除 config 对象本身
        // } catch (const std::exception& e) {
        //     std::cerr << "[LeftMemory] IOMMU mapping failed: " << e.what() << std::endl;
        //     munlock(virtual_addr, size);
        //     munmap(virtual_addr, size);
        //     virtual_addr = nullptr;
        //     throw;
        // }
        
        std::cout << "[LeftMemory] Allocated " << size << " bytes, VA=" << virtual_addr 
                  << ", DMA=" << iommu_addr << std::endl;
    }
    
    /**
     * @brief 释放内存和 IOMMU 映射
     */
    void free_memory(uint64_t domain_id) {
        if (virtual_addr == nullptr) {
            return;  // 已经释放或未分配
        }
        
        // 1. 销毁 IOMMU 映射
        if (iommu_addr != 0) {
            // uint64_t va_itr = va + std::get<0>(itr);
            // struct rknpu_mem_destroy destroy = {.handle = std::get<4>(itr), 
            //         .page_num = uint32_t(std::get<1>(itr) / PAGE_SIZE), .usr_va = va_itr};
            
            // rknpu_ioctl(DRM_IOCTL_RKNPU_MEM_DESTROY, &destroy, std::get<3>(itr));
            struct rknpu_mem_destroy mem_destroy = {};
            mem_destroy.handle = mem_obj_handle;
            mem_destroy.page_num = uint32_t(size / PAGE_SIZE);
            mem_destroy.usr_va = reinterpret_cast<__u64>(virtual_addr);
            
            try {
                rknpu_ioctl(DRM_IOCTL_RKNPU_MEM_DESTROY, &mem_destroy, domain_id);
                std::cout << "[LeftMemory] IOMMU mapping destroyed for DMA=" << iommu_addr << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[LeftMemory] Warning: Failed to destroy IOMMU mapping: " 
                          << e.what() << std::endl;
            }
            
            mem_obj_handle = 0;
            iommu_addr = 0;
        }
        
        // 2. 解锁和释放虚拟内存
        if (virtual_addr != nullptr) {
            munlock(virtual_addr, size);
            munmap(virtual_addr, size);
            std::cout << "[LeftMemory] Freed " << size << " bytes at VA=" << virtual_addr << std::endl;
            virtual_addr = nullptr;
        }
        
        size = 0;
    }
    
    ~LeftMemory() {
        // 析构函数不调用 free_memory，因为需要 domain_id
        // 由 Domain 显式调用
    }
};

struct Domain{
    uint64_t id;
    LeftMemory* input;      // 输入缓冲区（用于临时存储输入数据）
    LeftMemory* output;     // 输出缓冲区（NPU 写入计算结果）
    LeftMemory* regcmd;     // 寄存器命令内存（存储 108 个寄存器配置）
    LeftMemory* tasks_mem;  // 任务描述符内存（rknpu_task 数组）
    std::map<std::string, std::tuple<ggml_tensor*, TensorStorage*, IommuConfig*>> tensors;

    Domain(uint64_t domain_id)
        : id(domain_id), input(nullptr), output(nullptr), regcmd(nullptr), tasks_mem(nullptr) {
        
        std::cout << "[Domain " << id << "] Initializing with 4 LeftMemory buffers..." << std::endl;
        
        try {
            // 1. 分配输入缓冲区（100MB）
            input = new LeftMemory();
            input->allocate_and_map(NPU_INPUT_BUFFER_SIZE, domain_id);
            std::cout << "[Domain " << id << "] Input buffer allocated: " 
                      << NPU_INPUT_BUFFER_SIZE / (1024*1024) << " MB" << std::endl;
            
            // 2. 分配输出缓冲区（100MB）
            output = new LeftMemory();
            output->allocate_and_map(NPU_OUTPUT_BUFFER_SIZE, domain_id);
            std::cout << "[Domain " << id << "] Output buffer allocated: " 
                      << NPU_OUTPUT_BUFFER_SIZE / (1024*1024) << " MB" << std::endl;
            
            // 3. 分配寄存器命令内存（64KB，可存储多组寄存器配置）
            regcmd = new LeftMemory();
            regcmd->allocate_and_map(REGCMD_SIZE, domain_id);
            std::cout << "[Domain " << id << "] RegCmd buffer allocated: " 
                      << REGCMD_SIZE / 1024 << " KB" << std::endl;
            
            // 4. 分配任务描述符内存（4KB，存储 rknpu_task 数组）
            tasks_mem = new LeftMemory();
            tasks_mem->allocate_and_map(TASKS_MEM_SIZE, domain_id);
            std::cout << "[Domain " << id << "] TasksMem buffer allocated: " 
                      << TASKS_MEM_SIZE / 1024 << " KB" << std::endl;
            
            std::cout << "[Domain " << id << "] All buffers initialized successfully" << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "[Domain " << id << "] Initialization failed: " << e.what() << std::endl;
            // 清理已分配的内存
            cleanup();
            throw;
        }
    }

    // 禁用拷贝构造和拷贝赋值（防止双重释放）
    Domain(const Domain&) = delete;
    Domain& operator=(const Domain&) = delete;


    // 析构函数：自动释放资源
    ~Domain() {
        cleanup();
    }

private:
    void cleanup() {
        std::cout << "[Domain " << id << "] Cleaning up resources..." << std::endl;
        
        // 释放所有 LeftMemory 缓冲区
        if (input) {
            input->free_memory(id);
            delete input;
            input = nullptr;
        }
        
        if (output) {
            output->free_memory(id);
            delete output;
            output = nullptr;
        }
        
        if (regcmd) {
            regcmd->free_memory(id);
            delete regcmd;
            regcmd = nullptr;
        }
        
        if (tasks_mem) {
            tasks_mem->free_memory(id);
            delete tasks_mem;
            tasks_mem = nullptr;
        }
        
        std::cout << "[Domain " << id << "] Cleanup complete" << std::endl;
    }

    void iommu_destroy_domain() {
        // 已在 LeftMemory::free_memory() 中处理
        // 保留此函数以兼容旧代码
    }
public:
   
};

// ============================================================================
// IOMMU Memory Management Functions (统一接口)
// ============================================================================

/**
 * @brief 为已有内存创建 IOMMU 映射（用于 model 加载的 weight tensor）
 * 
 * @param virtual_addr 已存在的虚拟地址
 * @param domain_id IOMMU 域 ID
 * @param used_size 内存大小
 * @return IommuConfig* 映射配置（包含 DMA 地址和 handle）
 */
inline IommuConfig * iommu_create_domain(void *virtual_addr, uint64_t domain_id, size_t used_size) {
    uintptr_t addr_int = reinterpret_cast<uintptr_t>(virtual_addr);
    uintptr_t page_offset = addr_int & (PAGE_SIZE - 1);
    uintptr_t aligned_addr = addr_int - page_offset;
    
    // ✅ 对齐到页边界：包含页内偏移并向上对齐到 PAGE_SIZE
    // 例如：offset=0x100, size=0x500 => aligned_size=0x1000 (4KB)
    //      offset=0, size=44564480 => aligned_size=44564480 (已对齐)
    size_t aligned_size = (page_offset + used_size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
    std::cout << "[IOMMU] Creating domain for VA=" << virtual_addr 
              << ", size=" << used_size << " bytes (aligned to " << aligned_size << " bytes)" 
              << ", domain_id=" << domain_id << std::endl;
    struct rknpu_mem_create mem_create = {};
    mem_create.flags = RKNPU_MEM_ALLOCATED;
    
    mem_create.size = aligned_size;        // ✅ 页对齐的大小
    mem_create.usr_va = aligned_addr;      // ✅ 页对齐的地址
    mem_create.iommu_domain_id = domain_id;
    
    rknpu_ioctl(DRM_IOCTL_RKNPU_MEM_CREATE, &mem_create, domain_id);
    
    // ✅ 返回DMA地址时加上页内偏移
    IommuConfig* config = new IommuConfig();
    config->iommu_addr = (uint64_t)(mem_create.dma_addr + page_offset);
    config->mem_obj_handle = mem_create.handle;
    config->domain_id = domain_id;
    return config;
}
// inline IommuConfig * iommu_create_domain(void *virtual_addr, uint64_t domain_id, size_t used_size) {
//     // Implementation for creating an IOMMU domain
//     if (virtual_addr == nullptr) {  // ✅ 加个安全检查
//         throw std::runtime_error("virtual_addr is null, call mmap_domain_data first!");
//     }
    
//     // ✅ CRITICAL FIX: Align address to page boundary (4KB = 4096 bytes)
//     // NPU IOMMU requires page-aligned addresses
//     // PAGE_SIZE is already defined in rk-mem.hpp
//     uintptr_t addr_int = reinterpret_cast<uintptr_t>(virtual_addr);
//     uintptr_t page_offset = addr_int & (PAGE_SIZE - 1);  // Offset within page
//     uintptr_t aligned_addr = addr_int - page_offset;     // Round down to page boundary
//     size_t aligned_size = (used_size + page_offset + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);  // Round up size
    
//     struct rknpu_mem_create mem_create = {};
//     mem_create.flags = RKNPU_MEM_ALLOCATED ;
//     //               RKNPU_MEM_CACHEABLE |
//     //               RKNPU_MEM_NON_CONTIGUOUS |              // 允许非连续物理内存
//     //               RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT;    // 限制IOVA对齐（内核日志提示）

//     mem_create.size = aligned_size;  // Use aligned size
//     mem_create.usr_va = aligned_addr;  // Use aligned address
//     mem_create.iommu_domain_id = domain_id;
    
//     rknpu_ioctl(DRM_IOCTL_RKNPU_MEM_CREATE, &mem_create, domain_id);
    
//     if (mem_create.dma_addr == 0) {
//         throw std::runtime_error("IOMMU mapping failed: dma_addr is 0");
//     }
    
//     IommuConfig* config = new IommuConfig();
//     // Add the page offset back to the DMA address so it points to the actual tensor data
//     config->iommu_addr = (void*)(mem_create.dma_addr + page_offset);
//     config->mem_obj_handle = new uint64_t(mem_create.handle);
//     return config;
// }

struct FileDomains{
    std::vector<Domain*> domains;
    
    // 析构函数：清理所有域
    ~FileDomains() {
        for (auto* domain : domains) {
            delete domain;  // Domain的析构函数会自动清理资源
        }
        domains.clear();
    }
};


inline std::map<std::string,FileDomains*> file_mapping;
inline std::map<std::string, TensorInfo> tensor_info;

// ============================================================================
// Domain 查找辅助函数
// ============================================================================

/**
 * @brief 根据 domain_id 查找 Domain 对象
 * @param domain_id IOMMU 域 ID
 * @return Domain* 找到的 Domain，未找到返回 nullptr
 */
inline Domain* find_domain_by_id(uint64_t domain_id) {
    for (auto& [file_path, file_domains] : file_mapping) {
        if (!file_domains) continue;
        for (auto* domain : file_domains->domains) {
            if (domain && domain->id == domain_id) {
                return domain;
            }
        }
    }
    return nullptr;
}


// 全局清理函数：在程序退出时自动释放所有资源
inline void cleanup_all_domains() {
    static bool cleaning = false;
    if (cleaning) return;  // 防止重复清理
    cleaning = true;
    
    std::cout << "[RKMEM] Cleaning up all domains..." << std::endl;
    
    // 清理 file_mapping 中的所有 FileDomains
    for (auto& pair : file_mapping) {
        delete pair.second;  // FileDomains 的析构函数会清理内部的 Domain
    }
    file_mapping.clear();
    tensor_info.clear();
    std::cout << "[RKMEM] All domains cleaned up" << std::endl;
}

// 信号处理器：捕获崩溃信号并清理资源
inline void signal_handler(int signum) {
    const char* signame = "UNKNOWN";
    switch(signum) {
        case SIGBUS:  signame = "SIGBUS"; break;
        case SIGSEGV: signame = "SIGSEGV"; break;
        case SIGABRT: signame = "SIGABRT"; break;
        case SIGINT:  signame = "SIGINT"; break;
        case SIGTERM: signame = "SIGTERM"; break;
    }
    std::cerr << "\n[RKMEM] Caught signal " << signum << " (" << signame << "), cleaning up..." << std::endl;
    cleanup_all_domains();
    
    // 恢复默认信号处理并重新触发
    signal(signum, SIG_DFL);
    raise(signum);
}

// 注册 atexit 处理器和信号处理器（自动在程序退出或崩溃时调用）
inline void register_cleanup_handler() {
    static bool registered = false;
    if (!registered) {
        std::atexit(cleanup_all_domains);
        
        // 注册信号处理器
        signal(SIGBUS, signal_handler);   // Bus error
        signal(SIGSEGV, signal_handler);  // Segmentation fault
        signal(SIGABRT, signal_handler);  // Abort
        signal(SIGINT, signal_handler);   // Ctrl+C
        signal(SIGTERM, signal_handler);  // Terminate
        
        registered = true;
        std::cout << "[RKMEM] Cleanup handler and signal handlers registered" << std::endl;
    }
}



inline const char* recallmem_file_path = "/dev/recallmem-ioctl";  // recallmem 驱动设备路径
inline int _fd = -1;                            // recallmem 驱动文件描述符

inline struct MapOps* mem_recallmem_mmap(size_t file_len) {
    ioctl_MM_CREATE_t value;  // ioctl 创建映射命令结构
    
    // ===== 第1步：打开 recallmem 驱动设备（首次调用） =====
    if(_fd < 0) {
        _fd = open(recallmem_file_path, O_RDWR);
        if(_fd < 0) {
            check(0, "open recallmem ioctl device failed");
            return nullptr;
        }
        // printf("[recallmem]: opened recallmem ioctl device: %s\n", recallmem_file_path);
    }
    
    // 匿名映射模式
    // printf("[recallmem]: no file_path provided, using anonymous mmap\n");
    file_len = (file_len + 4096 - 1) / 4096 * 4096;  // 对齐到页边界
    value.req.model_path[0] = '\0';  // 空字符串表示匿名映射
    value.req.len = file_len;         // 显式指定映射大小

    
    // 调用 ioctl 创建匿名映射
    if(ioctl(_fd, IOCTL_MM_CREATE, &value) < 0) {
        // printf("[recallmem]: ioctl failed");
        return nullptr;
    }

    
    // ===== 第3步：保存映射信息到 maps 表 =====
    struct MapOps* ops = new struct MapOps();
    ops->fp =  std::string("");  // 保存文件路径
    ops->va = value.ret.va;             // 保存 VA
    ops->len = value.ret.len;           // 保存长度
    ops->handle = value.ret.handle;     // 保存 handle（驱动返回的标识符）
    printf("[recallmem]: mapped on %lx - %lx\n", ops->va, ops->va + ops->len);

    // 检查地址是否页对齐
    check_op(((uint64_t)ops->va) % PAGE_SIZE, ==, 0, "addr is not aligned");
    return ops;
}


inline std::tuple<void*, uint64_t, uint64_t, uint64_t> mem_allocate(size_t size, uint32_t flags,
                   uint64_t domain_id) {

    int ret;
    struct rknpu_mem_create mem_create = {};
    // printf("Enter mem_allocate: size %zu, flags 0x%x, domain_id %u\n", size, flags, domain_id);

    mem_create.flags = RKNPU_MEM_NON_CACHEABLE | RKNPU_MEM_KERNEL_MAPPING;
    mem_create.size = size;
    mem_create.iommu_domain_id = domain_id;

    ret = ioctl(g_npu_fd, DRM_IOCTL_RKNPU_MEM_CREATE, &mem_create);
    if (ret < 0) {
        printf("RKNPU_MEM_CREATE failed %d\n", ret);
        return std::make_tuple(nullptr, 0ULL, 0ULL, 0ULL);
    }
    // printf("mem_allocate rknpu_mem_create done, dma 0x%llx, size 0x%lx, domain_id: %x\n", 
    //                                     mem_create.dma_addr, (uint64_t)mem_create.size, mem_create.iommu_domain_id);
    

    struct rknpu_mem_map mem_map = {.handle = mem_create.handle, .reserved = 0, .offset = 0};
    ret = ioctl(g_npu_fd, DRM_IOCTL_RKNPU_MEM_MAP, &mem_map);
    if (ret < 0) {
        printf("RKNPU_MEM_MAP failed %d\n", ret);
        return std::make_tuple(nullptr, 0ULL, 0ULL, 0ULL);
    }

    void *map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, g_npu_fd, mem_map.offset);

    // TODO: Fix these undefined variables (dma_addr, obj, handle)
    // *dma_addr = mem_create.dma_addr;
    // *obj = mem_create.obj_addr;
    // *handle = mem_create.handle;
    return std::make_tuple(map, mem_create.obj_addr, mem_create.dma_addr, mem_create.handle);
}
