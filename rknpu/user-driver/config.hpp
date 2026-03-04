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

// Forward declaration - TensorStorage is defined in model.h
struct TensorStorage;

#define WEIGHT_SIZE (3696UL * 1024 * 1024)
#define DOMAIN_SIZE (4096UL * 1024 * 1024)
#define REGCMD_SIZE (64 * 1024)  // 64KB for register commands
#define TASKS_MEM_SIZE (4 * 1024)  // 4KB for task descriptors
#define NPU_INPUT_BUFFER_SIZE (100 * 1024 * 1024)  // 100MB for input/output buffers
#define NPU_OUTPUT_BUFFER_SIZE (100 * 1024 * 1024)  // 100MB for output buffers

inline uint64_t cur_max_domain_index = 0;   
inline uint64_t cur_used_size = 0;

class IommuConfig{

public:
    void *iommu_addr;
    uint64_t *mem_obj_handle;
    uint64_t domain_id;
    IommuConfig() : iommu_addr(nullptr), mem_obj_handle(nullptr) {}
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

class TensorInfo {
public:
    uint64_t offset;
    uint64_t size;
    TensorInfo() : offset(0), size(0) {}  // Default constructor
    TensorInfo(uint64_t offset, uint64_t size) : offset(offset), size(size) {}
};

class LeftMemory {
public:    
    size_t size;             // 内存大小
    void* virtual_addr;      // 虚拟地址（CPU 访问用）
    void* iommu_addr;        // IOMMU DMA 地址（NPU 访问用）
    uint64_t *mem_obj_handle; // IOMMU handle（用于释放）
    
    LeftMemory() : size(0), virtual_addr(nullptr), iommu_addr(nullptr), mem_obj_handle(nullptr) {}
    
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
        virtual_addr = mmap(nullptr, size, 
                           PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS,
                           -1, 0);
        
        if (virtual_addr == MAP_FAILED) {
            std::cerr << "[LeftMemory] mmap failed: " << strerror(errno) << std::endl;
            throw std::runtime_error("LeftMemory: mmap allocation failed");
        }
        
        // 2. 锁定内存（防止被 swap）
        if (mlock(virtual_addr, size) != 0) {
            std::cerr << "[LeftMemory] mlock failed: " << strerror(errno) << std::endl;
            munmap(virtual_addr, size);
            virtual_addr = nullptr;
            throw std::runtime_error("LeftMemory: mlock failed");
        }
        
        // 3. 创建 IOMMU 映射
        try {
            IommuConfig* config = iommu_create_domain(virtual_addr, domain_id, size);
            iommu_addr = config->iommu_addr;
            mem_obj_handle = config->mem_obj_handle;
            // 注意：不要 delete config，因为它的成员已经被我们保存了
            delete config;  // 只删除 config 对象本身
        } catch (const std::exception& e) {
            std::cerr << "[LeftMemory] IOMMU mapping failed: " << e.what() << std::endl;
            munlock(virtual_addr, size);
            munmap(virtual_addr, size);
            virtual_addr = nullptr;
            throw;
        }
        
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
        // if (iommu_addr != nullptr) {
        //     struct rknpu_mem_destroy mem_destroy = {};
        //     mem_destroy.dma_addr = reinterpret_cast<__u64>(iommu_addr);
        //     mem_destroy.iommu_domain_id = domain_id;
            
        //     try {
        //         rknpu_ioctl(DRM_IOCTL_RKNPU_MEM_DESTROY, &mem_destroy, domain_id);
        //         std::cout << "[LeftMemory] IOMMU mapping destroyed for DMA=" << iommu_addr << std::endl;
        //     } catch (const std::exception& e) {
        //         std::cerr << "[LeftMemory] Warning: Failed to destroy IOMMU mapping: " 
        //                   << e.what() << std::endl;
        //     }
            
        //     if (mem_obj_handle) {
        //         delete mem_obj_handle;
        //         mem_obj_handle = nullptr;
        //     }
        //     iommu_addr = nullptr;
        // }
        
        // 2. 解锁和释放虚拟内存
        // if (virtual_addr != nullptr) {
        //     munlock(virtual_addr, size);
        //     munmap(virtual_addr, size);
        //     std::cout << "[LeftMemory] Freed " << size << " bytes at VA=" << virtual_addr << std::endl;
        //     virtual_addr = nullptr;
        // }
        
        // size = 0;
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
    
    struct rknpu_mem_create mem_create = {};
    mem_create.flags = RKNPU_MEM_ALLOCATED;
    
    // ✅ 传递原始大小，不要对齐
    mem_create.size = used_size;          // 原始大小
    mem_create.usr_va = aligned_addr;      // 对齐的地址
    mem_create.iommu_domain_id = domain_id;
    
    rknpu_ioctl(DRM_IOCTL_RKNPU_MEM_CREATE, &mem_create, domain_id);
    
    // ✅ 返回DMA地址时加上页内偏移
    IommuConfig* config = new IommuConfig();
    config->iommu_addr = (void*)(mem_create.dma_addr + page_offset);
    config->mem_obj_handle = new uint64_t(mem_create.handle);
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

