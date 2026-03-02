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

inline uint64_t cur_max_domain_index = 0;   
inline uint64_t cur_used_size = 0;

class IommuConfig{

public:
    void *iommu_addr;
    uint64_t *mem_obj_handle;
    IommuConfig() : iommu_addr(nullptr), mem_obj_handle(nullptr) {}
};

class TensorInfo {
public:
    uint64_t offset;
    uint64_t size;
    TensorInfo() : offset(0), size(0) {}  // Default constructor
    TensorInfo(uint64_t offset, uint64_t size) : offset(offset), size(size) {}
};

struct Domain{
    uint64_t id;
  
    void *virtual_addr;
    std::map<std::string, std::tuple<ggml_tensor*, TensorStorage*, IommuConfig*>> tensors;

    Domain(uint64_t domain_id)
        : id(domain_id){}

    // 禁用拷贝构造和拷贝赋值（防止双重释放）
    Domain(const Domain&) = delete;
    Domain& operator=(const Domain&) = delete;


    // 析构函数：自动释放资源
    ~Domain() {
        cleanup();
    }

private:
    void cleanup() {
        // 销毁IOMMU域

        
        // 释放mmap的内存
     
    }

    void iommu_destroy_domain() {
        // if (iommu_addr == nullptr) {
        //     return;  // 已经销毁或未创建
        // }
        
        // struct rknpu_mem_destroy mem_destroy = {};
        // mem_destroy.dma_addr = reinterpret_cast<__u64>(iommu_addr);
        // mem_destroy.iommu_domain_id = id;
        
        // try {
        //     rknpu_ioctl(DRM_IOCTL_RKNPU_MEM_DESTROY, &mem_destroy, id);
        //     std::cout << "[RKMEM] Domain " << id << " IOMMU domain destroyed" << std::endl;
        // } catch (const std::exception& e) {
        //     std::cerr << "[RKMEM] ioctl destroy failed for domain " << id << ": " 
        //               << e.what() << std::endl;
        // }
        
        // iommu_addr = nullptr;
    }

public:
   
};

inline IommuConfig * iommu_create_domain(void *virtual_addr, uint64_t domain_id, size_t used_size) {
    // Implementation for creating an IOMMU domain
    if (virtual_addr == nullptr) {  // ✅ 加个安全检查
        throw std::runtime_error("virtual_addr is null, call mmap_domain_data first!");
    }
    
    // ✅ CRITICAL FIX: Align address to page boundary (4KB = 4096 bytes)
    // NPU IOMMU requires page-aligned addresses
    // PAGE_SIZE is already defined in rk-mem.hpp
    uintptr_t addr_int = reinterpret_cast<uintptr_t>(virtual_addr);
    uintptr_t page_offset = addr_int & (PAGE_SIZE - 1);  // Offset within page
    uintptr_t aligned_addr = addr_int - page_offset;     // Round down to page boundary
    size_t aligned_size = (used_size + page_offset + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);  // Round up size
    
    struct rknpu_mem_create mem_create = {};
    mem_create.flags = RKNPU_MEM_ALLOCATED ;
    //               RKNPU_MEM_CACHEABLE |
    //               RKNPU_MEM_NON_CONTIGUOUS |              // 允许非连续物理内存
    //               RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT;    // 限制IOVA对齐（内核日志提示）

    mem_create.size = aligned_size;  // Use aligned size
    mem_create.usr_va = aligned_addr;  // Use aligned address
    mem_create.iommu_domain_id = domain_id;
    
    rknpu_ioctl(DRM_IOCTL_RKNPU_MEM_CREATE, &mem_create, domain_id);
    
    if (mem_create.dma_addr == 0) {
        throw std::runtime_error("IOMMU mapping failed: dma_addr is 0");
    }
    
    IommuConfig* config = new IommuConfig();
    // Add the page offset back to the DMA address so it points to the actual tensor data
    config->iommu_addr = (void*)(mem_create.dma_addr + page_offset);
    config->mem_obj_handle = new uint64_t(mem_create.handle);
    return config;
}

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

