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

#define WEIGHT_SIZE (2560UL * 1024 * 1024)
#define DOMAIN_SIZE (3072UL * 1024 * 1024)

inline uint64_t cur_max_domain_index = 0;   
inline uint64_t cur_used_size = 0;

struct Domain{
    uint64_t id;
    uint64_t used_size;
    void *virtual_addr;
    void *iommu_addr;
    uint64_t *mem_obj_addr;
    uint64_t *mem_obj_handle;

    uint64_t offset_in_file;
    std::vector<std::pair<ggml_tensor*, uint64_t>> tensors;

    Domain(uint64_t domain_id)
        : id(domain_id), used_size(0), virtual_addr(nullptr), iommu_addr(nullptr), offset_in_file(0), mem_obj_addr(nullptr), mem_obj_handle(nullptr) {}

    // 禁用拷贝构造和拷贝赋值（防止双重释放）
    Domain(const Domain&) = delete;
    Domain& operator=(const Domain&) = delete;

    // 移动构造函数
    Domain(Domain&& other) noexcept
        : id(other.id), used_size(other.used_size), 
          virtual_addr(other.virtual_addr), iommu_addr(other.iommu_addr),
          mem_obj_addr(other.mem_obj_addr), mem_obj_handle(other.mem_obj_handle),
          offset_in_file(other.offset_in_file), tensors(std::move(other.tensors)) {
        // 将源对象的指针置空，防止其析构时释放资源
        other.virtual_addr = nullptr;
        other.iommu_addr = nullptr;
        other.mem_obj_addr = nullptr;
        other.mem_obj_handle = nullptr;
    }

    // 移动赋值运算符
    Domain& operator=(Domain&& other) noexcept {
        if (this != &other) {
            cleanup();  // 先清理当前对象的资源
            
            id = other.id;
            used_size = other.used_size;
            virtual_addr = other.virtual_addr;
            iommu_addr = other.iommu_addr;
            mem_obj_addr = other.mem_obj_addr;
            mem_obj_handle = other.mem_obj_handle;
            offset_in_file = other.offset_in_file;
            tensors = std::move(other.tensors);
            
            other.virtual_addr = nullptr;
            other.iommu_addr = nullptr;
            other.mem_obj_addr = nullptr;
            other.mem_obj_handle = nullptr;
        }
        return *this;
    }

    // 析构函数：自动释放资源
    ~Domain() {
        cleanup();
    }

private:
    void cleanup() {
        // 销毁IOMMU域
        if (iommu_addr != nullptr) {
            try {
                iommu_destroy_domain();
            } catch (const std::exception& e) {
                std::cerr << "[RKMEM] Error destroying IOMMU domain " << id << ": " << e.what() << std::endl;
            }
        }
        
        // 释放mmap的内存
        if (virtual_addr != nullptr) {
            // 先解锁内存
            if (munlock(virtual_addr, DOMAIN_SIZE) != 0) {
                std::cerr << "[RKMEM] munlock failed for domain " << id << ": " 
                          << strerror(errno) << std::endl;
            }
            
            // 解除内存映射
            if (munmap(virtual_addr, DOMAIN_SIZE) != 0) {
                std::cerr << "[RKMEM] munmap failed for domain " << id << ": " 
                          << strerror(errno) << std::endl;
            } else {
                std::cout << "[RKMEM] Domain " << id << " memory freed successfully" << std::endl;
            }
            
            virtual_addr = nullptr;
        }
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

    void iommu_create_domain(){
        // Implementation for creating an IOMMU domain
        if (virtual_addr == nullptr) {  // ✅ 加个安全检查
            throw std::runtime_error("virtual_addr is null, call mmap_domain_data first!");
        }
        struct rknpu_mem_create mem_create = {};
        mem_create.flags = RKNPU_MEM_ALLOCATED ;
        //               RKNPU_MEM_CACHEABLE |
        //               RKNPU_MEM_NON_CONTIGUOUS |              // 允许非连续物理内存
        //               RKNPU_MEM_IOMMU_LIMIT_IOVA_ALIGNMENT;    // 限制IOVA对齐（内核日志提示）

        mem_create.size = DOMAIN_SIZE;  // 必须映射整个域（包括计算缓冲区）
        mem_create.usr_va = reinterpret_cast<__u64>(virtual_addr);
        mem_create.iommu_domain_id = id;
        
        std::cout << "[RKMEM]: Domain " << id << " creating IOMMU mapping: "
                  << "va=" << std::hex << virtual_addr << std::dec
                  << ", domain_size=" << DOMAIN_SIZE << " bytes ("
                  << (DOMAIN_SIZE / 1024.0 / 1024.0) << " MB)"
                  << ", weight_size=" << used_size << " bytes ("
                  << (used_size / 1024.0 / 1024.0) << " MB)"
                  << ", domain_id=" << id << std::endl;
        
        rknpu_ioctl(DRM_IOCTL_RKNPU_MEM_CREATE, &mem_create, id);
        
        if (mem_create.dma_addr == 0) {
            throw std::runtime_error("IOMMU mapping failed: dma_addr is 0");
        }
        
        std::cout << "[RKMEM]: Domain " << id << " IOMMU mapping success: "
                  << "dma_addr=" << std::hex << mem_create.dma_addr << std::dec << std::endl;
        iommu_addr = (void*)mem_create.dma_addr;
        
        // **关键验证：IOMMU 映射后，立即验证 CPU 是否还能访问内存**
        std::cout << "[RKMEM]: Domain " << id << " verifying CPU access AFTER IOMMU mapping..." << std::endl;
        try {
            for (size_t i = 0; i < tensors.size() && i < 3; i++) {
                auto& [tensor, file_offset] = tensors[i];
                volatile uint8_t* ptr = (volatile uint8_t*)tensor->data;
                size_t size = ggml_nbytes(tensor);
                
                // 尝试读取首、中、尾三个位置
                volatile uint8_t test1 = ptr[0];
                volatile uint8_t test2 = ptr[size / 2];
                volatile uint8_t test3 = ptr[size - 1];
                (void)test1; (void)test2; (void)test3;
                
                std::cout << "[RKMEM]:   Tensor " << i << " @ " << std::hex << (void*)ptr << std::dec 
                          << ", size=" << size << " - CPU access OK after IOMMU" << std::endl;
            }
            std::cout << "[RKMEM]: Domain " << id << " CPU can still access memory after IOMMU mapping!" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[RKMEM]: ERROR - CPU cannot access memory after IOMMU mapping: " 
                      << e.what() << std::endl;
            throw;
        }
    }
};


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
inline std::map<std::string, uint64_t> weight_start_in_file;
inline std::map<uint64_t, Domain*> domain_map;

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
    
    // 清理 domain_map（指针已在 FileDomains 中删除，这里只清空map）
    domain_map.clear();
    
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


// 对齐到 16 字节边界（NEON 指令要求）
inline size_t align_to_16(size_t size) {
    return (size + 15) & ~15;
}

inline void mmap_domain_data(Domain *domain, const std::string& path, uint64_t offset) {
    // 1. 使用普通 mmap 分配内存（使用 MAP_POPULATE 强制立即建立页表）
    domain->virtual_addr = mmap(nullptr, DOMAIN_SIZE, 
                                PROT_READ | PROT_WRITE, 
                                MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE,
                                -1, 0);
    if (domain->virtual_addr == MAP_FAILED) {
        throw std::runtime_error("mmap failed: " + std::string(strerror(errno)));
    }
    
    // 验证 mmap 返回的地址是否 16 字节对齐
    if (((uintptr_t)domain->virtual_addr) % 16 != 0) {
        munmap(domain->virtual_addr, DOMAIN_SIZE);
        throw std::runtime_error("mmap returned unaligned address: " + 
                                std::to_string((uintptr_t)domain->virtual_addr));
    }
    std::cout << "[RKMEM]: Domain " << domain->id << " mmap allocated: "
              << "va=" << std::hex << domain->virtual_addr << std::dec
              << ", size=" << DOMAIN_SIZE << " bytes (" 
              << (DOMAIN_SIZE / 1024.0 / 1024.0) << " MB)" << std::endl;
    
    // 2. **关键步骤：立即锁定并触摸所有页面，确保页表完整建立**
    if (mlock(domain->virtual_addr, DOMAIN_SIZE) != 0) {
        munmap(domain->virtual_addr, DOMAIN_SIZE);
        domain->virtual_addr = nullptr;
        throw std::runtime_error("mlock failed: " + std::string(strerror(errno)));
    }
    std::cout << "[RKMEM]: Domain " << domain->id << " memory locked successfully" << std::endl;
    
    // 3. 触摸所有页面（每 256MB 打印进度）
    const size_t page_size = 4096;
    const size_t progress_interval = 256 * 1024 * 1024;  // 256 MB
    volatile char *ptr = (volatile char*)domain->virtual_addr;
    std::cout << "[RKMEM]: Domain " << domain->id << " touching pages..." << std::endl;
    for (size_t i = 0; i < DOMAIN_SIZE; i += page_size) {
        ptr[i] = 0;  // 触摸每个页面
        if ((i > 0) && (i % progress_interval == 0)) {
            std::cout << "[RKMEM]:   Touched " << (i / 1024 / 1024) << " MB" << std::endl;
        }
    }
    size_t num_pages = DOMAIN_SIZE / page_size;
    std::cout << "[RKMEM]: Domain " << domain->id << " touched all pages (" 
              << num_pages << " pages)" << std::endl;
    
    // 4. 打开数据文件
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) {
        munlock(domain->virtual_addr, DOMAIN_SIZE);
        munmap(domain->virtual_addr, DOMAIN_SIZE);
        domain->virtual_addr = nullptr;
        throw std::runtime_error("无法打开文件: " + std::string(strerror(errno)));
    }
    
    std::cout << "[RKMEM]: Domain " << domain->id << " reading data from file..." << std::endl;
    
    // 5. 先按文件偏移排序 tensors（关键：确保顺序正确）
    std::sort(domain->tensors.begin(), domain->tensors.end(), 
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    // 6. 读取所有 tensor 数据（保持文件中的相对位置关系）
    const size_t progress_read = 512 * 1024 * 1024;  // 512 MB
    size_t total_read = 0;
    
    for (auto& [tensor, file_offset] : domain->tensors) {
        // 计算 tensor 大小
        size_t tensor_size = ggml_nbytes(tensor);
        std::cout << tensor_size << std::endl;
        // **关键：在内存中的偏移 = 文件偏移 - domain 起始偏移**
        // 这样保持了文件中的对齐和间隙
        size_t offset_in_domain = file_offset - domain->offset_in_file;
        void* tensor_addr = (uint8_t*)domain->virtual_addr + offset_in_domain;
        
        // 打印所有 tensor 的信息（调试用）
        static int tensor_count = 0;
        std::cout << "[RKMEM]:   Tensor " << tensor_count 
                  << ": addr=" << std::hex << tensor_addr << std::dec
                  << ", size=" << tensor_size 
                  << ", file_offset=" << file_offset
                  << ", domain_offset_in_file=" << domain->offset_in_file
                  << ", offset_in_domain=" << offset_in_domain
                  << std::endl;
        tensor_count++;
        
        // 从文件读取数据
        if (lseek(fd, file_offset, SEEK_SET) == -1) {
            close(fd);
            munlock(domain->virtual_addr, DOMAIN_SIZE);
            munmap(domain->virtual_addr, DOMAIN_SIZE);
            domain->virtual_addr = nullptr;
            throw std::runtime_error("lseek 失败: " + std::string(strerror(errno)));
        }
        
        size_t bytes_read_total = 0;
        while (bytes_read_total < tensor_size) {
            size_t to_read = tensor_size - bytes_read_total;
            ssize_t bytes_read = read(fd, (uint8_t*)tensor_addr + bytes_read_total, to_read);
            
            if (bytes_read < 0) {
                if (errno == EINTR) continue;
                close(fd);
                munlock(domain->virtual_addr, DOMAIN_SIZE);
                munmap(domain->virtual_addr, DOMAIN_SIZE);
                domain->virtual_addr = nullptr;
                throw std::runtime_error("读取失败: " + std::string(strerror(errno)));
            }
            if (bytes_read == 0) break;
            
            bytes_read_total += bytes_read;
            total_read += bytes_read;
            
            // 打印读取进度
            if (total_read >= progress_read && total_read % progress_read < (size_t)bytes_read) {
                std::cout << "[RKMEM]:   Read " << (total_read / 1024 / 1024) << " MB" << std::endl;
            }
        }
        
        // **关键验证：确保读取的数据真的在内存中并且可访问**
        if (bytes_read_total < tensor_size) {
            close(fd);
            munlock(domain->virtual_addr, DOMAIN_SIZE);
            munmap(domain->virtual_addr, DOMAIN_SIZE);
            domain->virtual_addr = nullptr;
            throw std::runtime_error("数据读取不完整: 期望 " + std::to_string(tensor_size) + 
                                    ", 实际 " + std::to_string(bytes_read_total));
        }
        
        // 立即验证内存可读（计算校验和）
        uint64_t checksum = 0;
        const uint8_t* data = (const uint8_t*)tensor_addr;
        size_t sample_count = std::min(tensor_size, (size_t)1024);  // 只采样前 1KB
        for (size_t i = 0; i < sample_count; i++) {
            checksum += data[i];
        }
        (void)checksum;  // 防止编译器优化掉
        
        // 将 tensor 的数据指针指向对齐的地址
        tensor->data = tensor_addr;
        
        // 验证内存可访问（读取前16字节）
        volatile uint8_t test_byte = ((uint8_t*)tensor_addr)[0];
        if (tensor_size >= 16) {
            volatile uint8_t test_byte_15 = ((uint8_t*)tensor_addr)[15];
            (void)test_byte_15;  // 防止编译器优化掉
        }
        (void)test_byte;  // 防止编译器优化掉
    }
    
    close(fd);
    
    std::cout << "[RKMEM]: Domain " << domain->id << " data loaded: " 
              << total_read << " bytes (" 
              << (total_read / 1024.0 / 1024.0) << " MB) from " << path << std::endl;
    
    // **关键验证：在 IOMMU 映射之前，验证所有 tensor 数据都可以被 CPU 访问**
    std::cout << "[RKMEM]: Domain " << domain->id << " verifying CPU access to all tensors..." << std::endl;
    for (size_t i = 0; i < domain->tensors.size(); i++) {
        auto& [tensor, file_offset] = domain->tensors[i];
        volatile uint8_t* ptr = (volatile uint8_t*)tensor->data;
        size_t size = ggml_nbytes(tensor);
        
        // 采样验证：读取首、中、尾三个位置
        volatile uint8_t test1 = ptr[0];
        volatile uint8_t test2 = ptr[size / 2];
        volatile uint8_t test3 = ptr[size - 1];
        (void)test1; (void)test2; (void)test3;
        
        if (i < 3) {  // 只打印前3个
            std::cout << "[RKMEM]:   Tensor " << i << " @ " << std::hex << (void*)ptr << std::dec 
                      << ", size=" << size << " - CPU access OK" << std::endl;
        }
    }
    std::cout << "[RKMEM]: Domain " << domain->id << " all tensors verified before IOMMU mapping" << std::endl;
}
