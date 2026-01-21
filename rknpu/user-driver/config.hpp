#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <zip.h>
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
#include "rknpu-ioctl.h"
#include "rk-mem.hpp"
#include "ggml.h"

#define WEIGHT_SIZE (2560UL * 1024 * 1024)
#define DOMAIN_SIZE (3072UL * 1024 * 1024)

inline uint64_t cur_max_domain_index = 0;   
inline uint64_t cur_used_size = 0;

struct Domain{
    uint64_t id;
    uint64_t used_size;
    void *virtual_addr;
    uint64_t *iommu_addr;
    uint64_t *mem_obj_addr;
    uint64_t *mem_obj_handle;

    uint64_t offset_in_file;
    std::vector<std::pair<ggml_tensor*, uint64_t>> tensors;

    Domain(uint64_t domain_id)
        : id(domain_id), used_size(0), iommu_addr(nullptr), offset_in_file(0) {}

    // void iommu_create_domain(){
    //     // Implementation for creating an IOMMU domain
    //     struct rknpu_mem_create mem_create = {};
    //     mem_create.flags = RKNPU_MEM_NON_CACHEABLE;

    //     mem_create.size = DOMAIN_SIZE;
    //     mem_create.usr_va = reinterpret_cast<__u64>(virtual_addr);
    //     mem_create.iommu_domain_id = id;
    //     // // printf("[RKMEM]: locking rkmem: %lx - %lx\n", va_itr, va_itr + itr.second);
    //     // // printf("[RKMEM]: size: %zu, flags: 0x%x, domain_id: %u\n", itr.second, mem_create.flags, mem_create.iommu_domain_id);
    //     rknpu_ioctl(DRM_IOCTL_RKNPU_MEM_CREATE, &mem_create, id);
    //     std::cout << "[RKMEM]: Domain id " << id << " iommu addr: " << std::hex << mem_create.dma_addr << std::dec << std::endl;
    //     iommu_addr = (void*)mem_create.dma_addr;
    // }
};


struct FileDomains{
    std::vector<Domain*> domains;
};


inline std::map<std::string,FileDomains*> file_mapping;
inline std::map<std::string, uint64_t> weight_start_in_file;
inline std::map<uint64_t, Domain*> domain_map;
// void mmap_domain_data(Domain *domain, std::string path, uint64_t offset) {
//     int fd = open(path.c_str(), O_RDONLY);
//     if (fd == -1) throw std::runtime_error("无法打开文件");

//     // 1. 分配一块干净的匿名内存
//     domain->virtual_addr = mmap(nullptr, DOMAIN_SIZE, PROT_READ | PROT_WRITE, 
//                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

//     // 立刻锁定！
//     if (mlock(domain->virtual_addr, DOMAIN_SIZE) != 0) {
//         perror("mlock 失败");
//         // 如果内存不足以承载这 2.5GB，这里会报错输出 "Cannot allocate memory"
//     }
    
//     uint64_t total_to_read = domain->used_size > WEIGHT_SIZE ? WEIGHT_SIZE : domain->used_size;
//     uint64_t bytes_accumulated = 0;

//     // --- 核心修改：循环读取 ---
//     while (bytes_accumulated < total_to_read) {
//         ssize_t bytes_read = pread(fd, 
//                                    (char*)domain->virtual_addr + bytes_accumulated, 
//                                    total_to_read - bytes_accumulated, 
//                                    offset + bytes_accumulated);
        
//         if (bytes_read <= 0) {
//             if (bytes_read < 0 && errno == EINTR) continue; // 被信号中断则重试
//             throw std::runtime_error("读取文件失败或到达文件末尾");
//         }
//         bytes_accumulated += bytes_read;
//     }

//     close(fd);
//     std::cout << "Domain id " << domain->id << " [RKMEM]: Mapped " << bytes_accumulated << " bytes from " << path 
//               << " at offset " << offset << " to virtual address " << domain->virtual_addr << std::endl;
// }

void memalloc_domain_data(Domain *domain, std::string path, uint64_t offset) {
    int npu_fd = Memory::get_fd();
    int ret;
    struct rknpu_mem_create mem_create = {};
    // printf("Enter mem_allocate: size %zu, flags 0x%x, domain_id %u\n", size, flags, domain_id);

    mem_create.flags = RKNPU_MEM_NON_CACHEABLE;
    mem_create.size = DOMAIN_SIZE;
    mem_create.iommu_domain_id = domain->id;

    //建立npu内存
    ret = ioctl(npu_fd, DRM_IOCTL_RKNPU_MEM_CREATE, &mem_create);
    if (ret < 0) {
        printf("RKNPU_MEM_CREATE failed %d\n", ret);
        return;
    }
    // printf("mem_allocate rknpu_mem_create done, dma 0x%llx, size 0x%lx, domain_id: %x\n", 
    //                                     mem_create.dma_addr, (uint64_t)mem_create.size, mem_create.iommu_domain_id);
    
    // 返回npu内存对象位置(offset)
    struct rknpu_mem_map mem_map = {.handle = mem_create.handle, .reserved = 0, .offset = 0};
    ret = ioctl(npu_fd, DRM_IOCTL_RKNPU_MEM_MAP, &mem_map);
    if (ret < 0) {
        printf("RKNPU_MEM_MAP failed %d\n", ret);
        return;
    }
    
    void *map = mmap(NULL, DOMAIN_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, npu_fd, mem_map.offset);

    *(domain->iommu_addr) = mem_create.dma_addr;
    *(domain->mem_obj_addr) = mem_create.obj_addr;
    *(domain->mem_obj_handle) = mem_create.handle;
    domain->virtual_addr = map;


    //读取文件到npu内存
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) throw std::runtime_error("无法打开文件");
    uint64_t total_to_read = domain->used_size;
    uint64_t bytes_accumulated = 0;

    // --- 核心修改：循环读取 ---
    while (bytes_accumulated < total_to_read) {
        ssize_t bytes_read = pread(fd, 
                                   (char*)domain->virtual_addr + bytes_accumulated, 
                                   total_to_read - bytes_accumulated, 
                                   offset + bytes_accumulated);
        
        if (bytes_read <= 0) {
            if (bytes_read < 0 && errno == EINTR) continue; // 被信号中断则重试
            throw std::runtime_error("读取文件失败或到达文件末尾");
        }
        bytes_accumulated += bytes_read;
    }

    close(fd);
    std::cout << "Domain id " << domain->id << " [RKMEM]: Mapped " << bytes_accumulated << " bytes from " << path 
              << " at offset " << offset << " to virtual address " << domain->virtual_addr << std::endl;
}