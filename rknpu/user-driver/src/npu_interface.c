/*
 * Copyright (C) 2024  Jasbir Matharu, <jasjnuk@gmail.com>
 *
 * This file is part of rk3588-npu.
 *
 * rk3588-npu is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * rk3588-npu is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with rk3588-npu.  If not, see <https://www.gnu.org/licenses/>.
 *
 */

#include <stdint.h>
#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include "rknpu-ioctl.h"
#include "npu_hw.h"

void *mem_allocate(int fd, size_t size, uint64_t *dma_addr, uint64_t *obj, uint32_t flags, uint32_t *handle,
                   uint32_t domain_id) {

    int ret;
    struct rknpu_mem_create mem_create = {};
    // printf("Enter mem_allocate: size %zu, flags 0x%x, domain_id %u\n", size, flags, domain_id);

    mem_create.flags = flags | RKNPU_MEM_NON_CACHEABLE;
    mem_create.size = size;
    mem_create.iommu_domain_id = domain_id;

    ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_CREATE, &mem_create);
    if (ret < 0) {
        printf("RKNPU_MEM_CREATE failed %d\n", ret);
        return NULL;
    }
    // printf("mem_allocate rknpu_mem_create done, dma 0x%llx, size 0x%lx, domain_id: %x\n", 
    //                                     mem_create.dma_addr, (uint64_t)mem_create.size, mem_create.iommu_domain_id);
    

    struct rknpu_mem_map mem_map = {.handle = mem_create.handle, .reserved = 0, .offset = 0};
    ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_MAP, &mem_map);
    if (ret < 0) {
        printf("RKNPU_MEM_MAP failed %d\n", ret);
        return NULL;
    }

    void *map = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mem_map.offset);

    *dma_addr = mem_create.dma_addr;
    *obj = mem_create.obj_addr;
    *handle = mem_create.handle;
    return map;
}

void mem_destroy(int fd, uint32_t handle, uint64_t obj_addr, uint32_t reserved) {

    int ret;
    struct rknpu_mem_destroy destroy = {.handle = handle, .reserved = reserved, .obj_addr = obj_addr};
    // printf("Enter mem_destroy: handle %u, obj_addr 0x%lx, reserved %u\n", handle, obj_addr, reserved);

    ret = ioctl(fd, DRM_IOCTL_RKNPU_MEM_DESTROY, &destroy);
    if (ret < 0) {
        printf("RKNPU_MEM_DESTROY failed %s\n", strerror(errno));
    }
}
