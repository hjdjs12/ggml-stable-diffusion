#pragma once

#include <stdint.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <mutex>
#include <atomic>
#include <sys/ioctl.h>
#include "utils.hpp"
#include "utils.h"
#include <sys/syscall.h>
extern "C" {
#include "rknpu-ioctl.h"
#include "npu_interface.h"
#include "npu_matmul.h"
}

// some comments say the length of npu_regs must be 112; currently 108 still seems ok
#define NPU_REGS_SIZE 108
#define TASK_REG_AMOUNT (NPU_REGS_SIZE - RKNPU_PC_DATA_EXTRA_AMOUNT)
#ifndef PAGE_SIZE
#define PAGE_SIZE 4096

class MemoryInternal {

  public:
    mutable void *va{};
    mutable uint64_t dma{};
    mutable uint64_t obj{};
    mutable uint32_t handle{};

  protected:
    MemoryInternal(size_t size, uint32_t flags, uint32_t domain_id)
        : _size(size), _flags(flags), _domain_id(domain_id) {
    }
    MemoryInternal(MemoryInternal &) = delete;
    MemoryInternal(MemoryInternal &&) = default;
    MemoryInternal &operator=(MemoryInternal &&) = default;

    mutable size_t _size;
    mutable uint32_t _flags;
    mutable uint32_t _domain_id;
};
class Memory : public MemoryInternal {

  public:
    using MemoryInternal::_flags;
    using MemoryInternal::_size;
    using MemoryInternal::dma;
    using MemoryInternal::handle;
    using MemoryInternal::obj;
    using MemoryInternal::va;
    // inline static std::mutex ctx_map_mutex{};
    inline static std::atomic<int> current_domain{0};
    inline static std::atomic<int> running_num{0};
    static int get_fd() {
        static RESGUARD(fd, int, int _fd = npu_open(); check(_fd >= 0, "open fd failed"); return _fd;
                        , check(npu_close(fd) >= 0, "close fd failed"););
        return fd.get();
    }

    static inline void check_before_ioctl(int domain_id) {
        if(current_domain.load(std::memory_order_acquire) != domain_id) {
            while(running_num.load(std::memory_order_acquire) != 0) {};
            current_domain.store(domain_id, std::memory_order_release);
        }

        running_num.fetch_add(1, std::memory_order_acq_rel);
    }

    static inline void recall_num() {
        check_op(running_num.fetch_sub(1, std::memory_order_acq_rel), >, 0, "recall_num error");
    }
    static void _rknpu_ioctl(uint32_t cmd, void *act, int domain_id) {
        // std::lock_guard<std::mutex> lock(ctx_map_mutex);
        check_before_ioctl(domain_id);
        (void)cmd;
        (void)act;
        if (cmd == (uint32_t)DRM_IOCTL_RKNPU_MEM_DESTROY) {
            int ret;
            ret = ioctl(Memory::get_fd(), cmd, act);
            if(ret < 0)
                printf("DRM_IOCTL_RKNPU_MEM_DESTROY return value: %d\n", ret);
        } else {
            int ret = ioctl(Memory::get_fd(), cmd, act);
            if (ret != 0) {
                printf("rknpu ioctl failed: cmd=0x%x, ret=%d, errno=%d (%s)\n", 
                       cmd, ret, errno, strerror(errno));
            }
            check(ret == 0, "rknpu ioctl failed\n");
        }
        recall_num();
    }
    template <typename T> T get_va() const {
        static_assert(std::is_pointer<T>::value, "only accept pointer type");
        return static_cast<T>(va);
    }
    static size_t roundup(size_t s) {
        return (s + 4095) & ~4095;
    }
    /**
     * @brief Construct a new Memory object
     *
     * @param addr the written memory; should not be wrote anymore
     * @param size
     */
    Memory(void *addr, size_t size, uint32_t domain_id) : MemoryInternal(size, RKNPU_MEM_ALLOCATED, domain_id) {
        _size = roundup(_size);
        va = addr;

        _setup_va();
    }
    Memory(size_t size, uint32_t flags, uint32_t domain_id) : MemoryInternal(size, flags, domain_id) {
        _size = (size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
        _setup_va();
    }
    ~Memory() {
        // printf("Memory destructor: va %p, size %zu, flags 0x%x, domain_id %u\n", va, _size, _flags, _domain_id);
        // std::lock_guard<std::mutex> lock(ctx_map_mutex);
        check_before_ioctl(_domain_id);
        if (va) {
            if (_flags & RKNPU_MEM_ALLOCATED) {
                mem_destroy(get_fd(), 0, reinterpret_cast<uint64_t>(va), _size / PAGE_SIZE);
            } else {
                check_eq(syscall(__NR_munlock, va, _size), 0, "munlock failed");
                munmap(va, _size);
                mem_destroy(get_fd(), handle, obj, 0);
            }
        }
        recall_num();
    }

    Memory(const Memory &m) = delete;
    Memory(Memory &&m) : MemoryInternal(std::move(m)) {
        m.va = nullptr; // set as invalid;
    }
    Memory &operator=(Memory &&m) {
        if (this != &m) {
            MemoryInternal::operator=(std::move(m));
            m.va = nullptr;
        }
        return *this;
    }
    void switch_to_this(int domain_id) {
        // Reset the NPU
        // std::lock_guard<std::mutex> lock(ctx_map_mutex);
        check_before_ioctl(domain_id);
        _domain_id = domain_id;
        struct rknpu_action act = {
            .flags = RKNPU_SET_IOMMU_DOMAIN_ID,
            .value = _domain_id,
        };

        check_eq(ioctl(Memory::get_fd(), DRM_IOCTL_RKNPU_ACTION, &act), 0, "ioctl failed");
        recall_num();
    }
    uint32_t get_domain() const {
        return _domain_id;
    }

  private:
    void _setup_va() {
        // std::lock_guard<std::mutex> lock(ctx_map_mutex);
        check_before_ioctl(_domain_id);
        if (_flags & RKNPU_MEM_ALLOCATED) {
            struct rknpu_mem_create create;
            create.usr_va = (uint64_t)va;
            create.size = _size;
            create.flags = RKNPU_MEM_ALLOCATED;
            create.iommu_domain_id = _domain_id;
            // log_print("locking Memory: %llx - %llx\n", (uint64_t)va, (uint64_t)va + _size);
            // log_print("size: %zu, flags: 0x%x, domain_id: %u\n", _size, create.flags,
            //               create.iommu_domain_id);
            check_eq(ioctl(Memory::get_fd(), DRM_IOCTL_RKNPU_MEM_CREATE, &create), 0, "ioctl failed");
            // log_print("rknpu_mem_create done, dma: %llx, size: 0x%lx\n", create.dma_addr, (uint64_t)create.size);
            dma = create.dma_addr;
            handle = create.handle;
        } else {
            va = mem_allocate(get_fd(), _size, &dma, &obj, _flags, &handle, _domain_id);
            check(va, "mem alloc failed");
        }
        recall_num();
    }
};

class RegCmd {
  public:
    RegCmd(uint16_t m, uint16_t k, uint16_t n, Memory &mem, uint64_t &off) : regcmd{mem}, offset(off) {
        matmul_params_t params = {.m = m, .k = k, .n = n, .tasks = getVa()};
        check_eq(gen_matmul_int8(&params), 0,"gen matmul failed");
        off += NPU_REGS_SIZE * sizeof(uint64_t);
        check_op(off, <=, regcmd._size, "off exceeds");
    }
    void setupAddr(uint64_t input, uint64_t weights, uint64_t output) {
        is_addr_setup = true;
        update_matmul_addr(getVa(), input, weights, output);
    }
    void setupAddr(const Memory &input, const Memory &weights, const Memory &output) {
        is_addr_setup = true;
        if (input.get_domain() != get_domain() || weights.get_domain() != get_domain() ||
            output.get_domain() != get_domain()) {
            check(0, "domain is not same");
        }
        update_matmul_addr(getVa(), input.dma, weights.dma, output.dma);
    }
    uint64_t *getVa() const {
        return (uint64_t *)(regcmd.get_va<uint8_t *>() + offset);
    }
    uint32_t getCmdDma() const {
        check(is_addr_setup, "addr is not setup");
        return regcmd.dma + offset;
    }
    uint32_t get_domain() const {
        return regcmd.get_domain();
    }

  private:
    const Memory &regcmd;
    const uint64_t offset;
    bool is_addr_setup = false;
};

inline void rknpu_ioctl(uint32_t cmd, void *act, int domain_id) {
    Memory::_rknpu_ioctl(cmd, act, domain_id);
}
#endif // PAGE_SIZE