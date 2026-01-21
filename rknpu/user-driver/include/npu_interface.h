#ifndef NPU_INTERFACE_H
#define NPU_INTERFACE_H

#include <stdint.h>
#include <unistd.h>
#include <sys/ioctl.h>

void *mem_allocate(int fd, size_t size, uint64_t *dma_addr, uint64_t *obj, uint32_t flags, uint32_t *handle,
                   uint32_t domain_id);
void mem_destroy(int fd, uint32_t handle, uint64_t obj_addr, uint32_t reserved);

inline static int npu_open() {

    char buf1[256], buf2[256], buf3[256];

    memset(buf1, 0, sizeof(buf1));
    memset(buf2, 0, sizeof(buf2));
    memset(buf3, 0, sizeof(buf3));

    // Open DRI called "rknpu"
    int fd = open("/dev/dri/card1", O_RDWR);
    if (fd < 0) {
        log_print("Failed to open /dev/dri/card1 %d\n", errno);
        return fd;
    }

    struct drm_version dv;
    memset(&dv, 0, sizeof(dv));
    dv.name = buf1;
    dv.name_len = sizeof(buf1);
    dv.date = buf2;
    dv.date_len = sizeof(buf2);
    dv.desc = buf3;
    dv.desc_len = sizeof(buf3);

    int ret = ioctl(fd, DRM_IOCTL_VERSION, &dv);
    if (ret < 0) {
        log_print("DRM_IOCTL_VERISON failed %d\n", ret);
        return ret;
    }
    log_print("drm name is %s - %s - %s\n", dv.name, dv.date, dv.desc);
    return fd;
}

inline static int npu_close(int fd) {
    return close(fd);
}

inline static int npu_reset(int fd) {

    // Reset the NPU
    struct rknpu_action act = {

    };
    act.flags = RKNPU_ACT_RESET;
    return ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &act);
}

#endif // NPU_INTERFACE_H
