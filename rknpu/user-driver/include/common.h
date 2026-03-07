#pragma once

#include "utils.h"

#define MAX_MODEL_PATH_BUF_LEN 64

#define RECALLMEM_DRIVER_NAME "recallmem-ioctl"
#define NAMED_FIFO_MM "/tmp/fifo-mm"
#define NAMED_FIFO_MM_DECRYPT "/tmp/fifo-mm-decrypt"
#define NAMED_FIFO_TA "/tmp/fifo-ta"
#define FIFO_KU(name) NAMED_FIFO_##name "-ku"
#define FIFO_UK(name) NAMED_FIFO_##name "-uk"
#define DAEMON_LOG "/mnt/nvme/root/recallmem-daemon.log"

#define IOC_MAGIC '\x66'

typedef uint64_t mm_handle_t;

typedef struct {
    char model_path[MAX_MODEL_PATH_BUF_LEN];
    uint64_t len;
} mm_create_req_t;

typedef struct {
    mm_handle_t handle;
    uint64_t va;
    uint64_t len;
} mm_create_resp_t;

typedef struct {
    mm_handle_t handle;
    uint64_t va;
    uint64_t len;
} mm_common_req_t;

typedef struct {
    mm_handle_t handle;
    uint64_t len;
} mm_lockahead_req_t;

typedef mm_handle_t mm_release_req_t;
typedef mm_common_req_t mm_mlock_req_t;
typedef mm_common_req_t mm_munlock_req_t;
typedef mm_common_req_t mm_test_req_t;
typedef mm_common_req_t mm_decrypt_req_t;

typedef struct {
    uint64_t ta_id;
    uint64_t func_id;
} rpc_proxy_req_t;

typedef uint8_t void_struct_t[0];
typedef uint32_t status_t;

#define CROSS_BOUND_FUNC_TABLE()                                                                                       \
    FUNC_DECLARE(MM_CREATE, mm_create_req_t, mm_create_resp_t)                                                         \
    FUNC_DECLARE(MM_GET_PID, void_struct_t, uint64_t)                                                                  \
    FUNC_DECLARE(MM_RELEASE, mm_release_req_t, status_t)                                                               \
    FUNC_DECLARE(MM_LOCKAHEAD, mm_lockahead_req_t, uint64_t)                                                           \
    FUNC_DECLARE(MM_UNLOCKAHEAD, mm_lockahead_req_t, uint64_t)                                                         \
    FUNC_DECLARE(MM_MLOCK, mm_mlock_req_t, status_t)                                                                   \
    FUNC_DECLARE(MM_MUNLOCK, mm_munlock_req_t, status_t)                                                               \
    FUNC_DECLARE(MM_PING_PONG, void_struct_t, status_t)                                                                \
    FUNC_DECLARE(MM_TEST, mm_test_req_t, status_t)
    // FUNC_DECLARE(MM_DECRYPT, mm_decrypt_req_t, status_t)

#define FUNC_DECLARE(name, req_type, ret_type) FUNC_##name,
enum { PLACEHOLDER, CROSS_BOUND_FUNC_TABLE() FUNC_MM_DECRYPT };

#undef FUNC_DECLARE

#define MAX_MM_NUM 16

#define FUNC_DECLARE(name, req_type, ret_type)                                                                         \
    typedef union {                                                                                                    \
        req_type req;                                                                                                  \
        ret_type ret;                                                                                                  \
    } ioctl_##name##_t;

CROSS_BOUND_FUNC_TABLE()
typedef union {
        mm_decrypt_req_t req;
        status_t ret;
    } ioctl_MM_DECRYPT_t;
#undef FUNC_DECLARE

#define IOCTL_MM_CREATE _IOWR(IOC_MAGIC, FUNC_MM_CREATE, ioctl_MM_CREATE_t)
#define IOCTL_MM_RELEASE _IOWR(IOC_MAGIC, FUNC_MM_RELEASE, ioctl_MM_RELEASE_t)
#define IOCTL_LOCKAHEAD _IOWR(IOC_MAGIC, FUNC_MM_LOCKAHEAD, ioctl_MM_LOCKAHEAD_t)
#define IOCTL_UNLOCKAHEAD _IOWR(IOC_MAGIC, FUNC_MM_UNLOCKAHEAD, ioctl_MM_UNLOCKAHEAD_t)
#define IOCTL_MM_LOCK _IOWR(IOC_MAGIC, FUNC_MM_MLOCK, ioctl_MM_MLOCK_t)
#define IOCTL_MM_UNLOCK _IOWR(IOC_MAGIC, FUNC_MM_MUNLOCK, ioctl_MM_MUNLOCK_t)
#define IOCTL_MM_TEST _IOWR(IOC_MAGIC, FUNC_MM_TEST, ioctl_MM_TEST_t)
#define IOCTL_MM_DECRYPT _IOWR(IOC_MAGIC, FUNC_MM_DECRYPT, ioctl_MM_DECRYPT_t)
