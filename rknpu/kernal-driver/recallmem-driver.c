#include <linux/cdev.h>
#include <linux/fs.h>
#include <linux/init.h>
#include <linux/ioctl.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/version.h>
#include "../inc/common.h"
#include "../inc/rpc.h"
#include <linux/buffer_head.h>
#include <linux/recallmem.h>
#include <kvm/arm_hypercalls.h>
#include <../drivers/tee/optee/optee_private.h>
#include <../drivers/tee/optee/optee_msg.h>
#include <../drivers/tee/optee/optee_rpc_cmd.h>
#include <linux/cma.h>
#include <linux/dma-map-ops.h>
#include <linux/sort.h>
#include <linux/mutex.h>
// Module metadata
MODULE_AUTHOR("Yitong Cheng");
MODULE_DESCRIPTION("RecaLLMem Normal World Driver");
MODULE_LICENSE("GPL");

static unsigned int ioctl_major = 0;
static unsigned int num_of_dev = 1;
static struct cdev ioctl_cdev;
static struct cma *cma_area;

struct file *fifo_write_mm;
struct file *fifo_read_mm;
struct file *fifo_write_ta[2];
struct file *fifo_read_ta[2];
#define KERNEL_WRITE(fifo, ptr, len)                                                                                   \
    { check_eq(kernel_write(fifo, ptr, len, 0), (len), "kernel write " #ptr "failed"); }

#define KERNEL_READ(fifo, ptr, len)                                                                                    \
    { check_eq(kernel_read(fifo, ptr, len, 0), (len), "kernel read " #ptr " failed"); }
static int fifo_resources = 0x03; // 两个资源可用，使用 0b11 表示
static DEFINE_MUTEX(ta_fifo_mutex);
static DEFINE_MUTEX(mm_fifo_mutex);
int get_fifo(void) {
    int idx = -1;

    mutex_lock(&ta_fifo_mutex);
    if (fifo_resources & 0x01) {
        idx = 0;
        fifo_resources ^= 0x1;
    } else if (fifo_resources & 0x2) {
        idx = 1;
        fifo_resources ^= 0x2;
    }
    mutex_unlock(&ta_fifo_mutex);
    check(idx != -1, "fifo not get");
    // pr_info("get fifo idx:%d\n", idx);
    return idx;
}
void release_fifo(int idx) {
    // pr_info("release fifo idx:%d\n", idx);
    check(idx >= 0 && idx < 2, "invalid idx");
    mutex_lock(&ta_fifo_mutex);
    fifo_resources |= (1 << idx); // 释放资源
    mutex_unlock(&ta_fifo_mutex);
}
#define FUNC_DECLARE(name, req_type, ret_type, fifo)                                                                   \
    void func_##name(const req_type *req, const void *vla, const size_t vla_len, void *resp, size_t *resp_len) {       \
        uint64_t func_type = FUNC_##name;                                                                              \
        struct file *cur[2] = {NULL, NULL};                                                                            \
        int idx = -1;                                                                                                  \
        if (strcmp(#fifo, "mm") == 0) {                                                                                \
            mutex_lock(&mm_fifo_mutex);                                                                                \
            cur[0] = fifo_write_mm;                                                                                    \
            cur[1] = fifo_read_mm;                                                                                     \
        } else {                                                                                                       \
            idx = get_fifo();                                                                                          \
            cur[0] = fifo_write_ta[idx];                                                                               \
            cur[1] = fifo_read_ta[idx];                                                                                \
        }                                                                                                              \
        KERNEL_WRITE(cur[0], &func_type, sizeof(uint64_t));                                                            \
        KERNEL_WRITE(cur[0], req, sizeof(req_type));                                                                   \
        KERNEL_WRITE(cur[0], &vla_len, sizeof(size_t));                                                                \
        KERNEL_WRITE(cur[0], vla, vla_len);                                                                            \
        size_t len;                                                                                                    \
        KERNEL_READ(cur[1], &len, sizeof(size_t));                                                                     \
        if (*resp_len < len)                                                                                           \
            panic("resp buf too small");                                                                               \
        KERNEL_READ(cur[1], resp, len);                                                                                \
        if (strcmp(#fifo, "mm") == 0) {                                                                                \
            mutex_unlock(&mm_fifo_mutex);                                                                              \
        } else {                                                                                                       \
            check(idx != -1, "idx uninit");                                                                            \
            release_fifo(idx);                                                                                         \
        }                                                                                                              \
        *resp_len = len;                                                                                               \
    }

CROSS_BOUND_FUNC_TABLE_KU()
static uint64_t daemon_pid = 0;

typedef struct {
    uint64_t region_num;
    uint64_t recalled_num;
    memory_region *regions;
} mm_metadata_t;
static mm_metadata_t mm_metadatas[MAX_MM_NUM + 1];
static inline void count_mem_region(struct page **page, uint64_t page_nr, uint64_t *region_num,
                                    memory_region *regions) {
    uint64_t last_page_pa = 0;
    uint64_t region_num_tmp = 0;
    uint64_t last_region_size = 0;
    for (uint64_t i = 0; i < page_nr + 1; i++) { // +1 is for special ending condition
        // if (!*region_num) {
        //     pr_info("page addr:%x\n", page_to_phys(page[i]));
        // }
        if (last_page_pa && (i == page_nr || page_to_phys(page[i]) != last_page_pa + PAGE_SIZE)) {
            if (*region_num) {
                check(region_num_tmp < *region_num, "num exceeds");
                regions[region_num_tmp].addr = last_page_pa - last_region_size + PAGE_SIZE;
                regions[region_num_tmp].len = last_region_size;
            } else {
                check(regions == NULL, "should be null");
            }
            region_num_tmp++;
            last_region_size = 0;
        }
        if (i < page_nr) {
            // check_eq(page_size(page[i]), PAGE_SIZE, "currently only consider small page");
            last_page_pa = page_to_phys(page[i]);
            last_region_size += PAGE_SIZE;
        }
    }
    if (*region_num) {
        check(*region_num == region_num_tmp, "should be equal");
        check(page_to_phys(page[0]) == regions[0].addr, "pa checksum failed");
    }
    *region_num = region_num_tmp;
}
// static int compare_page(const void *a, const void *b) {
//     uint64_t va = page_to_phys(*(struct page **)a), vb = page_to_phys(*(struct page **)b);
//     if (va < vb)
//         return -1;
//     if (va > vb)
//         return 1;
//     return 0;
// }
static void handle_mmap(const char *file_path, size_t path_len, uint64_t *mm_handle, uint64_t *region_num,
                        uint64_t *len) {
    check(path_len <= MAX_MODEL_PATH_BUF_LEN, "file path oversize");

    mm_create_req_t req;
    if (file_path) {
        memcpy(req.model_path, file_path, path_len);
        check_eq(req.model_path[path_len - 1], 0, "need a ending zero");
        req.len = 0; // remember to set it zero!!!
        // pr_info("model file path:%s path len:%zu\n", req.model_path, path_len);
    } else {
        check(*len, "len should not be zero");
        req.len = *len;
    }

    mm_create_resp_t resp;
    {
        size_t siz = sizeof(resp);
        func_MM_CREATE_KU(&req, NULL, 0, &resp, &siz);
        check_eq(siz, sizeof(mm_create_resp_t), "unexpected ret size");
    }

    uint64_t va = resp.va;
    *len = resp.len;

    // --------- cma alloc ------ >
    // cma alloc should be done before holding lock
    const uint64_t page_nr = (*len + PAGE_SIZE - 1) / PAGE_SIZE;

    struct page **page = page_to_virt(cma_alloc(cma_area, (page_nr * sizeof(struct page *) + PAGE_SIZE - 1) / PAGE_SIZE,
                                                CONFIG_CMA_ALIGNMENT, false));
    // alloc_pages_exact(page_nr * sizeof(struct page *), GFP_KERNEL);
    check_eq_watch(!!page, 1, "alloc failed", page_nr);

    // --------- cma alloc ------ <

    struct task_struct *task = find_get_task_by_vpid(daemon_pid);
    check(task != NULL, "task not found");
    task_lock(task);

    struct mm_struct *mm = task->mm;
    check(mm != NULL, "mm not found");
    mmap_read_lock(mm);

    check(page_nr >= 1, "page num at least be 1");

    check(get_user_pages_remote(mm, va, page_nr, 0, page, NULL) == page_nr, "get page failed");

    // don't do sort for now, since data inside pages should be swapped
    // sort(page, page_nr, sizeof(page[0]), compare_page, NULL);

    *region_num = 0;
    count_mem_region(page, page_nr, region_num, NULL);

    check(mm_metadatas[resp.handle].regions == NULL && resp.handle <= MAX_MM_NUM, "mm handle invalid");

    void *buf = kmalloc((*region_num) * sizeof(memory_region), GFP_KERNEL);
    check_eq_watch(!!buf, 1, "kmalloc failed", region_num);
    mm_metadatas[resp.handle].region_num = *region_num;
    mm_metadatas[resp.handle].regions = buf;
    mm_metadatas[resp.handle].recalled_num = 0;

    count_mem_region(page, page_nr, region_num, mm_metadatas[resp.handle].regions);

    mmap_read_unlock(mm);
    task_unlock(task);
    put_task_struct(task);
    // *region_num = region_num_tmp;
    *len = resp.len;
    *mm_handle = resp.handle;
    check_eq(cma_release(cma_area, virt_to_page(page), (page_nr * sizeof(struct page *) + PAGE_SIZE - 1) / PAGE_SIZE),
             true, "cma_release failed");
}

static void handle_recall_mem(mm_handle_t handle, memory_region *regions, uint64_t recall_num) {
    // there maybe a concurrent issue between handle_map and handle_recall_mem, will be fixed after fully implemented
    check(handle && handle <= MAX_MM_NUM && mm_metadatas[handle].regions != NULL, "handle not valid");
    check(mm_metadatas[handle].region_num - mm_metadatas[handle].recalled_num >= recall_num, "region num is wrong");
    mm_metadata_t *metadata = &mm_metadatas[handle];
    const memory_region *src_regions = &metadata->regions[metadata->recalled_num];
    // pr_info("recall_num:%lld recalled_num:%lld regions:0x%llx\n", recall_num, metadata->recalled_num,
    //         (uint64_t)src_regions);
    for (uint64_t i = 0; i < recall_num; i++) {
        struct arm_smccc_res res;
        // pr_info("pa:0x%016llx - 0x%016llx len:0x%llx va:%llx region first bit:0x%x\n", src_regions[i].addr,
        //         src_regions[i].addr + src_regions[i].len - 1, src_regions[i].len,
        //         (uint64_t)phys_to_virt(src_regions[i].addr), *(char *)phys_to_virt(src_regions[i].addr));
        arm_smccc_1_1_hvc(KVM_HOST_SMCCC_FUNC(__recallmem_recall_mem), src_regions[i].addr, src_regions[i].len, &res);
        check(res.a0 == SMCCC_RET_SUCCESS, "smc failed");
        check(res.a1 == 0, "ret result wrong");
        uint64_t real_pa = src_regions[i].addr;

        regions[i].addr = real_pa;
        regions[i].len = src_regions[i].len;
    }
    metadata->recalled_num += recall_num;
}

static void handle_release_mem(mm_handle_t handle) {
    mm_metadata_t *meta = &mm_metadatas[handle];
    check(meta->regions != NULL && handle <= MAX_MM_NUM, "mm handle invalid");

    for (uint64_t i = 0; i < meta->region_num; i++) {
        struct arm_smccc_res res;
        arm_smccc_1_1_hvc(KVM_HOST_SMCCC_FUNC(__recallmem_release_mem), meta->regions[i].addr, meta->regions[i].len,
                          &res);
        check(res.a0 == SMCCC_RET_SUCCESS, "smc failed");
        check(res.a1 == 0, "ret result wrong");
    }

    mm_release_req_t req = {.handle = handle};
    size_t siz = sizeof(void_struct_t); // keep same size as the return type
    func_MM_RELEASE_KU(&req, NULL, 0, NULL, &siz);

    kfree(meta->regions);
    meta->regions = NULL;
}

static void handle_setup_tablepool(struct optee_msg_param *param) {
    const uint64_t page_nr = 10ull * 1024 * 1024 * 1024 / 4096; // reserve pagetable for 10GB mem
    uint64_t need_table = (page_nr + 512 - 1) / 512 + (page_nr + 512ull * 512 - 1) / 512 / 512 +
                          (page_nr + 512ull * 512 * 512 - 1) / 512 / 512 / 512;
    // pr_info("num of pages for table: 0x%llx\n", need_table);
    struct page *table_pages = cma_alloc(cma_area, need_table, CONFIG_CMA_ALIGNMENT, false);

    check_eq_watch(!!table_pages, 1, "cma alloc failed", need_table);

    param->u.value.a = (uint64_t)page_to_phys(table_pages);
    param->u.value.b = need_table * PAGE_SIZE;
}

static void handle_rpc_func_cmd_recallmem_impl_internal(struct tee_context *ctx, struct optee *optee,
                                                        struct optee_msg_arg *arg) {

    if (unlikely(daemon_pid == 0)) {
        {
            size_t siz = sizeof(daemon_pid);
            func_MM_GET_PID_KU(NULL, NULL, 0, &daemon_pid, &siz);
            check_eq(siz, sizeof(daemon_pid), "unexpected ret size");
        }

        // pr_info("mm daemon pid:%llu\n", daemon_pid);
    }
    enum SK_RPC_TYPE type = arg->params[0].u.value.a;
    switch (type) {
    case RECALLMEM_RPC:
        check(arg->num_params == 3, "param num is wrong");
        {
            rpc_proxy_req_t req = {.ta_id = arg->params[0].u.value.b, .func_id = arg->params[0].u.value.c};
            size_t len = arg->params[1].u.value.a;
            struct tee_shm *shm = (struct tee_shm *)arg->params[2].u.rmem.shm_ref;

            size_t buf_len = arg->params[2].u.rmem.size;
            func_RPC_PROXY_KU(&req, shm->kaddr, len, shm->kaddr, &buf_len);
            // pr_info("buf len after rpc:%lu\n", buf_len);
            arg->params[1].u.value.a = buf_len;
        }

        break;
    case RECALLMEM_MMAP:
        // pr_info("num_params:%d\n", arg->num_params);
        if (arg->num_params == 2) {

            // pr_info("mem attr type:0x%llx\n", arg->params[1].attr & OPTEE_MSG_ATTR_TYPE_MASK);
            check((arg->params[1].attr & OPTEE_MSG_ATTR_TYPE_MASK) == OPTEE_MSG_ATTR_TYPE_RMEM_INPUT,
                  "memref type wrong");
            struct tee_shm *shm = (struct tee_shm *)arg->params[1].u.rmem.shm_ref;
            check(arg->params[0].u.value.c == 0, "len should be zero");
            handle_mmap(shm->kaddr, arg->params[1].u.rmem.size, &arg->params[0].u.value.a, &arg->params[0].u.value.b,
                        &arg->params[0].u.value.c);
        } else {
            check(arg->num_params == 1, "param num is wrong");
            handle_mmap(NULL, 0, &arg->params[0].u.value.a, &arg->params[0].u.value.b, &arg->params[0].u.value.c);
        }
        break;

    case RECALLMEM_RECALL_MEM: {

        check(arg->num_params == 2, "param num is wrong");
        mm_handle_t handle = arg->params[0].u.value.b;
        uint64_t recall_num = arg->params[0].u.value.c;

        check((arg->params[1].attr & OPTEE_MSG_ATTR_TYPE_MASK) == OPTEE_MSG_ATTR_TYPE_RMEM_OUTPUT, "value type wrong");
        check(((struct tee_shm *)arg->params[1].u.rmem.shm_ref)->size >= recall_num * sizeof(memory_region),
              "buffer size is wrong");
        handle_recall_mem(handle, ((struct tee_shm *)arg->params[1].u.rmem.shm_ref)->kaddr, recall_num);
        break;
    }

    case RECALLMEM_RELEASE_MEM: {
        check(arg->num_params == 1, "param num is wrong");
        mm_handle_t handle = arg->params[0].u.value.b;
        handle_release_mem(handle);
        break;
    }
    case RECALLMEM_SETUP_TABLEPOOL:
        check(arg->num_params == 1, "param num is wrong");
        handle_setup_tablepool(&arg->params[0]);
        break;
    default:
        check(false, "unexpected enum");
    }

    arg->ret = TEEC_SUCCESS;
}
// static int create_mm_proc(uint64_t model_idx, size_t len) {

//     // if (len) { // Do malloc

//     // } else { // Map a file
//     // }
// }

// uint64_t handle_mm_create(uint64_t model_handle, uint64_t len) {
//     mm_create_req_t req = {.model_handle = model_handle, .len = len};
//     mm_create_resp_t resp = func_MM_CREATE_KU(&req);

//     uint64_t va = resp.va;
//     len = resp.len;

//     // struct pid *pid = find_vpid(daemon_pid);
//     // check(pid != NULL, "pid not found");
//     // struct task_struct *task = get_pid_task(pid, PIDTYPE_PID);
//     struct task_struct *task = find_get_task_by_vpid(daemon_pid);
//     check(task != NULL, "task not found");
//     task_lock(task);

//     struct mm_struct *mm = task->mm;
//     check(mm != NULL, "mm not found");
//     mmap_read_lock(mm);

//     uint64_t page_nr = (len + PAGE_SIZE - 1) / PAGE_SIZE;
//     struct page **page = kmalloc(page_nr * sizeof(struct page *), GFP_KERNEL);
//     void *buf = kmalloc(PAGE_SIZE, GFP_KERNEL);
//     check(page != NULL, "kmalloc failed");
//     check(buf != NULL, "kmalloc failed");
//     check(((uint64_t)buf) % PAGE_SIZE == 0, "page not aligned");
//     *(char *)buf = 'a';

//     check(get_user_pages_remote(mm, va, page_nr, 0, page, NULL) == page_nr, "get page failed");

//     uint64_t ret = page_to_phys(page[0]);
//     pr_info("gpa:%llx gva:%llx\n", ret, (uint64_t)phys_to_virt(ret));
//     check(*(uint8_t *)phys_to_virt(ret) == ('1'), "access");
//     struct arm_smccc_res res;
//     // arm_smccc_1_1_hvc(KVM_HOST_SMCCC_FUNC(__recallmem_recall_mem), ret, PAGE_SIZE * page_nr, &res);
//     // check(res.a0 == SMCCC_RET_SUCCESS, "smc failed");
//     // pr_info("smc func ret:%ld\n", res.a1);
//     recallmem_debug_point();
//     arm_smccc_1_1_hvc(KVM_HOST_SMCCC_FUNC(__recallmem_recall_mem), virt_to_phys(buf), PAGE_SIZE, &res);
//     recallmem_debug_point();
//     check(res.a0 == SMCCC_RET_SUCCESS, "smc failed");
//     pr_info("smc func ret:%ld\n", res.a1);
//     check(res.a1 == 0, "ret result wrong");
//     // pr_info("modified result is:%c\n", *(char *)buf);
//     recallmem_debug_point();
//     check(*(char *)buf == 'a', "modified result wrong");
//     recallmem_debug_point();

//     arm_smccc_1_1_hvc(KVM_HOST_SMCCC_FUNC(__recallmem_recall_mem), virt_to_phys(buf), PAGE_SIZE, &res);
//     check(res.a0 == SMCCC_RET_SUCCESS, "smc failed");
//     pr_info("smc func ret:%ld\n", res.a1);
//     check(res.a1 == 0, "ret result wrong");
//     pr_info("modified result is:%c\n", *(char *)buf);

//     flush_tlb_all();
//     // check(res.a1 == 0, "recall failed");

//     recallmem_debug_point();
//     check(*(uint8_t *)phys_to_virt(ret) == ('x'), "access");
//     pr_info("phys addr:%llx virt addr:%llx accessed bit:%c\n", ret, (uint64_t)phys_to_virt(ret),
//             *(uint8_t *)phys_to_virt(ret));

//     for (uint64_t i = 0; i < page_nr; i++) {
//         uint64_t pa = page_to_phys(page[i]);
//         check(*(uint8_t *)phys_to_virt(pa) == (i == 0 ? 'x' : '1'), "access");
//     }

//     for (uint64_t i = 0; i < page_nr; i++) {
//         uint64_t pa = page_to_phys(page[i]);
//         check(*(uint8_t *)phys_to_virt(pa) == (i == 0 ? 'x' : '1'), "access");
//         put_page(page[i]);
//     }

//     kfree(page);
//     mmap_read_unlock(mm);
//     task_unlock(task);
//     put_task_struct(task);

//     return ret;
// }

static int handle_user_req(ioctl_rpc_arg *arg) {
    switch (arg->func_type) {
    case FUNC_MM_CREATE_KU:
        arg->status = -1;
        // handle_mm_create(arg->data, 0);
        break;
    default:
        panic("func type unknown");
    }
    return 0;
}

static long ioctl_ioctl(struct file *filp, unsigned int cmd, unsigned long arg) {
    int retval = 0;
    ioctl_rpc_arg data;
    memset(&data, 0, sizeof(data));

    check(daemon_pid, "daemon pid should be set");

    switch (cmd) {

    case IOCTL_TEST:
        if (copy_from_user(&data, (int __user *)arg, sizeof(data))) {
            retval = -EFAULT;
            goto done;
        }
        pr_debug("Got data from user, func type:%lld\n", data.func_type);
        if ((retval = handle_user_req(&data))) {
            panic("handle user req failed: %d\n", retval);
            retval = -EFAULT;
            goto done;
        }
        if (copy_to_user((int __user *)arg, &data, sizeof(data))) {
            retval = -EFAULT;
            goto done;
        }

        break;

    default:
        retval = -ENOTTY;
    }

done:
    return retval;
}

// DEFINE_MUTEX(user_daemon_lock);
// static unsigned int connect_counter = 0;
// static int ioctl_open(struct inode *inode, struct file *filp) {
//     mutex_lock(&user_daemon_lock);
//     if (connect_counter > 0) {
//         panic("Only allowed one connection\n");
//     }
//     connect_counter++;
//     mutex_unlock(&user_daemon_lock);
//     return 0;
// }
static struct file_operations fops = {
    .open = NULL,
    .release = NULL,
    .read = NULL,
    .unlocked_ioctl = ioctl_ioctl,
};

// static struct miscdevice cma_malloc_miscdevice = {
//     .minor = MISC_DYNAMIC_MINOR,
//     .name = "cma_malloc",
//     .fops = NULL,
//     .mode = S_IRUGO | S_IWUGO,
// };

// Custom init and exit methods
static int __init recallmem_init(void) {
    dev_t dev;
    int alloc_ret = -1;
    int cdev_ret = -1;
    alloc_ret = alloc_chrdev_region(&dev, 0, num_of_dev, RECALLMEM_DRIVER_NAME);

    if (alloc_ret)
        goto error;

    ioctl_major = MAJOR(dev);
    cdev_init(&ioctl_cdev, &fops);
    cdev_ret = cdev_add(&ioctl_cdev, dev, num_of_dev);

    if (cdev_ret)
        goto error;

    cma_area = dma_contiguous_default_area;
    // cma_ret = misc_register(&cma_malloc_miscdevice);
    // check(!cma_ret, "cma register failed");
    // cma_dev = cma_malloc_miscdevice.this_device;

    // check_eq(!!cma_dev->cma_area, 1, "cma_area is empty");
    struct page *page = cma_alloc(cma_area, 4096, CONFIG_CMA_ALIGNMENT, false);
    check_eq(!!page, 1, "cma_alloc failed");
    check_eq(cma_release(cma_area, page, 4096), true, "cma_release failed");

    // pr_alert("%s driver(major: %d) installed.\n", RECALLMEM_DRIVER_NAME, ioctl_major);
    if (ERR_PTR == (void *)device_create(class_create(RECALLMEM_DRIVER_NAME), NULL, MKDEV(ioctl_major, 0), NULL,
                                         RECALLMEM_DRIVER_NAME)) {
        pr_err("fail to create node in sysfs\n");
    }
    // else {
    //     pr_info("Device created on /dev/%s\n", RECALLMEM_DRIVER_NAME);
    // }

#define INIT_FIFO(name, fname)                                                                                         \
    name = filp_open(fname, O_RDWR, 0);                                                                                \
    if (IS_ERR(name)) {                                                                                                \
        pr_err("Error opening FIFO: %ld\n", PTR_ERR(name));                                                            \
        goto error;                                                                                                    \
    }

    INIT_FIFO(fifo_write_mm, FIFO_KU(MM)); // Use O_RDWR to avoid waiting for partner
    INIT_FIFO(fifo_read_mm, FIFO_UK(MM));

    INIT_FIFO(fifo_write_ta[0], FIFO_KU(TA) "-0");
    INIT_FIFO(fifo_read_ta[0], FIFO_UK(TA) "-0");

    INIT_FIFO(fifo_write_ta[1], FIFO_KU(TA) "-1");
    INIT_FIFO(fifo_read_ta[1], FIFO_UK(TA) "-1");

    handle_rpc_func_cmd_recallmem_impl = handle_rpc_func_cmd_recallmem_impl_internal;

    return 0;
error:
    if (cdev_ret == 0)
        cdev_del(&ioctl_cdev);
    if (alloc_ret == 0)
        unregister_chrdev_region(dev, num_of_dev);

    return -1;
}

static void __exit recallmem_exit(void) {
    printk(KERN_INFO "Goodbye my friend, I shall miss you dearly...\n");
}
module_init(recallmem_init);
module_exit(recallmem_exit);