#pragma once
#ifdef __cplusplus
extern "C" {
#endif
#ifdef __OPTEE__
#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#else
#ifdef __KERNEL__
#include <linux/types.h>
#include <linux/stdarg.h>
#include <linux/string.h>
#else
#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#include <sched.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/time.h>
#endif
#endif
typedef struct {
    uint64_t addr;
    uint64_t len;
} memory_region;
#ifndef log_format
#define log_format "%llx"
#endif
#define RECALLMEM_NW 0
#define RECALLMEM_SHARED 1
#define RECALLMEM_HOST 2

#ifndef THREADS_SCHED_SIZE
#define THREADS_SCHED_SIZE 1
#endif

#ifndef THREADS_MATMUL_SIZE
#define THREADS_MATMUL_SIZE 8
#endif

#ifndef THREADS_DECRYPT_SIZE
#define THREADS_DECRYPT_SIZE 1
#endif

#ifndef log_panic
__attribute__((noreturn)) inline void log_panic(const char *msg) {
    fprintf(stderr, "PANIC: %s\n", msg);
    abort();
}
#endif
inline void log_print(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

#ifdef RECALLMEM_DEBUG
#define log_debug(fmt, ...) log_print(fmt, ##__VA_ARGS__)
#else
#define log_debug(fmt, ...)
#endif

#define VA_LIST_ENDING_MAGIC 1349880437ull

#ifdef check_watch_errlog
#error "check_watch_errlog is defined"
#endif

#ifdef check_watch_log
#error "check_watch_log is defined"
#endif

#ifdef check_watch_panic
#error "check_watch_panic is defined"
#endif

#ifdef check_watch_format
#error "check_watch_format is defined"
#endif

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define LINE_STR TOSTRING(__LINE__)
#define check_panic(msg) log_panic("On " __FILE__ ":" LINE_STR " " msg)

inline static void print_watch_value(const char *var_names, ...) {
    char buf[256];
    va_list args;
    va_start(args, var_names);
    // log_print("<-------\n");
    for (;;) {
        uint64_t v = va_arg(args, uint64_t);
        if (v == VA_LIST_ENDING_MAGIC) {
            break;
        }
        const char *comma_pos = var_names;
        while (*comma_pos != ',')
            comma_pos++;
        memcpy(buf, var_names, comma_pos - var_names);
        buf[comma_pos - var_names] = '\0';
        var_names = comma_pos + 1;
        if (var_names[0] == ' ') {
            var_names++;
        }

        log_print("%s: 0x" log_format "\n", buf, v);
    }
    // log_print("------->\n");
    va_end(args);
}
#ifdef RECALLMEM_PRINT_CHECK
#define PRINT_CHECK_MARCO(str) log_print(str)
#else
#define PRINT_CHECK_MARCO(str)
#endif

#define check_op_watch(a, op, b, msg, ...)                                                                             \
    do {                                                                                                               \
        PRINT_CHECK_MARCO("checking [" #a " " #op " " #b "]\n");                                                       \
        int64_t __check_watch_va = (int64_t)(a), __check_watch_vb = (int64_t)(b);                                          \
        if (!(__check_watch_va op __check_watch_vb)) {                                                                     \
            log_print("[%s]:0x" log_format " op:" #op " [%s]:0x" log_format "\n", "" #a, __check_watch_va, "" #b,        \
                      __check_watch_vb);                                                                                 \
            print_watch_value(#__VA_ARGS__ ",", __VA_ARGS__, VA_LIST_ENDING_MAGIC);                                    \
            check_panic(msg);                                                                                          \
        }                                                                                                              \
    } while (0)

#define check_eq_watch(a, b, msg, ...) check_op_watch(a, ==, b, msg, __VA_ARGS__)

#define check_eq(a, b, msg) check_eq_watch(a, b, msg, VA_LIST_ENDING_MAGIC)
#define check_op(a, op, b, msg) check_op_watch(a, op, b, msg, VA_LIST_ENDING_MAGIC)
// #define check_le(a, b, msg) check_op(a, <=, b, msg)
#define check(expr, msg) check_eq_watch(!!(expr), 1, msg, VA_LIST_ENDING_MAGIC)

#ifdef recallmem_timestamp
#define timestamp_begin(name) log_print("[TIMESTAMP_BEGIN]" #name " 0x" log_format "\n", recallmem_timestamp)
#define timestamp_end(name) log_print("[TIMESTAMP_END]" #name " 0x" log_format "\n", recallmem_timestamp)
// #define timestamp_begin(name)
// #define timestamp_end(name)
#endif

#define ARR_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

#define RECALLMEM_STATUS_CODE_TABLE()                                                                                  \
    RECALLMEM_STATUS_CODE_DECL(success, 0)                                                                             \
    RECALLMEM_STATUS_CODE_DECL(idmap_failed, 1)                                                                        \
    RECALLMEM_STATUS_CODE_DECL(entry_not_found, 2)                                                                     \
    RECALLMEM_STATUS_CODE_DECL(not_aligned, 3)                                                                         \
    RECALLMEM_STATUS_CODE_DECL(external_failure, 4) RECALLMEM_STATUS_CODE_DECL(idmap_not_performed, 5)

#define RECALLMEM_STATUS_CODE_DECL(name, num)                                                                          \
    inline static uint64_t RECALLMEM_CODE_##name(void) {                                                               \
        return num;                                                                                                    \
    }
RECALLMEM_STATUS_CODE_TABLE();
#undef RECALLMEM_STATUS_CODE_DECL

#ifdef __cplusplus
}
#endif

#ifndef __KERNEL__
#ifndef __OPTEE__

inline void thread_pin_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    
    // 绑定当前线程
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        check(0 ,"pthread_setaffinity_np failed");
    }
}

inline void process_pin_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
       check(0, "sched_setaffinity failed");
    }
}

inline void thread_pin_to_cores(int *core_ids, int num) {
    if (!core_ids) {
        return; // 如果列表为空，则不执行任何操作
    }

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    // 循环遍历所有指定的核心ID，并将它们添加到 cpuset 中
    for (int core_id; core_id < num; ++core_id) {
        CPU_SET(core_ids[core_id], &cpuset);
    }

    // 绑定当前线程到配置好的 cpuset
    // pthread_self() 获取当前线程的ID
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    check(rc == 0, "pthread_setaffinity_np failed");
}

inline void process_pin_to_cores(int *core_ids, int num) {
    if (!core_ids) {
        return; // 如果列表为空，则不执行任何操作
    }
    
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    // 循环遍历所有指定的核心ID，并将它们添加到 cpuset 中
    for (int core_id = 0; core_id < num; ++core_id) {
        CPU_SET(core_ids[core_id], &cpuset);
    }

    // 绑定当前进程到配置好的 cpuset
    // pid_t 为 0 表示操作当前进程
    int rc = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    check(rc == 0, "sched_setaffinity failed");
}
#endif
#endif
