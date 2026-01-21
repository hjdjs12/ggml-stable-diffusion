#include "utils.h"
#include <linux/printk.h>
#include <linux/panic.h>
#include <linux/module.h>

MODULE_LICENSE("GPL");

void log_panic(void) {
    panic("abort");
    while (1) {
    }
}

void log_print(const char *fmt, ...) {
    va_list args;

    va_start(args, fmt);
    vprintk(fmt, args);
    va_end(args);
}
