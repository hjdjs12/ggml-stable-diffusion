#pragma once
#include <functional>
#include <chrono>

// 获取微秒
inline uint64_t get_current_time(void) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}
// 获取毫秒
inline uint64_t get_current_time_ms(void) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count() / 1000;
}
template <typename T> class ResGuard {
  public:
    ResGuard(std::function<T(void)> &&enter, const std::function<void(T &&)> &&exit)
        : res(std::move(enter())), onExitScope(std::move(exit)) {
    }
    ResGuard(const ResGuard &) = delete;
    ResGuard &operator=(const ResGuard &) = delete;
    ~ResGuard() {
        onExitScope(std::move(res));
    }

    operator const T &() {
        return this->get();
    }

    const T &get() {
        return res;
    }

  private:
    T res;
    std::function<void(T &&)> onExitScope;
};
#define RESGUARD(name, type, enter, exit)                                                                              \
    ResGuard<type> name([&]() { enter },                                                                               \
                        [&](type &&_##name) {                                                                          \
                            (void)_##name;                                                                             \
                            exit                                                                                       \
                        })
#ifdef recallmem_timestamp
class TimeStamp {

  public:
    TimeStamp(const TimeStamp &) = default;
    TimeStamp &operator=(const TimeStamp &) = default;
    TimeStamp(const char *const _name) : name(_name) {
        log_print("[TIMESTAMP_BEGIN]%s 0x" log_format "\n", name, recallmem_timestamp);
    }
    ~TimeStamp() {
        log_print("[TIMESTAMP_END]%s 0x" log_format "\n", name, recallmem_timestamp);
    }

  private:
    const char *const name;
};
#define TIMESTAMP(name) TimeStamp ts_##name(#name);

#define TIMESTAMP_DEFER(name, var)                                                                                     \
    RESGUARD(name, uint64_t, return recallmem_timestamp;, var = recallmem_timestamp - name;)
#else
#define TIMESTAMP(name)
#define TIMESTAMP_DEFER(name, var)
#endif
