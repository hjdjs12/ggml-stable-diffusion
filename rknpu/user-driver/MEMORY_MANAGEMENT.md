# 自动内存管理说明

## 概述

本项目实现了基于 RAII（Resource Acquisition Is Initialization）的自动内存管理机制，确保在程序出错、崩溃或正常退出时，所有资源都能被正确释放。

## 主要特性

### 1. **Domain 对象自动清理**
`Domain` 结构体实现了完整的资源管理：
- **析构函数**：自动释放 mmap 分配的内存和 IOMMU 域
- **禁用拷贝**：防止双重释放导致的崩溃
- **移动语义**：支持高效的所有权转移

当 `Domain` 对象被销毁时（通过 `delete` 或离开作用域），会自动：
1. 销毁 IOMMU 域（如果已创建）
2. 解锁锁定的内存（munlock）
3. 解除内存映射（munmap）

### 2. **FileDomains 容器管理**
`FileDomains` 结构体管理一组 `Domain` 对象：
- 析构时自动删除所有管理的 `Domain` 指针
- 级联触发每个 `Domain` 的资源清理

### 3. **全局退出时清理**
程序退出时自动清理所有全局资源：
```cpp
cleanup_all_domains()  // 通过 atexit 自动调用
```

## 使用方法

### 初始化时注册清理处理器
在程序的 `main()` 函数或初始化代码中调用：
```cpp
#include "ggml/rknpu/user-driver/config.hpp"

int main() {
    // 注册自动清理处理器（仅需调用一次）
    register_cleanup_handler();
    
    // 你的代码...
    
    return 0;  // 退出时自动清理所有资源
}
```

### 创建和使用 Domain
```cpp
// 创建 Domain（使用 new）
Domain* domain = new Domain(0);

// 加载数据
mmap_domain_data(domain, "/path/to/file.bin", 0);

// 创建 IOMMU 域
domain->iommu_create_domain();

// 使用 domain...

// 不需要手动清理！
// Domain 会在程序退出时自动清理
// 或者你可以手动删除：delete domain;
```

### FileDomains 管理
```cpp
FileDomains* file_domains = new FileDomains();

// 添加多个 Domain
file_domains->domains.push_back(new Domain(0));
file_domains->domains.push_back(new Domain(1));
file_domains->domains.push_back(new Domain(2));

// 加载数据...

// 不需要手动删除每个 Domain
// 删除 FileDomains 会自动清理所有 Domain
delete file_domains;  // 或者等待程序退出时自动清理
```

### 全局映射管理
```cpp
// 添加到全局映射
file_mapping["model.bin"] = file_domains;
domain_map[0] = domain;

// 程序退出时，cleanup_all_domains() 会自动：
// 1. 遍历 file_mapping，删除所有 FileDomains
// 2. FileDomains 会删除所有 Domain
// 3. Domain 会释放所有 mmap 内存和 IOMMU 域
```

## 错误处理

### 异常安全
`mmap_domain_data()` 函数在所有错误路径上都会正确清理已分配的资源：

```cpp
void mmap_domain_data(Domain *domain, const std::string& path, uint64_t offset) {
    // 分配 mmap 内存
    domain->virtual_addr = mmap(...);
    if (domain->virtual_addr == MAP_FAILED) {
        throw std::runtime_error("mmap failed");  // 无需清理
    }
    
    // 锁定内存
    if (mlock(domain->virtual_addr, DOMAIN_SIZE) != 0) {
        munmap(domain->virtual_addr, DOMAIN_SIZE);  // ✅ 清理 mmap
        domain->virtual_addr = nullptr;
        throw std::runtime_error("mlock failed");
    }
    
    // 打开文件
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) {
        munlock(domain->virtual_addr, DOMAIN_SIZE);  // ✅ 清理 mlock
        munmap(domain->virtual_addr, DOMAIN_SIZE);   // ✅ 清理 mmap
        domain->virtual_addr = nullptr;
        throw std::runtime_error("无法打开文件");
    }
    
    // ... 其他操作都有类似的错误处理
}
```

### 崩溃恢复
即使程序崩溃（如段错误），操作系统会自动回收：
- mmap 分配的内存
- 打开的文件描述符
- IOMMU 资源（通过内核驱动）

但为了更优雅的清理，建议捕获信号：
```cpp
#include <signal.h>

void signal_handler(int signum) {
    std::cerr << "\n[SIGNAL] Received signal " << signum << std::endl;
    cleanup_all_domains();  // 手动清理
    exit(signum);
}

int main() {
    signal(SIGSEGV, signal_handler);  // 段错误
    signal(SIGINT, signal_handler);   // Ctrl+C
    signal(SIGTERM, signal_handler);  // 终止信号
    
    register_cleanup_handler();
    
    // 你的代码...
}
```

## 注意事项

### 1. 不要手动 munmap
一旦 Domain 对象创建并分配了内存，**不要手动调用 munmap**，析构函数会自动处理。

❌ **错误示例：**
```cpp
Domain* domain = new Domain(0);
mmap_domain_data(domain, "file.bin", 0);
munmap(domain->virtual_addr, DOMAIN_SIZE);  // ❌ 不要这样做！
delete domain;  // 会再次尝试 munmap，导致错误
```

✅ **正确示例：**
```cpp
Domain* domain = new Domain(0);
mmap_domain_data(domain, "file.bin", 0);
// 使用 domain...
delete domain;  // ✅ 自动清理
```

### 2. 智能指针（可选）
如果想要更安全的内存管理，可以使用智能指针：
```cpp
#include <memory>

std::unique_ptr<Domain> domain = std::make_unique<Domain>(0);
mmap_domain_data(domain.get(), "file.bin", 0);
// 离开作用域时自动删除
```

### 3. 异常传播
确保异常能正确传播，触发析构函数：
```cpp
try {
    Domain* domain = new Domain(0);
    mmap_domain_data(domain, "file.bin", 0);
    
    // 可能抛出异常的操作
    risky_operation();
    
    delete domain;  // 正常情况下执行
} catch (const std::exception& e) {
    // 如果抛出异常，domain 不会被自动删除
    // 建议使用智能指针或 RAII 包装
}
```

更好的方式：
```cpp
{
    std::unique_ptr<Domain> domain = std::make_unique<Domain>(0);
    mmap_domain_data(domain.get(), "file.bin", 0);
    
    risky_operation();  // 即使抛出异常，domain 也会被自动清理
}
```

## 调试信息

程序运行时会输出以下日志，帮助追踪资源管理：

```
[RKMEM] Cleanup handler registered          # 清理处理器已注册
[RKMEM]: Touched all pages for domain 0     # 页面已触摸
Domain id 0 [RKMEM]: Mapped 3072 MB...      # 内存映射成功
[RKMEM]: Domain id 0 iommu addr: 0xXXXX     # IOMMU 域创建成功
[RKMEM] Cleaning up all domains...          # 开始清理
[RKMEM] Domain 0 IOMMU domain destroyed     # IOMMU 域已销毁
[RKMEM] Domain 0 memory freed successfully   # 内存已释放
[RKMEM] All domains cleaned up              # 所有域已清理
```

## 总结

通过 RAII 机制，本项目实现了自动、安全、异常安全的内存管理：
✅ 无需手动清理资源  
✅ 防止内存泄漏  
✅ 异常安全  
✅ 程序崩溃时自动清理  
✅ 代码简洁易维护  

只需在程序初始化时调用 `register_cleanup_handler()`，即可享受自动内存管理的便利！
