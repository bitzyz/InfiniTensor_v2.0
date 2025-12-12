#pragma once
#ifndef RUNTIME_H
#define RUNTIME_H
#include "core/graph.h"
#include "core/kernel.h"
#include <infiniop/handle.h>
#include <infinirt.h>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>

namespace infini {
struct ContextObj {
    infiniDevice_t device = INFINI_DEVICE_CPU;
    int deviceId = 0;
    infinirtStream_t stream = nullptr;
};
using Context = Ref<ContextObj>;

class RuntimeObj : public std::enable_shared_from_this<RuntimeObj> {
  private:
    // 全局 map: thread_id -> Context
    mutable std::unordered_map<std::thread::id, Context> threadContexts;
    mutable std::shared_mutex ctx_mutex;
    static thread_local Context tls_context_cache;
    size_t workspaceSize;
    void *workspace;

  public:
    RuntimeObj() { allocworkspace(); }
    RuntimeObj(const RuntimeObj &) = delete;
    RuntimeObj &operator=(const RuntimeObj &) = delete;

    // 每个线程唯一的 Runtime
    static Runtime &getInstance();

    // 每个线程初始化自己的 Context
    void initThreadContext(infiniDevice_t device, int deviceId = 0);

    // 获取活跃 Context
    Context getCurrentThreadContext() const;
    void setCurrentDevice(infiniDevice_t device, int deviceId = 0);

    static void init();
    static void getAllDeviceCount(int *count_array);
    void run(const Graph &graph) const;
    void dataMalloc(const Graph &graph);
    void *allocHost(size_t size);
    void *allocDevice(size_t size);
    void deallocHost(void *ptr);
    void deallocDevice(void *ptr);
    void memcpy(void *dst, const void *src, size_t size,
                infinirtMemcpyKind_t kind);
    void memcpyAsync(void *dst, const void *src, size_t size,
                     infinirtMemcpyKind_t kind, infinirtStream_t stream);
    void *mallocAsync(size_t size, infinirtStream_t stream);
    void freeAsync(void *ptr, infinirtStream_t stream);
    void synchronize() const;
    size_t getWorkspaceSize() const;
    void *getWorkspace(size_t size) const;

    bool isCpu() const;

    // string toString() const;
  private:
    void allocworkspace();
};
} // namespace infini
#endif // RUNTIME_H
