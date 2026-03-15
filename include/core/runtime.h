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
    void *workspace = nullptr;
    size_t workspaceSize = 0;
};
using Context = Ref<ContextObj>;

class RuntimeObj : public std::enable_shared_from_this<RuntimeObj> {
  private:
    mutable std::unordered_map<std::thread::id, Context> threadContexts;
    mutable std::shared_mutex ctx_mutex;
    static thread_local Context tls_context_cache;
    static thread_local std::thread::id tls_thread_id;
    void ensureWorkspace(size_t size) const;
    void runOperators(const Graph &graph, const Context &ctx) const;
    bool isCudaGraphSupported(const Context &ctx) const;
    void compileCudaGraph(const Graph &graph, const Context &ctx) const;
    void launchCudaGraph(const Graph &graph, const Context &ctx) const;

  public:
    RuntimeObj() = default;
    RuntimeObj(const RuntimeObj &) = delete;
    RuntimeObj &operator=(const RuntimeObj &) = delete;
    ~RuntimeObj();

    // Unique Runtime per thread
    static Runtime &getInstance();

    // Initialize each thread's own Context
    void initThreadContext(infiniDevice_t device, int deviceId = 0);

    // Get active Context
    Context getCurrentThreadContext() const;
    // Switch device for current thread
    void setCurrentDevice(infiniDevice_t device, int deviceId = 0);

    static void init();
    static void getAllDeviceCount(int *count_array);
    void run(const Graph &graph) const;
    void dataMalloc(const Graph &graph);
    void *allocHost(size_t size);
    void *allocDevice(size_t size) const;
    void deallocHost(void *ptr);
    void deallocDevice(void *ptr) const;
    void memcpy(void *dst, const void *src, size_t size,
                infinirtMemcpyKind_t kind);
    void memcpyAsync(void *dst, const void *src, size_t size,
                     infinirtMemcpyKind_t kind, infinirtStream_t stream);
    void *mallocAsync(size_t size, infinirtStream_t stream);
    void freeAsync(void *ptr, infinirtStream_t stream);
    // Synchronize device for current thread
    void synchronize() const;
    // Get workspace of current Context
    size_t getWorkspaceSize() const;
    void *getWorkspace(size_t size) const;

    bool isCpu() const;
};
} // namespace infini
#endif // RUNTIME_H
