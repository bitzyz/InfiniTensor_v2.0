#include "core/runtime.h"

namespace infini {
thread_local Context RuntimeObj::tls_context_cache = nullptr;
thread_local std::thread::id RuntimeObj::tls_thread_id;

Runtime &RuntimeObj::getInstance() {
    static Runtime instance = make_ref<RuntimeObj>();
    return instance;
}

RuntimeObj::~RuntimeObj() {
    // Clean up all thread Contexts
    std::unique_lock<std::shared_mutex> lock(ctx_mutex);
    // Only clear the map, do not manually release CUDA resources
    // CUDA runtime will automatically clean up all resources on program exit
    threadContexts.clear();
}

void RuntimeObj::initThreadContext(infiniDevice_t device, int deviceId) {
    auto current_tid = std::this_thread::get_id();
    // Check for thread reuse
    if (tls_context_cache && tls_thread_id == current_tid &&
        tls_context_cache->device == device &&
        tls_context_cache->deviceId == deviceId) {
        return;
    }

    CHECK_INFINI_ERROR(infinirtSetDevice(device, deviceId));

    // Create new stream
    infinirtStream_t stream = nullptr;
    CHECK_INFINI_ERROR(infinirtStreamCreate(&stream));

    // Create new Context
    Context ctx = std::make_shared<ContextObj>();
    ctx->device = device;
    ctx->deviceId = deviceId;
    ctx->stream = stream;
    ctx->workspaceSize = 0;
    ctx->workspace = nullptr;

    // Update cache and global map
    tls_context_cache = ctx;
    tls_thread_id = current_tid;

    {
        std::unique_lock<std::shared_mutex> lock(ctx_mutex);
        threadContexts[current_tid] = ctx;
    }
}

void RuntimeObj::ensureWorkspace(size_t size) const {
    if (size == 0) {
        return;
    }

    auto ctx = getCurrentThreadContext();
    if (ctx->workspace != nullptr && ctx->workspaceSize >= size) {
        return;
    }

    CHECK_INFINI_ERROR(infinirtSetDevice(ctx->device, ctx->deviceId));

    if (ctx->workspace != nullptr) {
        CHECK_INFINI_ERROR(infinirtFree(ctx->workspace));
        ctx->workspace = nullptr;
        ctx->workspaceSize = 0;
    }

    CHECK_INFINI_ERROR(infinirtMalloc(&ctx->workspace, size));
    ctx->workspaceSize = size;
}

Context RuntimeObj::getCurrentThreadContext() const {
    auto current_tid = std::this_thread::get_id();

    // Check cache validity
    if (tls_context_cache && tls_thread_id == current_tid) {
        return tls_context_cache;
    }

    // Search in global map
    {
        std::shared_lock<std::shared_mutex> lock(ctx_mutex);
        auto it = threadContexts.find(current_tid);
        if (it != threadContexts.end()) {
            tls_context_cache = it->second;
            tls_thread_id = current_tid;
            return it->second;
        }
    }

    throw std::runtime_error(
        "Thread context not initialized! Call initThreadContext() first.");
}

void RuntimeObj::setCurrentDevice(infiniDevice_t device, int deviceId) {
    auto ctx = getCurrentThreadContext();

    // If device is the same, return directly
    if (ctx->device == device && ctx->deviceId == deviceId) {
        return;
    }

    // Re-initialize Context (force=true)
    initThreadContext(device, deviceId);
}

void RuntimeObj::init() { CHECK_INFINI_ERROR(infinirtInit()); }

void RuntimeObj::getAllDeviceCount(int *count_array) {
    CHECK_INFINI_ERROR(infinirtGetAllDeviceCount(count_array));
}

void RuntimeObj::runOperators(const Graph &graph, const Context &ctx) const {
    const auto &kernelRegistry = KernelRegistry::getInstance();
    for (auto &op : graph->getOperators()) {
        auto kernelAttrs =
            KernelAttrs{ctx->device, op->getOpType().underlying()};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        kernel->compute(op, this);
    }
}

bool RuntimeObj::isCudaGraphSupported(const Context &ctx) const {
    return ctx->device == infiniDevice_t::INFINI_DEVICE_NVIDIA;
}

void RuntimeObj::compileCudaGraph(const Graph &graph,
                                  const Context &ctx) const {
    // Warmup once to materialize op descriptors/workspace before capture.
    runOperators(graph, ctx);

    CHECK_INFINI_ERROR(infinirtStreamBeginCapture(
        ctx->stream, INFINIRT_STREAM_CAPTURE_MODE_GLOBAL));

    infinirtGraph_t graphHandle = nullptr;
    try {
        runOperators(graph, ctx);
        CHECK_INFINI_ERROR(infinirtStreamEndCapture(ctx->stream, &graphHandle));
    } catch (...) {
        infinirtGraph_t abandoned = nullptr;
        if (infinirtStreamEndCapture(ctx->stream, &abandoned) ==
                INFINI_STATUS_SUCCESS &&
            abandoned != nullptr) {
            infinirtGraphDestroy(abandoned);
        }
        throw;
    }

    infinirtGraphExec_t graphExec = nullptr;
    CHECK_INFINI_ERROR(
        infinirtGraphInstantiate(&graphExec, graphHandle, nullptr, nullptr, 0));
    CHECK_INFINI_ERROR(infinirtGraphDestroy(graphHandle));

    graph->setCudaGraphExecForCurrentThread(graphExec);
    graph->markCudaGraphCompiledForCurrentThread();
}

void RuntimeObj::launchCudaGraph(const Graph &graph, const Context &ctx) const {
    auto graphExec = graph->getCudaGraphExecForCurrentThread();
    IT_ASSERT(graphExec != nullptr,
              "Compiled CUDA graph exec is missing for current thread");
    CHECK_INFINI_ERROR(infinirtGraphLuanch(graphExec, ctx->stream));
    graph->markCudaGraphLaunched();
}

void RuntimeObj::run(const Graph &graph) const {
    auto ctx = getCurrentThreadContext();

    IT_ASSERT(graph->checkBeforRun());
    if (!graph->isCudaGraphEnabled() || !isCudaGraphSupported(ctx)) {
        // TODO: Currently only supports single device, multi-device support
        // coming later.
        runOperators(graph, ctx);
        return;
    }

    if (!graph->hasCompiledCudaGraphForCurrentThread()) {
        compileCudaGraph(graph, ctx);
        return;
    }

    launchCudaGraph(graph, ctx);
}

void RuntimeObj::dataMalloc(const Graph &graph) {
    IT_ASSERT(graph->checkBeforRun());
    for (auto &tensor : graph->getTensors()) {
        tensor->dataMalloc(shared_from_this());
    }
}

void *RuntimeObj::allocHost(size_t size) {
    void *ptr;
    CHECK_INFINI_ERROR(infinirtMallocHost(&ptr, size));
    return ptr;
}

void *RuntimeObj::allocDevice(size_t size) const {
    void *ptr = nullptr;
    CHECK_INFINI_ERROR(infinirtMalloc(&ptr, size));
    return ptr;
}

void RuntimeObj::deallocHost(void *ptr) {
    CHECK_INFINI_ERROR(infinirtFreeHost(ptr));
}

void RuntimeObj::deallocDevice(void *ptr) const {
    CHECK_INFINI_ERROR(infinirtFree(ptr));
}

void RuntimeObj::memcpy(void *dst, const void *src, size_t size,
                        infinirtMemcpyKind_t kind) {
    // Basic pointer validity check
    if (dst == nullptr || src == nullptr) {
        std::cerr << "[ERROR] memcpy called with null pointer!" << std::endl;
        // Should throw exception or return error here, not continue
        throw std::runtime_error("Null pointer in memcpy");
    }

    CHECK_INFINI_ERROR(infinirtMemcpy(dst, src, size, kind));
}

void RuntimeObj::memcpyAsync(void *dst, const void *src, size_t size,
                             infinirtMemcpyKind_t kind,
                             infinirtStream_t stream) {
    CHECK_INFINI_ERROR(infinirtMemcpyAsync(dst, src, size, kind, stream));
}

void *RuntimeObj::mallocAsync(size_t size, infinirtStream_t stream) {
    void *ptr = nullptr;
    CHECK_INFINI_ERROR(infinirtMallocAsync(&ptr, size, stream));
    return ptr;
}

void RuntimeObj::freeAsync(void *ptr, infinirtStream_t stream) {
    CHECK_INFINI_ERROR(infinirtFreeAsync(ptr, stream));
}

void RuntimeObj::synchronize() const {
    CHECK_INFINI_ERROR(infinirtDeviceSynchronize());
}

void *RuntimeObj::getWorkspace(size_t size) const {
    if (size == 0) {
        return nullptr;
    }

    ensureWorkspace(size);
    auto ctx = getCurrentThreadContext();
    return ctx->workspace;
}

size_t RuntimeObj::getWorkspaceSize() const {
    auto ctx = getCurrentThreadContext();
    return ctx->workspaceSize;
}

bool RuntimeObj::isCpu() const {
    auto context = getCurrentThreadContext();
    return context->device == INFINI_DEVICE_CPU;
}

} // namespace infini
