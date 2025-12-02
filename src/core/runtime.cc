#include "core/runtime.h"

namespace infini {
thread_local Context g_currentCtx = nullptr;

Runtime &RuntimeObj::getInstance() {
  static Runtime instance = make_ref<RuntimeObj>();
  ;
  return instance;
}

void RuntimeObj::initThreadContext(infiniDevice_t device, int deviceId) {
  // thread_local Context currentCtx;

  if (!g_currentCtx) {
    infinirtStream_t stream = nullptr;
    CHECK_INFINI_ERROR(infinirtSetDevice(device, deviceId));
    CHECK_INFINI_ERROR(infinirtStreamCreate(&stream)); // 创建线程专属 stream
    g_currentCtx = std::make_shared<ContextObj>();
    g_currentCtx->device = device;
    g_currentCtx->deviceId = deviceId;
    g_currentCtx->stream = stream;

    std::lock_guard<std::mutex> lock(mtx);
    threadContexts[std::this_thread::get_id()] = g_currentCtx;
  }
}

Context RuntimeObj::getCurrentThreadContext() const {
  // thread_local Context currentCtx;
  if (!g_currentCtx) {
    throw std::runtime_error("Thread context not initialized!");
  }
  return g_currentCtx;
}

void RuntimeObj::setCurrentDevice(infiniDevice_t device, int deviceId) {
  CHECK_INFINI_ERROR(infinirtSetDevice(device, deviceId));
}

void RuntimeObj::init() { CHECK_INFINI_ERROR(infinirtInit()); }

void RuntimeObj::getAllDeviceCount(int *count_array) {
  CHECK_INFINI_ERROR(infinirtGetAllDeviceCount(count_array));
}

void RuntimeObj::run(const Graph &graph) const {
  // TODO: 目前仅支持单卡，后续支持多卡

  const auto &kernelRegistry = KernelRegistry::getInstance();
  for (auto &op : graph->getOperators()) {
    auto context = getCurrentThreadContext();
    auto device = context->device;
    auto kernelAttrs = KernelAttrs{device, op->getOpType().underlying()};
    Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
    kernel->compute(op, this);
  }
}

void RuntimeObj::dataMalloc(const Graph &graph) {
  for (auto &tensor : graph->getTensors()) {
    tensor->dataMalloc(shared_from_this());
  }
}

void *RuntimeObj::allocHost(size_t size) {
  void *ptr;
  CHECK_INFINI_ERROR(infinirtMallocHost(&ptr, size));
  return ptr;
}

void *RuntimeObj::allocDevice(size_t size) {
  void *ptr;
  CHECK_INFINI_ERROR(infinirtMalloc(&ptr, size));
  return ptr;
}

void RuntimeObj::deallocHost(void *ptr) {
  CHECK_INFINI_ERROR(infinirtFreeHost(ptr));
}

void RuntimeObj::deallocDevice(void *ptr) {
  CHECK_INFINI_ERROR(infinirtFree(ptr));
}

void RuntimeObj::memcpy(void *dst, const void *src, size_t size,
                        infinirtMemcpyKind_t kind) {
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
  IT_ASSERT(size < getWorkspaceSize(), "Workspace size is too small");
  return workspace;
}

size_t RuntimeObj::getWorkspaceSize() const { return workspaceSize; }

bool RuntimeObj::isCpu() const {
  auto context = getCurrentThreadContext();
  auto device = context->device;
  return device == INFINI_DEVICE_CPU;
}

void RuntimeObj::allocworkspace() {
  workspaceSize = 7ll << 30;
  workspace = allocDevice(workspaceSize);
}
} // namespace infini
