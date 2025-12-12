#pragma once
#include "core/operator.h"
#include <infinirt.h>
namespace infini {
class RuntimeObj;
using KernelAttrs = std::tuple<infiniDevice_t, OpType::underlying_t>;
class Kernel {
  public:
    Kernel() {}
    virtual ~Kernel() {}
    virtual void compute(const Operator &op,
                         const RuntimeObj *context) const = 0;
};

class KernelRegistry {
  public:
    using KernelRecord =
        tuple<Kernel *const, const string, const int>; // Kernel, name, ID

  private:
    std::map<KernelAttrs, KernelRecord> kernels;
    int nKernels = 0;

  public:
    ~KernelRegistry() {
        for (auto &[k, v] : kernels)
            delete std::get<0>(v);
    }
    static KernelRegistry &getInstance() {
        static KernelRegistry instance;
        return instance;
    }
    bool registerKernel(const KernelAttrs &key, Kernel *kernel, string name) {
        IT_ASSERT(kernels.find(key) == kernels.end(),
                  "Kernel already registered");
        kernels.emplace(key, KernelRecord{kernel, name, ++nKernels});
        return true;
    }
    Kernel *getKernel(const KernelAttrs &kernelAttrs) const {
        auto it = kernels.find(kernelAttrs);
        IT_ASSERT(it != kernels.end(), "Kernel not found");
        return std::get<0>(it->second);
    }
    const KernelRecord &getKernelItem(const KernelAttrs &kernelAttrs) const {
        return kernels.at(kernelAttrs);
    }
};

} // namespace infini

#define _REGISTER_KERNEL_1(device, opType, kernel, name, cnt)                  \
    namespace infini {                                                         \
    static const bool _CAT(_register_kernel_, cnt) =                           \
        KernelRegistry::getInstance().registerKernel(KernelAttrs{device,       \
                                                                 opType},      \
                                                     new kernel(), name);      \
    }

#define REGISTER_KERNEL(device, opType, kernel, name)                          \
    _REGISTER_KERNEL_1(device, opType, kernel, name, __COUNTER__)

#define REGISTER_KERNEL_ALL_DEVICES(opType, kernel)                            \
    REGISTER_KERNEL(infiniDevice_t::INFINI_DEVICE_NVIDIA, opType, kernel,      \
                    TOSTRING(_CAT(kernel, _NVIDIA)));                          \
    REGISTER_KERNEL(infiniDevice_t::INFINI_DEVICE_CPU, opType, kernel,         \
                    TOSTRING(_CAT(kernel, _CPU)));                             \
    REGISTER_KERNEL(infiniDevice_t::INFINI_DEVICE_CAMBRICON, opType, kernel,   \
                    TOSTRING(_CAT(kernel, _CAMBRICON)));                       \
    REGISTER_KERNEL(infiniDevice_t::INFINI_DEVICE_ASCEND, opType, kernel,      \
                    TOSTRING(_CAT(kernel, _ASCEND)));                          \
    REGISTER_KERNEL(infiniDevice_t::INFINI_DEVICE_METAX, opType, kernel,       \
                    TOSTRING(_CAT(kernel, _METAX)));                           \
    REGISTER_KERNEL(infiniDevice_t::INFINI_DEVICE_MOORE, opType, kernel,       \
                    TOSTRING(_CAT(kernel, _MOORE)));                           \
    REGISTER_KERNEL(infiniDevice_t::INFINI_DEVICE_ILUVATAR, opType, kernel,    \
                    TOSTRING(_CAT(kernel, _ILUVATAR)));                        \
    REGISTER_KERNEL(infiniDevice_t::INFINI_DEVICE_KUNLUN, opType, kernel,      \
                    TOSTRING(_CAT(kernel, _KUNLUN)))
