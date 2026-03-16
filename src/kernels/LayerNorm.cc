#include "operators/LayerNorm.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include <infiniop/ops/layer_norm.h>

namespace infini {
class LayerNormKernel : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<LayerNormObj>(_op);
        op->createOpDesc();
        auto desc = (infiniopLayerNormDescriptor_t)op->getInfiniOpDesc();
        
        void *workspace = nullptr;
        size_t workspace_size = 0;
        CHECK_INFINI_ERROR(infiniopGetLayerNormWorkspaceSize(desc, &workspace_size));

        size_t std_size = op->getInputs()[0]->getTotalBytes();
        size_t std_dev_size = std_size / op->getInputs()[0]->getShape()->getConstantValue().back();
        
        size_t total_workspace = workspace_size + std_size + std_dev_size;
        void *base_ptr = total_workspace > 0 ? _context->getWorkspace(total_workspace) : nullptr;

        workspace = base_ptr;
        void *std_ptr = base_ptr ? (char*)base_ptr + workspace_size : nullptr;
        void *std_dev_ptr = base_ptr ? (char*)base_ptr + workspace_size + std_size : nullptr;

        void *x = op->getInputs()[0]->getRawDataPtr<void *>();
        void *w = nullptr;
        if (op->getInputs().size() > 1 && op->getInputs()[1]) {
            w = op->getInputs()[1]->getRawDataPtr<void *>();
        }
        void *bias = nullptr;
        if (op->getInputs().size() > 2 && op->getInputs()[2]) {
            bias = op->getInputs()[2]->getRawDataPtr<void *>();
        }
        void *y = op->getOutput(0)->getRawDataPtr<void *>();

        CHECK_INFINI_ERROR(infiniopLayerNorm(desc, workspace, workspace_size, y, std_ptr, std_dev_ptr, x, w, bias, _context->getCurrentThreadContext()->stream));
    }
};

REGISTER_KERNEL_ALL_DEVICES(OpType::LayerNorm, LayerNormKernel);

} // namespace infini
