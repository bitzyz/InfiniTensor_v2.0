#include "operators/Conv.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include <infiniop/ops/conv.h>

namespace infini {
class ConvKernel : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *_context) const override {
        auto op = as<ConvObj>(_op);
        op->createOpDesc();
        auto desc = (infiniopConvDescriptor_t)op->getInfiniOpDesc();
        
        void *workspace = nullptr;
        size_t workspace_size = 0;
        CHECK_INFINI_ERROR(infiniopGetConvWorkspaceSize(desc, &workspace_size));
        if (workspace_size > 0) {
            workspace = _context->getWorkspace(workspace_size);
        }

        void *x = op->getInputs()[0]->getRawDataPtr<void *>();
        void *w = op->getInputs()[1]->getRawDataPtr<void *>();
        void *bias = nullptr;
        if (op->getInputs().size() > 2) {
            bias = op->getInputs()[2]->getRawDataPtr<void *>();
        }
        void *y = op->getOutput(0)->getRawDataPtr<void *>();

        CHECK_INFINI_ERROR(infiniopConv(desc, workspace, workspace_size, y, x, w, bias, _context->getCurrentThreadContext()->stream));
    }
};

REGISTER_KERNEL_ALL_DEVICES(OpType::Conv, ConvKernel);

} // namespace infini
