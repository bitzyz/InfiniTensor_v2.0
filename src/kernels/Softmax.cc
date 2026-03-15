#include "operators/Softmax.h"
#include "core/runtime.h"

namespace infini {

class SoftmaxOp : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *runtime) const override {
        auto op = as<SoftmaxObj>(_op);
        op->createOpDesc();
        void *yData = (op->getOutput(0)->getRawDataPtr<void *>());
        void *const xData = (op->getInput(0)->getRawDataPtr<void *>());

        size_t workspace_size = 0;
        CHECK_INFINI_ERROR(infiniopGetSoftmaxWorkspaceSize(
            (infiniopSoftmaxDescriptor_t)op->getInfiniOpDesc(),
            &workspace_size));
        void *workspace = runtime->getWorkspace(workspace_size);
        CHECK_INFINI_ERROR(
            infiniopSoftmax((infiniopSoftmaxDescriptor_t)op->getInfiniOpDesc(),
                            workspace, workspace_size, yData, xData,
                            runtime->getCurrentThreadContext()->stream));
    }
};

REGISTER_KERNEL_ALL_DEVICES(OpType::Softmax, SoftmaxOp);
} // namespace infini
