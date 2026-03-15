#include "operators/RmsNorm.h"
#include "core/runtime.h"

namespace infini {

class RmsNormOp : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *runtime) const override {
        auto op = as<RmsNormObj>(_op);
        op->createOpDesc();
        void *yData = (op->getOutput(0)->getRawDataPtr<void *>());
        void *const xData = (op->getInput(0)->getRawDataPtr<void *>());
        void *const weightData = (op->getInput(1)->getRawDataPtr<void *>());

        size_t workspace_size = 0;
        CHECK_INFINI_ERROR(infiniopGetRMSNormWorkspaceSize(
            (infiniopRMSNormDescriptor_t)op->getInfiniOpDesc(),
            &workspace_size));
        void *workspace = runtime->getWorkspace(workspace_size);
        CHECK_INFINI_ERROR(
            infiniopRMSNorm((infiniopRMSNormDescriptor_t)op->getInfiniOpDesc(),
                            workspace, workspace_size, yData, xData, weightData,
                            runtime->getCurrentThreadContext()->stream));
    }
};

REGISTER_KERNEL_ALL_DEVICES(OpType::RmsNorm, RmsNormOp);
} // namespace infini
