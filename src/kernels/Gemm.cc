#include "operators/Gemm.h"
#include "core/runtime.h"

namespace infini {

class GemmOp : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *runtime) const override {
        auto op = as<GemmObj>(_op);
        op->createOpDesc();
        void *yData = (op->getOutput(0)->getRawDataPtr<void *>());
        void *const aData = (op->getInput(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInput(1)->getRawDataPtr<void *>());
        size_t workspace_size = 0;
        CHECK_INFINI_ERROR(infiniopGetGemmWorkspaceSize(
            (infiniopGemmDescriptor_t)op->getInfiniOpDesc(), &workspace_size));
        void *workspace = runtime->getWorkspace(workspace_size);
        CHECK_INFINI_ERROR(infiniopGemm(
            (infiniopGemmDescriptor_t)op->getInfiniOpDesc(), workspace,
            workspace_size, yData, aData, bData, op->getAlpha(), op->getBeta(),
            runtime->getCurrentThreadContext()->stream));
    }
};

REGISTER_KERNEL_ALL_DEVICES(OpType::Gemm, GemmOp);
} // namespace infini
