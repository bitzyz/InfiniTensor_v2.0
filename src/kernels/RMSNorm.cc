#include "operators/RMSNorm.h"
#include "core/runtime.h"

namespace infini
{

    class RMSNormOp : public Kernel
    {
        void compute(const Operator &_op,
                     const RuntimeObj *runtime) const override
        {
            auto op = as<RMSNormObj>(_op);
            op->createOpDesc();
            void *yData = (op->getOutput(0)->getRawDataPtr<void *>());
            void *const xData = (op->getInput(0)->getRawDataPtr<void *>());
            void *const wData = (op->getInput(1)->getRawDataPtr<void *>());
            size_t workspace_size = 0;
            CHECK_INFINI_ERROR(infiniopGetRMSNormWorkspaceSize(
                (infiniopRMSNormDescriptor_t)op->getInfiniOpDesc(), &workspace_size));
            void *workspace = runtime->getWorkspace(workspace_size);
            CHECK_INFINI_ERROR(infiniopRMSNorm(
                (infiniopRMSNormDescriptor_t)op->getInfiniOpDesc(), workspace,
                workspace_size, yData, xData, wData,
                runtime->getCurrentThreadContext()->stream));
        }
    };

    REGISTER_KERNEL_ALL_DEVICES(OpType::RMSNorm, RMSNormOp);
} // namespace infini
