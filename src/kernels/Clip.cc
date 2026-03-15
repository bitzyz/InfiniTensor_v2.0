#include "operators/Clip.h"
#include "core/runtime.h"

namespace infini {

class ClipOp : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *runtime) const override {
        auto op = as<ClipObj>(_op);
        op->createOpDesc();
        void *yData = (op->getOutput(0)->getRawDataPtr<void *>());
        void *const inData = (op->getInput(0)->getRawDataPtr<void *>());
        void *const minData = (op->getInput(1)->getRawDataPtr<void *>());
        void *const maxData = (op->getInput(2)->getRawDataPtr<void *>());

        size_t workspace_size = 0;
        CHECK_INFINI_ERROR(infiniopGetClipWorkspaceSize(
            (infiniopClipDescriptor_t)op->getInfiniOpDesc(), &workspace_size));
        void *workspace = runtime->getWorkspace(workspace_size);
        CHECK_INFINI_ERROR(
            infiniopClip((infiniopClipDescriptor_t)op->getInfiniOpDesc(),
                         workspace, workspace_size, yData, inData, minData,
                         maxData, runtime->getCurrentThreadContext()->stream));
    }
};

REGISTER_KERNEL_ALL_DEVICES(OpType::Clip, ClipOp);
} // namespace infini
