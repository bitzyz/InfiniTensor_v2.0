#include "core/runtime.h"
#include "operators/Clip.h"

namespace infini {

class ClipOp : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *runtime) const override {
        auto op = as<ClipObj>(_op);
        op->createOpDesc();
        void *yData = (op->getOutput(0)->getRawDataPtr<void *>());
        void *const aData = (op->getInput(0)->getRawDataPtr<void *>());
        void *const min_val = (op->getInput(1)->getRawDataPtr<void *>());
        void *const max_val = (op->getInput(2)->getRawDataPtr<void *>());
        size_t workspace_size = 0;
        CHECK_INFINI_ERROR(infiniopGetClipWorkspaceSize(
            (infiniopClipDescriptor_t)op->getInfiniOpDesc(), &workspace_size));
        void *workspace = runtime->getWorkspace(workspace_size);
        CHECK_INFINI_ERROR(infiniopClip(
            (infiniopClipDescriptor_t)op->getInfiniOpDesc(), workspace,
            workspace_size, yData, aData, min_val, max_val,
            runtime->getCurrentThreadContext()->stream));
    }
};
// 执行注册机制，将算子和对应的计算方式进行绑定并添加到对应的注册表中
REGISTER_KERNEL_ALL_DEVICES(OpType::Clip, ClipOp);
} // namespace infini   