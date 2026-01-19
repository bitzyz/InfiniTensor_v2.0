#include "core/runtime.h"
#include "operators/ElementWise.h"

namespace infini {

class ElementWiseOp : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *runtime) const override {
        auto op = as<ElementWiseObj>(_op);
        op->createOpDesc();
        auto type = op->getElemenwiseOpType();
        void *yData = (op->getOutput(0)->getRawDataPtr<void *>());
        void *const aData = (op->getInput(0)->getRawDataPtr<void *>());
        void *const bData = (op->getInput(1)->getRawDataPtr<void *>());
        size_t workspace_size = 0;
        if (type == OpType::Add) {
            CHECK_INFINI_ERROR(infiniopGetAddWorkspaceSize(
                (infiniopAddDescriptor_t)op->getInfiniOpDesc(),
                &workspace_size));
            void *workspace = runtime->getWorkspace(workspace_size);
            CHECK_INFINI_ERROR(
                infiniopAdd((infiniopAddDescriptor_t)op->getInfiniOpDesc(),
                            workspace, workspace_size, yData, aData, bData,
                            runtime->getCurrentThreadContext()->stream));
        } else if (type == OpType::Mul) {
            CHECK_INFINI_ERROR(infiniopGetMulWorkspaceSize(
                (infiniopMulDescriptor_t)op->getInfiniOpDesc(),
                &workspace_size));
            void *workspace = runtime->getWorkspace(workspace_size);
            CHECK_INFINI_ERROR(
                infiniopMul((infiniopMulDescriptor_t)op->getInfiniOpDesc(),
                            workspace, workspace_size, yData, aData, bData,
                            runtime->getCurrentThreadContext()->stream));
        } else if (type == OpType::Sub) {
            CHECK_INFINI_ERROR(infiniopGetSubWorkspaceSize(
                (infiniopSubDescriptor_t)op->getInfiniOpDesc(),
                &workspace_size));
            void *workspace = runtime->getWorkspace(workspace_size);
            CHECK_INFINI_ERROR(
                infiniopSub((infiniopSubDescriptor_t)op->getInfiniOpDesc(),
                            workspace, workspace_size, yData, aData, bData,
                            runtime->getCurrentThreadContext()->stream));
        } else {
            IT_TODO_HALT_MSG("ElemenWise operator not supported");
        }
    }
};

REGISTER_KERNEL_ALL_DEVICES(OpType::Add, ElementWiseOp);
REGISTER_KERNEL_ALL_DEVICES(OpType::Mul, ElementWiseOp);
REGISTER_KERNEL_ALL_DEVICES(OpType::Sub, ElementWiseOp);
} // namespace infini
