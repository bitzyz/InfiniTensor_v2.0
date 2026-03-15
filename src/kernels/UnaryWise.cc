#include "core/runtime.h"
#include "operators/UnaryOp.h"

namespace infini {

class UnaryOpOp : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *runtime) const override {
        auto op = as<UnaryWiseObj>(_op);
        op->createOpDesc();
        auto type = op->getUnaryWiseOpType();
        void *yData = (op->getOutput(0)->getRawDataPtr<void *>());
        void *const aData = (op->getInput(0)->getRawDataPtr<void *>());
        size_t workspace_size = 0;
        if (type == OpType::Relu) {
            CHECK_INFINI_ERROR(infiniopGetReluWorkspaceSize(
                (infiniopReluDescriptor_t)op->getInfiniOpDesc(),
                &workspace_size));
            void *workspace = runtime->getWorkspace(workspace_size);
            CHECK_INFINI_ERROR(
                infiniopRelu((infiniopReluDescriptor_t)op->getInfiniOpDesc(),
                             workspace, workspace_size, yData, aData,
                             runtime->getCurrentThreadContext()->stream));
        } else if (type == OpType::Sigmoid) {
            CHECK_INFINI_ERROR(infiniopGetSigmoidWorkspaceSize(
                (infiniopSigmoidDescriptor_t)op->getInfiniOpDesc(),
                &workspace_size));
            void *workspace = runtime->getWorkspace(workspace_size);
            CHECK_INFINI_ERROR(infiniopSigmoid(
                (infiniopSigmoidDescriptor_t)op->getInfiniOpDesc(), workspace,
                workspace_size, yData, aData,
                runtime->getCurrentThreadContext()->stream));
        } else if (type == OpType::Silu) {
            CHECK_INFINI_ERROR(infiniopGetSiluWorkspaceSize(
                (infiniopSiluDescriptor_t)op->getInfiniOpDesc(),
                &workspace_size));
            void *workspace = runtime->getWorkspace(workspace_size);
            CHECK_INFINI_ERROR(
                infiniopSilu((infiniopSiluDescriptor_t)op->getInfiniOpDesc(),
                             workspace, workspace_size, yData, aData,
                             runtime->getCurrentThreadContext()->stream));
        } else if (type == OpType::Gelu) {
            CHECK_INFINI_ERROR(infiniopGetGeluWorkspaceSize(
                (infiniopGeluDescriptor_t)op->getInfiniOpDesc(),
                &workspace_size));
            void *workspace = runtime->getWorkspace(workspace_size);
            CHECK_INFINI_ERROR(
                infiniopGelu((infiniopGeluDescriptor_t)op->getInfiniOpDesc(),
                             workspace, workspace_size, yData, aData,
                             runtime->getCurrentThreadContext()->stream));
        } else if (type == OpType::Softplus) {
            CHECK_INFINI_ERROR(infiniopGetSoftplusWorkspaceSize(
                (infiniopSoftplusDescriptor_t)op->getInfiniOpDesc(),
                &workspace_size));
            void *workspace = runtime->getWorkspace(workspace_size);
            CHECK_INFINI_ERROR(infiniopSoftplus(
                (infiniopSoftplusDescriptor_t)op->getInfiniOpDesc(), workspace,
                workspace_size, yData, aData,
                runtime->getCurrentThreadContext()->stream));
        } else if (type == OpType::Tanh) {
            CHECK_INFINI_ERROR(infiniopGetTanhWorkspaceSize(
                (infiniopTanhDescriptor_t)op->getInfiniOpDesc(),
                &workspace_size));
            void *workspace = runtime->getWorkspace(workspace_size);
            CHECK_INFINI_ERROR(
                infiniopTanh((infiniopTanhDescriptor_t)op->getInfiniOpDesc(),
                             workspace, workspace_size, yData, aData,
                             runtime->getCurrentThreadContext()->stream));
        } else {
            IT_TODO_HALT_MSG("ElemenWise operator not supported");
        }
    }
};

REGISTER_KERNEL_ALL_DEVICES(OpType::Relu, UnaryOpOp);
REGISTER_KERNEL_ALL_DEVICES(OpType::Sigmoid, UnaryOpOp);
REGISTER_KERNEL_ALL_DEVICES(OpType::Silu, UnaryOpOp);
REGISTER_KERNEL_ALL_DEVICES(OpType::Gelu, UnaryOpOp);
REGISTER_KERNEL_ALL_DEVICES(OpType::Softplus, UnaryOpOp);
REGISTER_KERNEL_ALL_DEVICES(OpType::Tanh, UnaryOpOp);
} // namespace infini
