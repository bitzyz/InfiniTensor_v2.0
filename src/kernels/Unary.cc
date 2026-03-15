#include "operators/Unary.h"
#include "core/runtime.h"

namespace infini {

class UnaryOp : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *runtime) const override {
        auto op = as<UnaryObj>(_op);
        if (op->getInfiniOpDesc() == nullptr) {
            op->createOpDesc();
        }
        auto opType = op->getUnaryOpType();
        void *yData = op->getOutput(0)->getRawDataPtr<void *>();
        void *const xData = op->getInput(0)->getRawDataPtr<void *>();
        size_t ws_size = 0;
        void *ws = nullptr;

        if (opType == OpType::Relu) {
            CHECK_INFINI_ERROR(infiniopGetReluWorkspaceSize(
                (infiniopReluDescriptor_t)op->getInfiniOpDesc(), &ws_size));
            ws = runtime->getWorkspace(ws_size);
            CHECK_INFINI_ERROR(infiniopRelu(
                (infiniopReluDescriptor_t)op->getInfiniOpDesc(), ws, ws_size,
                yData, xData, runtime->getCurrentThreadContext()->stream));
        } else if (opType == OpType::Sigmoid) {
            CHECK_INFINI_ERROR(infiniopGetSigmoidWorkspaceSize(
                (infiniopSigmoidDescriptor_t)op->getInfiniOpDesc(), &ws_size));
            ws = runtime->getWorkspace(ws_size);
            CHECK_INFINI_ERROR(infiniopSigmoid(
                (infiniopSigmoidDescriptor_t)op->getInfiniOpDesc(), ws, ws_size,
                yData, xData, runtime->getCurrentThreadContext()->stream));
        } else if (opType == OpType::Silu) {
            CHECK_INFINI_ERROR(infiniopGetSiluWorkspaceSize(
                (infiniopSiluDescriptor_t)op->getInfiniOpDesc(), &ws_size));
            ws = runtime->getWorkspace(ws_size);
            CHECK_INFINI_ERROR(infiniopSilu(
                (infiniopSiluDescriptor_t)op->getInfiniOpDesc(), ws, ws_size,
                yData, xData, runtime->getCurrentThreadContext()->stream));
        } else if (opType == OpType::Gelu) {
            CHECK_INFINI_ERROR(infiniopGetGeluWorkspaceSize(
                (infiniopGeluDescriptor_t)op->getInfiniOpDesc(), &ws_size));
            ws = runtime->getWorkspace(ws_size);
            CHECK_INFINI_ERROR(infiniopGelu(
                (infiniopGeluDescriptor_t)op->getInfiniOpDesc(), ws, ws_size,
                yData, xData, runtime->getCurrentThreadContext()->stream));
        } else if (opType == OpType::Softplus) {
            CHECK_INFINI_ERROR(infiniopGetSoftplusWorkspaceSize(
                (infiniopSoftplusDescriptor_t)op->getInfiniOpDesc(), &ws_size));
            ws = runtime->getWorkspace(ws_size);
            CHECK_INFINI_ERROR(infiniopSoftplus(
                (infiniopSoftplusDescriptor_t)op->getInfiniOpDesc(), ws,
                ws_size, yData, xData,
                runtime->getCurrentThreadContext()->stream));
        } else if (opType == OpType::Tanh) {
            CHECK_INFINI_ERROR(infiniopGetTanhWorkspaceSize(
                (infiniopTanhDescriptor_t)op->getInfiniOpDesc(), &ws_size));
            ws = runtime->getWorkspace(ws_size);
            CHECK_INFINI_ERROR(infiniopTanh(
                (infiniopTanhDescriptor_t)op->getInfiniOpDesc(), ws, ws_size,
                yData, xData, runtime->getCurrentThreadContext()->stream));
        } else {
            IT_TODO_HALT_MSG("Unary operator not supported");
        }
    }
};

REGISTER_KERNEL_ALL_DEVICES(OpType::Relu, UnaryOp);
REGISTER_KERNEL_ALL_DEVICES(OpType::Sigmoid, UnaryOp);
REGISTER_KERNEL_ALL_DEVICES(OpType::Silu, UnaryOp);
REGISTER_KERNEL_ALL_DEVICES(OpType::Gelu, UnaryOp);
REGISTER_KERNEL_ALL_DEVICES(OpType::Softplus, UnaryOp);
REGISTER_KERNEL_ALL_DEVICES(OpType::Tanh, UnaryOp);

} // namespace infini
