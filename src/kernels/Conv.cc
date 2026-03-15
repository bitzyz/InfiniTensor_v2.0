#include "operators/Conv.h"
#include "core/runtime.h"

namespace infini {

class ConvOp : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *runtime) const override {
        auto op = as<ConvObj>(_op);
        if (op->getInfiniOpDesc() == nullptr) {
            op->createOpDesc();
        }
        void *yData = op->getOutput(0)->getRawDataPtr<void *>();
        void *const xData = op->getInput(0)->getRawDataPtr<void *>();
        void *const wData = op->getInput(1)->getRawDataPtr<void *>();
        void *const biasData =
            op->hasBias() ? op->getInput(2)->getRawDataPtr<void *>() : nullptr;
        size_t ws_size = 0;
        CHECK_INFINI_ERROR(infiniopGetConvWorkspaceSize(
            (infiniopConvDescriptor_t)op->getInfiniOpDesc(), &ws_size));
        void *ws = runtime->getWorkspace(ws_size);
        CHECK_INFINI_ERROR(
            infiniopConv((infiniopConvDescriptor_t)op->getInfiniOpDesc(), ws,
                         ws_size, yData, xData, wData, biasData,
                         runtime->getCurrentThreadContext()->stream));
    }
};

REGISTER_KERNEL_ALL_DEVICES(OpType::Conv, ConvOp);

} // namespace infini
