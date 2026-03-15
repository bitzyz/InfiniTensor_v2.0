#include "operators/LayerNorm.h"
#include "core/runtime.h"

namespace infini {

static int normalizeAxis(int axis, size_t rank) {
    int norm = axis;
    if (norm < 0)
        norm += static_cast<int>(rank);
    IT_ASSERT(norm >= 0 && norm < static_cast<int>(rank),
              "axis out of range in LayerNorm kernel");
    return norm;
}

class LayerNormOp : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *runtime) const override {
        auto op = as<LayerNormObj>(_op);
        if (op->getInfiniOpDesc() == nullptr) {
            op->createOpDesc();
        }

        void *yData = op->getOutput(0)->getRawDataPtr<void *>();
        void *const xData = op->getInput(0)->getRawDataPtr<void *>();
        void *const wData = op->getInput(1)->getRawDataPtr<void *>();
        void *const bData = op->getInput(2)->getRawDataPtr<void *>();

        // Compute scratch buffer size for mean and std_dev tensors.
        // Their shape is input.shape[:axis] — the batch prefix.
        auto xShape = op->getInput(0)->getShape()->getConstantValue();
        int axis = normalizeAxis(op->getAxis(), xShape.size());
        IT_ASSERT(xShape.size() == 3,
                  "LayerNorm kernel only supports 3D input");
        IT_ASSERT(axis == 2, "LayerNorm kernel only supports axis=2");
        size_t prefixElems = 1;
        for (int i = 0; i < axis; ++i)
            prefixElems *= (size_t)xShape[i];
        size_t inputElems = prefixElems * (size_t)xShape.back();
        size_t elemBytes = op->getInput(0)->getDataType().getSize();
        size_t meanBytes = inputElems * elemBytes;
        size_t stdBytes = prefixElems * elemBytes;

        void *meanScratch = runtime->allocDevice(meanBytes);
        void *stdScratch = runtime->allocDevice(stdBytes);

        size_t ws_size = 0;
        CHECK_INFINI_ERROR(infiniopGetLayerNormWorkspaceSize(
            (infiniopLayerNormDescriptor_t)op->getInfiniOpDesc(), &ws_size));
        void *ws = runtime->getWorkspace(ws_size);

        CHECK_INFINI_ERROR(infiniopLayerNorm(
            (infiniopLayerNormDescriptor_t)op->getInfiniOpDesc(), ws, ws_size,
            yData, meanScratch, stdScratch, xData, wData, bData,
            runtime->getCurrentThreadContext()->stream));

        runtime->deallocDevice(meanScratch);
        runtime->deallocDevice(stdScratch);
    }
};

REGISTER_KERNEL_ALL_DEVICES(OpType::LayerNorm, LayerNormOp);

} // namespace infini
