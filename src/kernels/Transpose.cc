#include "operators/Transpose.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include <infiniop/ops/rearrange.h>

namespace infini {

class TransposeKernel : public Kernel {
    void compute(const Operator &_op, const RuntimeObj *_context) const override {
        auto op = as<TransposeObj>(_op);
        op->createOpDesc();
        auto desc = (infiniopRearrangeDescriptor_t)op->getInfiniOpDesc();

        void *x = op->getInputs()[0]->getRawDataPtr<void *>();
        void *y = op->getOutput(0)->getRawDataPtr<void *>();
        
        CHECK_INFINI_ERROR(infiniopRearrange(desc, y, x, _context->getCurrentThreadContext()->stream));
    }
};

REGISTER_KERNEL_ALL_DEVICES(OpType::Transpose, TransposeKernel);

} // namespace infini
