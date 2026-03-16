#include "operators/RMSNorm.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include <infiniop/ops/rms_norm.h>

namespace infini {

class RMSNormKernel : public Kernel {
    void compute(const Operator &_op, const RuntimeObj *_context) const override {
        auto op = as<RMSNormObj>(_op);
        void *x = op->getInputs()[0]->getRawDataPtr<void *>();
        void *w = op->getInputs()[1]->getRawDataPtr<void *>();
        void *y = op->getOutput(0)->getRawDataPtr<void *>();
        
        try {
            op->createOpDesc();
        } catch (const std::exception &e) {
             if (_context->isCpu()) {
                 computeCpu(op.get(), (float*)x, (float*)w, (float*)y);
                 return;
             }
             throw;
        }
        auto desc = (infiniopRMSNormDescriptor_t)op->getInfiniOpDesc();

        void *workspace = nullptr;
        size_t workspace_size = 0;
        infiniStatus_t status = infiniopGetRMSNormWorkspaceSize(desc, &workspace_size);
        if (status == INFINI_STATUS_SUCCESS) {
            if (workspace_size > 0) {
                workspace = _context->getWorkspace(workspace_size);
            }
            CHECK_INFINI_ERROR(infiniopRMSNorm(desc, workspace, workspace_size, y, x, w, _context->getCurrentThreadContext()->stream));
        } else {
            if (_context->isCpu()) {
                computeCpu(op.get(), (float*)x, (float*)w, (float*)y);
            } else {
                CHECK_INFINI_ERROR(status);
            }
        }
    }

    void computeCpu(const RMSNormObj* op, const float* x, const float* w, float* y) const {
        auto input = op->getInput(0);
        auto shape = input->getShape()->getConstantValue();
        size_t dim = shape.back(); 
        size_t total = 1;
        for(auto s : shape) total *= s;
        size_t outer = total / dim;
        float eps = op->getEps();
        
        for (size_t i = 0; i < outer; ++i) {
            float sum_sq = 0;
            for (size_t d = 0; d < dim; ++d) {
                float val = x[i * dim + d];
                sum_sq += val * val;
            }
            float rms = std::sqrt(sum_sq / dim + eps);
            float inv_rms = 1.0f / rms;
            
            for (size_t d = 0; d < dim; ++d) {
                y[i * dim + d] = x[i * dim + d] * inv_rms * w[d];
            }
        }
    }
};

REGISTER_KERNEL_ALL_DEVICES(OpType::RMSNorm, RMSNormKernel);

} // namespace infini
