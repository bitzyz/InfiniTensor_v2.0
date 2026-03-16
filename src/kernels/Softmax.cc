#include "operators/Softmax.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include <infiniop/ops/softmax.h>
#include <infiniop/ops/logsoftmax.h>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace infini {

class SoftmaxKernel : public Kernel {
    void compute(const Operator &_op, const RuntimeObj *_context) const override {
        auto op = as<SoftmaxObj>(_op);
        void *x = op->getInputs()[0]->getRawDataPtr<void *>();
        void *y = op->getOutput(0)->getRawDataPtr<void *>();
        
        try {
            op->createOpDesc();
        } catch (const std::exception &e) {
             if (_context->isCpu()) {
                 computeCpu(op.get(), (float*)x, (float*)y);
                 return;
             }
             throw;
        }
        auto desc = (infiniopSoftmaxDescriptor_t)op->getInfiniOpDesc();

        void *workspace = nullptr;
        size_t workspace_size = 0;
        infiniStatus_t status = infiniopGetSoftmaxWorkspaceSize(desc, &workspace_size);
        
        if (status == INFINI_STATUS_SUCCESS) {
            if (workspace_size > 0) {
                workspace = _context->getWorkspace(workspace_size);
            }
            CHECK_INFINI_ERROR(infiniopSoftmax(desc, workspace, workspace_size, y, x, _context->getCurrentThreadContext()->stream));
        } else {
            if (_context->isCpu()) {
                computeCpu(op.get(), (float*)x, (float*)y);
            } else {
                CHECK_INFINI_ERROR(status);
            }
        }
    }

    void computeCpu(const SoftmaxObj* op, const float* x, float* y) const {
        // Naive Softmax implementation for CPU
        // We need to handle arbitrary axis.
        // Flatten into [outer, axis, inner]
        auto shape = op->getInputs()[0]->getShape()->getConstantValue();
        int axis = op->getAxis();
        if (axis < 0) axis += shape.size();
        
        size_t outer = 1;
        for (int i = 0; i < axis; ++i) outer *= shape[i];
        size_t dim = shape[axis];
        size_t inner = 1;
        for (size_t i = axis + 1; i < shape.size(); ++i) inner *= shape[i];
        
        for (size_t o = 0; o < outer; ++o) {
            for (size_t i = 0; i < inner; ++i) {
                // Find max
                float max_val = -std::numeric_limits<float>::infinity();
                for (size_t d = 0; d < dim; ++d) {
                    size_t idx = o * dim * inner + d * inner + i;
                    max_val = std::max(max_val, x[idx]);
                }
                
                // Compute exp sum
                float sum = 0;
                for (size_t d = 0; d < dim; ++d) {
                    size_t idx = o * dim * inner + d * inner + i;
                    y[idx] = std::exp(x[idx] - max_val);
                    sum += y[idx];
                }
                
                // Normalize
                for (size_t d = 0; d < dim; ++d) {
                    size_t idx = o * dim * inner + d * inner + i;
                    y[idx] /= sum;
                }
            }
        }
    }
};

class LogSoftmaxKernel : public Kernel {
    void compute(const Operator &_op, const RuntimeObj *_context) const override {
        auto op = as<LogSoftmaxObj>(_op);
        void *x = op->getInputs()[0]->getRawDataPtr<void *>();
        void *y = op->getOutput(0)->getRawDataPtr<void *>();

        try {
            op->createOpDesc();
        } catch (const std::exception &e) {
             if (_context->isCpu()) {
                 computeCpu(op.get(), (float*)x, (float*)y);
                 return;
             }
             throw;
        }
        auto desc = (infiniopLogSoftmaxDescriptor_t)op->getInfiniOpDesc();

        void *workspace = nullptr;
        size_t workspace_size = 0;
        infiniStatus_t status = infiniopGetLogSoftmaxWorkspaceSize(desc, &workspace_size);
        
        if (status == INFINI_STATUS_SUCCESS) {
            if (workspace_size > 0) {
                workspace = _context->getWorkspace(workspace_size);
            }
            CHECK_INFINI_ERROR(infiniopLogSoftmax(desc, workspace, workspace_size, y, x, _context->getCurrentThreadContext()->stream));
        } else {
             // Fallback for CPU if InfiniCore doesn't support it or fails
             if (_context->isCpu()) {
                 computeCpu(op.get(), (float*)x, (float*)y);
             } else {
                 CHECK_INFINI_ERROR(status);
             }
         }
     }
     
     void computeCpu(const LogSoftmaxObj* op, const float* x, float* y) const {
         // Naive LogSoftmax
         auto shape = op->getInputs()[0]->getShape()->getConstantValue();
         int axis = op->getAxis();
         if (axis < 0) axis += shape.size();
         
         size_t outer = 1;
         for (int i = 0; i < axis; ++i) outer *= shape[i];
         size_t dim = shape[axis];
         size_t inner = 1;
         for (size_t i = axis + 1; i < shape.size(); ++i) inner *= shape[i];
        
        for (size_t o = 0; o < outer; ++o) {
            for (size_t i = 0; i < inner; ++i) {
                float max_val = -std::numeric_limits<float>::infinity();
                for (size_t d = 0; d < dim; ++d) {
                    size_t idx = o * dim * inner + d * inner + i;
                    max_val = std::max(max_val, x[idx]);
                }
                
                float sum = 0;
                for (size_t d = 0; d < dim; ++d) {
                    size_t idx = o * dim * inner + d * inner + i;
                    sum += std::exp(x[idx] - max_val);
                }
                float log_sum = std::log(sum);
                
                for (size_t d = 0; d < dim; ++d) {
                    size_t idx = o * dim * inner + d * inner + i;
                    y[idx] = x[idx] - max_val - log_sum;
                }
            }
        }
    }
};

REGISTER_KERNEL_ALL_DEVICES(OpType::Softmax, SoftmaxKernel);
REGISTER_KERNEL_ALL_DEVICES(OpType::LogSoftmax, LogSoftmaxKernel);

} // namespace infini
