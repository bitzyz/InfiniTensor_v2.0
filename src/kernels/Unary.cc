#include "operators/Unary.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include <infiniop/ops/relu.h>
#include <infiniop/ops/sigmoid.h>
#include <infiniop/ops/tanh.h>
#include <infiniop/ops/gelu.h>
#include <infiniop/ops/silu.h>
#include <infiniop/ops/softplus.h>

namespace infini {

template <typename ObjType, typename DescType, typename CreateFunc>
class UnaryKernel : public Kernel {
  public:
    using FuncType = CreateFunc;
    
    UnaryKernel(FuncType func) : func(func) {}

    void compute(const Operator &_op, const RuntimeObj *_context) const override {
        auto op = as<ObjType>(_op);
        op->createOpDesc();
        auto desc = (DescType)op->getInfiniOpDesc();

        void *x = op->getInputs()[0]->template getRawDataPtr<void *>();
        void *y = op->getOutput(0)->template getRawDataPtr<void *>();
        
        CHECK_INFINI_ERROR(func(desc, y, x, _context->getCurrentThreadContext()->stream));
    }

  private:
    FuncType func;
};

// Helper macro to register unary kernels
#define REGISTER_UNARY_KERNEL(OpTypeEnum, ObjType, DescType, ExecFunc) \
    class OpTypeEnum##Kernel : public Kernel { \
        void compute(const Operator &_op, const RuntimeObj *_context) const override { \
            auto op = as<ObjType>(_op); \
            op->createOpDesc(); \
            auto desc = (DescType)op->getInfiniOpDesc(); \
            void *x = op->getInputs()[0]->getRawDataPtr<void *>(); \
            void *y = op->getOutput(0)->getRawDataPtr<void *>(); \
            CHECK_INFINI_ERROR(ExecFunc(desc, y, x, _context->getCurrentThreadContext()->stream)); \
        } \
    }; \
    REGISTER_KERNEL_ALL_DEVICES(OpTypeEnum, OpTypeEnum##Kernel)

// We need to use concrete names for classes to avoid macro expansion issues with ##
// OpType::Relu##Kernel -> OpType::ReluKernel which is invalid syntax if OpType is a scope.
// Helper macro to register unary kernels
#define REGISTER_UNARY_KERNEL_NAMED(OpName, OpTypeEnum, ObjType, DescType, ExecFunc) \
    class OpName##Kernel : public Kernel { \
        void compute(const Operator &_op, const RuntimeObj *_context) const override { \
            auto op = as<ObjType>(_op); \
            op->createOpDesc(); \
            auto desc = (DescType)op->getInfiniOpDesc(); \
            void *x = op->getInputs()[0]->getRawDataPtr<void *>(); \
            void *y = op->getOutput(0)->getRawDataPtr<void *>(); \
            size_t workspace_size = 0; \
            infiniopGet##OpName##WorkspaceSize(desc, &workspace_size); \
            void *workspace = nullptr; \
            if (workspace_size > 0) { \
                workspace = _context->getWorkspace(workspace_size); \
            } \
            CHECK_INFINI_ERROR(ExecFunc(desc, workspace, workspace_size, y, x, _context->getCurrentThreadContext()->stream)); \
        } \
    }; \
    REGISTER_KERNEL_ALL_DEVICES(OpTypeEnum, OpName##Kernel)

#ifdef USE_MOORE
    // MOORE platform does not support Silu yet
    REGISTER_UNARY_KERNEL_NAMED(Relu, OpType::Relu, ReluObj, infiniopReluDescriptor_t, infiniopRelu);
    REGISTER_UNARY_KERNEL_NAMED(Sigmoid, OpType::Sigmoid, SigmoidObj, infiniopSigmoidDescriptor_t, infiniopSigmoid);
    REGISTER_UNARY_KERNEL_NAMED(Tanh, OpType::Tanh, TanhObj, infiniopTanhDescriptor_t, infiniopTanh);
    REGISTER_UNARY_KERNEL_NAMED(Gelu, OpType::Gelu, GeluObj, infiniopGeluDescriptor_t, infiniopGelu);
    // REGISTER_UNARY_KERNEL_NAMED(Silu, OpType::Silu, SiluObj, infiniopSiluDescriptor_t, infiniopSilu);
#else
    REGISTER_UNARY_KERNEL_NAMED(Relu, OpType::Relu, ReluObj, infiniopReluDescriptor_t, infiniopRelu);
    REGISTER_UNARY_KERNEL_NAMED(Sigmoid, OpType::Sigmoid, SigmoidObj, infiniopSigmoidDescriptor_t, infiniopSigmoid);
    REGISTER_UNARY_KERNEL_NAMED(Tanh, OpType::Tanh, TanhObj, infiniopTanhDescriptor_t, infiniopTanh);
    REGISTER_UNARY_KERNEL_NAMED(Gelu, OpType::Gelu, GeluObj, infiniopGeluDescriptor_t, infiniopGelu);
    REGISTER_UNARY_KERNEL_NAMED(Silu, OpType::Silu, SiluObj, infiniopSiluDescriptor_t, infiniopSilu);
#endif
// Softplus requires workspace?
// Based on error: infiniopSoftplus(desc, workspace, size, y, x, stream)
// Let's implement SoftplusKernel correctly.
class SoftplusKernel : public Kernel {
    void compute(const Operator &_op, const RuntimeObj *_context) const override {
        auto op = as<SoftplusObj>(_op);
        op->createOpDesc();
        auto desc = (infiniopSoftplusDescriptor_t)op->getInfiniOpDesc();
        
        void *workspace = nullptr;
        size_t workspace_size = 0;
        CHECK_INFINI_ERROR(infiniopGetSoftplusWorkspaceSize(desc, &workspace_size));
        if (workspace_size > 0) {
            workspace = _context->getWorkspace(workspace_size);
        }

        void *x = op->getInputs()[0]->getRawDataPtr<void *>();
        void *y = op->getOutput(0)->getRawDataPtr<void *>();
        CHECK_INFINI_ERROR(infiniopSoftplus(desc, workspace, workspace_size, y, x, _context->getCurrentThreadContext()->stream));
    }
};
REGISTER_KERNEL_ALL_DEVICES(OpType::Softplus, SoftplusKernel);

} // namespace infini
