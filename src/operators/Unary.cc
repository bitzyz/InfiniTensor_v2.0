#include "operators/Unary.h"
#include "core/runtime.h"
#include <infiniop/ops/relu.h>
#include <infiniop/ops/sigmoid.h>
#include <infiniop/ops/tanh.h>
#include <infiniop/ops/gelu.h>
#include <infiniop/ops/silu.h>
#include <infiniop/ops/softplus.h>

namespace infini {

UnaryObj::UnaryObj(OpType type, GraphObj *graph, Tensor input, Tensor output)
    : OperatorObj(type, {input}, {output}) {
    IT_ASSERT(checkValid(graph));
}

std::optional<std::vector<ShapeExpr>> UnaryObj::inferShape() {
    auto inputShape = inputs[0]->getShape();
    std::vector<Expr> shape_vec;
    for (size_t i = 0; i < inputShape->size(); ++i) {
        shape_vec.push_back((*inputShape)[i]);
    }
    ShapeExpr ret = make_ref<ShapeExprObj>(ShapeExprObj(shape_vec));
    return {{ret}};
}

std::vector<DataType> UnaryObj::inferDataType() const {
    return {inputs[0]->getDataType()};
}

std::string UnaryObj::toString() const {
    std::ostringstream os;
    os << OpType(type).toString() << "[" << getGuid() << "]";
    os << "(";
    os << vecToString(inputs[0]->getShape()->getConstantValue()) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid();
    os << ")";
    return os.str();
}

void UnaryObj::createOpDesc() {
    auto yShape = outputs[0]->getShape();
    auto xShape = inputs[0]->getShape();
    
    auto yStride = outputs[0]->getStride();
    auto xStride = inputs[0]->getStride();
    
    infiniopTensorDescriptor_t yTensor, xTensor;
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &yTensor, yShape->size(), yShape->getConstantValue().data(),
        yStride->getConstantValue().data(), outputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &xTensor, xShape->size(), xShape->getConstantValue().data(),
        xStride->getConstantValue().data(), inputs[0]->getDataType().getType()));

    infiniopHandle_t handle = nullptr;
    CHECK_INFINI_ERROR(infiniopCreateHandle(&handle));

    switch (type.underlying()) {
    case OpType::Relu:
        CHECK_INFINI_ERROR(infiniopCreateReluDescriptor(
            handle, (infiniopReluDescriptor_t *)&infiniOpDesc, yTensor, xTensor));
        break;
    case OpType::Sigmoid:
        CHECK_INFINI_ERROR(infiniopCreateSigmoidDescriptor(
            handle, (infiniopSigmoidDescriptor_t *)&infiniOpDesc, yTensor, xTensor));
        break;
    case OpType::Tanh:
        CHECK_INFINI_ERROR(infiniopCreateTanhDescriptor(
            handle, (infiniopTanhDescriptor_t *)&infiniOpDesc, yTensor, xTensor));
        break;
    case OpType::Gelu:
        CHECK_INFINI_ERROR(infiniopCreateGeluDescriptor(
            handle, (infiniopGeluDescriptor_t *)&infiniOpDesc, yTensor, xTensor));
        break;
    case OpType::Silu:
        CHECK_INFINI_ERROR(infiniopCreateSiluDescriptor(
            handle, (infiniopSiluDescriptor_t *)&infiniOpDesc, yTensor, xTensor));
        break;
    case OpType::Softplus:
        CHECK_INFINI_ERROR(infiniopCreateSoftplusDescriptor(
            handle, (infiniopSoftplusDescriptor_t *)&infiniOpDesc, yTensor, xTensor));
        break;
    default:
        // IT_TODO_HALT() is not available?
        // Let's use standard assert or skip.
        // Or include correct header.
        // It should be in common.h or exception.h
        // Let's just throw or abort.
        abort();
    }

    CHECK_INFINI_ERROR(infiniopDestroyHandle(handle));

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(xTensor));
}

UnaryObj::~UnaryObj() {
    if (infiniOpDesc) {
        switch (type.underlying()) {
        case OpType::Relu:
            infiniopDestroyReluDescriptor((infiniopReluDescriptor_t)infiniOpDesc);
            break;
        case OpType::Sigmoid:
            infiniopDestroySigmoidDescriptor((infiniopSigmoidDescriptor_t)infiniOpDesc);
            break;
        case OpType::Tanh:
            infiniopDestroyTanhDescriptor((infiniopTanhDescriptor_t)infiniOpDesc);
            break;
        case OpType::Gelu:
            infiniopDestroyGeluDescriptor((infiniopGeluDescriptor_t)infiniOpDesc);
            break;
        case OpType::Silu:
            infiniopDestroySiluDescriptor((infiniopSiluDescriptor_t)infiniOpDesc);
            break;
        case OpType::Softplus:
            infiniopDestroySoftplusDescriptor((infiniopSoftplusDescriptor_t)infiniOpDesc);
            break;
        default:
            break;
        }
    }
}

} // namespace infini
