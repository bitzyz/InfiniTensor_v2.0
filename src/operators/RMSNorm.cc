#include "operators/RMSNorm.h"
#include "core/runtime.h"
#include <infiniop/ops/rms_norm.h>

namespace infini {

RMSNormObj::RMSNormObj(GraphObj *graph, Tensor input, Tensor weight, Tensor output, float eps)
    : OperatorObj(OpType::RMSNorm, {input, weight}, {output}), eps(eps) {
    IT_ASSERT(checkValid(graph));
}

std::optional<std::vector<ShapeExpr>> RMSNormObj::inferShape() {
    auto inputShape = inputs[0]->getShape();
    std::vector<Expr> shape_vec;
    for (size_t i = 0; i < inputShape->size(); ++i) {
        shape_vec.push_back((*inputShape)[i]);
    }
    ShapeExpr ret = make_ref<ShapeExprObj>(ShapeExprObj(shape_vec));
    return {{ret}};
}

std::vector<DataType> RMSNormObj::inferDataType() const {
    return {inputs[0]->getDataType()};
}

std::string RMSNormObj::toString() const {
    std::ostringstream os;
    os << "RMSNorm[" << getGuid() << "]";
    os << "(";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "weight=" << inputs[1]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ",";
    os << "eps=" << eps;
    os << ")";
    return os.str();
}

void RMSNormObj::createOpDesc() {
    auto yShape = outputs[0]->getShape();
    auto xShape = inputs[0]->getShape();
    auto wShape = inputs[1]->getShape();
    
    auto yStride = outputs[0]->getStride();
    auto xStride = inputs[0]->getStride();
    auto wStride = inputs[1]->getStride();
    
    infiniopTensorDescriptor_t yTensor, xTensor, wTensor;
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &yTensor, yShape->size(), yShape->getConstantValue().data(),
        yStride->getConstantValue().data(), outputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &xTensor, xShape->size(), xShape->getConstantValue().data(),
        xStride->getConstantValue().data(), inputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &wTensor, wShape->size(), wShape->getConstantValue().data(),
        wStride->getConstantValue().data(), inputs[1]->getDataType().getType()));

    infiniopHandle_t handle = nullptr;
    CHECK_INFINI_ERROR(infiniopCreateHandle(&handle));

    CHECK_INFINI_ERROR(infiniopCreateRMSNormDescriptor(
        handle, (infiniopRMSNormDescriptor_t *)&infiniOpDesc, yTensor, xTensor, wTensor, eps));

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(xTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(wTensor));
}

RMSNormObj::~RMSNormObj() {
    if (infiniOpDesc) {
        infiniopDestroyRMSNormDescriptor((infiniopRMSNormDescriptor_t)infiniOpDesc);
    }
}

} // namespace infini
