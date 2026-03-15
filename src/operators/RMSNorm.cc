#include "operators/RMSNorm.h"
#include "core/runtime.h"

namespace infini {

RMSNormObj::RMSNormObj(GraphObj *graph, Tensor input, Tensor weight,
                       Tensor output, float epsilon)
    : OperatorObj(OpType::RMSNorm, {input, weight}, {output}),
      epsilon(epsilon) {
    IT_ASSERT(checkValid(graph));
}

string RMSNormObj::toString() const {
    std::ostringstream os;
    os << "RMSNorm(eps=" << epsilon << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "weight=" << inputs[1]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

optional<vector<ShapeExpr>> RMSNormObj::inferShape() {
    return {{inputs[0]->getShape()}};
}

vector<DataType> RMSNormObj::inferDataType() const {
    return {inputs[0]->getDataType()};
}

RMSNormObj::~RMSNormObj() {
    if (infiniOpDesc)
        infiniopDestroyRMSNormDescriptor(
            (infiniopRMSNormDescriptor_t)infiniOpDesc);
}

void RMSNormObj::createOpDesc() {
    auto yShape = outputs[0]->getShape();
    auto xShape = inputs[0]->getShape();
    auto wShape = inputs[1]->getShape();
    auto yStride = outputs[0]->getStride();
    auto xStride = inputs[0]->getStride();
    auto wStride = inputs[1]->getStride();

    infiniopTensorDescriptor_t yDesc, xDesc, wDesc;
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &yDesc, yShape->size(), yShape->getConstantValue().data(),
        yStride->getConstantValue().data(),
        outputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &xDesc, xShape->size(), xShape->getConstantValue().data(),
        xStride->getConstantValue().data(),
        inputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &wDesc, wShape->size(), wShape->getConstantValue().data(),
        wStride->getConstantValue().data(),
        inputs[1]->getDataType().getType()));

    infiniopHandle_t handle = nullptr;
    CHECK_INFINI_ERROR(infiniopCreateHandle(&handle));
    CHECK_INFINI_ERROR(infiniopCreateRMSNormDescriptor(
        handle, (infiniopRMSNormDescriptor_t *)&infiniOpDesc, yDesc, xDesc,
        wDesc, epsilon));

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yDesc));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(xDesc));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(wDesc));
}

float RMSNormObj::getEpsilon() const { return epsilon; }

} // namespace infini
