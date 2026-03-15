#include "operators/LogSoftmax.h"
#include "core/runtime.h"

namespace infini {

LogSoftmaxObj::LogSoftmaxObj(GraphObj *graph, Tensor input, Tensor output)
    : OperatorObj(OpType::LogSoftmax, {input}, {output}) {
    IT_ASSERT(checkValid(graph));
}

string LogSoftmaxObj::toString() const {
    std::ostringstream os;
    os << "LogSoftmax(";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

optional<vector<ShapeExpr>> LogSoftmaxObj::inferShape() {
    return {{inputs[0]->getShape()}};
}

vector<DataType> LogSoftmaxObj::inferDataType() const {
    return {inputs[0]->getDataType()};
}

LogSoftmaxObj::~LogSoftmaxObj() {
    if (infiniOpDesc)
        infiniopDestroyLogSoftmaxDescriptor(
            (infiniopLogSoftmaxDescriptor_t)infiniOpDesc);
}

void LogSoftmaxObj::createOpDesc() {
    auto yShape = outputs[0]->getShape();
    auto xShape = inputs[0]->getShape();
    auto yStride = outputs[0]->getStride();
    auto xStride = inputs[0]->getStride();

    infiniopTensorDescriptor_t yDesc, xDesc;
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &yDesc, yShape->size(), yShape->getConstantValue().data(),
        yStride->getConstantValue().data(),
        outputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &xDesc, xShape->size(), xShape->getConstantValue().data(),
        xStride->getConstantValue().data(),
        inputs[0]->getDataType().getType()));

    infiniopHandle_t handle = nullptr;
    CHECK_INFINI_ERROR(infiniopCreateHandle(&handle));
    CHECK_INFINI_ERROR(infiniopCreateLogSoftmaxDescriptor(
        handle, (infiniopLogSoftmaxDescriptor_t *)&infiniOpDesc, yDesc, xDesc));

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yDesc));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(xDesc));
}

} // namespace infini
