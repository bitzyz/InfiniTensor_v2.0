#include "operators/LogSoftmax.h"
#include "core/runtime.h"

namespace infini {

LogSoftmaxObj::LogSoftmaxObj(GraphObj *graph, Tensor x, int dim, Tensor Y)
    : OperatorObj(OpType::LogSoftmax, TensorVec{x}, {Y}), dim(dim) {
    IT_ASSERT(checkValid(graph));
}

string LogSoftmaxObj::toString() const {
    std::ostringstream os;
    os << "LogSoftmax( x=" << inputs[0]->getGuid()
       << ", Y=" << outputs[0]->getGuid() << " )";
    return os.str();
}

LogSoftmaxObj::~LogSoftmaxObj() {
    if (infiniOpDesc) {
        infiniStatus_t err = INFINI_STATUS_SUCCESS;
        err = infiniopDestroyLogSoftmaxDescriptor(
            (infiniopLogSoftmaxDescriptor_t)infiniOpDesc);
        if (err != INFINI_STATUS_SUCCESS) {
            std::cerr << "Warning: LogSoftmax descriptor destroy failed with "
                         "error code "
                      << err << std::endl;
        }
    }
}

optional<vector<ShapeExpr>> LogSoftmaxObj::inferShape() {
    auto x = inputs[0];
    auto shapeX = x->getShape();
    return {{shapeX}};
}

vector<DataType> LogSoftmaxObj::inferDataType() const {
    return {inputs[0]->getDataType()};
}

void LogSoftmaxObj::createOpDesc() {
    auto xShape = inputs[0]->getShape();
    auto yShape = outputs[0]->getShape();

    auto xStride = inputs[0]->getStride();
    auto yStride = outputs[0]->getStride();

    infiniopTensorDescriptor_t xTensor, yTensor;
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &yTensor, yShape->size(), yShape->getConstantValue().data(),
        yStride->getConstantValue().data(),
        outputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &xTensor, xShape->size(), xShape->getConstantValue().data(),
        xStride->getConstantValue().data(),
        inputs[0]->getDataType().getType()));

    infiniopHandle_t handle = nullptr;
    CHECK_INFINI_ERROR(infiniopCreateHandle(&handle));
    // create LogSoftmax op descriptor
    CHECK_INFINI_ERROR(infiniopCreateLogSoftmaxDescriptor(
        handle, (infiniopLogSoftmaxDescriptor_t *)&infiniOpDesc, yTensor,
        xTensor));

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(xTensor));
}

} // namespace infini
