#include "operators/Softmax.h"
#include "core/runtime.h"

namespace infini {

SoftmaxObj::SoftmaxObj(GraphObj *graph, Tensor x, int axis, Tensor Y)
    : OperatorObj(OpType::Softmax, TensorVec{x}, {Y}), axis(axis) {
    IT_ASSERT(checkValid(graph));
}

string SoftmaxObj::toString() const {
    std::ostringstream os;
    os << "Softmax( x=" << inputs[0]->getGuid() << ", axis=" << axis
       << ", Y=" << outputs[0]->getGuid() << " )";
    return os.str();
}

SoftmaxObj::~SoftmaxObj() {
    if (infiniOpDesc) {
        infiniStatus_t err = INFINI_STATUS_SUCCESS;
        err = infiniopDestroySoftmaxDescriptor(
            (infiniopSoftmaxDescriptor_t)infiniOpDesc);
        if (err != INFINI_STATUS_SUCCESS) {
            std::cerr << "Warning: Softmax descriptor destroy failed with "
                         "error code "
                      << err << std::endl;
        }
    }
}

optional<vector<ShapeExpr>> SoftmaxObj::inferShape() {
    auto x = inputs[0];
    auto shapeX = x->getShape();
    return {{shapeX}};
}

vector<DataType> SoftmaxObj::inferDataType() const {
    return {inputs[0]->getDataType()};
}

void SoftmaxObj::createOpDesc() {
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
    // create Softmax op descriptor
    CHECK_INFINI_ERROR(infiniopCreateSoftmaxDescriptor(
        handle, (infiniopSoftmaxDescriptor_t *)&infiniOpDesc, yTensor, xTensor,
        axis));

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(xTensor));
}

} // namespace infini
