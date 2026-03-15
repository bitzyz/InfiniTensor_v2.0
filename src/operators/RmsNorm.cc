#include "operators/RmsNorm.h"
#include "core/runtime.h"

namespace infini {

RmsNormObj::RmsNormObj(GraphObj *graph, Tensor x, Tensor weight, float eps,
                       Tensor Y)
    : OperatorObj(OpType::RmsNorm, TensorVec{x, weight}, {Y}), _eps(eps) {
    IT_ASSERT(checkValid(graph));
}

string RmsNormObj::toString() const {
    std::ostringstream os;
    os << "RmsNorm( x=" << inputs[0]->getGuid()
       << ", weight=" << inputs[1]->getGuid() << ", eps=" << _eps
       << ", Y=" << outputs[0]->getGuid() << " )";
    return os.str();
}

RmsNormObj::~RmsNormObj() {
    if (infiniOpDesc) {
        infiniStatus_t err = INFINI_STATUS_SUCCESS;
        err = infiniopDestroyRMSNormDescriptor(
            (infiniopRMSNormDescriptor_t)infiniOpDesc);
        if (err != INFINI_STATUS_SUCCESS) {
            std::cerr << "Warning: RmsNorm descriptor destroy failed with "
                         "error code "
                      << err << std::endl;
        }
    }
}

optional<vector<ShapeExpr>> RmsNormObj::inferShape() {
    auto x = inputs[0];
    auto shapeX = x->getShape();
    return {{shapeX}};
}

vector<DataType> RmsNormObj::inferDataType() const {
    return {inputs[0]->getDataType()};
}

void RmsNormObj::createOpDesc() {
    auto xShape = inputs[0]->getShape();
    auto weightShape = inputs[1]->getShape();
    auto yShape = outputs[0]->getShape();

    auto xStride = inputs[0]->getStride();
    auto weightStride = inputs[1]->getStride();
    auto yStride = outputs[0]->getStride();

    infiniopTensorDescriptor_t xTensor, weightTensor, yTensor;
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &yTensor, yShape->size(), yShape->getConstantValue().data(),
        yStride->getConstantValue().data(),
        outputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &xTensor, xShape->size(), xShape->getConstantValue().data(),
        xStride->getConstantValue().data(),
        inputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(
        infiniopCreateTensorDescriptor(&weightTensor, weightShape->size(),
                                       weightShape->getConstantValue().data(),
                                       weightStride->getConstantValue().data(),
                                       inputs[1]->getDataType().getType()));

    infiniopHandle_t handle = nullptr;
    CHECK_INFINI_ERROR(infiniopCreateHandle(&handle));
    // create RmsNorm op descriptor
    CHECK_INFINI_ERROR(infiniopCreateRMSNormDescriptor(
        handle, (infiniopRMSNormDescriptor_t *)&infiniOpDesc, yTensor, xTensor,
        weightTensor, _eps));

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(xTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(weightTensor));
}

} // namespace infini
