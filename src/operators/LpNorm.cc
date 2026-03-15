#include "operators/LpNorm.h"
#include "core/runtime.h"

namespace infini {

LpNormObj::LpNormObj(GraphObj *graph, Tensor x, int axis, int p, float eps,
                     Tensor Y)
    : OperatorObj(OpType::LpNorm, TensorVec{x}, {Y}), axis(axis), p(p),
      eps(eps) {
    IT_ASSERT(checkValid(graph));
}

string LpNormObj::toString() const {
    std::ostringstream os;
    os << "LpNorm( x=" << inputs[0]->getGuid() << ", axis=" << axis
       << ", p=" << p << ", eps=" << eps << ", Y=" << outputs[0]->getGuid()
       << " )";
    return os.str();
}

LpNormObj::~LpNormObj() {
    if (infiniOpDesc) {
        infiniStatus_t err = INFINI_STATUS_SUCCESS;
        err = infiniopDestroyLPNormDescriptor(
            (infiniopLPNormDescriptor_t)infiniOpDesc);
        if (err != INFINI_STATUS_SUCCESS) {
            std::cerr << "Warning: LpNorm descriptor destroy failed with "
                         "error code "
                      << err << std::endl;
        }
    }
}

optional<vector<ShapeExpr>> LpNormObj::inferShape() {
    auto x = inputs[0];
    auto shapeX = x->getShape();
    return {{shapeX}};
}

vector<DataType> LpNormObj::inferDataType() const {
    return {inputs[0]->getDataType()};
}

void LpNormObj::createOpDesc() {
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
    // create LpNorm op descriptor
    CHECK_INFINI_ERROR(infiniopCreateLPNormDescriptor(
        handle, (infiniopLPNormDescriptor_t *)&infiniOpDesc, yTensor, xTensor,
        axis, p, eps));

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(xTensor));
}

} // namespace infini
