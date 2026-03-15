#include "operators/LPNorm.h"
#include "core/runtime.h"

namespace infini {

static int normalizeAxis(int axis, size_t rank) {
    int norm = axis;
    if (norm < 0)
        norm += static_cast<int>(rank);
    IT_ASSERT(norm >= 0 && norm < static_cast<int>(rank),
              "axis out of range in LPNorm");
    return norm;
}

LPNormObj::LPNormObj(GraphObj *graph, Tensor input, Tensor output, int axis,
                     int p, float eps)
    : OperatorObj(OpType::LPNorm, {input}, {output}), axis(axis), p(p),
      eps(eps) {
    IT_ASSERT(checkValid(graph));
}

string LPNormObj::toString() const {
    std::ostringstream os;
    os << "LPNorm(axis=" << axis << ",p=" << p << ",eps=" << eps << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

optional<vector<ShapeExpr>> LPNormObj::inferShape() {
    return {{inputs[0]->getShape()}};
}

vector<DataType> LPNormObj::inferDataType() const {
    return {inputs[0]->getDataType()};
}

LPNormObj::~LPNormObj() {
    if (infiniOpDesc)
        infiniopDestroyLPNormDescriptor(
            (infiniopLPNormDescriptor_t)infiniOpDesc);
}

void LPNormObj::createOpDesc() {
    auto yShape = outputs[0]->getShape();
    auto xShape = inputs[0]->getShape();
    auto yStride = outputs[0]->getStride();
    auto xStride = inputs[0]->getStride();
    int axisNorm = normalizeAxis(axis, xShape->size());

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
    CHECK_INFINI_ERROR(infiniopCreateLPNormDescriptor(
        handle, (infiniopLPNormDescriptor_t *)&infiniOpDesc, yDesc, xDesc,
        axisNorm, p, eps));

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yDesc));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(xDesc));
}

int LPNormObj::getAxis() const { return axis; }
int LPNormObj::getP() const { return p; }
float LPNormObj::getEps() const { return eps; }

} // namespace infini
