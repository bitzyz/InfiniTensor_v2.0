#include "operators/Softmax.h"
#include "core/runtime.h"

namespace infini {

static int normalizeAxis(int axis, size_t rank) {
    int norm = axis;
    if (norm < 0)
        norm += static_cast<int>(rank);
    IT_ASSERT(norm >= 0 && norm < static_cast<int>(rank),
              "axis out of range in Softmax");
    return norm;
}

SoftmaxObj::SoftmaxObj(GraphObj *graph, Tensor input, Tensor output, int axis)
    : OperatorObj(OpType::Softmax, {input}, {output}), axis(axis) {
    IT_ASSERT(checkValid(graph));
}

string SoftmaxObj::toString() const {
    std::ostringstream os;
    os << "Softmax(axis=" << axis << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

optional<vector<ShapeExpr>> SoftmaxObj::inferShape() {
    return {{inputs[0]->getShape()}};
}

vector<DataType> SoftmaxObj::inferDataType() const {
    return {inputs[0]->getDataType()};
}

SoftmaxObj::~SoftmaxObj() {
    if (infiniOpDesc)
        infiniopDestroySoftmaxDescriptor(
            (infiniopSoftmaxDescriptor_t)infiniOpDesc);
}

void SoftmaxObj::createOpDesc() {
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
    CHECK_INFINI_ERROR(infiniopCreateSoftmaxDescriptor(
        handle, (infiniopSoftmaxDescriptor_t *)&infiniOpDesc, yDesc, xDesc,
        axisNorm));

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yDesc));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(xDesc));
}

int SoftmaxObj::getAxis() const { return axis; }

} // namespace infini
