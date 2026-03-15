#include "operators/LayerNorm.h"
#include "core/runtime.h"

namespace infini {

static int normalizeAxis(int axis, size_t rank) {
    int norm = axis;
    if (norm < 0)
        norm += static_cast<int>(rank);
    IT_ASSERT(norm >= 0 && norm < static_cast<int>(rank),
              "axis out of range in LayerNorm");
    return norm;
}

// inputs = {input, weight, bias}, outputs = {y}
LayerNormObj::LayerNormObj(GraphObj *graph, Tensor input, Tensor weight,
                           Tensor bias, Tensor output, int axis, float eps)
    : OperatorObj(OpType::LayerNorm, {input, weight, bias}, {output}),
      axis(axis), eps(eps) {
    IT_ASSERT(checkValid(graph));
}

string LayerNormObj::toString() const {
    std::ostringstream os;
    os << "LayerNorm(axis=" << axis << ",eps=" << eps << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "weight=" << inputs[1]->getGuid() << ",";
    os << "bias=" << inputs[2]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

optional<vector<ShapeExpr>> LayerNormObj::inferShape() {
    return {{inputs[0]->getShape()}};
}

vector<DataType> LayerNormObj::inferDataType() const {
    return {inputs[0]->getDataType()};
}

LayerNormObj::~LayerNormObj() {
    if (infiniOpDesc)
        infiniopDestroyLayerNormDescriptor(
            (infiniopLayerNormDescriptor_t)infiniOpDesc);
}

void LayerNormObj::createOpDesc() {
    // inputs[0]=x, inputs[1]=weight, inputs[2]=bias, outputs[0]=y
    auto xShape = inputs[0]->getShape();
    auto wShape = inputs[1]->getShape();
    auto bShape = inputs[2]->getShape();
    auto yShape = outputs[0]->getShape();
    auto xStride = inputs[0]->getStride();
    auto wStride = inputs[1]->getStride();
    auto bStride = inputs[2]->getStride();
    auto yStride = outputs[0]->getStride();

    auto xVals = xShape->getConstantValue();
    IT_ASSERT(xVals.size() == 3, "LayerNorm only supports 3D input");
    int axisNorm = normalizeAxis(axis, xVals.size());
    IT_ASSERT(axisNorm == 2, "LayerNorm only supports axis=2 for 3D input");
    size_t featureSize = xVals.back();
    auto wVals = wShape->getConstantValue();
    auto bVals = bShape->getConstantValue();
    IT_ASSERT(wVals.size() == 1 && wVals[0] == featureSize,
              "LayerNorm weight must be 1D and match last dimension");
    IT_ASSERT(bVals.size() == 1 && bVals[0] == featureSize,
              "LayerNorm bias must be 1D and match last dimension");

    // Std deviation descriptor uses prefix shape: x.shape[:-1].
    vector<int64_t> stdShape(xVals.begin(), xVals.end() - 1);
    vector<int64_t> stdStride(stdShape.size(), 1);
    for (int i = (int)stdShape.size() - 2; i >= 0; --i)
        stdStride[i] = stdStride[i + 1] * (int64_t)stdShape[i + 1];
    vector<size_t> stdShapeSz(stdShape.begin(), stdShape.end());
    vector<ptrdiff_t> stdStrideP(stdStride.begin(), stdStride.end());

    infiniDtype_t dtype = inputs[0]->getDataType().getType();

    infiniopTensorDescriptor_t yDesc, meanDesc, stdDesc, xDesc, wDesc, bDesc;
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &yDesc, yShape->size(), yShape->getConstantValue().data(),
        yStride->getConstantValue().data(), dtype));
    // input_standardization matches input shape/stride
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &meanDesc, xShape->size(), xShape->getConstantValue().data(),
        xStride->getConstantValue().data(), dtype));
    // input_std_deviation uses prefix shape (ndim-1)
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &stdDesc, (uint64_t)stdShapeSz.size(), stdShapeSz.data(),
        stdStrideP.data(), dtype));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &xDesc, xShape->size(), xShape->getConstantValue().data(),
        xStride->getConstantValue().data(), dtype));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &wDesc, wShape->size(), wShape->getConstantValue().data(),
        wStride->getConstantValue().data(),
        inputs[1]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &bDesc, bShape->size(), bShape->getConstantValue().data(),
        bStride->getConstantValue().data(),
        inputs[2]->getDataType().getType()));

    infiniopHandle_t handle = nullptr;
    CHECK_INFINI_ERROR(infiniopCreateHandle(&handle));
    CHECK_INFINI_ERROR(infiniopCreateLayerNormDescriptor(
        handle, (infiniopLayerNormDescriptor_t *)&infiniOpDesc, yDesc, meanDesc,
        stdDesc, xDesc, wDesc, bDesc, eps));

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yDesc));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(meanDesc));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(stdDesc));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(xDesc));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(wDesc));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(bDesc));
}

int LayerNormObj::getAxis() const { return axis; }
float LayerNormObj::getEps() const { return eps; }

} // namespace infini
