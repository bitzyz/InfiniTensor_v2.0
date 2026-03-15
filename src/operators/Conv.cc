#include "operators/Conv.h"
#include "core/runtime.h"

namespace infini {

ConvObj::ConvObj(GraphObj *graph, Tensor x, Tensor w, Tensor bias, Tensor y,
                 vector<int64_t> pads_, vector<int64_t> strides_,
                 vector<int64_t> dilations_)
    : OperatorObj(OpType::Conv, bias ? TensorVec{x, w, bias} : TensorVec{x, w},
                  {y}),
      pads(std::move(pads_)), strides(std::move(strides_)),
      dilations(std::move(dilations_)) {
    IT_ASSERT(checkValid(graph));
}

string ConvObj::toString() const {
    std::ostringstream os;
    os << "Conv(";
    os << "x=" << inputs[0]->getGuid() << ",";
    os << "w=" << inputs[1]->getGuid() << ",";
    if (hasBias())
        os << "bias=" << inputs[2]->getGuid() << ",";
    os << "y=" << outputs[0]->getGuid() << ")";
    return os.str();
}

// output shape: [N, K, out_0, out_1, ...]
// out_i = floor((in_i + 2*pads[i] - dilations[i]*(kernel_i-1) - 1) /
// strides[i]) + 1
optional<vector<ShapeExpr>> ConvObj::inferShape() {
    auto xShape = inputs[0]->getShape();
    auto wShape = inputs[1]->getShape();
    IT_ASSERT(xShape->size() >= 2 && wShape->size() >= 2);
    IT_ASSERT(xShape->size() == wShape->size());

    size_t nDims = strides.size();
    IT_ASSERT(nDims == xShape->size() - 2);
    IT_ASSERT(nDims == pads.size() && nDims == dilations.size());

    auto xVals = xShape->getConstantValue();
    auto wVals = wShape->getConstantValue();

    // output: [N, K, ...]
    vector<Expr> outDims;
    outDims.push_back((*xShape)[0]); // N
    outDims.push_back((*wShape)[0]); // K (number of filters)
    for (size_t i = 0; i < nDims; ++i) {
        int64_t in_i = xVals[i + 2];
        int64_t k_i = wVals[i + 2];
        int64_t out_i =
            (in_i + 2 * pads[i] - dilations[i] * (k_i - 1) - 1) / strides[i] +
            1;
        outDims.push_back(ExprObj::constant(out_i));
    }

    auto ret = make_ref<ShapeExprObj>(ShapeExprObj(outDims));
    return {{ret}};
}

vector<DataType> ConvObj::inferDataType() const {
    return {inputs[0]->getDataType()};
}

ConvObj::~ConvObj() {
    if (infiniOpDesc)
        infiniopDestroyConvDescriptor((infiniopConvDescriptor_t)infiniOpDesc);
}

void ConvObj::createOpDesc() {
    auto yShape = outputs[0]->getShape();
    auto xShape = inputs[0]->getShape();
    auto wShape = inputs[1]->getShape();
    auto yStride = outputs[0]->getStride();
    auto xStride = inputs[0]->getStride();
    auto wStride = inputs[1]->getStride();

    infiniopTensorDescriptor_t yDesc, xDesc, wDesc, bDesc = nullptr;
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
    if (hasBias()) {
        auto bShape = inputs[2]->getShape();
        auto bStride = inputs[2]->getStride();
        CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
            &bDesc, bShape->size(), bShape->getConstantValue().data(),
            bStride->getConstantValue().data(),
            inputs[2]->getDataType().getType()));
    }

    size_t n = strides.size(); // number of spatial dims
    infiniopHandle_t handle = nullptr;
    CHECK_INFINI_ERROR(infiniopCreateHandle(&handle));
    CHECK_INFINI_ERROR(infiniopCreateConvDescriptor(
        handle, (infiniopConvDescriptor_t *)&infiniOpDesc, yDesc, xDesc, wDesc,
        bDesc, (void *)pads.data(), (void *)strides.data(),
        (void *)dilations.data(), n));

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yDesc));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(xDesc));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(wDesc));
    if (bDesc)
        CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(bDesc));
}

const vector<int64_t> &ConvObj::getPads() const { return pads; }
const vector<int64_t> &ConvObj::getStrides() const { return strides; }
const vector<int64_t> &ConvObj::getDilations() const { return dilations; }
bool ConvObj::hasBias() const { return inputs.size() == 3; }

} // namespace infini
