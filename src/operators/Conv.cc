#include "operators/Conv.h"
#include "core/runtime.h"

namespace infini {

ConvObj::ConvObj(GraphObj *graph, Tensor x, Tensor weight, Tensor bias,
                 const vector<size_t> &strides, const vector<size_t> &dilations,
                 const vector<size_t> &paddings, Tensor Y)
    : OperatorObj(OpType::Conv, TensorVec{x, weight}, {Y}),
      _has_bias(bias != nullptr), _strides(strides), _dilations(dilations),
      _paddings(paddings), _n(strides.size()) {
    if (bias != nullptr) {
        inputs.emplace_back(bias);
    }
    IT_ASSERT(checkValid(graph));
}

string ConvObj::toString() const {
    std::ostringstream os;
    os << "Conv( x=" << inputs[0]->getGuid()
       << ", weight=" << inputs[1]->getGuid();
    if (_has_bias) {
        os << ", bias=" << inputs[2]->getGuid();
    }
    os << ", Y=" << outputs[0]->getGuid() << " )";
    return os.str();
}

ConvObj::~ConvObj() {
    if (infiniOpDesc) {
        infiniStatus_t err = INFINI_STATUS_SUCCESS;
        err = infiniopDestroyConvDescriptor(
            (infiniopConvDescriptor_t)infiniOpDesc);
        if (err != INFINI_STATUS_SUCCESS) {
            std::cerr << "Warning: Conv descriptor destroy failed with "
                         "error code "
                      << err << std::endl;
        }
    }
}

optional<vector<ShapeExpr>> ConvObj::inferShape() {
    auto x = inputs[0];
    auto weight = inputs[1];
    auto shapeX = x->getShape();
    auto shapeW = weight->getShape();

    // Conv output shape: [N, C_out, ...]
    // N = batch size from input
    // C_out = output channels from weight
    // Spatial dims computed from input shape, kernel size, stride, padding,
    // dilation

    std::vector<Expr> dims;
    dims.reserve(2 + _n);

    // Batch size
    dims.push_back((*shapeX)[0]);

    // Output channels (first dim of weight)
    dims.push_back((*shapeW)[0]);

    // Spatial dimensions
    for (int i = 0; i < _n; ++i) {
        // out_dim = (in_dim + 2*padding - dilation*(kernel_size-1) - 1) /
        // stride + 1
        Expr inDim = (*shapeX)[2 + i];
        Expr kernelSize = (*shapeW)[2 + i];
        Expr stride = ExprObj::constant(_strides[i]);
        Expr padding = ExprObj::constant(_paddings[i]);
        Expr dilation = ExprObj::constant(_dilations[i]);
        Expr _1 = ExprObj::constant(1);
        Expr _2 = ExprObj::constant(2);

        Expr outDim =
            (inDim + _2 * padding - dilation * (kernelSize - _1) - _1) /
                stride +
            _1;
        dims.push_back(outDim);
    }

    ShapeExpr ret = make_ref<ShapeExprObj>(dims);
    return {{ret}};
}

vector<DataType> ConvObj::inferDataType() const {
    return {inputs[0]->getDataType()};
}

void ConvObj::createOpDesc() {
    auto aShape = inputs[0]->getShape();
    auto weightShape = inputs[1]->getShape();
    auto yShape = outputs[0]->getShape();
    auto aStride = inputs[0]->getStride();
    auto weightStride = inputs[1]->getStride();
    auto yStride = outputs[0]->getStride();
    infiniopTensorDescriptor_t yTensor, aTensor, weightTensor,
        biasTensor = nullptr;
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &yTensor, yShape->size(), yShape->getConstantValue().data(),
        yStride->getConstantValue().data(),
        outputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &aTensor, aShape->size(), aShape->getConstantValue().data(),
        aStride->getConstantValue().data(),
        inputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(
        infiniopCreateTensorDescriptor(&weightTensor, weightShape->size(),
                                       weightShape->getConstantValue().data(),
                                       weightStride->getConstantValue().data(),
                                       inputs[1]->getDataType().getType()));

    if (_has_bias) {
        auto biasShape = inputs[2]->getShape();
        auto biasStride = inputs[2]->getStride();
        CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
            &biasTensor, biasShape->size(),
            biasShape->getConstantValue().data(),
            biasStride->getConstantValue().data(),
            inputs[2]->getDataType().getType()));
    }
    infiniopHandle_t handle = nullptr;
    CHECK_INFINI_ERROR(infiniopCreateHandle(&handle));
    // create Clip op descriptor
    CHECK_INFINI_ERROR(infiniopCreateConvDescriptor(
        handle, (infiniopConvDescriptor_t *)&infiniOpDesc, yTensor, aTensor,
        weightTensor, biasTensor, _paddings.data(), _strides.data(),
        _dilations.data(), _n));

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(aTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(weightTensor));
    if (_has_bias) {
        CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(biasTensor));
    }
}

} // namespace infini
