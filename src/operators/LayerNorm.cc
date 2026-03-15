#include "operators/LayerNorm.h"
#include "core/runtime.h"

namespace infini {

LayerNormObj::LayerNormObj(GraphObj *graph, Tensor x, Tensor weight,
                           Tensor bias, float eps, Tensor Y, Tensor Norm,
                           Tensor Std)
    : OperatorObj(OpType::LayerNorm, TensorVec{x, weight}, {Y, Norm, Std}),
      _has_bias(bias != nullptr), _eps(eps) {
    if (_has_bias) {
        inputs.emplace_back(bias);
    }
    IT_ASSERT(checkValid(graph));
}

string LayerNormObj::toString() const {
    std::ostringstream os;
    os << "LayerNorm( x=" << inputs[0]->getGuid()
       << ", weight=" << inputs[1]->getGuid();
    if (inputs[2] != nullptr) {
        os << ", bias=" << inputs[2]->getGuid();
    }
    os << ", Y=" << outputs[0]->getGuid() << " )";
    return os.str();
}

LayerNormObj::~LayerNormObj() {
    if (infiniOpDesc) {
        infiniStatus_t err = INFINI_STATUS_SUCCESS;
        err = infiniopDestroyLayerNormDescriptor(
            (infiniopLayerNormDescriptor_t)infiniOpDesc);
        if (err != INFINI_STATUS_SUCCESS) {
            std::cerr << "Warning: LayerNorm descriptor destroy failed with "
                         "error code "
                      << err << std::endl;
        }
    }
}

optional<vector<ShapeExpr>> LayerNormObj::inferShape() {
    auto x = inputs[0];
    auto shapeX = x->getShape();

    auto stdShapeDims = shapeX->dims;
    stdShapeDims.pop_back();
    auto stdShape = make_ref<ShapeExprObj>(stdShapeDims);
    return {{shapeX, shapeX, stdShape}};
}

vector<DataType> LayerNormObj::inferDataType() const {
    return {inputs[0]->getDataType(), inputs[0]->getDataType(),
            inputs[0]->getDataType()};
}

void LayerNormObj::createOpDesc() {
    auto xShape = inputs[0]->getShape();
    auto weightShape = inputs[1]->getShape();
    auto yShape = outputs[0]->getShape();
    auto normShape = outputs[1]->getShape();
    auto stdShape = outputs[2]->getShape();

    auto xStride = inputs[0]->getStride();
    auto weightStride = inputs[1]->getStride();
    auto yStride = outputs[0]->getStride();
    auto normStride = outputs[1]->getStride();
    auto stdStride = outputs[2]->getStride();

    infiniopTensorDescriptor_t xTensor, weightTensor, biasTensor = nullptr;
    infiniopTensorDescriptor_t yTensor, normTensor, stdTensor;
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &yTensor, yShape->size(), yShape->getConstantValue().data(),
        yStride->getConstantValue().data(),
        outputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &normTensor, normShape->size(), normShape->getConstantValue().data(),
        normStride->getConstantValue().data(),
        outputs[1]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &stdTensor, stdShape->size(), stdShape->getConstantValue().data(),
        stdStride->getConstantValue().data(),
        outputs[2]->getDataType().getType()));

    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &xTensor, xShape->size(), xShape->getConstantValue().data(),
        xStride->getConstantValue().data(),
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
    CHECK_INFINI_ERROR(infiniopCreateLayerNormDescriptor(
        handle, (infiniopLayerNormDescriptor_t *)&infiniOpDesc, yTensor,
        normTensor, stdTensor, xTensor, weightTensor, biasTensor, _eps));

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(normTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(stdTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(xTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(weightTensor));
    if (_has_bias) {
        CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(biasTensor));
    }
}

} // namespace infini
