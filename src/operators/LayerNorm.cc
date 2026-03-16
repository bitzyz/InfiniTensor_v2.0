#include "operators/LayerNorm.h"
#include "core/runtime.h"
#include <infiniop/ops/layer_norm.h>
#include <iostream>
#include <sstream>

namespace infini {

LayerNormObj::LayerNormObj(GraphObj *graph, Tensor input, Tensor weight, Tensor bias, Tensor output, float eps)
    : OperatorObj(OpType::LayerNorm, bias ? TensorVec{input, weight, bias} : TensorVec{input, weight}, {output}),
      eps(eps) {
    IT_ASSERT(checkValid(graph));
}

std::optional<std::vector<ShapeExpr>> LayerNormObj::inferShape() {
    auto inputShape = inputs[0]->getShape();
    std::vector<Expr> shape_vec;
    for (size_t i = 0; i < inputShape->size(); ++i) {
        shape_vec.push_back((*inputShape)[i]);
    }
    ShapeExpr ret = make_ref<ShapeExprObj>(ShapeExprObj(shape_vec));
    return {{ret}};
}

std::vector<DataType> LayerNormObj::inferDataType() const {
    return {inputs[0]->getDataType()};
}

std::string LayerNormObj::toString() const {
    std::ostringstream os;
    os << "LayerNorm(in=" << inputs[0]->getGuid() << ", w=" << inputs[1]->getGuid();
    if (inputs.size() > 2) {
        os << ", bias=" << inputs[2]->getGuid();
    }
    os << ", out=" << outputs[0]->getGuid() << ", eps=" << eps << ")";
    return os.str();
}

void LayerNormObj::createOpDesc() {
    auto yShape = outputs[0]->getShape();
    auto xShape = inputs[0]->getShape();
    auto wShape = inputs[1]->getShape();
    
    auto yStride = outputs[0]->getStride();
    auto xStride = inputs[0]->getStride();
    auto wStride = inputs[1]->getStride();
    
    infiniopTensorDescriptor_t yTensor, xTensor, wTensor;
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &yTensor, yShape->size(), yShape->getConstantValue().data(),
        yStride->getConstantValue().data(), outputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &xTensor, xShape->size(), xShape->getConstantValue().data(),
        xStride->getConstantValue().data(), inputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &wTensor, wShape->size(), wShape->getConstantValue().data(),
        wStride->getConstantValue().data(), inputs[1]->getDataType().getType()));

    infiniopTensorDescriptor_t bTensor = nullptr;
    if (inputs.size() > 2) {
        auto bShape = inputs[2]->getShape();
        auto bStride = inputs[2]->getStride();
        CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
            &bTensor, bShape->size(), bShape->getConstantValue().data(),
            bStride->getConstantValue().data(), inputs[2]->getDataType().getType()));
    }

    infiniopTensorDescriptor_t std_tensor, std_dev_tensor;
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &std_tensor, xShape->size(), xShape->getConstantValue().data(),
        xStride->getConstantValue().data(), inputs[0]->getDataType().getType()));
        
    std::vector<size_t> std_dev_shape(xShape->size() - 1);
    std::vector<ptrdiff_t> std_dev_stride(xShape->size() - 1);
    for (size_t i = 0; i < xShape->size() - 1; ++i) {
        std_dev_shape[i] = xShape->getConstantValue()[i];
        std_dev_stride[i] = xStride->getConstantValue()[i];
    }
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &std_dev_tensor, std_dev_shape.size(), std_dev_shape.data(),
        std_dev_stride.data(), inputs[0]->getDataType().getType()));

    infiniopHandle_t handle = nullptr;
    CHECK_INFINI_ERROR(infiniopCreateHandle(&handle));

    CHECK_INFINI_ERROR(infiniopCreateLayerNormDescriptor(
        handle, (infiniopLayerNormDescriptor_t *)&infiniOpDesc, yTensor, std_tensor, std_dev_tensor, xTensor,
        wTensor, bTensor, eps));

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(xTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(wTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(std_tensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(std_dev_tensor));
    if (bTensor) {
        CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(bTensor));
    }
}

LayerNormObj::~LayerNormObj() {
    if (infiniOpDesc) {
        infiniStatus_t err = infiniopDestroyLayerNormDescriptor((infiniopLayerNormDescriptor_t)infiniOpDesc);
        if (err != INFINI_STATUS_SUCCESS) {
            std::cerr << "Warning: LayerNorm descriptor destroy failed with error code " << err << std::endl;
        }
    }
}

} // namespace infini
