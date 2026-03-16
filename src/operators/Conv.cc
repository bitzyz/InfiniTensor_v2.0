#include "operators/Conv.h"
#include "core/runtime.h"
#include <infiniop/ops/conv.h>
#include <iostream>
#include <sstream>

namespace infini {

ConvObj::ConvObj(GraphObj *graph, Tensor input, Tensor weight, Tensor output,
                 std::vector<int> pads, std::vector<int> strides,
                 std::vector<int> dilations, Tensor bias)
    : OperatorObj(OpType::Conv, bias ? TensorVec{input, weight, bias} : TensorVec{input, weight}, {output}),
      pads(std::move(pads)), strides(std::move(strides)), dilations(std::move(dilations)) {
    IT_ASSERT(checkValid(graph));
}

std::optional<std::vector<ShapeExpr>> ConvObj::inferShape() {
    auto inputShape = inputs[0]->getShape();
    auto weightShape = inputs[1]->getShape();
    IT_ASSERT(inputShape->size() >= 3 && weightShape->size() >= 3);
    IT_ASSERT(inputShape->size() == weightShape->size());
    
    size_t ndim = inputShape->size() - 2;
    IT_ASSERT(pads.size() == ndim);
    IT_ASSERT(strides.size() == ndim);
    IT_ASSERT(dilations.size() == ndim);

    std::vector<Expr> shape_vec;
    shape_vec.push_back((*inputShape)[0]); // batch
    shape_vec.push_back((*weightShape)[0]); // out_channels
    
    for (size_t i = 0; i < ndim; ++i) {
        // out = (in + 2*pad - dilation*(kernel-1) - 1) / stride + 1
        Expr in_dim = (*inputShape)[i + 2];
        Expr kernel_dim = (*weightShape)[i + 2];
        Expr pad2 = ExprObj::constant(2 * pads[i]);
        Expr dil = ExprObj::constant(dilations[i]);
        Expr str = ExprObj::constant(strides[i]);
        
        Expr numerator = in_dim + pad2 - dil * (kernel_dim - ExprObj::constant(1)) - ExprObj::constant(1);
        Expr out_dim = (numerator / str) + ExprObj::constant(1);
        auto evaluated = out_dim->evaluate({});
        if (evaluated.has_value()) {
            shape_vec.push_back(ExprObj::constant(evaluated.value()));
        } else {
            shape_vec.push_back(out_dim);
        }
    }
    
    ShapeExpr ret = make_ref<ShapeExprObj>(ShapeExprObj(shape_vec));
    return {{ret}};
}

std::vector<DataType> ConvObj::inferDataType() const {
    return {inputs[0]->getDataType()};
}

std::string ConvObj::toString() const {
    std::ostringstream os;
    os << "Conv(in=" << inputs[0]->getGuid() << ", w=" << inputs[1]->getGuid();
    if (inputs.size() > 2) {
        os << ", bias=" << inputs[2]->getGuid();
    }
    os << ", out=" << outputs[0]->getGuid() << ")";
    return os.str();
}

void ConvObj::createOpDesc() {
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

    infiniopHandle_t handle = nullptr;
    CHECK_INFINI_ERROR(infiniopCreateHandle(&handle));

    // Notice: pads, strides, dilations expected to be void* (likely casting to uint64_t* or int64_t* or int* internally). 
    // Wait, the API takes `void* pads, void* strides, void* dilations, size_t n`.
    // In infiniop, pads, strides, dilations are usually size_t / uint64_t.
    // Let's create local uint64_t arrays.
    size_t ndim = pads.size();
    std::vector<uint64_t> pads_u64(ndim);
    std::vector<uint64_t> strides_u64(ndim);
    std::vector<uint64_t> dilations_u64(ndim);
    for(size_t i=0; i<ndim; ++i) {
        pads_u64[i] = pads[i];
        strides_u64[i] = strides[i];
        dilations_u64[i] = dilations[i];
    }

    CHECK_INFINI_ERROR(infiniopCreateConvDescriptor(
        handle, (infiniopConvDescriptor_t *)&infiniOpDesc, yTensor, xTensor,
        wTensor, bTensor, pads_u64.data(), strides_u64.data(), dilations_u64.data(), ndim));

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(xTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(wTensor));
    if (bTensor) {
        CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(bTensor));
    }
}

ConvObj::~ConvObj() {
    if (infiniOpDesc) {
        infiniStatus_t err = infiniopDestroyConvDescriptor((infiniopConvDescriptor_t)infiniOpDesc);
        if (err != INFINI_STATUS_SUCCESS) {
            std::cerr << "Warning: Conv descriptor destroy failed with error code " << err << std::endl;
        }
    }
}

} // namespace infini
