#include "operators/Transpose.h"
#include "core/runtime.h"
#include <infiniop/ops/rearrange.h>

namespace infini {

TransposeObj::TransposeObj(GraphObj *graph, Tensor input, Tensor output, std::vector<int> perm)
    : OperatorObj(OpType::Transpose, {input}, {output}), perm(perm) {
    IT_ASSERT(checkValid(graph));
}

std::optional<std::vector<ShapeExpr>> TransposeObj::inferShape() {
    auto inputShape = inputs[0]->getShape();
    std::vector<Expr> shape_vec;
    for (int p : perm) {
        shape_vec.push_back((*inputShape)[p]);
    }
    ShapeExpr ret = make_ref<ShapeExprObj>(ShapeExprObj(shape_vec));
    return {{ret}};
}

std::vector<DataType> TransposeObj::inferDataType() const {
    return {inputs[0]->getDataType()};
}

std::string TransposeObj::toString() const {
    std::ostringstream os;
    os << "Transpose[" << getGuid() << "]";
    os << "(";
    os << "perm=" << vecToString(perm) << ",";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid();
    os << ")";
    return os.str();
}

void TransposeObj::createOpDesc() {
    auto yShape = outputs[0]->getShape();
    auto xShape = inputs[0]->getShape();
    
    auto yStride = outputs[0]->getStride();
    auto xStride = inputs[0]->getStride();
    
    infiniopTensorDescriptor_t yTensor, xTensor;
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &yTensor, yShape->size(), yShape->getConstantValue().data(),
        yStride->getConstantValue().data(), outputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &xTensor, xShape->size(), xShape->getConstantValue().data(),
        xStride->getConstantValue().data(), inputs[0]->getDataType().getType()));

    infiniopHandle_t handle = nullptr;
    CHECK_INFINI_ERROR(infiniopCreateHandle(&handle));

    // Rearrange(dst=y, src=x)
    // The descriptors contain shape and stride.
    // If x is contiguous and y is contiguous, but y shape is permuted,
    // InfiniCore Rearrange should detect it needs to transpose based on shape mismatch?
    // Or it expects strides to define the layout.
    // Actually, Transpose implementation in CUDNN/etc usually needs:
    // Input: shape A, stride SA
    // Output: shape perm(A), stride SB
    // If we want to copy Input -> Output with transpose:
    // We can view Input as: shape perm(A), stride perm(SA)
    // And copy to Output: shape perm(A), stride SB (contiguous)
    
    // So, we should create a 'virtual' source descriptor that has permuted shape and permuted strides of original input.
    // Wait, xShape and xStride are from input tensor (original).
    // If we pass xTensor as is, it has shape A and stride SA.
    // yTensor has shape perm(A) and stride SB.
    // Does Rearrange handle this?
    // Rearrange usually implies "copy from src to dst".
    // If src and dst have different shapes, it might error unless it's just reshape (same element count).
    // But transpose changes strides order.
    
    // Let's assume InfiniCore Rearrange is smart enough or works like `cudnnTransformTensor`.
    // cudnnTransformTensor takes srcDesc and dstDesc. If dimensions match but strides differ, it permutes.
    // Here dimensions differ (order is permuted).
    // So we might need to permute xDesc to match yDesc shape, but keeping x's strides permuted?
    
    // Yes: create a descriptor for X that has Y's shape, but strides permuted according to perm.
    // Then src and dst have same shape, but different strides.
    
    // But `inputs[0]` has fixed shape/stride. We can't change it.
    // We can create a temporary descriptor.
    
    std::vector<size_t> x_dims_permuted;
    std::vector<ptrdiff_t> x_strides_permuted;
    auto x_dims_orig = xShape->getConstantValue();
    auto x_strides_orig = xStride->getConstantValue();
    
    for (int p : perm) {
        x_dims_permuted.push_back(x_dims_orig[p]);
        x_strides_permuted.push_back(x_strides_orig[p]);
    }
    
    infiniopTensorDescriptor_t xTensorPermuted;
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &xTensorPermuted, x_dims_permuted.size(), x_dims_permuted.data(),
        x_strides_permuted.data(), inputs[0]->getDataType().getType()));
        
    CHECK_INFINI_ERROR(infiniopCreateRearrangeDescriptor(
        handle, (infiniopRearrangeDescriptor_t *)&infiniOpDesc, yTensor, xTensorPermuted));

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(xTensor)); // We didn't use this one
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(xTensorPermuted));
}

TransposeObj::~TransposeObj() {
    if (infiniOpDesc) {
        infiniopDestroyRearrangeDescriptor((infiniopRearrangeDescriptor_t)infiniOpDesc);
    }
}

} // namespace infini
