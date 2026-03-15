#include "operators/Unary.h"
#include "core/runtime.h"

namespace infini {

UnaryObj::UnaryObj(GraphObj *graph, OpType type_, Tensor input, Tensor output)
    : OperatorObj(type_, {input}, {output}) {
    IT_ASSERT(checkValid(graph));
}

string UnaryObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "(";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

optional<vector<ShapeExpr>> UnaryObj::inferShape() {
    return {{inputs[0]->getShape()}};
}

vector<DataType> UnaryObj::inferDataType() const {
    return {inputs[0]->getDataType()};
}

UnaryObj::~UnaryObj() {
    if (!infiniOpDesc)
        return;
    infiniStatus_t err = INFINI_STATUS_SUCCESS;
    if (type == OpType::Relu)
        err = infiniopDestroyReluDescriptor(
            (infiniopReluDescriptor_t)infiniOpDesc);
    else if (type == OpType::Sigmoid)
        err = infiniopDestroySigmoidDescriptor(
            (infiniopSigmoidDescriptor_t)infiniOpDesc);
    else if (type == OpType::Silu)
        err = infiniopDestroySiluDescriptor(
            (infiniopSiluDescriptor_t)infiniOpDesc);
    else if (type == OpType::Gelu)
        err = infiniopDestroyGeluDescriptor(
            (infiniopGeluDescriptor_t)infiniOpDesc);
    else if (type == OpType::Softplus)
        err = infiniopDestroySoftplusDescriptor(
            (infiniopSoftplusDescriptor_t)infiniOpDesc);
    else if (type == OpType::Tanh)
        err = infiniopDestroyTanhDescriptor(
            (infiniopTanhDescriptor_t)infiniOpDesc);
    if (err != INFINI_STATUS_SUCCESS)
        std::cerr << "Warning: " << type.toString()
                  << " descriptor destroy failed with error " << err << "\n";
}

void UnaryObj::createOpDesc() {
    auto yShape = outputs[0]->getShape();
    auto xShape = inputs[0]->getShape();
    auto yStride = outputs[0]->getStride();
    auto xStride = inputs[0]->getStride();

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

    if (type == OpType::Relu) {
        CHECK_INFINI_ERROR(infiniopCreateReluDescriptor(
            handle, (infiniopReluDescriptor_t *)&infiniOpDesc, yDesc, xDesc));
    } else if (type == OpType::Sigmoid) {
        CHECK_INFINI_ERROR(infiniopCreateSigmoidDescriptor(
            handle, (infiniopSigmoidDescriptor_t *)&infiniOpDesc, yDesc,
            xDesc));
    } else if (type == OpType::Silu) {
        CHECK_INFINI_ERROR(infiniopCreateSiluDescriptor(
            handle, (infiniopSiluDescriptor_t *)&infiniOpDesc, yDesc, xDesc));
    } else if (type == OpType::Gelu) {
        CHECK_INFINI_ERROR(infiniopCreateGeluDescriptor(
            handle, (infiniopGeluDescriptor_t *)&infiniOpDesc, yDesc, xDesc));
    } else if (type == OpType::Softplus) {
        CHECK_INFINI_ERROR(infiniopCreateSoftplusDescriptor(
            handle, (infiniopSoftplusDescriptor_t *)&infiniOpDesc, yDesc,
            xDesc));
    } else if (type == OpType::Tanh) {
        CHECK_INFINI_ERROR(infiniopCreateTanhDescriptor(
            handle, (infiniopTanhDescriptor_t *)&infiniOpDesc, yDesc, xDesc));
    } else {
        IT_TODO_HALT_MSG("Unary operator not supported");
    }

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yDesc));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(xDesc));
}

OpType UnaryObj::getUnaryOpType() const { return type; }

} // namespace infini
