#include "core/runtime.h"
#include "operators/UnaryOp.h"
namespace infini {

UnaryWiseObj::UnaryWiseObj(GraphObj *graph, OpType type_, Tensor input,
                           Tensor output)
    : OperatorObj(type_, TensorVec{input}, {output}), type(type_) {
    IT_ASSERT(checkValid(graph));
}

string UnaryWiseObj::toString() const {
    std::ostringstream os;
    os << type.toString();
    os << "(";
    os << "input0=" << inputs[0]->getGuid() << ",";
    os << "input1=" << inputs[1]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

optional<vector<ShapeExpr>> UnaryWiseObj::inferShape() {
    auto A = inputs[0];
    auto shapeA = A->getShape();
    return {{shapeA}};
}

vector<DataType> UnaryWiseObj::inferDataType() const {
    return {inputs[0]->getDataType()};
}

UnaryWiseObj::~UnaryWiseObj() {
    if (infiniOpDesc) {
        infiniStatus_t err = INFINI_STATUS_SUCCESS;
        if (type == OpType::Relu) {
            err = infiniopDestroyReluDescriptor(
                (infiniopReluDescriptor_t)infiniOpDesc);
        } else if (type == OpType::Sigmoid) {
            err = infiniopDestroySigmoidDescriptor(
                (infiniopSigmoidDescriptor_t)infiniOpDesc);
        } else if (type == OpType::Silu) {
            err = infiniopDestroySiluDescriptor(
                (infiniopSiluDescriptor_t)infiniOpDesc);
        } else if (type == OpType::Gelu) {
            err = infiniopDestroyGeluDescriptor(
                (infiniopGeluDescriptor_t)infiniOpDesc);
        } else if (type == OpType::Softplus) {
            err = infiniopDestroySoftplusDescriptor(
                (infiniopSoftplusDescriptor_t)infiniOpDesc);
        } else if (type == OpType::Tanh) {
            err = infiniopDestroyTanhDescriptor(
                (infiniopTanhDescriptor_t)infiniOpDesc);
        }
        if (err != INFINI_STATUS_SUCCESS) {
            std::cerr << "Warning: " << type.toString()
                      << " descriptor destroy failed with error code " << err
                      << std::endl;
        }
    }
}

void UnaryWiseObj::createOpDesc() {
    auto yShape = outputs[0]->getShape();
    auto aShape = inputs[0]->getShape();
    auto aStride = broadcastStride(aShape, inputs[0]->getStride(), yShape);
    auto yStride = outputs[0]->getStride();

    infiniopTensorDescriptor_t yTensor, aTensor;
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &yTensor, yShape->size(), yShape->getConstantValue().data(),
        yStride->getConstantValue().data(),
        outputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &aTensor, yShape->size(), yShape->getConstantValue().data(),
        aStride->getConstantValue().data(),
        inputs[0]->getDataType().getType()));

    infiniopHandle_t handle = nullptr;
    CHECK_INFINI_ERROR(infiniopCreateHandle(&handle));
    if (type == OpType::Relu) {
        CHECK_INFINI_ERROR(infiniopCreateReluDescriptor(
            handle, (infiniopReluDescriptor_t *)&infiniOpDesc, yTensor,
            aTensor));
    } else if (type == OpType::Sigmoid) {
        CHECK_INFINI_ERROR(infiniopCreateSigmoidDescriptor(
            handle, (infiniopSigmoidDescriptor_t *)&infiniOpDesc, yTensor,
            aTensor));
    } else if (type == OpType::Silu) {
        CHECK_INFINI_ERROR(infiniopCreateSiluDescriptor(
            handle, (infiniopSiluDescriptor_t *)&infiniOpDesc, yTensor,
            aTensor));
    } else if (type == OpType::Gelu) {
        CHECK_INFINI_ERROR(infiniopCreateGeluDescriptor(
            handle, (infiniopGeluDescriptor_t *)&infiniOpDesc, yTensor,
            aTensor));
    } else if (type == OpType::Softplus) {
        CHECK_INFINI_ERROR(infiniopCreateSoftplusDescriptor(
            handle, (infiniopSoftplusDescriptor_t *)&infiniOpDesc, yTensor,
            aTensor));
    } else if (type == OpType::Tanh) {
        CHECK_INFINI_ERROR(infiniopCreateTanhDescriptor(
            handle, (infiniopTanhDescriptor_t *)&infiniOpDesc, yTensor,
            aTensor));
    } else {
        IT_TODO_HALT_MSG("UnaryOp operator not supported yet");
    }

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(aTensor));
}

OpType UnaryWiseObj::getUnaryWiseOpType() const { return type; }
} // namespace infini
