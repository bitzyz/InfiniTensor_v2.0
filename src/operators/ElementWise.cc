#include "operators/ElementWise.h"
#include "core/runtime.h"
namespace infini {
ElementWiseObj::ElementWiseObj(GraphObj *graph, OpType type_, Tensor input0,
                               Tensor input1, Tensor output)
    : OperatorObj(type_, TensorVec{input0, input1}, {output}), type(type_) {
    IT_ASSERT(checkValid(graph));
}

string ElementWiseObj::toString() const {
    std::ostringstream os;
    os << type.toString();
    os << "(";
    os << "input0=" << inputs[0]->getGuid() << ",";
    os << "input1=" << inputs[1]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

optional<vector<ShapeExpr>> ElementWiseObj::inferShape() {
    auto A = inputs[0], B = inputs[1];
    auto shapeA = A->getShape();
    auto shapeB = B->getShape();
    auto ret = infer_broadcast(shapeA, shapeB);
    return {{ret}};
}

vector<DataType> ElementWiseObj::inferDataType() const {
    IT_ASSERT(inputs[0]->getDataType() == inputs[1]->getDataType());
    return {inputs[0]->getDataType()};
}

ElementWiseObj::~ElementWiseObj() {
    if (infiniOpDesc) {
        infiniStatus_t err = INFINI_STATUS_SUCCESS;
        if (type == OpType::Add) {
            err = infiniopDestroyAddDescriptor(
                (infiniopAddDescriptor_t)infiniOpDesc);
        } else if (type == OpType::Mul) {
            err = infiniopDestroyMulDescriptor(
                (infiniopMulDescriptor_t)infiniOpDesc);
        } else if (type == OpType::Sub) {
            err = infiniopDestroySubDescriptor(
                (infiniopSubDescriptor_t)infiniOpDesc);
        }
        if (err != INFINI_STATUS_SUCCESS) {
            std::cerr << "Warning: " << type.toString()
                      << " descriptor destroy failed with error code " << err
                      << std::endl;
        }
    }
}

void ElementWiseObj::createOpDesc() {
    auto yShape = outputs[0]->getShape();
    auto aShape = inputs[0]->getShape();
    auto aStride = broadcastStride(aShape, inputs[0]->getStride(), yShape);

    auto bShape = inputs[1]->getShape();
    auto bStride = broadcastStride(bShape, inputs[1]->getStride(), yShape);

    auto yStride = outputs[0]->getStride();
    infiniopTensorDescriptor_t yTensor, aTensor, bTensor;
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &yTensor, yShape->size(), yShape->getConstantValue().data(),
        yStride->getConstantValue().data(),
        outputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &aTensor, yShape->size(), yShape->getConstantValue().data(),
        aStride->getConstantValue().data(),
        inputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &bTensor, yShape->size(), yShape->getConstantValue().data(),
        bStride->getConstantValue().data(),
        inputs[1]->getDataType().getType()));
    infiniopHandle_t handle = nullptr;
    CHECK_INFINI_ERROR(infiniopCreateHandle(&handle));
    if (type == OpType::Add) {
        CHECK_INFINI_ERROR(infiniopCreateAddDescriptor(
            handle, (infiniopAddDescriptor_t *)&infiniOpDesc, yTensor, aTensor,
            bTensor));
    } else if (type == OpType::Mul) {
        CHECK_INFINI_ERROR(infiniopCreateMulDescriptor(
            handle, (infiniopMulDescriptor_t *)&infiniOpDesc, yTensor, aTensor,
            bTensor));
    } else if (type == OpType::Sub) {
        CHECK_INFINI_ERROR(infiniopCreateSubDescriptor(
            handle, (infiniopSubDescriptor_t *)&infiniOpDesc, yTensor, aTensor,
            bTensor));
    } else {
        IT_TODO_HALT_MSG("ElementWise operator not supported yet");
    }

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(aTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(bTensor));
}

OpType ElementWiseObj::getElemenwiseOpType() const { return type; }
} // namespace infini
