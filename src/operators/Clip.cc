#include "operators/Clip.h"
#include "core/runtime.h"

namespace infini {

ClipObj::ClipObj(GraphObj *graph, Tensor IN, Tensor Y, Tensor MIN, Tensor MAX)
    : OperatorObj(OpType::Clip, TensorVec{IN, MIN, MAX}, {Y}) {
    IT_ASSERT(checkValid(graph));
}

string ClipObj::toString() const {
    std::ostringstream os;
    os << "Clip( [IN,MIN,MAX],IN=" << inputs[0]->getGuid()
       << ",MIN=" << inputs[1]->getGuid() << ",MAX=" << inputs[2]->getGuid()
       << ",Y=" << outputs[0]->getGuid() << " )";
    return os.str();
}

ClipObj::~ClipObj() {
    if (infiniOpDesc) {
        infiniStatus_t err = INFINI_STATUS_SUCCESS;
        err = infiniopDestroyClipDescriptor(
            (infiniopClipDescriptor_t)infiniOpDesc);
        if (err != INFINI_STATUS_SUCCESS) {
            std::cerr << "Warning: Clip descriptor destroy failed with "
                         "error code "
                      << err << std::endl;
        }
    }
}

optional<vector<ShapeExpr>> ClipObj::inferShape() {
    auto IN = inputs[0], MIN = inputs[1], MAX = inputs[2];
    auto shapeIn = IN->getShape();
    auto shapeMin = MIN->getShape();
    // auto shapeMax = MAX->getShape();
    // // support broadcast // infiniteCore不支持广播
    // IT_ASSERT(shapeIn->size() >= shapeMin->size() && shapeIn->size() >=
    // shapeMax->size()); for (int i = 0; i <
    // static_cast<int>(shapeMin->size()); ++i) {
    //     IT_ASSERT((*shapeMin)[shapeMin->size() - 1 - i] ==
    //     ExprObj::constant(1)
    //             || (*shapeMin)[shapeMin->size() - 1 - i] ==
    //             (*shapeIn)[shapeIn->size() - 1 - i],
    //         "batch dimensions of IN and shapeMin must be equal or one of them
    //         is 1");
    // }
    // for (int i = 0; i < static_cast<int>(shapeMax->size()); ++i) {
    //     IT_ASSERT((*shapeMax)[shapeMax->size() - 1 - i] ==
    //     ExprObj::constant(1)
    //             || (*shapeMax)[shapeMax->size() - 1 - i] ==
    //             (*shapeIn)[shapeIn->size() - 1 - i],
    //         "batch dimensions of IN and shapeMax must be equal or one of them
    //         is 1");
    // }
    return {{shapeIn}};
}

vector<DataType> ClipObj::inferDataType() const {
    return {inputs[0]->getDataType()};
}

void ClipObj::createOpDesc() {
    auto inShape = inputs[0]->getShape();
    auto minShape = inputs[1]->getShape();
    auto maxShape = inputs[2]->getShape();
    auto yShape = outputs[0]->getShape();
    auto inStride = inputs[0]->getStride();
    auto minStride = inputs[1]->getStride();
    auto maxStride = inputs[2]->getStride();
    auto yStride = outputs[0]->getStride();
    infiniopTensorDescriptor_t yTensor, inTensor, minTensor, maxTensor;
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &yTensor, yShape->size(), yShape->getConstantValue().data(),
        yStride->getConstantValue().data(),
        outputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &inTensor, inShape->size(), inShape->getConstantValue().data(),
        inStride->getConstantValue().data(),
        inputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &minTensor, minShape->size(), minShape->getConstantValue().data(),
        minStride->getConstantValue().data(),
        inputs[1]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &maxTensor, maxShape->size(), maxShape->getConstantValue().data(),
        maxStride->getConstantValue().data(),
        inputs[2]->getDataType().getType()));
    infiniopHandle_t handle = nullptr;
    CHECK_INFINI_ERROR(infiniopCreateHandle(&handle));
    // create Clip op descriptor
    CHECK_INFINI_ERROR(infiniopCreateClipDescriptor(
        handle, (infiniopClipDescriptor_t *)&infiniOpDesc, yTensor, inTensor,
        minTensor, maxTensor));

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(inTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(minTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(maxTensor));
}

} // namespace infini
