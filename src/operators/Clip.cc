#include "operators/Clip.h"
#include "core/runtime.h"

namespace infini {

ClipObj::ClipObj(GraphObj *graph, Tensor input, Tensor min_val, Tensor max_val,
                 Tensor output)
    : OperatorObj(OpType::Clip, TensorVec{input, min_val, max_val}, {output}) {
    IT_ASSERT(checkValid(graph));
}

string ClipObj::toString() const {
    std::ostringstream os;
    os << "Clip(";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "min_val=" << inputs[1]->getGuid() << ",";
    os << "max_val=" << inputs[2]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

ClipObj::~ClipObj() {
    if (infiniOpDesc) {
        infiniStatus_t err = INFINI_STATUS_SUCCESS;
        err = infiniopDestroyClipDescriptor((infiniopClipDescriptor_t)infiniOpDesc);
        if (err != INFINI_STATUS_SUCCESS) {
            std::cerr << "Warning: Clip descriptor destroy failed with error code "
                      << err << std::endl;
        }
    }
}

optional<vector<ShapeExpr>> ClipObj::inferShape() {
    // Clip does not change the shape of the input tensor
    // Simply return the input shape as-is (supports both concrete and symbolic shapes)
    auto inputShape = inputs[0]->getShape();
    return {{inputShape}};
}

vector<DataType> ClipObj::inferDataType() const {
    return {inputs[0]->getDataType()};
}

void ClipObj::createOpDesc() {
    auto yShape = outputs[0]->getShape();
    auto yStride = outputs[0]->getStride();

    auto xShape = inputs[0]->getShape();
    auto xStride = inputs[0]->getStride();

    auto minValShape = inputs[1]->getShape();
    auto minValStride = inputs[1]->getStride();

    auto maxValShape = inputs[2]->getShape();
    auto maxValStride = inputs[2]->getStride();

    infiniopTensorDescriptor_t yTensor, xTensor, minValTensor, maxValTensor;

    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &yTensor, yShape->size(), yShape->getConstantValue().data(),
        yStride->getConstantValue().data(),
        outputs[0]->getDataType().getType()));

    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &xTensor, xShape->size(), xShape->getConstantValue().data(),
        xStride->getConstantValue().data(),
        inputs[0]->getDataType().getType()));

    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &minValTensor, minValShape->size(), minValShape->getConstantValue().data(),
        minValStride->getConstantValue().data(),
        inputs[1]->getDataType().getType()));

    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &maxValTensor, maxValShape->size(), maxValShape->getConstantValue().data(),
        maxValStride->getConstantValue().data(),
        inputs[2]->getDataType().getType()));

    infiniopHandle_t handle = nullptr;
    CHECK_INFINI_ERROR(infiniopCreateHandle(&handle));

    CHECK_INFINI_ERROR(infiniopCreateClipDescriptor(
        handle, (infiniopClipDescriptor_t *)&infiniOpDesc, yTensor, xTensor,
        minValTensor, maxValTensor));

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(xTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(minValTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(maxValTensor));
}

} // namespace infini