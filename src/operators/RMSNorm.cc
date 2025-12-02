#include "operators/RMSNorm.h"
#include "core/runtime.h"
namespace infini
{
    RMSNormObj::RMSNormObj(GraphObj *graph, Tensor X, Tensor Y,
                             Tensor W, float epsilon)
        : OperatorObj(OpType::RMSNorm, TensorVec{X, W}, {Y}), epsilon(epsilon)
    {
        IT_ASSERT(checkValid(graph));
    }

    string RMSNormObj::toString() const
    {
        std::ostringstream os;
        os << "RMSNorm( X=" << inputs[0]->getGuid()
           << ",W=" << inputs[1]->getGuid()
           << ",Y=" << outputs[0]->getGuid()
           << ",epsilon=" << epsilon
           << " )";
        return os.str();
    }

    optional<vector<Shape>> RMSNormObj::inferShape() 
    {
        const auto X = inputs[0];
        auto input_dim = X->getShape();
        return {{input_dim}};
    }

    vector<DataType> RMSNormObj::inferDataType() const
    {
        return {inputs[0]->getDataType()};
    }

    void RMSNormObj::createOpDesc()
    {
        auto xShape = inputs[0]->getShape();
        auto wShape = inputs[1]->getShape();
        auto yShape = outputs[0]->getShape();
        auto xStride = inputs[0]->getStride();
        auto wStride = inputs[1]->getStride();
        auto yStride = outputs[0]->getStride();
        infiniopTensorDescriptor_t yTensor, xTensor, wTensor;
        CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
            &yTensor, yShape.size(), yShape.data(), yStride.data(),
            outputs[0]->getDataType().getType()));
        CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
            &xTensor, xShape.size(), xShape.data(), xStride.data(),
            inputs[0]->getDataType().getType()));
        CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
            &wTensor, wShape.size(), wShape.data(), wStride.data(),
            inputs[1]->getDataType().getType()));
        infiniopHandle_t handle = nullptr;
        CHECK_INFINI_ERROR(infiniopCreateHandle(&handle));
        // create RMSNorm op descriptor
        CHECK_INFINI_ERROR(infiniopCreateRMSNormDescriptor(
            handle, (infiniopRMSNormDescriptor_t *)&infiniOpDesc, yTensor, xTensor,
            wTensor, epsilon));

        CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yTensor));
        CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(xTensor));
        CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(wTensor));
    }

} // namespace infini
