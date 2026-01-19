#include "operators/Gemm.h"
#include "core/runtime.h"
namespace infini {

GemmObj::GemmObj(GraphObj *graph, Tensor A, Tensor B, Tensor Y, Tensor C,
                 float alpha, float beta, bool transA, bool transB)
    : OperatorObj(OpType::Gemm, TensorVec{A, B}, {Y}), alpha(alpha), beta(beta),
      transA(transA), transB(transB) {
    IT_ASSERT(checkValid(graph));
    if (C) {
        graph->getRuntime()->memcpy(
            Y->getRawDataPtr<void *>(), C->getRawDataPtr<void *>(),
            Y->getTotalBytes(), infinirtMemcpyKind_t::INFINIRT_MEMCPY_D2D);
    }
}

string GemmObj::toString() const {
    std::ostringstream os;
    os << "Gemm( [" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B")
       << "],A=" << inputs[0]->getGuid() << ",B=" << inputs[1]->getGuid()
       << ",C="
       << (inputs.size() == 3 ? std::to_string(inputs[2]->getGuid()) : "null")
       << ",Y=" << outputs[0]->getGuid() << " )";
    return os.str();
}

GemmObj::~GemmObj() {
    if (infiniOpDesc) {
        infiniStatus_t err = INFINI_STATUS_SUCCESS;
        err = infiniopDestroyGemmDescriptor(
            (infiniopGemmDescriptor_t)infiniOpDesc);
        if (err != INFINI_STATUS_SUCCESS) {
            std::cerr << "Warning: Gemm descriptor destroy failed with "
                         "error code "
                      << err << std::endl;
        }
    }
}

optional<vector<ShapeExpr>> GemmObj::inferShape() {
    auto A = inputs[0], B = inputs[1];
    auto shapeA = A->getShape();
    auto shapeB = B->getShape();
    IT_ASSERT(shapeA->size() >= 2 && shapeB->size() >= 2);
    Expr batchA = (shapeA->size() == 3) ? (*shapeA)[0] : ExprObj::constant(1);
    Expr batchB = (shapeB->size() == 3) ? (*shapeB)[0] : ExprObj::constant(1);
    // 广播 batch 维度
    Expr batch;
    if (batchA == batchB)
        batch = batchA;
    else if (batchA == ExprObj::constant(1))
        batch = batchB;
    else if (batchB == ExprObj::constant(1))
        batch = batchA;
    else
        IT_ASSERT(
            false,
            "batch dimensions of A and B must be equal or one of them is 1");
    Expr m =
        transA ? (*shapeA)[shapeA->size() - 1] : (*shapeA)[shapeA->size() - 2];
    Expr kA =
        transA ? (*shapeA)[shapeA->size() - 2] : (*shapeA)[shapeA->size() - 1];
    Expr kB =
        transB ? (*shapeB)[shapeB->size() - 1] : (*shapeB)[shapeB->size() - 2];
    Expr n =
        transB ? (*shapeB)[shapeB->size() - 2] : (*shapeB)[shapeB->size() - 1];
    IT_ASSERT(kA == kB);
    ShapeExpr ret;
    ret = make_ref<ShapeExprObj>(ShapeExprObj({batch, m, n}));
    return {{ret}};
}

vector<DataType> GemmObj::inferDataType() const {
    IT_ASSERT(inputs[0]->getDataType() == inputs[1]->getDataType());
    return {inputs[0]->getDataType()};
}

void GemmObj::createOpDesc() {
    auto aShape = inputs[0]->getShape();
    auto bShape = inputs[1]->getShape();
    auto yShape = outputs[0]->getShape();
    auto aStride = inputs[0]->getStride();
    auto bStride = inputs[1]->getStride();
    auto yStride = outputs[0]->getStride();
    infiniopTensorDescriptor_t yTensor, aTensor, bTensor;
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &yTensor, yShape->size(), yShape->getConstantValue().data(),
        yStride->getConstantValue().data(),
        outputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &aTensor, aShape->size(), aShape->getConstantValue().data(),
        aStride->getConstantValue().data(),
        inputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &bTensor, bShape->size(), bShape->getConstantValue().data(),
        bStride->getConstantValue().data(),
        inputs[1]->getDataType().getType()));
    infiniopHandle_t handle = nullptr;
    CHECK_INFINI_ERROR(infiniopCreateHandle(&handle));
    // create gemm op descriptor
    CHECK_INFINI_ERROR(infiniopCreateGemmDescriptor(
        handle, (infiniopGemmDescriptor_t *)&infiniOpDesc, yTensor, aTensor,
        bTensor));

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(aTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(bTensor));
}

bool GemmObj::getTransA() const { return transA; }
bool GemmObj::getTransB() const { return transB; }
float GemmObj::getAlpha() const { return alpha; }
float GemmObj::getBeta() const { return beta; }

} // namespace infini
