#include "operators/Gemm.h"
#include "core/runtime.h"
namespace infini {

GemmObj::GemmObj(GraphObj *graph, Tensor A, Tensor B, Tensor Y, Tensor C,
                 float alpha, float beta, bool transA, bool transB)
    : OperatorObj(OpType::Gemm, TensorVec{A, B}, {Y}), alpha(alpha), beta(beta),
      transA(transA), transB(transB) {
  IT_ASSERT(checkValid(graph));
  if (C) {
    graph->getRuntime()->memcpy(Y->getRawDataPtr<void *>(),
                                C->getRawDataPtr<void *>(), Y->getTotalBytes(),
                                infinirtMemcpyKind_t::INFINIRT_MEMCPY_D2D);
  }
}

string GemmObj::toString() const {
  std::ostringstream os;
  os << "Gemm( [" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B")
     << "],A=" << inputs[0]->getGuid() << ",B=" << inputs[1]->getGuid() << ",C="
     << (inputs.size() == 3 ? std::to_string(inputs[2]->getGuid()) : "null")
     << ",Y=" << outputs[0]->getGuid() << " )";
  return os.str();
}

optional<vector<Shape>> GemmObj::inferShape() {
  auto A = inputs[0], B = inputs[1];
  auto shapeA = A->getShape();
  auto shapeB = B->getShape();
  IT_ASSERT(shapeA.size() >= 2 && shapeB.size() >= 2);
  size_t batchA = (shapeA.size() == 3) ? shapeA[0] : 1;
  size_t batchB = (shapeB.size() == 3) ? shapeB[0] : 1;
  // 广播 batch 维度
  ShapeElem batch;
  if (batchA == batchB)
    batch = batchA;
  else if (batchA == 1)
    batch = batchB;
  else if (batchB == 1)
    batch = batchA;
  else
    IT_ASSERT(false,
              "batch dimensions of A and B must be equal or one of them is 1");
  ShapeElem m = transA ? shapeA[shapeA.size() - 1] : shapeA[shapeA.size() - 2];
  ShapeElem kA = transA ? shapeA[shapeA.size() - 2] : shapeA[shapeA.size() - 1];
  ShapeElem kB = transB ? shapeB[shapeB.size() - 1] : shapeB[shapeB.size() - 2];
  ShapeElem n = transB ? shapeB[shapeB.size() - 2] : shapeB[shapeB.size() - 1];
  IT_ASSERT(kA == kB);
  Shape ret;
  if (batch > 1)
    ret = {batch, m, n}; // 3D
  else
    ret = {m, n}; // 2D
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
      &yTensor, yShape.size(), yShape.data(), yStride.data(),
      outputs[0]->getDataType().getType()));
  CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
      &aTensor, aShape.size(), aShape.data(), aStride.data(),
      inputs[0]->getDataType().getType()));
  CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
      &bTensor, bShape.size(), bShape.data(), bStride.data(),
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
