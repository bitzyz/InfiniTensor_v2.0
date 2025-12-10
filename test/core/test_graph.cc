#include "core/graph.h"
#include "core/runtime.h"
#include "operators/Gemm.h"
#include "gtest/gtest.h"

namespace infini {
class GraphBasicTest : public testing::Test {
protected:
  Runtime runtime;

  void SetUp() override { runtime = make_ref<RuntimeObj>(); }
};

// 测试Graph构造和运行时获取
TEST_F(GraphBasicTest, GraphConstruction) {
  auto graph = make_ref<GraphObj>(runtime);

  EXPECT_EQ(graph->getRuntime(), runtime);
  EXPECT_TRUE(graph->getTensors().empty());
  EXPECT_TRUE(graph->getOperators().empty());
}

// 测试添加常量形状Tensor
TEST_F(GraphBasicTest, AddConcreteTensor) {
  auto graph = make_ref<GraphObj>(runtime);

  auto tensor1 = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
  auto tensor2 = graph->addTensor({5, 6}, DataType(INFINI_DTYPE_F16));

  EXPECT_EQ(graph->getTensors().size(), 2);
  EXPECT_EQ(tensor1->getRank(), 3);
  EXPECT_EQ(tensor2->getRank(), 2);
  EXPECT_EQ(tensor1->getDataType(), DataType(INFINI_DTYPE_F32));
  EXPECT_EQ(tensor2->getDataType(), DataType(INFINI_DTYPE_F16));
}

// 测试添加符号形状Tensor
TEST_F(GraphBasicTest, AddSymbolicTensor) {
  auto graph = make_ref<GraphObj>(runtime);

  auto batch = ExprObj::variable("batch");
  auto height = ExprObj::constant(224);
  auto width = ExprObj::constant(224);

  auto shapeExpr = ShapeExpr(new ShapeExprObj({batch, height, width}));
  auto tensor = graph->addTensor(shapeExpr, DataType(INFINI_DTYPE_F32));

  EXPECT_EQ(graph->getTensors().size(), 1);
  EXPECT_EQ(tensor->getRank(), 3);
  EXPECT_FALSE(tensor->getShape()->isConcrete());
  EXPECT_TRUE(tensor->getShape()->isDynamic());
}

// 测试添加带步长的Tensor
TEST_F(GraphBasicTest, AddTensorWithStride) {
  auto graph = make_ref<GraphObj>(runtime);

  // 常量步长
  auto tensor1 =
      graph->addTensor({2, 3, 4}, {12, 4, 1}, DataType(INFINI_DTYPE_F32));
  EXPECT_TRUE(tensor1->getStride()->isConcrete());

  // 步长表达式
  auto strideExpr = StrideExpr(new StrideExprObj(
      {ExprObj::constant(12), ExprObj::constant(4), ExprObj::constant(1)}));
  auto tensor2 =
      graph->addTensor({2, 3, 4}, strideExpr, DataType(INFINI_DTYPE_F32));
  EXPECT_TRUE(tensor2->getStride()->isConcrete());
}

// 测试批量添加Tensor
TEST_F(GraphBasicTest, AddTensorVector) {
  auto graph = make_ref<GraphObj>(runtime);

  TensorVec tensors = {
      make_ref<TensorObj>(Shape{2, 3}, DataType(INFINI_DTYPE_F32)),
      make_ref<TensorObj>(Shape{4, 5}, DataType(INFINI_DTYPE_F16)),
      make_ref<TensorObj>(Shape{6, 7, 8}, DataType(INFINI_DTYPE_F32))};

  auto added = graph->addTensor(tensors);

  EXPECT_EQ(graph->getTensors().size(), 3);
  EXPECT_EQ(added.size(), 3);
  EXPECT_EQ(added[0], tensors[0]);
  EXPECT_EQ(added[1], tensors[1]);
  EXPECT_EQ(added[2], tensors[2]);
}

// 测试移除Tensor和Operator
TEST_F(GraphBasicTest, RemoveTensorAndOperator) {
  auto graph = make_ref<GraphObj>(runtime);

  // 创建Tensor和Operator
  auto A = graph->addTensor({2, 3}, DataType(INFINI_DTYPE_F32));
  auto B = graph->addTensor({3, 4}, DataType(INFINI_DTYPE_F32));
  auto Y = graph->addTensor({1, 2, 4}, DataType(INFINI_DTYPE_F32));
  auto gemm = graph->addOpWithOutputs<GemmObj>(A, B, Y, nullptr);

  EXPECT_EQ(graph->getTensors().size(), 3);
  EXPECT_EQ(graph->getOperators().size(), 1);

  // 移除Tensor
  graph->removeTensor(A);
  EXPECT_EQ(graph->getTensors().size(), 2);

  // 移除Operator
  graph->removeOperator(gemm);
  EXPECT_EQ(graph->getOperators().size(), 0);
}
} // namespace infini
