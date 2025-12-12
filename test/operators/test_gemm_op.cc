#include "core/runtime.h"
#include "operators/Gemm.h"
#include "gtest/gtest.h"

namespace infini {
class GemmBasicTest : public testing::Test {
  protected:
    Runtime runtime;
    Graph graph;

    void SetUp() override {
        runtime = make_ref<RuntimeObj>();
        graph = make_ref<GraphObj>(runtime);
    }
};

// 测试Gemm的基本构造
TEST_F(GemmBasicTest, BasicConstruction) {
    auto A = graph->addTensor({2, 3}, DataType(INFINI_DTYPE_F32));
    auto B = graph->addTensor({3, 4}, DataType(INFINI_DTYPE_F32));

    auto gemm =
        graph->addOp<GemmObj>(A, B, nullptr, nullptr, 1.0f, 0.0f, false, false);

    EXPECT_EQ(gemm->getOpType(), OpType::Gemm);
    EXPECT_EQ(gemm->getNumInputs(), 2);
    EXPECT_EQ(gemm->getNumOutputs(), 1);
    EXPECT_EQ(gemm->getAlpha(), 1.0f);
    EXPECT_EQ(gemm->getBeta(), 0.0f);
    EXPECT_FALSE(gemm->getTransA());
    EXPECT_FALSE(gemm->getTransB());
}

// 测试Gemm形状推导（不转置）
TEST_F(GemmBasicTest, ShapeInferenceNoTranspose) {
    auto A = graph->addTensor({2, 3}, DataType(INFINI_DTYPE_F32));
    auto B = graph->addTensor({3, 4}, DataType(INFINI_DTYPE_F32));

    auto gemm = graph->addOp<GemmObj>(A, B, nullptr, nullptr);

    auto inferredShapes = gemm->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());
    ASSERT_EQ(inferredShapes->size(), 1);

    auto outputShape = (*inferredShapes)[0];
    EXPECT_TRUE(outputShape->isConcrete());
    EXPECT_EQ(outputShape->size(), 3);

    auto shapeValues = outputShape->getConstantValue();
    EXPECT_EQ(shapeValues[0], 1); // batch
    EXPECT_EQ(shapeValues[1], 2); // M
    EXPECT_EQ(shapeValues[2], 4); // N
}

// 测试Gemm形状推导（双转置）
TEST_F(GemmBasicTest, ShapeInferenceBothTranspose) {
    auto A =
        graph->addTensor({3, 2}, DataType(INFINI_DTYPE_F32)); // A^T will be 2x3
    auto B =
        graph->addTensor({4, 3}, DataType(INFINI_DTYPE_F32)); // B^T will be 3x4

    auto gemm =
        graph->addOp<GemmObj>(A, B, nullptr, nullptr, 1.0f, 0.0f, true, true);

    auto inferredShapes = gemm->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());

    auto outputShape = (*inferredShapes)[0];
    auto shapeValues = outputShape->getConstantValue();
    EXPECT_EQ(shapeValues[0], 1); // batch
    EXPECT_EQ(shapeValues[1], 2); // M from A^T
    EXPECT_EQ(shapeValues[2], 4); // N from B^T
}

// 测试batch维度的广播
TEST_F(GemmBasicTest, ShapeInferenceBatchBroadcast) {
    // A: [1, M, K], B: [batch, K, N] -> [batch, M, N]
    auto A = graph->addTensor({1, 2, 3}, DataType(INFINI_DTYPE_F32));
    auto B = graph->addTensor({5, 3, 4}, DataType(INFINI_DTYPE_F32));

    auto gemm = graph->addOp<GemmObj>(A, B, nullptr, nullptr);

    auto inferredShapes = gemm->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());

    auto outputShape = (*inferredShapes)[0];
    auto shapeValues = outputShape->getConstantValue();
    EXPECT_EQ(shapeValues[0], 5); // broadcast batch
}

// 测试K维度匹配检查
TEST_F(GemmBasicTest, KDimensionMismatch) {
    auto A = graph->addTensor({2, 3}, DataType(INFINI_DTYPE_F32));
    auto B =
        graph->addTensor({5, 4},
                         DataType(INFINI_DTYPE_F32)); // K维度不匹配：3 != 5

    EXPECT_THROW(graph->addOp<GemmObj>(A, B, nullptr, nullptr), Exception);
}

// 测试数据类型推断
TEST_F(GemmBasicTest, DataTypeInference) {
    auto A = graph->addTensor({2, 3}, DataType(INFINI_DTYPE_F32));
    auto B = graph->addTensor({3, 4}, DataType(INFINI_DTYPE_F32));

    auto gemm = graph->addOp<GemmObj>(A, B, nullptr, nullptr);

    auto inferredTypes = gemm->inferDataType();
    ASSERT_EQ(inferredTypes.size(), 1);
    EXPECT_EQ(inferredTypes[0], DataType(INFINI_DTYPE_F32));
}

TEST_F(GemmBasicTest, SymbolicShapeInference) {
    auto batch = ExprObj::variable("batch");
    auto m = ExprObj::variable("m");
    auto k = ExprObj::constant(256);
    auto n = ExprObj::constant(512);

    auto shapeA = ShapeExpr(new ShapeExprObj({batch, m, k}));
    auto shapeB = ShapeExpr(new ShapeExprObj({batch, k, n}));
    auto shapeY = ShapeExpr(new ShapeExprObj({batch, m, n}));

    auto A = graph->addTensor(shapeA, DataType(INFINI_DTYPE_F32));
    auto B = graph->addTensor(shapeB, DataType(INFINI_DTYPE_F32));
    auto Y = graph->addTensor(shapeY, DataType(INFINI_DTYPE_F32));

    auto gemm = graph->addOpWithOutputs<GemmObj>(A, B, Y, nullptr);

    auto inferredShapes = gemm->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());

    auto outputShape = (*inferredShapes)[0];
    EXPECT_FALSE(outputShape->isConcrete());
    EXPECT_EQ(outputShape->size(), 3);
    EXPECT_EQ(outputShape->toString(), "[batch, m, 512]");
}
} // namespace infini
