#include "core/runtime.h"
#include "operators/RMSNorm.h"
#include "gtest/gtest.h"

namespace infini {

class RMSNormBasicTest : public testing::Test {
  protected:
    Runtime runtime;
    Graph graph;
    void SetUp() override {
        runtime = make_ref<RuntimeObj>();
        graph = make_ref<GraphObj>(runtime);
    }
};

TEST_F(RMSNormBasicTest, BasicConstruction) {
    auto x = graph->addTensor({2, 8}, DataType(INFINI_DTYPE_F32));
    auto w = graph->addTensor({8}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<RMSNormObj>(x, w, nullptr, 1e-6f);
    EXPECT_EQ(op->getOpType(), OpType::RMSNorm);
    EXPECT_EQ(op->getNumInputs(), 2);
    EXPECT_EQ(op->getNumOutputs(), 1);
    EXPECT_FLOAT_EQ(op->getEpsilon(), 1e-6f);
}

TEST_F(RMSNormBasicTest, ShapeInference) {
    auto x = graph->addTensor({4, 16}, DataType(INFINI_DTYPE_F32));
    auto w = graph->addTensor({16}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<RMSNormObj>(x, w, nullptr);
    auto shapes = op->inferShape();
    ASSERT_TRUE(shapes.has_value());
    auto vals = (*shapes)[0]->getConstantValue();
    ASSERT_EQ(vals.size(), 2u);
    EXPECT_EQ(vals[0], 4);
    EXPECT_EQ(vals[1], 16);
}

TEST_F(RMSNormBasicTest, DataTypeInference) {
    auto x = graph->addTensor({4, 16}, DataType(INFINI_DTYPE_F32));
    auto w = graph->addTensor({16}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<RMSNormObj>(x, w, nullptr);
    auto dtypes = op->inferDataType();
    ASSERT_EQ(dtypes.size(), 1u);
    EXPECT_EQ(dtypes[0], DataType(INFINI_DTYPE_F32));
}

// 3D input shape inference
TEST_F(RMSNormBasicTest, ShapeInference3D) {
    auto x = graph->addTensor({2, 4, 16}, DataType(INFINI_DTYPE_F32));
    auto w = graph->addTensor({16}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<RMSNormObj>(x, w, nullptr);
    auto shapes = op->inferShape();
    ASSERT_TRUE(shapes.has_value());
    auto vals = (*shapes)[0]->getConstantValue();
    ASSERT_EQ(vals.size(), 3u);
    EXPECT_EQ(vals[0], 2);
    EXPECT_EQ(vals[1], 4);
    EXPECT_EQ(vals[2], 16);
}

// Symbolic shape inference: batch dim can be symbolic
TEST_F(RMSNormBasicTest, SymbolicShapeInference) {
    auto batch = ExprObj::variable("batch");
    auto seq = ExprObj::variable("seq");
    auto shapeX =
        ShapeExpr(new ShapeExprObj({batch, seq, ExprObj::constant(32)}));
    auto x = graph->addTensor(shapeX, DataType(INFINI_DTYPE_F32));
    auto w = graph->addTensor({32}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<RMSNormObj>(x, w, nullptr);
    auto shapes = op->inferShape();
    ASSERT_TRUE(shapes.has_value());
    auto outShape = (*shapes)[0];
    EXPECT_FALSE(outShape->isConcrete());
    EXPECT_EQ(outShape->size(), 3u);
    EXPECT_EQ(outShape->toString(), "[batch, seq, 32]");
}

} // namespace infini
