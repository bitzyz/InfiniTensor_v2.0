#include "core/runtime.h"
#include "operators/LPNorm.h"
#include "gtest/gtest.h"

namespace infini {

class LPNormBasicTest : public testing::Test {
  protected:
    Runtime runtime;
    Graph graph;
    void SetUp() override {
        runtime = make_ref<RuntimeObj>();
        graph = make_ref<GraphObj>(runtime);
    }
};

TEST_F(LPNormBasicTest, BasicConstruction) {
    auto x = graph->addTensor({4, 8}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<LPNormObj>(x, nullptr, 1, 2, 1e-12f);
    EXPECT_EQ(op->getOpType(), OpType::LPNorm);
    EXPECT_EQ(op->getAxis(), 1);
    EXPECT_EQ(op->getP(), 2);
    EXPECT_FLOAT_EQ(op->getEps(), 1e-12f);
}

TEST_F(LPNormBasicTest, ShapeInference) {
    auto x = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<LPNormObj>(x, nullptr, 2, 1);
    auto shapes = op->inferShape();
    ASSERT_TRUE(shapes.has_value());
    auto vals = (*shapes)[0]->getConstantValue();
    ASSERT_EQ(vals.size(), 3u);
    EXPECT_EQ(vals[0], 2);
    EXPECT_EQ(vals[1], 3);
    EXPECT_EQ(vals[2], 4);
}

TEST_F(LPNormBasicTest, DataTypeInference) {
    auto x = graph->addTensor({4, 8}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<LPNormObj>(x, nullptr, 1, 2);
    auto dtypes = op->inferDataType();
    ASSERT_EQ(dtypes.size(), 1u);
    EXPECT_EQ(dtypes[0], DataType(INFINI_DTYPE_F32));
}

// L1 norm variant
TEST_F(LPNormBasicTest, L1Norm) {
    auto x = graph->addTensor({4, 8}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<LPNormObj>(x, nullptr, 1, 1);
    EXPECT_EQ(op->getP(), 1);
    EXPECT_EQ(op->getAxis(), 1);
    auto shapes = op->inferShape();
    ASSERT_TRUE(shapes.has_value());
    auto vals = (*shapes)[0]->getConstantValue();
    EXPECT_EQ(vals[0], 4);
    EXPECT_EQ(vals[1], 8);
}

// Symbolic shape inference: output carries symbolic dimensions
TEST_F(LPNormBasicTest, SymbolicShapeInference) {
    auto batch = ExprObj::variable("batch");
    auto shapeX = ShapeExpr(new ShapeExprObj({batch, ExprObj::constant(8)}));
    auto x = graph->addTensor(shapeX, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<LPNormObj>(x, nullptr, 1, 2);
    auto shapes = op->inferShape();
    ASSERT_TRUE(shapes.has_value());
    auto outShape = (*shapes)[0];
    EXPECT_FALSE(outShape->isConcrete());
    EXPECT_EQ(outShape->size(), 2u);
    EXPECT_EQ(outShape->toString(), "[batch, 8]");
}

} // namespace infini
