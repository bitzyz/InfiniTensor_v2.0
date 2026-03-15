#include "core/runtime.h"
#include "operators/Softmax.h"
#include "gtest/gtest.h"

namespace infini {

class SoftmaxBasicTest : public testing::Test {
  protected:
    Runtime runtime;
    Graph graph;
    void SetUp() override {
        runtime = make_ref<RuntimeObj>();
        graph = make_ref<GraphObj>(runtime);
    }
};

TEST_F(SoftmaxBasicTest, BasicConstruction) {
    auto x = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<SoftmaxObj>(x, nullptr, 1);
    EXPECT_EQ(op->getOpType(), OpType::Softmax);
    EXPECT_EQ(op->getAxis(), 1);
    EXPECT_EQ(op->getNumInputs(), 1);
    EXPECT_EQ(op->getNumOutputs(), 1);
}

TEST_F(SoftmaxBasicTest, ShapeInference) {
    auto x = graph->addTensor({4, 8}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<SoftmaxObj>(x, nullptr, -1);
    auto shapes = op->inferShape();
    ASSERT_TRUE(shapes.has_value());
    auto vals = (*shapes)[0]->getConstantValue();
    ASSERT_EQ(vals.size(), 2u);
    EXPECT_EQ(vals[0], 4);
    EXPECT_EQ(vals[1], 8);
}

TEST_F(SoftmaxBasicTest, DataTypeInference) {
    auto x = graph->addTensor({4, 8}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<SoftmaxObj>(x, nullptr, 1);
    auto dtypes = op->inferDataType();
    ASSERT_EQ(dtypes.size(), 1u);
    EXPECT_EQ(dtypes[0], DataType(INFINI_DTYPE_F32));
}

// Symbolic shape inference: output carries symbolic dimensions unchanged
TEST_F(SoftmaxBasicTest, SymbolicShapeInference) {
    auto batch = ExprObj::variable("batch");
    auto seq = ExprObj::variable("seq");
    auto shapeX =
        ShapeExpr(new ShapeExprObj({batch, seq, ExprObj::constant(64)}));
    auto x = graph->addTensor(shapeX, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<SoftmaxObj>(x, nullptr, -1);
    auto shapes = op->inferShape();
    ASSERT_TRUE(shapes.has_value());
    auto outShape = (*shapes)[0];
    EXPECT_FALSE(outShape->isConcrete());
    EXPECT_EQ(outShape->size(), 3u);
    EXPECT_EQ(outShape->toString(), "[batch, seq, 64]");
}

} // namespace infini
