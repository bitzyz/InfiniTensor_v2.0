#include "core/runtime.h"
#include "operators/LogSoftmax.h"
#include "gtest/gtest.h"

namespace infini {

class LogSoftmaxBasicTest : public testing::Test {
  protected:
    Runtime runtime;
    Graph graph;
    void SetUp() override {
        runtime = make_ref<RuntimeObj>();
        graph = make_ref<GraphObj>(runtime);
    }
};

TEST_F(LogSoftmaxBasicTest, BasicConstruction) {
    auto x = graph->addTensor({2, 10}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<LogSoftmaxObj>(x, nullptr);
    EXPECT_EQ(op->getOpType(), OpType::LogSoftmax);
    EXPECT_EQ(op->getNumInputs(), 1);
    EXPECT_EQ(op->getNumOutputs(), 1);
}

TEST_F(LogSoftmaxBasicTest, ShapeInference) {
    auto x = graph->addTensor({3, 5, 7}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<LogSoftmaxObj>(x, nullptr);
    auto shapes = op->inferShape();
    ASSERT_TRUE(shapes.has_value());
    auto vals = (*shapes)[0]->getConstantValue();
    ASSERT_EQ(vals.size(), 3u);
    EXPECT_EQ(vals[0], 3);
    EXPECT_EQ(vals[1], 5);
    EXPECT_EQ(vals[2], 7);
}

TEST_F(LogSoftmaxBasicTest, DataTypeInference) {
    auto x = graph->addTensor({2, 10}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<LogSoftmaxObj>(x, nullptr);
    auto dtypes = op->inferDataType();
    ASSERT_EQ(dtypes.size(), 1u);
    EXPECT_EQ(dtypes[0], DataType(INFINI_DTYPE_F32));
}

// Symbolic shape inference: output carries symbolic dimensions unchanged
TEST_F(LogSoftmaxBasicTest, SymbolicShapeInference) {
    auto batch = ExprObj::variable("batch");
    auto vocab = ExprObj::variable("vocab");
    auto shapeX = ShapeExpr(new ShapeExprObj({batch, vocab}));
    auto x = graph->addTensor(shapeX, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<LogSoftmaxObj>(x, nullptr);
    auto shapes = op->inferShape();
    ASSERT_TRUE(shapes.has_value());
    auto outShape = (*shapes)[0];
    EXPECT_FALSE(outShape->isConcrete());
    EXPECT_EQ(outShape->size(), 2u);
    EXPECT_EQ(outShape->toString(), "[batch, vocab]");
}

} // namespace infini
