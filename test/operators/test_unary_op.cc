#include "core/runtime.h"
#include "operators/Unary.h"
#include "gtest/gtest.h"

namespace infini {

class UnaryBasicTest : public testing::Test {
  protected:
    Runtime runtime;
    Graph graph;

    void SetUp() override {
        runtime = make_ref<RuntimeObj>();
        graph = make_ref<GraphObj>(runtime);
    }
};

// Helper to test shape/dtype inference for any unary op
static void testUnaryInference(Graph g, OpType opType) {
    auto x = g->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<UnaryObj>(opType, x, nullptr);
    EXPECT_EQ(op->getOpType(), opType);
    EXPECT_EQ(op->getNumInputs(), 1);
    EXPECT_EQ(op->getNumOutputs(), 1);

    auto shapes = op->inferShape();
    ASSERT_TRUE(shapes.has_value());
    auto vals = (*shapes)[0]->getConstantValue();
    ASSERT_EQ(vals.size(), 3u);
    EXPECT_EQ(vals[0], 2);
    EXPECT_EQ(vals[1], 3);
    EXPECT_EQ(vals[2], 4);

    auto dtypes = op->inferDataType();
    ASSERT_EQ(dtypes.size(), 1u);
    EXPECT_EQ(dtypes[0], DataType(INFINI_DTYPE_F32));
}

TEST_F(UnaryBasicTest, Relu) { testUnaryInference(graph, OpType::Relu); }
TEST_F(UnaryBasicTest, Sigmoid) { testUnaryInference(graph, OpType::Sigmoid); }
TEST_F(UnaryBasicTest, Silu) { testUnaryInference(graph, OpType::Silu); }
TEST_F(UnaryBasicTest, Gelu) { testUnaryInference(graph, OpType::Gelu); }
TEST_F(UnaryBasicTest, Softplus) {
    testUnaryInference(graph, OpType::Softplus);
}
TEST_F(UnaryBasicTest, Tanh) { testUnaryInference(graph, OpType::Tanh); }

TEST_F(UnaryBasicTest, SymbolicShape) {
    auto batch = ExprObj::variable("batch");
    auto h = ExprObj::variable("h");
    auto shapeX =
        ShapeExpr(new ShapeExprObj({batch, h, ExprObj::constant(64)}));
    auto x = graph->addTensor(shapeX, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<UnaryObj>(OpType::Relu, x, nullptr);
    auto shapes = op->inferShape();
    ASSERT_TRUE(shapes.has_value());
    EXPECT_FALSE((*shapes)[0]->isConcrete());
    EXPECT_EQ((*shapes)[0]->size(), 3u);
}

} // namespace infini
