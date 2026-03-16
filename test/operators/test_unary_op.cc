#include "core/runtime.h"
#include "operators/Unary.h"
#include "gtest/gtest.h"

namespace infini {

class UnaryOpTest : public testing::Test {
  protected:
    Runtime runtime;
    Graph graph;

    void SetUp() override {
        runtime = make_ref<RuntimeObj>();
        graph = make_ref<GraphObj>(runtime);
    }
};

TEST_F(UnaryOpTest, BasicConstruction) {
    auto input = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto relu = graph->addOp<ReluObj>(input, nullptr);
    EXPECT_EQ(relu->getOpType(), OpType::Relu);
    EXPECT_EQ(relu->getNumInputs(), 1);
    EXPECT_EQ(relu->getNumOutputs(), 1);
}

TEST_F(UnaryOpTest, ShapeInference) {
    auto input = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto relu = graph->addOp<ReluObj>(input, nullptr);

    auto inferredShapes = relu->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());
    ASSERT_EQ(inferredShapes->size(), 1);

    auto outputShape = (*inferredShapes)[0];
    EXPECT_TRUE(outputShape->isConcrete());
    auto shapeValues = outputShape->getConstantValue();
    EXPECT_EQ(shapeValues, Shape({2, 3, 4}));
}

TEST_F(UnaryOpTest, OtherUnaryOps) {
    auto input = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    
    auto sigmoid = graph->addOp<SigmoidObj>(input, nullptr);
    EXPECT_EQ(sigmoid->getOpType(), OpType::Sigmoid);

    auto tanh = graph->addOp<TanhObj>(input, nullptr);
    EXPECT_EQ(tanh->getOpType(), OpType::Tanh);

    auto gelu = graph->addOp<GeluObj>(input, nullptr);
    EXPECT_EQ(gelu->getOpType(), OpType::Gelu);
    
    auto silu = graph->addOp<SiluObj>(input, nullptr);
    EXPECT_EQ(silu->getOpType(), OpType::Silu);
    
    auto softplus = graph->addOp<SoftplusObj>(input, nullptr);
    EXPECT_EQ(softplus->getOpType(), OpType::Softplus);
}

} // namespace infini
