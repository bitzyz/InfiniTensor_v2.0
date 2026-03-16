#include "core/runtime.h"
#include "operators/RMSNorm.h"
#include "gtest/gtest.h"

namespace infini {

class RMSNormOpTest : public testing::Test {
  protected:
    Runtime runtime;
    Graph graph;

    void SetUp() override {
        runtime = make_ref<RuntimeObj>();
        graph = make_ref<GraphObj>(runtime);
    }
};

TEST_F(RMSNormOpTest, BasicConstruction) {
    auto input = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto weight = graph->addTensor({4}, DataType(INFINI_DTYPE_F32));
    auto rmsnorm = graph->addOp<RMSNormObj>(input, weight, nullptr, 1e-6);
    EXPECT_EQ(rmsnorm->getOpType(), OpType::RMSNorm);
    EXPECT_EQ(rmsnorm->getNumInputs(), 2);
    EXPECT_EQ(rmsnorm->getNumOutputs(), 1);
    EXPECT_EQ(rmsnorm->getEps(), 1e-6f);
}

TEST_F(RMSNormOpTest, ShapeInference) {
    auto input = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto weight = graph->addTensor({4}, DataType(INFINI_DTYPE_F32));
    auto rmsnorm = graph->addOp<RMSNormObj>(input, weight, nullptr, 1e-6);

    auto inferredShapes = rmsnorm->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());
    ASSERT_EQ(inferredShapes->size(), 1);

    auto outputShape = (*inferredShapes)[0];
    EXPECT_TRUE(outputShape->isConcrete());
    auto shapeValues = outputShape->getConstantValue();
    EXPECT_EQ(shapeValues, Shape({2, 3, 4}));
}

} // namespace infini
