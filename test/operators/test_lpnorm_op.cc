#include "core/runtime.h"
#include "operators/LpNorm.h"
#include "gtest/gtest.h"

namespace infini {

class LpNormOpTest : public testing::Test {
  protected:
    Runtime runtime;
    Graph graph;

    void SetUp() override {
        runtime = make_ref<RuntimeObj>();
        graph = make_ref<GraphObj>(runtime);
    }
};

TEST_F(LpNormOpTest, BasicConstruction) {
    auto input = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto lpnorm = graph->addOp<LpNormObj>(input, nullptr, 2.0f, std::vector<int>{1}, false);
    EXPECT_EQ(lpnorm->getOpType(), OpType::LpNorm);
    EXPECT_EQ(lpnorm->getNumInputs(), 1);
    EXPECT_EQ(lpnorm->getNumOutputs(), 1);
    EXPECT_EQ(lpnorm->getP(), 2.0f);
    EXPECT_EQ(lpnorm->getDims(), std::vector<int>{1});
    EXPECT_EQ(lpnorm->getKeepDim(), false);
}

TEST_F(LpNormOpTest, ShapeInferenceKeepDim) {
    auto input = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto lpnorm = graph->addOp<LpNormObj>(input, nullptr, 2.0f, std::vector<int>{1}, true);

    auto inferredShapes = lpnorm->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());
    ASSERT_EQ(inferredShapes->size(), 1);

    auto outputShape = (*inferredShapes)[0];
    EXPECT_TRUE(outputShape->isConcrete());
    auto shapeValues = outputShape->getConstantValue();
    EXPECT_EQ(shapeValues, Shape({2, 1, 4}));
}

TEST_F(LpNormOpTest, ShapeInferenceNoKeepDim) {
    auto input = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto lpnorm = graph->addOp<LpNormObj>(input, nullptr, 2.0f, std::vector<int>{1}, false);

    auto inferredShapes = lpnorm->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());
    ASSERT_EQ(inferredShapes->size(), 1);

    auto outputShape = (*inferredShapes)[0];
    EXPECT_TRUE(outputShape->isConcrete());
    auto shapeValues = outputShape->getConstantValue();
    EXPECT_EQ(shapeValues, Shape({2, 4}));
}

} // namespace infini
