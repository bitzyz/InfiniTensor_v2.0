#include "core/runtime.h"
#include "operators/Softmax.h"
#include "gtest/gtest.h"

namespace infini {

class SoftmaxOpTest : public testing::Test {
  protected:
    Runtime runtime;
    Graph graph;

    void SetUp() override {
        runtime = make_ref<RuntimeObj>();
        graph = make_ref<GraphObj>(runtime);
    }
};

TEST_F(SoftmaxOpTest, BasicConstruction) {
    auto input = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto softmax = graph->addOp<SoftmaxObj>(input, nullptr, 1);
    EXPECT_EQ(softmax->getOpType(), OpType::Softmax);
    EXPECT_EQ(softmax->getNumInputs(), 1);
    EXPECT_EQ(softmax->getNumOutputs(), 1);
    EXPECT_EQ(softmax->getAxis(), 1);
}

TEST_F(SoftmaxOpTest, ShapeInference) {
    auto input = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto softmax = graph->addOp<SoftmaxObj>(input, nullptr, -1);

    auto inferredShapes = softmax->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());
    ASSERT_EQ(inferredShapes->size(), 1);

    auto outputShape = (*inferredShapes)[0];
    EXPECT_TRUE(outputShape->isConcrete());
    auto shapeValues = outputShape->getConstantValue();
    EXPECT_EQ(shapeValues, Shape({2, 3, 4}));
}

class LogSoftmaxOpTest : public testing::Test {
  protected:
    Runtime runtime;
    Graph graph;

    void SetUp() override {
        runtime = make_ref<RuntimeObj>();
        graph = make_ref<GraphObj>(runtime);
    }
};

TEST_F(LogSoftmaxOpTest, BasicConstruction) {
    auto input = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto softmax = graph->addOp<LogSoftmaxObj>(input, nullptr, 1);
    EXPECT_EQ(softmax->getOpType(), OpType::LogSoftmax);
    EXPECT_EQ(softmax->getNumInputs(), 1);
    EXPECT_EQ(softmax->getNumOutputs(), 1);
    EXPECT_EQ(softmax->getAxis(), 1);
}

TEST_F(LogSoftmaxOpTest, ShapeInference) {
    auto input = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto softmax = graph->addOp<LogSoftmaxObj>(input, nullptr, -1);

    auto inferredShapes = softmax->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());
    ASSERT_EQ(inferredShapes->size(), 1);

    auto outputShape = (*inferredShapes)[0];
    EXPECT_TRUE(outputShape->isConcrete());
    auto shapeValues = outputShape->getConstantValue();
    EXPECT_EQ(shapeValues, Shape({2, 3, 4}));
}

} // namespace infini
