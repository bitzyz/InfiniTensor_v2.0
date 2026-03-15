#include "core/graph.h"
#include "core/runtime.h"
#include "operators/Clip.h"
#include "operators/ElementWise.h"
#include "operators/Softmax.h"
#include "operators/Unary.h"
#include "gtest/gtest.h"
#include <limits>

namespace infini {
class GraphOptimizerTest : public testing::Test {
  protected:
    Runtime runtime;

    void SetUp() override { runtime = make_ref<RuntimeObj>(); }

    Tensor makeScalar(Graph graph, float value) {
        auto tensor = graph->addTensor(Shape{}, DataType(INFINI_DTYPE_F32));
        auto *buffer = new float[1];
        buffer[0] = value;
        tensor->setData(buffer);
        return tensor;
    }
};

TEST_F(GraphOptimizerTest, IdentityEliminationPreservesOutput) {
    auto graph = make_ref<GraphObj>(runtime);
    auto input = graph->addTensor({2, 2}, DataType(INFINI_DTYPE_F32));
    auto zero = makeScalar(graph, 0.0f);
    auto add = graph->addOp<ElementWiseObj>(OpType::Add, input, zero, nullptr);
    graph->setOutputs({add->getOutput(0)});

    graph->optimize();

    EXPECT_EQ(graph->getOperators().size(), 0);
    ASSERT_EQ(graph->getOutputs().size(), 1);
    EXPECT_EQ(graph->getOutputs()[0], input);
    EXPECT_TRUE(graph->checkValid());
}

TEST_F(GraphOptimizerTest, ConstantFoldAddProducesConstantOutput) {
    auto graph = make_ref<GraphObj>(runtime);
    auto lhs = makeScalar(graph, 2.0f);
    auto rhs = makeScalar(graph, 3.0f);
    auto add = graph->addOp<ElementWiseObj>(OpType::Add, lhs, rhs, nullptr);
    auto output = add->getOutput(0);
    graph->setOutputs({output});

    graph->optimize();

    EXPECT_EQ(graph->getOperators().size(), 0);
    EXPECT_TRUE(output->isConstant());
    ASSERT_NE(output->getData(), nullptr);
    EXPECT_FLOAT_EQ(output->getRawDataPtr<float *>()[0], 5.0f);
    EXPECT_TRUE(graph->checkValid());
}

TEST_F(GraphOptimizerTest, DeadCodeEliminationDropsUnusedBranch) {
    auto graph = make_ref<GraphObj>(runtime);
    auto input = graph->addTensor({2, 2}, DataType(INFINI_DTYPE_F32));
    auto one = makeScalar(graph, 1.0f);
    auto two = makeScalar(graph, 2.0f);
    auto live = graph->addOp<ElementWiseObj>(OpType::Mul, input, one, nullptr);
    auto dead = graph->addOp<ElementWiseObj>(OpType::Mul, input, two, nullptr);
    auto deadOutputFuid = dead->getOutput(0)->getFuid();
    graph->setOutputs({live->getOutput(0)});

    graph->optimize();

    EXPECT_EQ(graph->getOperators().size(), 0);
    ASSERT_EQ(graph->getOutputs().size(), 1);
    EXPECT_EQ(graph->getOutputs()[0], input);
    EXPECT_EQ(graph->getTensor(deadOutputFuid), nullptr);
    EXPECT_TRUE(graph->checkValid());
}

TEST_F(GraphOptimizerTest, ConstantFoldReluProducesConstantOutput) {
    auto graph = make_ref<GraphObj>(runtime);
    auto input = makeScalar(graph, -2.5f);
    auto relu = graph->addOp<UnaryObj>(OpType::Relu, input, nullptr);
    auto output = relu->getOutput(0);
    graph->setOutputs({output});

    graph->optimize();

    EXPECT_EQ(graph->getOperators().size(), 0);
    EXPECT_TRUE(output->isConstant());
    ASSERT_NE(output->getData(), nullptr);
    EXPECT_FLOAT_EQ(output->getRawDataPtr<float *>()[0], 0.0f);
    EXPECT_TRUE(graph->checkValid());
}

TEST_F(GraphOptimizerTest, ConstantFoldClipProducesConstantOutput) {
    auto graph = make_ref<GraphObj>(runtime);
    auto input = makeScalar(graph, 3.0f);
    auto minVal = makeScalar(graph, 0.0f);
    auto maxVal = makeScalar(graph, 1.0f);
    auto clip = graph->addOp<ClipObj>(input, minVal, maxVal, nullptr);
    auto output = clip->getOutput(0);
    graph->setOutputs({output});

    graph->optimize();

    EXPECT_EQ(graph->getOperators().size(), 0);
    EXPECT_TRUE(output->isConstant());
    ASSERT_NE(output->getData(), nullptr);
    EXPECT_FLOAT_EQ(output->getRawDataPtr<float *>()[0], 1.0f);
    EXPECT_TRUE(graph->checkValid());
}

TEST_F(GraphOptimizerTest, IdentityEliminationDivByOnePreservesOutput) {
    auto graph = make_ref<GraphObj>(runtime);
    auto input = graph->addTensor({2, 2}, DataType(INFINI_DTYPE_F32));
    auto one = makeScalar(graph, 1.0f);
    auto div = graph->addOp<ElementWiseObj>(OpType::Div, input, one, nullptr);
    graph->setOutputs({div->getOutput(0)});

    graph->optimize();

    EXPECT_EQ(graph->getOperators().size(), 0);
    ASSERT_EQ(graph->getOutputs().size(), 1);
    EXPECT_EQ(graph->getOutputs()[0], input);
    EXPECT_TRUE(graph->checkValid());
}

TEST_F(GraphOptimizerTest,
       IdentityEliminationClipInfiniteBoundsPreservesOutput) {
    auto graph = make_ref<GraphObj>(runtime);
    auto input = graph->addTensor({2, 2}, DataType(INFINI_DTYPE_F32));
    auto negInf = makeScalar(graph, -std::numeric_limits<float>::infinity());
    auto posInf = makeScalar(graph, std::numeric_limits<float>::infinity());
    auto clip = graph->addOp<ClipObj>(input, negInf, posInf, nullptr);
    graph->setOutputs({clip->getOutput(0)});

    graph->optimize();

    EXPECT_EQ(graph->getOperators().size(), 0);
    ASSERT_EQ(graph->getOutputs().size(), 1);
    EXPECT_EQ(graph->getOutputs()[0], input);
    EXPECT_TRUE(graph->checkValid());
}

TEST_F(GraphOptimizerTest, IdempotenceEliminationReluRelu) {
    auto graph = make_ref<GraphObj>(runtime);
    auto input = graph->addTensor({2, 2}, DataType(INFINI_DTYPE_F32));
    auto relu1 = graph->addOp<UnaryObj>(OpType::Relu, input, nullptr);
    auto relu2 =
        graph->addOp<UnaryObj>(OpType::Relu, relu1->getOutput(0), nullptr);
    graph->setOutputs({relu2->getOutput(0)});

    graph->optimize();

    EXPECT_EQ(graph->getOperators().size(), 1);
    ASSERT_EQ(graph->getOutputs().size(), 1);
    EXPECT_EQ(graph->getOutputs()[0], relu1->getOutput(0));
    EXPECT_TRUE(graph->checkValid());
}

TEST_F(GraphOptimizerTest, IdempotenceEliminationClipClipSameBounds) {
    auto graph = make_ref<GraphObj>(runtime);
    auto input = graph->addTensor({2, 2}, DataType(INFINI_DTYPE_F32));
    auto minVal = makeScalar(graph, -1.0f);
    auto maxVal = makeScalar(graph, 1.0f);
    auto clip1 = graph->addOp<ClipObj>(input, minVal, maxVal, nullptr);
    auto clip2 =
        graph->addOp<ClipObj>(clip1->getOutput(0), minVal, maxVal, nullptr);
    graph->setOutputs({clip2->getOutput(0)});

    graph->optimize();

    EXPECT_EQ(graph->getOperators().size(), 1);
    ASSERT_EQ(graph->getOutputs().size(), 1);
    EXPECT_EQ(graph->getOutputs()[0], clip1->getOutput(0));
    EXPECT_TRUE(graph->checkValid());
}

TEST_F(GraphOptimizerTest, SoftmaxChainIsNotEliminated) {
    auto graph = make_ref<GraphObj>(runtime);
    auto input = graph->addTensor({2, 2}, DataType(INFINI_DTYPE_F32));
    auto softmax1 = graph->addOp<SoftmaxObj>(input, nullptr, -1);
    auto softmax2 =
        graph->addOp<SoftmaxObj>(softmax1->getOutput(0), nullptr, -1);
    graph->setOutputs({softmax2->getOutput(0)});

    graph->optimize();

    EXPECT_EQ(graph->getOperators().size(), 2);
    ASSERT_EQ(graph->getOutputs().size(), 1);
    EXPECT_EQ(graph->getOutputs()[0], softmax2->getOutput(0));
    EXPECT_TRUE(graph->checkValid());
}
} // namespace infini
