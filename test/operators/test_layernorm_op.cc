#include "core/runtime.h"
#include "operators/LayerNorm.h"
#include "gtest/gtest.h"

namespace infini {

class LayerNormOpTest : public testing::Test {
  protected:
    Runtime runtime;
    Graph graph;

    void SetUp() override {
        runtime = make_ref<RuntimeObj>();
        graph = make_ref<GraphObj>(runtime);
    }
};

TEST_F(LayerNormOpTest, BasicConstruction) {
    auto input = graph->addTensor({1, 3, 224, 224}, DataType(INFINI_DTYPE_F32));
    auto scale = graph->addTensor({224}, DataType(INFINI_DTYPE_F32));
    auto bias = graph->addTensor({224}, DataType(INFINI_DTYPE_F32));
    // LayerNormObj(GraphObj *graph, Tensor input, Tensor weight, Tensor bias, Tensor output, float eps = 1e-5);
    auto layernorm = graph->addOp<LayerNormObj>(input, scale, bias, nullptr, 1e-5);
    EXPECT_EQ(layernorm->getOpType(), OpType::LayerNorm);
    EXPECT_EQ(layernorm->getNumInputs(), 3);
    EXPECT_EQ(layernorm->getNumOutputs(), 1);
}

TEST_F(LayerNormOpTest, ShapeInference) {
    auto input = graph->addTensor({1, 3, 224, 224}, DataType(INFINI_DTYPE_F32));
    auto scale = graph->addTensor({224}, DataType(INFINI_DTYPE_F32));
    auto bias = graph->addTensor({224}, DataType(INFINI_DTYPE_F32));
    auto layernorm = graph->addOp<LayerNormObj>(input, scale, bias, nullptr, 1e-5);

    auto inferredShapes = layernorm->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());
    ASSERT_EQ(inferredShapes->size(), 1);

    auto outputShape = (*inferredShapes)[0];
    EXPECT_TRUE(outputShape->isConcrete());
    auto shapeValues = outputShape->getConstantValue();
    EXPECT_EQ(shapeValues, Shape({1, 3, 224, 224}));
}

} // namespace infini
