#include "core/runtime.h"
#include "operators/Conv.h"
#include "gtest/gtest.h"

namespace infini {

class ConvOpTest : public testing::Test {
  protected:
    Runtime runtime;
    Graph graph;

    void SetUp() override {
        runtime = make_ref<RuntimeObj>();
        graph = make_ref<GraphObj>(runtime);
    }
};

TEST_F(ConvOpTest, BasicConstruction) {
    auto input = graph->addTensor({1, 3, 224, 224}, DataType(INFINI_DTYPE_F32));
    auto weight = graph->addTensor({64, 3, 7, 7}, DataType(INFINI_DTYPE_F32));
    // ConvObj(GraphObj *graph, Tensor input, Tensor weight, Tensor output,
    //         std::vector<int> pads, std::vector<int> strides,
    //         std::vector<int> dilations, Tensor bias = nullptr);
    auto conv = graph->addOp<ConvObj>(input, weight, nullptr, 
                                      std::vector<int>{3, 3}, 
                                      std::vector<int>{2, 2}, 
                                      std::vector<int>{1, 1}, 
                                      nullptr);
    EXPECT_EQ(conv->getOpType(), OpType::Conv);
    EXPECT_EQ(conv->getNumInputs(), 2);
    EXPECT_EQ(conv->getNumOutputs(), 1);
}

TEST_F(ConvOpTest, ShapeInference) {
    auto input = graph->addTensor({1, 3, 224, 224}, DataType(INFINI_DTYPE_F32));
    auto weight = graph->addTensor({64, 3, 7, 7}, DataType(INFINI_DTYPE_F32));
    // pad=3, stride=2, dilation=1
    // H_out = (224 + 2*3 - 1*(7-1) - 1)/2 + 1 = (224 + 6 - 7)/2 + 1 = 223/2 + 1 = 111 + 1 = 112
    auto conv = graph->addOp<ConvObj>(input, weight, nullptr, 
                                      std::vector<int>{3, 3}, 
                                      std::vector<int>{2, 2}, 
                                      std::vector<int>{1, 1}, 
                                      nullptr);

    auto inferredShapes = conv->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());
    ASSERT_EQ(inferredShapes->size(), 1);

    auto outputShape = (*inferredShapes)[0];
    EXPECT_TRUE(outputShape->isConcrete());
    auto shapeValues = outputShape->getConstantValue();
    EXPECT_EQ(shapeValues, Shape({1, 64, 112, 112}));
}

} // namespace infini
