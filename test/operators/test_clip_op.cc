#include "core/runtime.h"
#include "operators/ElementWise.h"
#include "gtest/gtest.h"

namespace infini {

class ClipOpTest : public testing::Test {
  protected:
    Runtime runtime;
    Graph graph;

    void SetUp() override {
        runtime = make_ref<RuntimeObj>();
        graph = make_ref<GraphObj>(runtime);
    }
};

TEST_F(ClipOpTest, BasicConstruction) {
    auto input = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto min = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));
    auto max = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));
    
    // ElementWiseObj(GraphObj *graph, OpType type, Tensor input, Tensor min, Tensor max, Tensor output);
    auto clip = graph->addOp<ElementWiseObj>(OpType::Clip, input, min, max, nullptr);
    
    EXPECT_EQ(clip->getOpType(), OpType::Clip);
    EXPECT_EQ(clip->getNumInputs(), 3);
    EXPECT_EQ(clip->getNumOutputs(), 1);
}

TEST_F(ClipOpTest, ShapeInference) {
    auto input = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto min = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));
    auto max = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));
    
    auto clip = graph->addOp<ElementWiseObj>(OpType::Clip, input, min, max, nullptr);

    auto inferredShapes = clip->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());
    ASSERT_EQ(inferredShapes->size(), 1);

    auto outputShape = (*inferredShapes)[0];
    EXPECT_TRUE(outputShape->isConcrete());
    auto shapeValues = outputShape->getConstantValue();
    EXPECT_EQ(shapeValues, Shape({2, 3, 4}));
}

TEST_F(ClipOpTest, DataTypeInference) {
    auto input = graph->addTensor({2, 3}, DataType(INFINI_DTYPE_F32));
    auto min = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));
    auto max = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));
    
    auto clip = graph->addOp<ElementWiseObj>(OpType::Clip, input, min, max, nullptr);

    auto inferredTypes = clip->inferDataType();
    ASSERT_EQ(inferredTypes.size(), 1);
    EXPECT_EQ(inferredTypes[0], DataType(INFINI_DTYPE_F32));
}

} // namespace infini
