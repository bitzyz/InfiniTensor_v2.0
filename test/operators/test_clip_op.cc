#include "core/runtime.h"
#include "operators/Clip.h"
#include "gtest/gtest.h"

namespace infini {

class ClipBasicTest : public testing::Test {
  protected:
    Runtime runtime;
    Graph graph;

    void SetUp() override {
        runtime = make_ref<RuntimeObj>();
        graph = make_ref<GraphObj>(runtime);
    }
};

// Test basic construction of Clip
TEST_F(ClipBasicTest, BasicConstruction) {
    auto input = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto min_val = graph->addTensor(Shape{}, DataType(INFINI_DTYPE_F32));
    auto max_val = graph->addTensor(Shape{}, DataType(INFINI_DTYPE_F32));

    auto clip = graph->addOp<ClipObj>(input, min_val, max_val, nullptr);
    EXPECT_EQ(clip->getOpType(), OpType::Clip);
    EXPECT_EQ(clip->getNumInputs(), 3);
    EXPECT_EQ(clip->getNumOutputs(), 1);
}

// Test Clip shape inference - same shape as input
TEST_F(ClipBasicTest, ShapeInferenceSameShape) {
    auto input = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto min_val = graph->addTensor(Shape{}, DataType(INFINI_DTYPE_F32));
    auto max_val = graph->addTensor(Shape{}, DataType(INFINI_DTYPE_F32));

    auto clip = graph->addOp<ClipObj>(input, min_val, max_val, nullptr);

    auto inferredShapes = clip->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());
    ASSERT_EQ(inferredShapes->size(), 1);

    auto outputShape = (*inferredShapes)[0];
    EXPECT_TRUE(outputShape->isConcrete());

    auto shapeValues = outputShape->getConstantValue();
    EXPECT_EQ(shapeValues.size(), 3);
    EXPECT_EQ(shapeValues[0], 2);
    EXPECT_EQ(shapeValues[1], 3);
    EXPECT_EQ(shapeValues[2], 4);
}

// Test Clip shape inference - 1D tensor
TEST_F(ClipBasicTest, ShapeInference1D) {
    auto input = graph->addTensor({100}, DataType(INFINI_DTYPE_F32));
    auto min_val = graph->addTensor(Shape{}, DataType(INFINI_DTYPE_F32));
    auto max_val = graph->addTensor(Shape{}, DataType(INFINI_DTYPE_F32));

    auto clip = graph->addOp<ClipObj>(input, min_val, max_val, nullptr);

    auto inferredShapes = clip->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());

    auto outputShape = (*inferredShapes)[0];
    auto shapeValues = outputShape->getConstantValue();
    EXPECT_EQ(shapeValues.size(), 1);
    EXPECT_EQ(shapeValues[0], 100);
}

// Test Clip shape inference - 4D tensor (common deep learning shape)
TEST_F(ClipBasicTest, ShapeInference4D) {
    auto input = graph->addTensor({2, 3, 4, 5}, DataType(INFINI_DTYPE_F32));
    auto min_val = graph->addTensor(Shape{}, DataType(INFINI_DTYPE_F32));
    auto max_val = graph->addTensor(Shape{}, DataType(INFINI_DTYPE_F32));

    auto clip = graph->addOp<ClipObj>(input, min_val, max_val, nullptr);

    auto inferredShapes = clip->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());

    auto outputShape = (*inferredShapes)[0];
    auto shapeValues = outputShape->getConstantValue();
    EXPECT_EQ(shapeValues.size(), 4);
    EXPECT_EQ(shapeValues[0], 2);
    EXPECT_EQ(shapeValues[1], 3);
    EXPECT_EQ(shapeValues[2], 4);
    EXPECT_EQ(shapeValues[3], 5);
}

// Test Clip data type inference - Float32
TEST_F(ClipBasicTest, DataTypeInferenceFloat32) {
    auto input = graph->addTensor({2, 3}, DataType(INFINI_DTYPE_F32));
    auto min_val = graph->addTensor(Shape{}, DataType(INFINI_DTYPE_F32));
    auto max_val = graph->addTensor(Shape{}, DataType(INFINI_DTYPE_F32));

    auto clip = graph->addOp<ClipObj>(input, min_val, max_val, nullptr);

    auto inferredTypes = clip->inferDataType();
    ASSERT_EQ(inferredTypes.size(), 1);
    EXPECT_EQ(inferredTypes[0], DataType(INFINI_DTYPE_F32));
}

// Test Clip data type inference - Float64
TEST_F(ClipBasicTest, DataTypeInferenceFloat64) {
    auto input = graph->addTensor({2, 3}, DataType(INFINI_DTYPE_F64));
    auto min_val = graph->addTensor(Shape{}, DataType(INFINI_DTYPE_F64));
    auto max_val = graph->addTensor(Shape{}, DataType(INFINI_DTYPE_F64));

    auto clip = graph->addOp<ClipObj>(input, min_val, max_val, nullptr);

    auto inferredTypes = clip->inferDataType();
    ASSERT_EQ(inferredTypes.size(), 1);
    EXPECT_EQ(inferredTypes[0], DataType(INFINI_DTYPE_F64));
}

// Test Clip data type inference - Float16
TEST_F(ClipBasicTest, DataTypeInferenceFloat16) {
    auto input = graph->addTensor({2, 3}, DataType(INFINI_DTYPE_F16));
    auto min_val = graph->addTensor(Shape{}, DataType(INFINI_DTYPE_F16));
    auto max_val = graph->addTensor(Shape{}, DataType(INFINI_DTYPE_F16));

    auto clip = graph->addOp<ClipObj>(input, min_val, max_val, nullptr);

    auto inferredTypes = clip->inferDataType();
    ASSERT_EQ(inferredTypes.size(), 1);
    EXPECT_EQ(inferredTypes[0], DataType(INFINI_DTYPE_F16));
}

// Test symbolic shape inference
TEST_F(ClipBasicTest, SymbolicShapeInference) {
    auto batch = ExprObj::variable("batch");
    auto height = ExprObj::variable("h");
    auto width = ExprObj::constant(256);

    auto shapeInput = ShapeExpr(new ShapeExprObj({batch, height, width}));

    auto input = graph->addTensor(shapeInput, DataType(INFINI_DTYPE_F32));
    auto min_val = graph->addTensor(Shape{}, DataType(INFINI_DTYPE_F32));
    auto max_val = graph->addTensor(Shape{}, DataType(INFINI_DTYPE_F32));

    auto clip = graph->addOp<ClipObj>(input, min_val, max_val, nullptr);

    auto inferredShapes = clip->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());

    auto outputShape = (*inferredShapes)[0];
    EXPECT_FALSE(outputShape->isConcrete());
    EXPECT_EQ(outputShape->size(), 3);
    EXPECT_EQ(outputShape->toString(), "[batch, h, 256]");
}

} // namespace infini