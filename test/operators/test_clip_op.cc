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
    auto x = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto mn = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));
    auto mx = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));

    auto clip = graph->addOp<ClipObj>(x, mn, mx, nullptr);
    EXPECT_EQ(clip->getOpType(), OpType::Clip);
    EXPECT_EQ(clip->getNumInputs(), 3);
    EXPECT_EQ(clip->getNumOutputs(), 1);
}

// Shape inference: output shape matches input shape
TEST_F(ClipBasicTest, ShapeInference) {
    auto x = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto mn = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));
    auto mx = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));

    auto clip = graph->addOp<ClipObj>(x, mn, mx, nullptr);
    auto shapes = clip->inferShape();
    ASSERT_TRUE(shapes.has_value());
    ASSERT_EQ(shapes->size(), 1);

    auto outShape = (*shapes)[0];
    EXPECT_TRUE(outShape->isConcrete());
    auto vals = outShape->getConstantValue();
    ASSERT_EQ(vals.size(), 3u);
    EXPECT_EQ(vals[0], 2);
    EXPECT_EQ(vals[1], 3);
    EXPECT_EQ(vals[2], 4);
}

// Data type inference: output dtype matches input dtype
TEST_F(ClipBasicTest, DataTypeInference) {
    auto x = graph->addTensor({2, 3}, DataType(INFINI_DTYPE_F32));
    auto mn = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));
    auto mx = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));

    auto clip = graph->addOp<ClipObj>(x, mn, mx, nullptr);
    auto dtypes = clip->inferDataType();
    ASSERT_EQ(dtypes.size(), 1u);
    EXPECT_EQ(dtypes[0], DataType(INFINI_DTYPE_F32));
}

// Symbolic shape inference: output shape carries symbolic dims unchanged
TEST_F(ClipBasicTest, SymbolicShapeInference) {
    auto batch = ExprObj::variable("batch");
    auto h = ExprObj::variable("h");
    auto w = ExprObj::constant(64);
    auto shapeX = ShapeExpr(new ShapeExprObj({batch, h, w}));

    auto x = graph->addTensor(shapeX, DataType(INFINI_DTYPE_F32));
    auto mn = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));
    auto mx = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));

    auto clip = graph->addOp<ClipObj>(x, mn, mx, nullptr);
    auto shapes = clip->inferShape();
    ASSERT_TRUE(shapes.has_value());

    auto outShape = (*shapes)[0];
    EXPECT_FALSE(outShape->isConcrete());
    EXPECT_EQ(outShape->size(), 3u);
    EXPECT_EQ(outShape->toString(), "[batch, h, 64]");
}

} // namespace infini
