#include "core/runtime.h"
#include "operators/LayerNorm.h"
#include "gtest/gtest.h"

namespace infini {

class LayerNormBasicTest : public testing::Test {
  protected:
    Runtime runtime;
    Graph graph;
    void SetUp() override {
        runtime = make_ref<RuntimeObj>();
        graph = make_ref<GraphObj>(runtime);
    }
};

TEST_F(LayerNormBasicTest, BasicConstruction) {
    // Normalize over last dim for 3D input (axis=2)
    auto x = graph->addTensor({2, 4, 8}, DataType(INFINI_DTYPE_F32));
    auto w = graph->addTensor({8}, DataType(INFINI_DTYPE_F32));
    auto b = graph->addTensor({8}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<LayerNormObj>(x, w, b, nullptr, 2, 1e-5f);
    EXPECT_EQ(op->getOpType(), OpType::LayerNorm);
    EXPECT_EQ(op->getNumInputs(), 3);
    EXPECT_EQ(op->getNumOutputs(), 1);
    EXPECT_EQ(op->getAxis(), 2);
    EXPECT_FLOAT_EQ(op->getEps(), 1e-5f);
}

TEST_F(LayerNormBasicTest, ShapeInference3D) {
    auto x = graph->addTensor({2, 4, 8}, DataType(INFINI_DTYPE_F32));
    auto w = graph->addTensor({8}, DataType(INFINI_DTYPE_F32));
    auto b = graph->addTensor({8}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<LayerNormObj>(x, w, b, nullptr, 2);
    auto shapes = op->inferShape();
    ASSERT_TRUE(shapes.has_value());
    auto vals = (*shapes)[0]->getConstantValue();
    ASSERT_EQ(vals.size(), 3u);
    EXPECT_EQ(vals[0], 2);
    EXPECT_EQ(vals[1], 4);
    EXPECT_EQ(vals[2], 8);
}

TEST_F(LayerNormBasicTest, DataTypeInference) {
    auto x = graph->addTensor({2, 4, 8}, DataType(INFINI_DTYPE_F32));
    auto w = graph->addTensor({8}, DataType(INFINI_DTYPE_F32));
    auto b = graph->addTensor({8}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<LayerNormObj>(x, w, b, nullptr, 2);
    auto dtypes = op->inferDataType();
    ASSERT_EQ(dtypes.size(), 1u);
    EXPECT_EQ(dtypes[0], DataType(INFINI_DTYPE_F32));
}

// 3D input: axis = 2 normalizes the last dimension
TEST_F(LayerNormBasicTest, ShapeInference3D_LastDim) {
    auto x = graph->addTensor({2, 8, 16}, DataType(INFINI_DTYPE_F32));
    auto w = graph->addTensor({16}, DataType(INFINI_DTYPE_F32));
    auto b = graph->addTensor({16}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<LayerNormObj>(x, w, b, nullptr, 2);
    auto shapes = op->inferShape();
    ASSERT_TRUE(shapes.has_value());
    auto vals = (*shapes)[0]->getConstantValue();
    ASSERT_EQ(vals.size(), 3u);
    EXPECT_EQ(vals[0], 2);
    EXPECT_EQ(vals[1], 8);
    EXPECT_EQ(vals[2], 16);
}

// Symbolic shape inference: batch and sequence dims can be symbolic
TEST_F(LayerNormBasicTest, SymbolicShapeInference) {
    auto batch = ExprObj::variable("batch");
    auto seq = ExprObj::variable("seq");
    auto shapeX =
        ShapeExpr(new ShapeExprObj({batch, seq, ExprObj::constant(64)}));
    auto x = graph->addTensor(shapeX, DataType(INFINI_DTYPE_F32));
    auto w = graph->addTensor({64}, DataType(INFINI_DTYPE_F32));
    auto b = graph->addTensor({64}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<LayerNormObj>(x, w, b, nullptr, 2);
    auto shapes = op->inferShape();
    ASSERT_TRUE(shapes.has_value());
    auto outShape = (*shapes)[0];
    EXPECT_FALSE(outShape->isConcrete());
    EXPECT_EQ(outShape->size(), 3u);
    EXPECT_EQ(outShape->toString(), "[batch, seq, 64]");
}

} // namespace infini
