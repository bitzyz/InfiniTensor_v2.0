#include "core/runtime.h"
#include "operators/Conv.h"
#include "gtest/gtest.h"

namespace infini {

class ConvBasicTest : public testing::Test {
  protected:
    Runtime runtime;
    Graph graph;
    void SetUp() override {
        runtime = make_ref<RuntimeObj>();
        graph = make_ref<GraphObj>(runtime);
    }
};

// 2D conv: [N,C,H,W] * [K,C,kH,kW] -> [N,K,oH,oW]
// oH = (H + 2*pad - dilation*(kH-1) - 1) / stride + 1
TEST_F(ConvBasicTest, BasicConstruction) {
    auto x = graph->addTensor({1, 3, 8, 8}, DataType(INFINI_DTYPE_F32));
    auto w = graph->addTensor({16, 3, 3, 3}, DataType(INFINI_DTYPE_F32));
    auto op =
        graph->addOp<ConvObj>(x, w, nullptr, nullptr, vector<int64_t>{1, 1},
                              vector<int64_t>{1, 1}, vector<int64_t>{1, 1});
    EXPECT_EQ(op->getOpType(), OpType::Conv);
    EXPECT_FALSE(op->hasBias());
    EXPECT_EQ(op->getNumInputs(), 2);
    EXPECT_EQ(op->getNumOutputs(), 1);
}

TEST_F(ConvBasicTest, ShapeInferencePad1Stride1) {
    // [1,3,8,8] conv [16,3,3,3], pad=1, stride=1, dilation=1 -> [1,16,8,8]
    auto x = graph->addTensor({1, 3, 8, 8}, DataType(INFINI_DTYPE_F32));
    auto w = graph->addTensor({16, 3, 3, 3}, DataType(INFINI_DTYPE_F32));
    auto op =
        graph->addOp<ConvObj>(x, w, nullptr, nullptr, vector<int64_t>{1, 1},
                              vector<int64_t>{1, 1}, vector<int64_t>{1, 1});
    auto shapes = op->inferShape();
    ASSERT_TRUE(shapes.has_value());
    auto vals = (*shapes)[0]->getConstantValue();
    ASSERT_EQ(vals.size(), 4u);
    EXPECT_EQ(vals[0], 1);
    EXPECT_EQ(vals[1], 16);
    EXPECT_EQ(vals[2], 8);
    EXPECT_EQ(vals[3], 8);
}

TEST_F(ConvBasicTest, ShapeInferenceStride2) {
    // [1,3,8,8] conv [16,3,3,3], pad=0, stride=2, dilation=1 -> [1,16,3,3]
    auto x = graph->addTensor({1, 3, 8, 8}, DataType(INFINI_DTYPE_F32));
    auto w = graph->addTensor({16, 3, 3, 3}, DataType(INFINI_DTYPE_F32));
    auto op =
        graph->addOp<ConvObj>(x, w, nullptr, nullptr, vector<int64_t>{0, 0},
                              vector<int64_t>{2, 2}, vector<int64_t>{1, 1});
    auto shapes = op->inferShape();
    ASSERT_TRUE(shapes.has_value());
    auto vals = (*shapes)[0]->getConstantValue();
    ASSERT_EQ(vals.size(), 4u);
    EXPECT_EQ(vals[0], 1);
    EXPECT_EQ(vals[1], 16);
    EXPECT_EQ(vals[2], 3);
    EXPECT_EQ(vals[3], 3);
}

TEST_F(ConvBasicTest, ShapeInferenceDilation2) {
    // [1,3,8,8] conv [16,3,3,3], pad=0, stride=1, dilation=2 -> [1,16,4,4]
    // out = (8 - 2*(3-1) - 1)/1 + 1 = (8-4-1)/1+1 = 4
    auto x = graph->addTensor({1, 3, 8, 8}, DataType(INFINI_DTYPE_F32));
    auto w = graph->addTensor({16, 3, 3, 3}, DataType(INFINI_DTYPE_F32));
    auto op =
        graph->addOp<ConvObj>(x, w, nullptr, nullptr, vector<int64_t>{0, 0},
                              vector<int64_t>{1, 1}, vector<int64_t>{2, 2});
    auto shapes = op->inferShape();
    ASSERT_TRUE(shapes.has_value());
    auto vals = (*shapes)[0]->getConstantValue();
    ASSERT_EQ(vals.size(), 4u);
    EXPECT_EQ(vals[2], 4);
    EXPECT_EQ(vals[3], 4);
}

TEST_F(ConvBasicTest, WithBias) {
    auto x = graph->addTensor({1, 3, 8, 8}, DataType(INFINI_DTYPE_F32));
    auto w = graph->addTensor({16, 3, 3, 3}, DataType(INFINI_DTYPE_F32));
    auto bias = graph->addTensor({16}, DataType(INFINI_DTYPE_F32));
    auto op =
        graph->addOp<ConvObj>(x, w, bias, nullptr, vector<int64_t>{1, 1},
                              vector<int64_t>{1, 1}, vector<int64_t>{1, 1});
    EXPECT_TRUE(op->hasBias());
    EXPECT_EQ(op->getNumInputs(), 3);
}

TEST_F(ConvBasicTest, DataTypeInference) {
    auto x = graph->addTensor({1, 3, 8, 8}, DataType(INFINI_DTYPE_F32));
    auto w = graph->addTensor({16, 3, 3, 3}, DataType(INFINI_DTYPE_F32));
    auto op =
        graph->addOp<ConvObj>(x, w, nullptr, nullptr, vector<int64_t>{0, 0},
                              vector<int64_t>{1, 1}, vector<int64_t>{1, 1});
    auto dtypes = op->inferDataType();
    ASSERT_EQ(dtypes.size(), 1u);
    EXPECT_EQ(dtypes[0], DataType(INFINI_DTYPE_F32));
}

} // namespace infini
