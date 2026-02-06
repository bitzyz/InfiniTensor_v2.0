#include "core/runtime.h"
#include "operators/ElementWise.h"
#include "gtest/gtest.h"

namespace infini {

class ElementWiseBasicTest : public testing::Test {
  protected:
    Runtime runtime;
    Graph graph;

    void SetUp() override {
        runtime = make_ref<RuntimeObj>();
        graph = make_ref<GraphObj>(runtime);
    }
};

// 测试ElementWise的基本构造
TEST_F(ElementWiseBasicTest, BasicConstruction) {
    auto A = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto B = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto elementwise = graph->addOp<ElementWiseObj>(OpType::Add, A, B, nullptr);
    EXPECT_EQ(elementwise->getOpType(), OpType::Add);
    EXPECT_EQ(elementwise->getNumInputs(), 2);
    EXPECT_EQ(elementwise->getNumOutputs(), 1);
    EXPECT_EQ(elementwise->getElemenwiseOpType(), OpType::Add);
}

// 测试ElementWise形状推导 - 相同形状
TEST_F(ElementWiseBasicTest, ShapeInferenceSameShape) {
    auto A = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto B = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));

    auto elementwise = graph->addOp<ElementWiseObj>(OpType::Add, A, B, nullptr);

    auto inferredShapes = elementwise->inferShape();
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

// 测试ElementWise形状推导 - 广播（标量广播到张量）
TEST_F(ElementWiseBasicTest, ShapeInferenceScalarBroadcast) {
    // 标量广播到 [2, 3, 4]
    auto scalar = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));
    auto tensor = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));

    auto elementwise =
        graph->addOp<ElementWiseObj>(OpType::Mul, scalar, tensor, nullptr);

    auto inferredShapes = elementwise->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());

    auto outputShape = (*inferredShapes)[0];
    auto shapeValues = outputShape->getConstantValue();

    EXPECT_EQ(shapeValues.size(), 3); // 标量应该广播到 tensor 的维度
    EXPECT_EQ(shapeValues[0], 2);
    EXPECT_EQ(shapeValues[1], 3);
    EXPECT_EQ(shapeValues[2], 4);
}

// 测试ElementWise形状推导 - 广播（两个操作数都需要广播）
TEST_F(ElementWiseBasicTest, ShapeInferenceBothBroadcast) {
    // [1, 3, 1] 和 [2, 1, 4] 广播到 [2, 3, 4]
    auto A = graph->addTensor({1, 3, 1}, DataType(INFINI_DTYPE_F32));
    auto B = graph->addTensor({2, 1, 4}, DataType(INFINI_DTYPE_F32));

    auto elementwise = graph->addOp<ElementWiseObj>(OpType::Mul, A, B, nullptr);

    auto inferredShapes = elementwise->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());

    auto outputShape = (*inferredShapes)[0];
    auto shapeValues = outputShape->getConstantValue();

    EXPECT_EQ(shapeValues.size(), 3);
    EXPECT_EQ(shapeValues[0], 2);
    EXPECT_EQ(shapeValues[1], 3);
    EXPECT_EQ(shapeValues[2], 4);
}

// 测试ElementWise数据类型推断
TEST_F(ElementWiseBasicTest, DataTypeInference) {
    auto A = graph->addTensor({2, 3}, DataType(INFINI_DTYPE_F32));
    auto B = graph->addTensor({2, 3}, DataType(INFINI_DTYPE_F32));

    auto elementwise = graph->addOp<ElementWiseObj>(OpType::Add, A, B, nullptr);

    auto inferredTypes = elementwise->inferDataType();
    ASSERT_EQ(inferredTypes.size(), 1);
    EXPECT_EQ(inferredTypes[0], DataType(INFINI_DTYPE_F32));
}

// 测试符号形状推导
TEST_F(ElementWiseBasicTest, SymbolicShapeInference) {
    auto batch = ExprObj::variable("batch");
    auto height = ExprObj::variable("h");
    auto width = ExprObj::constant(256);

    auto shapeA = ShapeExpr(new ShapeExprObj({batch, height, width}));
    auto shapeB = ShapeExpr(new ShapeExprObj({height, width}));

    auto A = graph->addTensor(shapeA, DataType(INFINI_DTYPE_F32));
    auto B = graph->addTensor(shapeB, DataType(INFINI_DTYPE_F32));

    auto elementwise = graph->addOp<ElementWiseObj>(OpType::Add, A, B, nullptr);

    auto inferredShapes = elementwise->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());

    auto outputShape = (*inferredShapes)[0];
    EXPECT_FALSE(outputShape->isConcrete());
    EXPECT_EQ(outputShape->size(), 3);

    // 检查符号表达式
    EXPECT_EQ(outputShape->toString(), "[batch, h, 256]");
}

} // namespace infini
