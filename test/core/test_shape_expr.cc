#include "core/expr.h"
#include "gtest/gtest.h"

namespace infini {
class ShapeExprTest : public testing::Test {
  protected:
    void SetUp() override {}

    std::unordered_map<std::string, ElementType> createTestEnv() {
        return {
            {"batch", 32}, {"height", 224}, {"width", 224}, {"channels", 3}};
    }
};

// 测试常量形状
TEST_F(ShapeExprTest, ConcreteShape) {
    std::vector<Expr> dims = {ExprObj::constant(32), ExprObj::constant(224),
                              ExprObj::constant(224), ExprObj::constant(3)};

    auto shape = ShapeExpr(new ShapeExprObj(dims));

    EXPECT_EQ(shape->size(), 4);
    EXPECT_EQ(shape->toString(), "[32, 224, 224, 3]");
    EXPECT_TRUE(shape->isConcrete());
    EXPECT_FALSE(shape->isDynamic());

    auto constantValue = shape->getConstantValue();
    EXPECT_EQ(constantValue.size(), 4);
    EXPECT_EQ(constantValue[0], 32);
    EXPECT_EQ(constantValue[1], 224);
    EXPECT_EQ(constantValue[2], 224);
    EXPECT_EQ(constantValue[3], 3);

    auto evalResult = shape->evaluate({});
    EXPECT_TRUE(evalResult.has_value());
    EXPECT_EQ(evalResult->size(), 4);
}

// 测试符号形状
TEST_F(ShapeExprTest, SymbolicShape) {
    std::vector<Expr> dims = {ExprObj::variable("batch"),
                              ExprObj::constant(224), ExprObj::constant(224),
                              ExprObj::variable("channels")};

    auto shape = ShapeExpr(new ShapeExprObj(dims));

    EXPECT_EQ(shape->size(), 4);
    EXPECT_EQ(shape->toString(), "[batch, 224, 224, channels]");
    EXPECT_FALSE(shape->isConcrete());
    EXPECT_TRUE(shape->isDynamic());

    auto variables = shape->getVariables();
    EXPECT_EQ(variables.size(), 2);
    EXPECT_TRUE(variables.count("batch") > 0);
    EXPECT_TRUE(variables.count("channels") > 0);

    auto env = createTestEnv();
    auto evalResult = shape->evaluate(env);
    EXPECT_TRUE(evalResult.has_value());
    EXPECT_EQ(evalResult->size(), 4);
    EXPECT_EQ((*evalResult)[0], 32);  // batch = 32
    EXPECT_EQ((*evalResult)[1], 224); // height = 224
    EXPECT_EQ((*evalResult)[2], 224); // width = 224
    EXPECT_EQ((*evalResult)[3], 3);   // channels = 3
}

// 测试混合表达式形状
TEST_F(ShapeExprTest, MixedExpressionShape) {
    auto batch = ExprObj::variable("batch");
    auto height = ExprObj::constant(224);
    auto width = ExprObj::constant(224);
    auto channels = ExprObj::constant(3);

    // 使用表达式计算维度
    auto paddedHeight = height + ExprObj::constant(2);
    auto paddedWidth = width + ExprObj::constant(2);

    std::vector<Expr> dims = {batch, paddedHeight, paddedWidth, channels};
    auto shape = ShapeExpr(new ShapeExprObj(dims));

    EXPECT_EQ(shape->toString(), "[batch, (224 + 2), (224 + 2), 3]");
    EXPECT_FALSE(shape->isConcrete());

    auto env = createTestEnv();
    auto evalResult = shape->evaluate(env);
    EXPECT_TRUE(evalResult.has_value());
    EXPECT_EQ((*evalResult)[0], 32);  // batch
    EXPECT_EQ((*evalResult)[1], 226); // 224 + 2
    EXPECT_EQ((*evalResult)[2], 226); // 224 + 2
    EXPECT_EQ((*evalResult)[3], 3);   // 3

    // 测试简化
    auto simplified = shape->simplify();
    EXPECT_EQ(simplified->toString(), "[batch, 226, 226, 3]");
}
} // namespace infini
