#include "core/expr.h"
#include "gtest/gtest.h"

namespace infini {

class ExprTest : public testing::Test {
  protected:
    void SetUp() override {}

    // 辅助函数：创建测试环境
    std::unordered_map<std::string, ElementType> createTestEnv() {
        return {{"a", 5}, {"b", 3}, {"c", 2}};
    }
};
// 测试常量表达式
TEST_F(ExprTest, ConstantExpr) {
    auto expr = ExprObj::constant(42);

    EXPECT_EQ(expr->getType(), ExprObj::Type::CONSTANT);
    EXPECT_EQ(expr->toString(), "42");
    EXPECT_TRUE(expr->getVariables().empty());

    auto evalResult = expr->evaluate({});
    EXPECT_TRUE(evalResult.has_value());
    EXPECT_EQ(*evalResult, 42);

    auto constantValue = expr->asConstant();
    EXPECT_TRUE(constantValue.has_value());
    EXPECT_EQ(*constantValue, 42);

    EXPECT_TRUE(expr->equals(ExprObj::constant(42)));
    EXPECT_FALSE(expr->equals(ExprObj::constant(43)));
}

// 测试变量表达式
TEST_F(ExprTest, VariableExpr) {
    auto expr = ExprObj::variable("x");

    EXPECT_EQ(expr->getType(), ExprObj::Type::VARIABLE);
    EXPECT_EQ(expr->toString(), "x");

    auto variables = expr->getVariables();
    EXPECT_EQ(variables.size(), 1);
    EXPECT_TRUE(variables.count("x") > 0);

    // 未提供变量值
    EXPECT_FALSE(expr->evaluate({}).has_value());

    // 提供变量值
    auto env = createTestEnv();
    EXPECT_TRUE(expr->evaluate({{"x", 10}}).has_value());
    EXPECT_EQ(*expr->evaluate({{"x", 10}}), 10);

    EXPECT_TRUE(expr->equals(ExprObj::variable("x")));
    EXPECT_FALSE(expr->equals(ExprObj::variable("y")));
}

// 测试加法表达式
TEST_F(ExprTest, AddExpr) {
    auto a = ExprObj::variable("a");
    auto b = ExprObj::variable("b");
    auto expr = a + b;

    EXPECT_EQ(expr->getType(), ExprObj::Type::ADD);
    EXPECT_EQ(expr->toString(), "(a + b)");

    auto env = createTestEnv();
    auto result = expr->evaluate(env);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 8); // 5 + 3 = 8

    // 测试常量折叠
    auto constExpr = ExprObj::constant(2) + ExprObj::constant(3);
    auto constResult = constExpr->simplify();
    EXPECT_EQ(constResult->getType(), ExprObj::Type::CONSTANT);
    EXPECT_EQ(*constResult->asConstant(), 5);
}

// 测试减法表达式
TEST_F(ExprTest, SubExpr) {
    auto a = ExprObj::variable("a");
    auto b = ExprObj::variable("b");
    auto expr = a - b;

    EXPECT_EQ(expr->getType(), ExprObj::Type::SUB);
    EXPECT_EQ(expr->toString(), "(a - b)");

    auto env = createTestEnv();
    auto result = expr->evaluate(env);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 2); // 5 - 3 = 2
}

// 测试乘法表达式
TEST_F(ExprTest, MulExpr) {
    auto a = ExprObj::variable("a");
    auto b = ExprObj::variable("b");
    auto expr = a * b;

    EXPECT_EQ(expr->getType(), ExprObj::Type::MUL);
    EXPECT_EQ(expr->toString(), "(a * b)");

    auto env = createTestEnv();
    auto result = expr->evaluate(env);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 15); // 5 * 3 = 15
}

// 测试除法表达式
TEST_F(ExprTest, DivExpr) {
    auto a = ExprObj::variable("a");
    auto b = ExprObj::variable("b");
    auto expr = a / b;

    EXPECT_EQ(expr->getType(), ExprObj::Type::DIV);
    EXPECT_EQ(expr->toString(), "(a / b)");

    auto env = createTestEnv();
    auto result = expr->evaluate(env);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 1); // 5 / 3 = 1 (整数除法)

    // 测试除以0的情况
    auto zeroExpr = a / ExprObj::constant(0);
    auto zeroResult = zeroExpr->evaluate(env);
    EXPECT_FALSE(zeroResult.has_value());
}

// 测试取模表达式
TEST_F(ExprTest, ModExpr) {
    auto a = ExprObj::variable("a");
    auto b = ExprObj::variable("b");
    auto expr = a % b;

    EXPECT_EQ(expr->getType(), ExprObj::Type::MOD);
    EXPECT_EQ(expr->toString(), "(a % b)");

    auto env = createTestEnv();
    auto result = expr->evaluate(env);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 2); // 5 % 3 = 2
}

// 测试min表达式
TEST_F(ExprTest, MinExpr) {
    auto a = ExprObj::variable("a");
    auto b = ExprObj::variable("b");
    auto expr = ExprObj::createMin(a, b);

    EXPECT_EQ(expr->getType(), ExprObj::Type::MIN);
    EXPECT_EQ(expr->toString(), "min(a,b)");

    auto env = createTestEnv();
    auto result = expr->evaluate(env);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 3); // min(5, 3) = 3
}

// 测试max表达式
TEST_F(ExprTest, MaxExpr) {
    auto a = ExprObj::variable("a");
    auto b = ExprObj::variable("b");
    auto expr = ExprObj::createMax(a, b);

    EXPECT_EQ(expr->getType(), ExprObj::Type::MAX);
    EXPECT_EQ(expr->toString(), "max(a,b)");

    auto env = createTestEnv();
    auto result = expr->evaluate(env);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 5); // max(5, 3) = 5
}

TEST_F(ExprTest, ComplexExpression) {
    auto a = ExprObj::variable("a");
    auto b = ExprObj::variable("b");
    auto c = ExprObj::variable("c");

    // (a + b) * c
    auto expr = (a + b) * c;

    EXPECT_EQ(expr->toString(), "((a + b) * c)");

    auto env = createTestEnv();
    auto result = expr->evaluate(env);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(*result, 16); // (5 + 3) * 2 = 16

    // 测试嵌套表达式
    auto complexExpr = (a * b) + (b / c) - (a % c);
    EXPECT_EQ(complexExpr->toString(), "(((a * b) + (b / c)) - (a % c))");

    auto complexResult = complexExpr->evaluate(env);
    EXPECT_TRUE(complexResult.has_value());
    EXPECT_EQ(*complexResult, 15); // (5*3) + (3/2) - (5%2) = 15 + 1 - 1 = 15
}

// 测试表达式相等性
TEST_F(ExprTest, ExpressionEquality) {
    auto a = ExprObj::variable("a");
    auto b = ExprObj::variable("b");

    auto expr1 = a + b;
    auto expr2 = a + b;
    auto expr3 =
        b + a; // 交换律，但表达式结构不同，目前还不支持交换律、结合律等等

    EXPECT_TRUE(expr1->equals(expr2));
    EXPECT_FALSE(expr1->equals(expr3));

    // 常量折叠后的相等性
    auto const1 = (ExprObj::constant(2) + ExprObj::constant(3))->simplify();
    auto const2 = ExprObj::constant(5);

    EXPECT_TRUE(const1->equals(const2));
}
} // namespace infini