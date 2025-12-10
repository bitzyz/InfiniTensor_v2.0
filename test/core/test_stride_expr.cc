#include "core/expr.h"
#include "gtest/gtest.h"

namespace infini {
class StrideExprTest : public testing::Test {
protected:
  void SetUp() override {}
};

// 测试步长计算
TEST_F(StrideExprTest, StrideComputation) {
  auto shape = ShapeExpr(
      new ShapeExprObj({ExprObj::constant(32), ExprObj::constant(224),
                        ExprObj::constant(224), ExprObj::constant(3)}));

  std::vector<Expr> strides = {ExprObj::constant(224 * 224 * 3),
                               ExprObj::constant(224 * 3), ExprObj::constant(3),
                               ExprObj::constant(1)};

  auto stride = StrideExpr(new StrideExprObj(strides));

  EXPECT_EQ(stride->size(), 4);
  EXPECT_EQ(stride->toString(), "[150528, 672, 3, 1]");
  EXPECT_TRUE(stride->isConcrete());

  auto constantValue = stride->getConstantValue();
  EXPECT_EQ(constantValue.size(), 4);
  EXPECT_EQ(constantValue[0], 150528);
  EXPECT_EQ(constantValue[1], 672);
  EXPECT_EQ(constantValue[2], 3);
  EXPECT_EQ(constantValue[3], 1);
}

// 测试符号步长
TEST_F(StrideExprTest, SymbolicStride) {
  auto batch = ExprObj::variable("batch");
  auto height = ExprObj::variable("height");
  auto width = ExprObj::constant(224);
  auto channels = ExprObj::constant(3);

  // 符号步长计算
  std::vector<Expr> strides = {height * width * channels, width * channels,
                               channels, ExprObj::constant(1)};

  auto stride = StrideExpr(new StrideExprObj(strides));

  EXPECT_EQ(stride->toString(), "[((height * 224) * 3), (224 * 3), 3, 1]");
  EXPECT_FALSE(stride->isConcrete());

  // 简化后的表达式
  // TODO:目前不支持乘法交换律，因此无法简化
  auto simplified = stride->simplify();
  EXPECT_EQ(simplified->toString(), "[((height * 224) * 3), 672, 3, 1]");

  // 求值
  auto env = std::unordered_map<std::string, ElementType>{{"height", 256}};
  auto result = stride->evaluate(env);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ((*result)[0], 256 * 224 * 3);
  EXPECT_EQ((*result)[1], 224 * 3);
  EXPECT_EQ((*result)[2], 3);
  EXPECT_EQ((*result)[3], 1);
}
} // namespace infini
