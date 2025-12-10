#pragma once
#ifndef EXPR_H
#define EXPR_H

#include "core/ref.h"

namespace infini {

#define DECL_CLASS(name)                                                       \
  class name##Obj;                                                             \
  using name = Ref<name##Obj>;
DECL_CLASS(Expr)
DECL_CLASS(ConstantExpr)
DECL_CLASS(VariableExpr)
DECL_CLASS(AddExpr)
DECL_CLASS(SubExpr)
DECL_CLASS(MulExpr)
DECL_CLASS(DivExpr)
DECL_CLASS(ModExpr)
DECL_CLASS(MinExpr)
DECL_CLASS(MaxExpr)
DECL_CLASS(BaseExpr)
DECL_CLASS(ShapeExpr)
DECL_CLASS(StrideExpr)

//===============================================
// ExprObj
//===============================================
class ExprObj : public std::enable_shared_from_this<ExprObj> {
public:
  enum class Type { CONSTANT, VARIABLE, ADD, SUB, MUL, DIV, MOD, MIN, MAX };
  virtual ~ExprObj() = default;
  virtual Type getType() const = 0;
  virtual std::string toString() const = 0;
  virtual std::set<std::string> getVariables() const = 0;
  virtual std::optional<ElementType> evaluate(
      const std::unordered_map<std::string, ElementType> &values) const = 0;
  virtual Expr simplify() const = 0;
  virtual bool equals(const Expr &other) const = 0;
  virtual std::optional<ElementType> asConstant() const { return std::nullopt; }

  static Expr constant(ElementType value);
  static Expr variable(const std::string &name);
  static Expr createAdd(const Expr &lhs, const Expr &rhs);
  static Expr createSub(const Expr &lhs, const Expr &rhs);
  static Expr createMul(const Expr &lhs, const Expr &rhs);
  static Expr createDiv(const Expr &lhs, const Expr &rhs);
  static Expr createMod(const Expr &lhs, const Expr &rhs);
  static Expr createMin(const Expr &lhs, const Expr &rhs);
  static Expr createMax(const Expr &lhs, const Expr &rhs);
};
Expr operator+(const Expr &lhs, const Expr &rhs);

Expr operator-(const Expr &lhs, const Expr &rhs);

Expr operator*(const Expr &lhs, const Expr &rhs);

Expr operator/(const Expr &lhs, const Expr &rhs);

Expr operator%(const Expr &lhs, const Expr &rhs);

bool operator==(const Expr &lhs, const Expr &rhs);

bool operator!=(const Expr &lhs, const Expr &rhs);

//===============================================
// ConstantExprObj
//===============================================
class ConstantExprObj : public ExprObj {
public:
  ElementType value;
  ConstantExprObj(ElementType v);
  Type getType() const override;
  std::string toString() const override;
  std::set<std::string> getVariables() const override;
  std::optional<ElementType>
  evaluate(const std::unordered_map<std::string, ElementType> &) const override;
  Expr simplify() const override;
  bool equals(const Expr &other) const override;
  std::optional<ElementType> asConstant() const override;
};

//===============================================
// VariableExpr
//===============================================
class VariableExprObj : public ExprObj {
public:
  std::string name;
  VariableExprObj(std::string n);

  Type getType() const override;
  std::string toString() const override;
  std::set<std::string> getVariables() const override;
  std::optional<ElementType>
  evaluate(const std::unordered_map<std::string, ElementType> &values)
      const override;
  Expr simplify() const override;
  bool equals(const Expr &other) const override;
};

//===============================================
// BinaryExprObj
//===============================================
class BinaryExprObj : public ExprObj {
public:
  Expr lhs, rhs;
  BinaryExprObj(Expr l, Expr r);
  std::set<std::string> getVariables() const override;
  bool equals(const Expr &other) const override;
};

//===============================================
// Macro for simple binary expr
//===============================================
#define DECL_BINARY_EXPR(ClassName)                                            \
  class ClassName : public BinaryExprObj {                                     \
  public:                                                                      \
    using BinaryExprObj::BinaryExprObj;                                        \
    Type getType() const override;                                             \
    std::string toString() const override;                                     \
    std::optional<ElementType>                                                 \
    evaluate(const std::unordered_map<std::string, ElementType> &values)       \
        const override;                                                        \
    Expr simplify() const override;                                            \
  };

DECL_BINARY_EXPR(AddExprObj)
DECL_BINARY_EXPR(SubExprObj)
DECL_BINARY_EXPR(MulExprObj)
DECL_BINARY_EXPR(DivExprObj)
DECL_BINARY_EXPR(ModExprObj)

//===============================================
// MinExprObj
//===============================================
class MinExprObj : public BinaryExprObj {
  using BinaryExprObj::BinaryExprObj;
  Type getType() const override;
  std::string toString() const override;
  std::optional<ElementType>
  evaluate(const std::unordered_map<std::string, ElementType> &values)
      const override;
  Expr simplify() const override;
};

//===============================================
// MaxExprObj
//===============================================
class MaxExprObj : public BinaryExprObj {
  using BinaryExprObj::BinaryExprObj;
  Type getType() const override;
  std::string toString() const override;
  std::optional<ElementType>
  evaluate(const std::unordered_map<std::string, ElementType> &values)
      const override;
  Expr simplify() const override;
};

//===============================================
// BaseExprObj
//===============================================
class BaseExprObj : public std::enable_shared_from_this<BaseExprObj> {
public:
  std::vector<Expr> dims;

  BaseExprObj();
  explicit BaseExprObj(std::vector<Expr> v);
  virtual ~BaseExprObj() = default;

  virtual std::string toString() const;
  virtual std::set<std::string> getVariables() const;
  virtual bool equals(const BaseExpr &other) const;
  bool isConcrete() const;
  bool isDynamic() const;
  size_t size() const;
  Expr operator[](size_t idx) const;
};

//===============================================
// ShapeExprObj
//===============================================
class ShapeExprObj : public BaseExprObj {
public:
  using BaseExprObj::BaseExprObj;

  std::optional<std::vector<ShapeElem>>
  evaluate(const std::unordered_map<std::string, ElementType> &values) const;
  ShapeExpr simplify() const;
  Shape getConstantValue() const;
};

bool operator==(const ShapeExpr &lhs, const ShapeExpr &rhs);
bool operator!=(const ShapeExpr &lhs, const ShapeExpr &rhs);

//===============================================
// StrideExprObj
//===============================================
class StrideExprObj : public BaseExprObj {
public:
  using BaseExprObj::BaseExprObj;

  std::optional<std::vector<StrideElem>>
  evaluate(const std::unordered_map<std::string, ElementType> &values) const;
  StrideExpr simplify() const;
  Stride getConstantValue() const;
};

} // namespace infini
#endif // EXPR_H
