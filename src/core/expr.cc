#include "core/expr.h"

namespace infini {
//==================================
// Static Factory Functions 实现
//==================================
Expr ExprObj::constant(ElementType value) {
    return Expr(new ConstantExprObj(value));
}

Expr ExprObj::variable(const std::string &name) {
    return Expr(new VariableExprObj(name));
}

Expr ExprObj::createAdd(const Expr &lhs, const Expr &rhs) {
    return Expr(new AddExprObj(lhs, rhs));
}

Expr ExprObj::createSub(const Expr &lhs, const Expr &rhs) {
    return Expr(new SubExprObj(lhs, rhs));
}

Expr ExprObj::createMul(const Expr &lhs, const Expr &rhs) {
    return Expr(new MulExprObj(lhs, rhs));
}

Expr ExprObj::createDiv(const Expr &lhs, const Expr &rhs) {
    return Expr(new DivExprObj(lhs, rhs));
}

Expr ExprObj::createMod(const Expr &lhs, const Expr &rhs) {
    return Expr(new ModExprObj(lhs, rhs));
}

Expr ExprObj::createMin(const Expr &lhs, const Expr &rhs) {
    return Expr(new MinExprObj(lhs, rhs));
}

Expr ExprObj::createMax(const Expr &lhs, const Expr &rhs) {
    return Expr(new MaxExprObj(lhs, rhs));
}

//==================================
// 运算符重载实现
//==================================
Expr operator+(const Expr &lhs, const Expr &rhs) {
    return ExprObj::createAdd(lhs, rhs);
}

Expr operator-(const Expr &lhs, const Expr &rhs) {
    return ExprObj::createSub(lhs, rhs);
}

Expr operator*(const Expr &lhs, const Expr &rhs) {
    return ExprObj::createMul(lhs, rhs);
}

Expr operator/(const Expr &lhs, const Expr &rhs) {
    return ExprObj::createDiv(lhs, rhs);
}

Expr operator%(const Expr &lhs, const Expr &rhs) {
    return ExprObj::createMod(lhs, rhs);
}

bool operator==(const Expr &lhs, const Expr &rhs) { return lhs->equals(rhs); }

bool operator!=(const Expr &lhs, const Expr &rhs) { return !(lhs == rhs); }

//==================================
// ConstantExprObj 实现
//==================================
ConstantExprObj::ConstantExprObj(ElementType v) : value(v) {}

ExprObj::Type ConstantExprObj::getType() const { return Type::CONSTANT; }

std::string ConstantExprObj::toString() const { return std::to_string(value); }

std::set<std::string> ConstantExprObj::getVariables() const { return {}; }

std::optional<ElementType> ConstantExprObj::evaluate(
    const std::unordered_map<std::string, ElementType> &) const {
    return value;
}

Expr ConstantExprObj::simplify() const {
    return make_ref<ConstantExprObj>(*this);
}

bool ConstantExprObj::equals(const Expr &other) const {
    if (other->getType() != Type::CONSTANT) {
        return false;
    }
    return value == std::static_pointer_cast<ConstantExprObj>(other)->value;
}

std::optional<ElementType> ConstantExprObj::asConstant() const { return value; }

//==================================
// VariableExprObj 实现
//==================================
VariableExprObj::VariableExprObj(std::string n) : name(std::move(n)) {}

ExprObj::Type VariableExprObj::getType() const { return Type::VARIABLE; }

std::string VariableExprObj::toString() const { return name; }

std::set<std::string> VariableExprObj::getVariables() const { return {name}; }

std::optional<ElementType> VariableExprObj::evaluate(
    const std::unordered_map<std::string, ElementType> &values) const {
    auto it = values.find(name);
    if (it == values.end())
        return std::nullopt;
    return it->second;
}

Expr VariableExprObj::simplify() const {
    return make_ref<VariableExprObj>(*this);
}

bool VariableExprObj::equals(const Expr &other) const {
    if (other->getType() != Type::VARIABLE)
        return false;
    return name == std::static_pointer_cast<VariableExprObj>(other)->name;
}

//==================================
// BinaryExprObj 实现
//==================================
BinaryExprObj::BinaryExprObj(Expr l, Expr r)
    : lhs(std::move(l)), rhs(std::move(r)) {}

std::set<std::string> BinaryExprObj::getVariables() const {
    auto lv = lhs->getVariables();
    auto rv = rhs->getVariables();
    lv.insert(rv.begin(), rv.end());
    return lv;
}

bool BinaryExprObj::equals(const Expr &other) const {
    if (other->getType() != getType()) {
        return false;
    }
    auto binaryOther = std::static_pointer_cast<BinaryExprObj>(other);
    return lhs->equals(binaryOther->lhs) && rhs->equals(binaryOther->rhs);
}

//==================================
// Achieve for simple binary expr
//==================================
#define IMPLEMENT_BINARY_EXPR(CLASS, TYPE_ENUM, OP, STR)                       \
    ExprObj::Type CLASS::getType() const { return Type::TYPE_ENUM; }           \
                                                                               \
    std::string CLASS::toString() const {                                      \
        return "(" + lhs->toString() + STR + rhs->toString() + ")";            \
    }                                                                          \
                                                                               \
    std::optional<ElementType> CLASS::evaluate(                                \
        const std::unordered_map<std::string, ElementType> &values) const {    \
        auto a = lhs->evaluate(values);                                        \
        auto b = rhs->evaluate(values);                                        \
        if (!a || !b)                                                          \
            return std::nullopt;                                               \
        if (*b == 0 && (string(STR) == " / " || string(STR) == " % ")) {       \
            return std::nullopt;                                               \
        }                                                                      \
        return (*a OP * b);                                                    \
    }                                                                          \
                                                                               \
    Expr CLASS::simplify() const {                                             \
        auto L = lhs->simplify();                                              \
        auto R = rhs->simplify();                                              \
        auto c1 = L->asConstant();                                             \
        auto c2 = R->asConstant();                                             \
        if (c1 && c2)                                                          \
            return ExprObj::constant(*c1 OP * c2);                             \
        return Expr(new CLASS(L, R));                                          \
    }

IMPLEMENT_BINARY_EXPR(AddExprObj, ADD, +, " + ")
IMPLEMENT_BINARY_EXPR(SubExprObj, SUB, -, " - ")
IMPLEMENT_BINARY_EXPR(MulExprObj, MUL, *, " * ")
IMPLEMENT_BINARY_EXPR(DivExprObj, DIV, /, " / ")
IMPLEMENT_BINARY_EXPR(ModExprObj, MOD, %, " % ")

//==================================
// MinExprObj 实现
//==================================
ExprObj::Type MinExprObj::getType() const { return Type::MIN; }

std::string MinExprObj::toString() const {
    return "min(" + lhs->toString() + "," + rhs->toString() + ")";
}

std::optional<ElementType> MinExprObj::evaluate(
    const std::unordered_map<std::string, ElementType> &values) const {
    auto a = lhs->evaluate(values);
    auto b = rhs->evaluate(values);
    if (!a || !b)
        return std::nullopt;
    return std::min(*a, *b);
}

Expr MinExprObj::simplify() const {
    auto L = lhs->simplify();
    auto R = rhs->simplify();
    auto c1 = L->asConstant();
    auto c2 = R->asConstant();
    if (c1 && c2)
        return ExprObj::constant(std::min(*c1, *c2));
    return Expr(new MinExprObj(L, R));
}

//==================================
// MaxExprObj 实现
//==================================
ExprObj::Type MaxExprObj::getType() const { return Type::MAX; }

std::string MaxExprObj::toString() const {
    return "max(" + lhs->toString() + "," + rhs->toString() + ")";
}

std::optional<ElementType> MaxExprObj::evaluate(
    const std::unordered_map<std::string, ElementType> &values) const {
    auto a = lhs->evaluate(values);
    auto b = rhs->evaluate(values);
    if (!a || !b)
        return std::nullopt;
    return std::max(*a, *b);
}

Expr MaxExprObj::simplify() const {
    auto L = lhs->simplify();
    auto R = rhs->simplify();
    auto c1 = L->asConstant();
    auto c2 = R->asConstant();
    if (c1 && c2)
        return ExprObj::constant(std::max(*c1, *c2));
    return Expr(new MaxExprObj(L, R));
}

//==================================
// BaseExprObj 实现
//==================================
BaseExprObj::BaseExprObj() = default;

BaseExprObj::BaseExprObj(std::vector<Expr> v) : dims(std::move(v)) {}

std::string BaseExprObj::toString() const {
    std::string s = "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        s += dims[i]->toString();
        if (i + 1 < dims.size())
            s += ", ";
    }
    s += "]";
    return s;
}

std::set<std::string> BaseExprObj::getVariables() const {
    std::set<std::string> res;
    for (auto &d : dims) {
        auto vars = d->getVariables();
        res.insert(vars.begin(), vars.end());
    }
    return res;
}

bool BaseExprObj::equals(const BaseExpr &other) const {
    if (dims.size() != other->dims.size())
        return false;
    for (size_t i = 0; i < dims.size(); ++i) {
        if (!dims[i]->equals(other->dims[i]))
            return false;
    }
    return true;
}

bool BaseExprObj::isConcrete() const {
    for (auto &d : dims) {
        if (!d->asConstant().has_value())
            return false;
    }
    return true;
}

bool BaseExprObj::isDynamic() const { return !isConcrete(); }

size_t BaseExprObj::size() const { return dims.size(); }

Expr BaseExprObj::operator[](size_t idx) const {
    if (idx >= dims.size()) {
        throw std::out_of_range("BaseExpr index out of range");
    }
    return dims[idx];
}

void BaseExprObj::insert(size_t pos, const Expr &value) {
    if (pos > dims.size()) {
        throw std::out_of_range("insert: pos out of range");
    }
    dims.insert(dims.begin() + pos, value);
}

//==================================
// ShapeExprObj 实现
//==================================
std::optional<std::vector<ShapeElem>> ShapeExprObj::evaluate(
    const std::unordered_map<std::string, ElementType> &values) const {
    std::vector<ShapeElem> out;
    out.reserve(dims.size());
    for (auto &d : dims) {
        auto v = d->evaluate(values);
        if (!v)
            return std::nullopt;
        out.push_back(static_cast<ShapeElem>(*v));
    }
    return out;
}

ShapeExpr ShapeExprObj::simplify() const {
    std::vector<Expr> out;
    out.reserve(dims.size());
    for (auto &d : dims)
        out.push_back(d->simplify());
    return ShapeExpr(new ShapeExprObj(out));
}

Shape ShapeExprObj::getConstantValue() const {
    IT_ASSERT(isConcrete(), "ShapeExpr is not concrete");
    Shape out;
    for (auto &d : dims) {
        out.push_back(static_cast<ShapeElem>(d->asConstant().value()));
    }
    return out;
}

//==================================
// StrideExprObj 实现
//==================================
std::optional<std::vector<StrideElem>> StrideExprObj::evaluate(
    const std::unordered_map<std::string, ElementType> &values) const {
    std::vector<StrideElem> out;
    out.reserve(dims.size());
    for (auto &d : dims) {
        auto v = d->evaluate(values);
        if (!v)
            return std::nullopt;
        out.push_back(static_cast<StrideElem>(*v));
    }
    return out;
}

StrideExpr StrideExprObj::simplify() const {
    std::vector<Expr> out;
    out.reserve(dims.size());
    for (auto &d : dims)
        out.push_back(d->simplify());
    return StrideExpr(new StrideExprObj(out));
}

Stride StrideExprObj::getConstantValue() const {
    IT_ASSERT(isConcrete(), "StrideExpr is not concrete");
    Stride out;
    for (auto &d : dims) {
        out.push_back(static_cast<StrideElem>(d->asConstant().value()));
    }
    return out;
}

//==================================
// ShapeExpr比较运算符实现
//==================================
bool operator==(const ShapeExpr &lhs, const ShapeExpr &rhs) {
    return lhs->equals(rhs);
}

bool operator!=(const ShapeExpr &lhs, const ShapeExpr &rhs) {
    return !(lhs == rhs);
}
} // namespace infini
