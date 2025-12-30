#include "utils/utils.h"

namespace infini {
ShapeExpr infer_broadcast(const ShapeExpr &A, const ShapeExpr &B) {
    if (A->size() == 0 && B->size() == 0) {
        return ShapeExpr({});
    }
    auto A_ = A;
    auto B_ = B;
    size_t rankA = A->size();
    size_t rankB = B->size();
    size_t rank = std::max(rankA, rankB);
    if (rankA < rank) {
        for (size_t i = 0; i < rank - rankA; ++i) {
            A_->insert(0, ExprObj::constant(1));
        }
    }
    if (rankB < rank) {
        for (size_t i = 0; i < rank - rankB; ++i) {
            B_->insert(0, ExprObj::constant(1));
        }
    }
    // 逐维度计算广播结果
    std::vector<Expr> resultDims;
    for (size_t i = 0; i < rank; ++i) {
        IT_ASSERT((*A_)[i] == (*B_)[i] || (*A_)[i] == ExprObj::constant(1) ||
                  (*B_)[i] == ExprObj::constant(1));
        auto shapeEle = (*A_)[i] == ExprObj::constant(1) ? (*B_)[i] : (*A_)[i];
        resultDims.emplace_back(shapeEle);
    }
    return make_ref<ShapeExprObj>(ShapeExprObj(resultDims));
}

size_t calculateLinearOffset(size_t index, Shape shape, Stride stride) {
    size_t rank = shape.size();
    std::vector<size_t> indices(rank);
    size_t remaining = index;
    for (size_t i = 0; i < rank; ++i) {
        size_t dim = rank - 1 - i;
        indices[dim] = remaining % shape.at(dim);
        remaining /= shape.at(dim);
    }
    size_t offset = 0;
    for (size_t i = 0; i < rank; ++i) {
        offset += indices[i] * stride.at(i);
    }
    return offset;
}
} // namespace infini
