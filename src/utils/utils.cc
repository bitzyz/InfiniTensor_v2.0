#include "utils/utils.h"

namespace infini {
Shape infer_broadcast(const Shape &A, const Shape &B) {
  if (A.empty() && B.empty()) {
    return {};
  }
  auto A_ = A;
  auto B_ = B;
  int rankA = A.size();
  int rankB = B.size();
  int rank = std::max(rankA, rankB);
  if (rankA < rank) {
    for (int i = 0; i < rank - rankA; ++i) {
      A_.insert(A_.begin(), 1);
    }
  }
  if (rankB < rank) {
    for (int i = 0; i < rank - rankB; ++i) {
      B_.insert(B_.begin(), 1);
    }
  }
  Shape ret;
  for (int i = 0; i < rank; ++i) {
    IT_ASSERT(A_[i] == B_[i] || A_[i] == 1 || B_[i] == 1);
    auto shapeEle = std::max(A_[i], B_[i]);
    ret.emplace_back(shapeEle);
  }
  return ret;
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
