#pragma once
#ifndef UTIL_H
#define UTIL_H

#include "core/common.h"
#include <numeric>
namespace infini {
Shape infer_broadcast(const Shape &A, const Shape &B);
size_t calculateLinearOffset(size_t index, Shape shape, Stride stride);
} // namespace infini

#endif
