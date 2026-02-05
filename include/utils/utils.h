#pragma once
#ifndef UTIL_H
#define UTIL_H

#include "core/common.h"
#include "core/expr.h"
#include <numeric>
namespace infini {
ShapeExpr infer_broadcast(const ShapeExpr &A, const ShapeExpr &B);
size_t calculateLinearOffset(size_t index, Shape shape, Stride stride);

// FP16 to FP32 conversion utility
float fp16_to_fp32(uint16_t fp16);
} // namespace infini

#endif
