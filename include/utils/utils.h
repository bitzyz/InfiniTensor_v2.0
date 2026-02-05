#pragma once
#ifndef UTIL_H
#define UTIL_H

#include "core/common.h"
#include "core/expr.h"
#include <numeric>
namespace infini {
ShapeExpr infer_broadcast(const ShapeExpr &A, const ShapeExpr &B);
size_t calculateLinearOffset(size_t index, Shape shape, Stride stride);

// 计算广播后的stride
// inputShape: 输入张量的形状
// inputStride: 输入张量的原始stride
// outputShape: 输出张量的形状（广播后的形状）
// 返回: 广播后的stride，对于广播的维度，stride设置为0
StrideExpr broadcastStride(const ShapeExpr &inputShape,
                           const StrideExpr &inputStride,
                           const ShapeExpr &outputShape);

// FP16 to FP32 conversion utility
float fp16_to_fp32(uint16_t fp16);
} // namespace infini

#endif
