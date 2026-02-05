#include "utils/utils.h"

namespace infini {

StrideExpr broadcastStride(const ShapeExpr &inputShape,
                           const StrideExpr &inputStride,
                           const ShapeExpr &outputShape) {
    IT_ASSERT(inputShape->size() == inputStride->size(),
              "Input shape and stride must have the same rank");

    size_t inputRank = inputShape->size();
    size_t outputRank = outputShape->size();

    // 从右向左对齐维度（NumPy广播规则）
    std::vector<Expr> broadcastedStride(outputRank);

    for (size_t outIdx = 0; outIdx < outputRank; ++outIdx) {
        // 计算对应的输入维度索引（从右向左对齐）
        int inIdx = static_cast<int>(outIdx) - static_cast<int>(outputRank) +
                    static_cast<int>(inputRank);

        if (inIdx < 0) {
            // 输入维度不存在，相当于维度为1，stride为0
            broadcastedStride[outIdx] = ExprObj::constant(0);
        } else {
            // 检查输入维度是否为1（广播维度）或与输出维度相等
            const Expr &inputDim = (*inputShape)[inIdx];
            const Expr &outputDim = (*outputShape)[outIdx];

            // 如果输入维度是常量1，则是广播维度，stride为0
            auto inputDimConst = inputDim->asConstant();
            if (inputDimConst.has_value() && *inputDimConst == 1) {
                broadcastedStride[outIdx] = ExprObj::constant(0);
            } else {
                // 维度相等或不相等但有效（由infer_broadcast保证），使用原始stride
                broadcastedStride[outIdx] = (*inputStride)[inIdx];
            }
        }
    }

    return make_ref<StrideExprObj>(broadcastedStride);
}

ShapeExpr infer_broadcast(const ShapeExpr &A, const ShapeExpr &B) {
    size_t rankA = A->size();
    size_t rankB = B->size();
    size_t rank = std::max(rankA, rankB);

    std::vector<Expr> resultDims;
    for (size_t i = 0; i < rank; ++i) {
        // 从右向左对齐（NumPy广播规则）
        int idxA = static_cast<int>(i) - static_cast<int>(rank) +
                   static_cast<int>(rankA);
        int idxB = static_cast<int>(i) - static_cast<int>(rank) +
                   static_cast<int>(rankB);

        // 如果索引为负，表示该维度不存在，相当于维度为1
        Expr aDim = (idxA < 0) ? ExprObj::constant(1) : (*A)[idxA];
        Expr bDim = (idxB < 0) ? ExprObj::constant(1) : (*B)[idxB];

        // 验证广播规则：维度必须相等，或者其中一个为1
        IT_ASSERT(aDim == bDim || aDim == ExprObj::constant(1) ||
                  bDim == ExprObj::constant(1));

        // 广播结果：如果a维度为1则取b维度，否则取a维度
        auto shapeEle = aDim == ExprObj::constant(1) ? bDim : aDim;
        resultDims.emplace_back(shapeEle);
    }

    return make_ref<ShapeExprObj>(resultDims);
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

float fp16_to_fp32(uint16_t fp16) {
    // Union for safe type punning
    union {
        uint32_t u;
        float f;
    } converter;

    // Extract components from FP16
    uint32_t sign = (fp16 >> 15) & 0x1;
    uint32_t exponent = (fp16 >> 10) & 0x1F;
    uint32_t mantissa = fp16 & 0x3FF;

    // Handle special cases
    if (exponent == 0) {
        if (mantissa == 0) {
            // Zero
            converter.u = sign << 31;
            return converter.f;
        } else {
            // Subnormal number: normalize it
            while (!(mantissa & 0x400)) {
                mantissa <<= 1;
                exponent--;
            }
            exponent++;
            mantissa &= 0x3FF;
        }
    } else if (exponent == 31) {
        // Infinity or NaN
        converter.u = (sign << 31) | 0x7F800000;
        if (mantissa) {
            converter.u |= mantissa; // NaN
        }
        return converter.f;
    }

    // Convert to FP32
    // FP32: 1 sign bit, 8 exponent bits (bias 127), 23 mantissa bits
    // FP16: 1 sign bit, 5 exponent bits (bias 15), 10 mantissa bits
    converter.u = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13);
    return converter.f;
}
} // namespace infini
