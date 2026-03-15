#pragma once
#ifndef OP_TYPE_H
#define OP_TYPE_H

#include "core/common.h"

namespace infini {
struct OpType {
    using underlying_t = uint16_t;
    enum : underlying_t {
        Unknown,
        Add,
        Cast,
        Clip,
        Concat,
        Conv,
        Div,
        Gemm,
        Mul,
        MatMul,
        Relu,
        Sub,
        Transpose,
        LayerNorm,
        LogSoftmax,
        Softmax,
        LpNorm,
        RmsNorm,
        Sigmoid,
        Silu,
        Gelu,
        Softplus,
        Tanh

    } type;

    constexpr OpType(decltype(type) t) : type(t) {}
    constexpr explicit OpType(underlying_t val) : type((decltype(type))val) {}
    constexpr underlying_t underlying() const { return type; }

    bool operator==(OpType others) const { return type == others.type; }
    bool operator!=(OpType others) const { return type != others.type; }

    const char *toString() const {
#define CASE(NAME)                                                             \
    case OpType::NAME:                                                         \
        return #NAME

        switch (type) {
            CASE(Unknown);
            CASE(Add);
            CASE(Sub);
            CASE(Mul);
            CASE(Div);
            CASE(Cast);
            CASE(Clip);
            CASE(Conv);
            CASE(Relu);
            CASE(Transpose);
            CASE(Concat);
            CASE(MatMul);
            CASE(LayerNorm);
            CASE(LogSoftmax);
            CASE(Softmax);
            CASE(LpNorm);
            CASE(RmsNorm);
            CASE(Sigmoid);
            CASE(Silu);
            CASE(Gelu);
            CASE(Softplus);
            CASE(Tanh);

        default:
            return "Unknown";
        }

#undef CASE
    };
};

} // namespace infini

#endif // OP_TYPE_H
