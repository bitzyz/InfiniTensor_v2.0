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
        Gelu,
        Gemm,
        LayerNorm,
        LogSoftmax,
        LpNorm,
        Mul,
        MatMul,
        Relu,
        RMSNorm,
        Sigmoid,
        Silu,
        Softmax,
        Softplus,
        Sub,
        Tanh,
        Transpose,

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
            CASE(Cast);
            CASE(Clip);
            CASE(Concat);
            CASE(Conv);
            CASE(Div);
            CASE(Gelu);
            CASE(Gemm);
            CASE(LayerNorm);
            CASE(LogSoftmax);
            CASE(LpNorm);
            CASE(Mul);
            CASE(MatMul);
            CASE(Relu);
            CASE(RMSNorm);
            CASE(Sigmoid);
            CASE(Silu);
            CASE(Softmax);
            CASE(Softplus);
            CASE(Sub);
            CASE(Tanh);
            CASE(Transpose);

        default:
            return "Unknown";
        }

#undef CASE
    };
};

} // namespace infini

#endif // OP_TYPE_H
