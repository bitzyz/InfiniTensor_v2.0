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
        Gelu,
        LayerNorm,
        LogSoftmax,
        LPNorm,
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
            CASE(Sub);
            CASE(Mul);
            CASE(Div);
            CASE(Cast);
            CASE(Clip);
            CASE(Conv);
            CASE(Gemm);
            CASE(Gelu);
            CASE(LayerNorm);
            CASE(LogSoftmax);
            CASE(LPNorm);
            CASE(Relu);
            CASE(RMSNorm);
            CASE(Sigmoid);
            CASE(Silu);
            CASE(Softmax);
            CASE(Softplus);
            CASE(Tanh);
            CASE(Transpose);
            CASE(Concat);
            CASE(MatMul);

        default:
            return "Unknown";
        }

#undef CASE
    };
};

} // namespace infini

#endif // OP_TYPE_H
