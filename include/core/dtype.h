#pragma once
#ifndef DTYPE_H
#define DTYPE_H
#include "core/common.h"
#include <infinicore.h>

namespace infini {
class DataType {
  public:
    explicit DataType(infiniDtype_t dtype) : dtype_(dtype) {}

    size_t getSize() const {
        switch (dtype_) {
        case INFINI_DTYPE_BYTE:
            return 1;
        case INFINI_DTYPE_BOOL:
            return 1;
        case INFINI_DTYPE_I8:
            return 1;
        case INFINI_DTYPE_I16:
            return 2;
        case INFINI_DTYPE_I32:
            return 4;
        case INFINI_DTYPE_I64:
            return 8;
        case INFINI_DTYPE_U8:
            return 1;
        case INFINI_DTYPE_U16:
            return 2;
        case INFINI_DTYPE_U32:
            return 4;
        case INFINI_DTYPE_U64:
            return 8;
        case INFINI_DTYPE_F8:
            return 1; // 自定义 8-bit float
        case INFINI_DTYPE_F16:
            return 2;
        case INFINI_DTYPE_F32:
            return 4;
        case INFINI_DTYPE_F64:
            return 8;
        case INFINI_DTYPE_C16:
            return 2; // complex16: 2*8bit
        case INFINI_DTYPE_C32:
            return 4; // 2*16bit
        case INFINI_DTYPE_C64:
            return 8; // 2*32bit
        case INFINI_DTYPE_C128:
            return 16; // 2*64bit
        case INFINI_DTYPE_BF16:
            return 2; // bfloat16
        default:
            throw std::runtime_error("Invalid data type");
        }
    }

    infiniDtype_t getType() const { return dtype_; }

    std::string toString() const {
        switch (dtype_) {
        case INFINI_DTYPE_BYTE:
            return "BYTE";
        case INFINI_DTYPE_BOOL:
            return "BOOL";
        case INFINI_DTYPE_I8:
            return "I8";
        case INFINI_DTYPE_I16:
            return "I16";
        case INFINI_DTYPE_I32:
            return "I32";
        case INFINI_DTYPE_I64:
            return "I64";
        case INFINI_DTYPE_U8:
            return "U8";
        case INFINI_DTYPE_U16:
            return "U16";
        case INFINI_DTYPE_U32:
            return "U32";
        case INFINI_DTYPE_U64:
            return "U64";
        case INFINI_DTYPE_F8:
            return "F8";
        case INFINI_DTYPE_F16:
            return "F16";
        case INFINI_DTYPE_F32:
            return "F32";
        case INFINI_DTYPE_F64:
            return "F64";
        case INFINI_DTYPE_C16:
            return "C16";
        case INFINI_DTYPE_C32:
            return "C32";
        case INFINI_DTYPE_C64:
            return "C64";
        case INFINI_DTYPE_C128:
            return "C128";
        case INFINI_DTYPE_BF16:
            return "BF16";
        default:
            return "INVALID";
        }
    }

    bool operator==(const DataType &other) const {
        return getType() == other.getType();
    }

    bool operator!=(const DataType &other) const {
        return getType() != other.getType();
    }

  private:
    infiniDtype_t dtype_;
};
} // namespace infini
#endif // DTYPE_H
