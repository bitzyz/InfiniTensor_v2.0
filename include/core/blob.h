#pragma once
#ifndef BLOB_H
#define BLOB_H

#include "core/ref.h"

namespace infini {

class BlobObj {
    void *ptr;

  public:
    BlobObj(void *ptr) : ptr(ptr) {}
    BlobObj(BlobObj &other) = delete;
    BlobObj &operator=(BlobObj const &) = delete;
    ~BlobObj() {};

    template <typename T> T getPtr() const { return reinterpret_cast<T>(ptr); }
};

} // namespace infini

#endif // BLOB_H
