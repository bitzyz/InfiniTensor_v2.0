#pragma once
#ifndef TENSOR_H
#define TENSOR_H

#include "core/blob.h"
#include "core/dtype.h"
#include "core/expr.h"
#include "core/object.h"
#include "utils/utils.h"

namespace infini {

class TensorObj : public Object {
    friend class GraphObj;

  private:
    Fuid fuid;
    DataType dtype;
    ShapeExpr shape;
    StrideExpr stride;
    Blob data = nullptr;
    vector<WRef<OperatorObj>> targets;
    WRef<OperatorObj> source;
    infiniDevice_t device = INFINI_DEVICE_CPU;

  public:
    TensorObj(ShapeExpr symbolic_shape, DataType dtype);
    TensorObj(ShapeExpr symbolic_shape, StrideExpr stride, DataType dtype);
    TensorObj(ShapeExpr symbolic_shape, Stride stride, DataType dtype);
    TensorObj(Shape shape, DataType dtype);
    TensorObj(Shape shape, StrideExpr stride, DataType dtype);
    TensorObj(Shape shape, Stride stride, DataType dtype);
    virtual ~TensorObj() {}

    // =============Get TensorObj attributes=================
    UidBaseType getFuid() const;
    DataType getDataType() const;
    ShapeExpr getShape() const;
    void setShape(ShapeExpr shape_);
    void setShape(Shape shape_);
    StrideExpr getStride() const;
    void setStride(Stride stride_);
    void setStride(StrideExpr stride_);
    Blob getData() const;
    ElementType getElement() const;
    ElementType getStorageSize() const;
    ElementType getTotalBytes() const;
    ElementType getRank() const;
    OpVec getTargets() const;
    Operator getSource() const;

    string toString() const override;
    // ============= TensorObj Data Operations==============
    void setData(void *data_);
    void dataMalloc(const Runtime &runtime);

    template <typename T> T getRawDataPtr() const {
        static_assert(std::is_pointer_v<T>,
                      "Raw data pointer has a type of pointer");
        IT_ASSERT(data != nullptr);
        return data->getPtr<T>();
    }

    void printData(const Runtime &runtime, size_t maxElements = 0,
                   int precision = 4) const;
    void copyToHost(const Runtime &runtime);
    void copyToDevice(const Runtime &runtime);

  private:
    // ============= Change Graph Operations==============
    void addTarget(const Operator &op);
    void setSource(const Operator &op);
    void removeTarget(const Operator &op);
    StrideExpr computeContiguousStride(const ShapeExpr &shape) const;
    bool checkValid() const;
    ShapeExpr makeShapeExpr(const Shape &shape) const;
    StrideExpr makeStrideExpr(const Stride &stride) const;

    template <typename T>
    void printDataImpl(const Runtime &runtime, size_t maxElements = 0,
                       int precision = 4) const;
};

} // namespace infini
#endif // TENSOR_H
