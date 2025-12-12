#include "core/tensor.h"
#include "core/operator.h"
#include "core/runtime.h"

#include <cmath>
#include <iomanip>
#include <numeric>

namespace infini {

TensorObj::TensorObj(ShapeExpr symbolic_shape, DataType dtype)
    : dtype(dtype), shape(symbolic_shape) {
    stride = computeContiguousStride(shape);
    IT_ASSERT(checkValid());
}

TensorObj::TensorObj(ShapeExpr symbolic_shape, StrideExpr stride,
                     DataType dtype)
    : dtype(dtype), shape(symbolic_shape), stride(stride) {
    IT_ASSERT(checkValid());
}

TensorObj::TensorObj(ShapeExpr symbolic_shape, Stride stride_, DataType dtype)
    : dtype(dtype), shape(symbolic_shape) {
    stride = makeStrideExpr(stride_);
    IT_ASSERT(checkValid());
}

TensorObj::TensorObj(Shape shape_, DataType dtype) : dtype(dtype) {
    shape = makeShapeExpr(shape_);
    stride = computeContiguousStride(shape);
    IT_ASSERT(checkValid());
}

TensorObj::TensorObj(Shape shape_, StrideExpr stride_, DataType dtype)
    : dtype(dtype), stride(std::move(stride_)) {
    shape = makeShapeExpr(shape_);
    IT_ASSERT(checkValid());
}

TensorObj::TensorObj(Shape shape_, Stride stride_, DataType dtype)
    : dtype(dtype) {
    shape = makeShapeExpr(shape_);
    stride = makeStrideExpr(stride_);
    IT_ASSERT(checkValid());
}

UidBaseType TensorObj::getFuid() const { return fuid; }
DataType TensorObj::getDataType() const { return dtype; }

ShapeExpr TensorObj::getShape() const { return shape; }

void TensorObj::setShape(ShapeExpr shape_) {
    shape = std::move(shape_);
    stride = computeContiguousStride(shape);
}

void TensorObj::setShape(Shape shape_) {
    shape = makeShapeExpr(shape_);
    stride = computeContiguousStride(shape);
}

StrideExpr TensorObj::getStride() const { return stride; }

void TensorObj::setStride(StrideExpr stride_) { stride = std::move(stride_); }

void TensorObj::setStride(Stride stride_) { stride = makeStrideExpr(stride_); }

Blob TensorObj::getData() const { return data; }

void TensorObj::setData(void *data_) {
    IT_ASSERT(data_ != nullptr);
    data = std::make_shared<BlobObj>(data_);
}

void TensorObj::dataMalloc(const Runtime &runtime) {
    if (data == nullptr) {
        data = make_ref<BlobObj>(runtime->allocDevice(getTotalBytes()));
        device = runtime->getCurrentThreadContext()->device;
    } else {
        if (runtime->getCurrentThreadContext()->device != device &&
            device == INFINI_DEVICE_CPU) {
            void *data_ptr = runtime->allocDevice(getTotalBytes());
            runtime->memcpy(data_ptr, data->getPtr<void *>(), getTotalBytes(),
                            INFINIRT_MEMCPY_H2D);
            setData(data_ptr);
        }
    }
}

ElementType TensorObj::getElement() const {
    Shape constant_shape = shape->getConstantValue();
    return std::accumulate(constant_shape.begin(), constant_shape.end(), 1,
                           std::multiplies{});
}

ElementType TensorObj::getStorageSize() const {
    Shape constant_shape = shape->getConstantValue();
    Stride constant_stride = stride->getConstantValue();
    size_t max_offset = 0;
    size_t min_offset = 0;
    size_t storageSize = 1;
    if (constant_shape.empty()) {
        return storageSize; // 标量 Tensor
    }
    for (auto i = 0; i < getRank(); ++i) {
        if (constant_stride[i] >= 0) {
            max_offset += (constant_shape[i] - 1) * constant_stride[i];
        } else {
            min_offset += (constant_shape[i] - 1) * constant_stride[i];
        }
    }
    storageSize = max_offset - min_offset + 1;
    return storageSize;
}

ElementType TensorObj::getTotalBytes() const {
    return getStorageSize() * dtype.getSize();
}

ElementType TensorObj::getRank() const { return shape->size(); }

OpVec TensorObj::getTargets() const { return wrefs_to_refs(targets); }

Operator TensorObj::getSource() const { return source.lock(); }

string TensorObj::toString() const {
    // Convert data pointer to string
    std::stringstream ss;
    if (data != nullptr)
        ss << data->getPtr<void *>();
    else
        ss << "nullptr data";
    string ret =
        "Tensor " + std::to_string(guid) + ", Fuid = " + std::to_string(fuid) +
        ", shape = " + shape->toString() + ", stride = " + stride->toString() +
        ", dtype = " + dtype.toString() + ", " + ss.str() + "\n";
    vector<UidBaseType> targetGuids;
    for (const auto &op : targets)
        targetGuids.emplace_back(op.lock()->getGuid());
    if (auto o = source.lock())
        ret += ", source " + std::to_string(o->getGuid());
    else
        ret += ", source None";
    ret += ", targets " + vecToString(targetGuids);
    return ret;
}

void TensorObj::addTarget(const Operator &op) { targets.emplace_back(op); }
void TensorObj::setSource(const Operator &op) { source = op; }
void TensorObj::removeTarget(const Operator &op) {
    for (auto itr = targets.begin(); itr != targets.end();) {
        if (itr->lock() == op)
            itr = targets.erase(itr);
        else
            ++itr;
    }
}

StrideExpr TensorObj::computeContiguousStride(const ShapeExpr &shape) const {
    auto rank = shape->size();
    vector<Expr> strides(rank);
    Expr acc = ExprObj::constant(1);
    for (auto i = rank; i > 0; --i) {
        strides[i - 1] = acc;
        acc = acc * (*shape)[i - 1];
    }
    auto strides_expr = make_ref<StrideExprObj>(strides);
    if (shape->isConcrete()) {
        strides_expr = strides_expr->simplify();
    }
    return strides_expr;
}

bool TensorObj::checkValid() const {
    IT_ASSERT(shape->size() == stride->size());
    for (size_t i = 0; i < shape->size(); ++i) {
        const Expr &dim = (*shape)[i];
        if (dim->getType() == ExprObj::Type::CONSTANT) {
            IT_ASSERT(as<ConstantExprObj>(dim)->asConstant() > 0);
        }
    }
    return true;
}

ShapeExpr TensorObj::makeShapeExpr(const Shape &shape_) const {
    vector<Expr> dims;
    dims.reserve(shape_.size());
    for (auto v : shape_) {
        dims.push_back(ExprObj::constant(v));
    }
    return make_ref<ShapeExprObj>(dims);
}

StrideExpr TensorObj::makeStrideExpr(const Stride &stride_) const {
    vector<Expr> strides;
    strides.reserve(stride_.size());
    for (auto v : stride_) {
        strides.push_back(ExprObj::constant(v));
    }
    return make_ref<StrideExprObj>(strides);
}

void TensorObj::printData(const Runtime &runtime, size_t maxElements,
                          int precision) const {
    IT_ASSERT(data != nullptr);
    switch (dtype.getType()) {
    case INFINI_DTYPE_F32:
        printDataImpl<float>(runtime, maxElements, precision);
        break;
    case INFINI_DTYPE_F64:
        printDataImpl<double>(runtime, maxElements, precision);
        break;
    case INFINI_DTYPE_I32:
        printDataImpl<int32_t>(runtime, maxElements, precision);
        break;
    default:
        IT_TODO_HALT_MSG("unsupported data type");
    }
}

template <typename T>
void TensorObj::printDataImpl(const Runtime &runtime, size_t maxElements,
                              int precision) const {
    IT_ASSERT(data != nullptr && shape->isConcrete() && stride->isConcrete());
    auto constant_shape = shape->getConstantValue();
    auto constant_stride = stride->getConstantValue();
    void *data_ptr = runtime->allocHost(getTotalBytes());
    runtime->memcpy(data_ptr, data->getPtr<void *>(), getTotalBytes(),
                    INFINIRT_MEMCPY_D2H);
    size_t totalElements = getElement();
    if (maxElements == 0) {
        maxElements = totalElements;
    }
    size_t printCount = std::min(totalElements, maxElements);
    T *typed_data = static_cast<T *>(data_ptr);
    std::cout << "Data: [";
    for (size_t i = 0; i < printCount; ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        size_t offset =
            calculateLinearOffset(i, constant_shape, constant_stride);

        if constexpr (std::is_floating_point_v<T> ||
                      std::is_same_v<T, double>) {
            std::cout << std::setprecision(precision) << typed_data[offset];
        } else if constexpr (std::is_same_v<T, bool>) {
            std::cout << (typed_data[offset] ? "true" : "false");
        } else {
            std::cout << static_cast<int64_t>(typed_data[offset]);
        }
    }
    if (printCount < totalElements) {
        std::cout << ", ... (" << totalElements - printCount << " more)";
    }
    std::cout << "]" << std::endl;
    runtime->deallocHost(data_ptr);
}

template void TensorObj::printDataImpl<float>(const Runtime &, size_t,
                                              int) const;
template void TensorObj::printDataImpl<double>(const Runtime &, size_t,
                                               int) const;
template void TensorObj::printDataImpl<int32_t>(const Runtime &, size_t,
                                                int) const;

void TensorObj::copyToHost(const Runtime &runtime) {
    IT_ASSERT(data != nullptr && shape->isConcrete() && stride->isConcrete());
    IT_ASSERT(device != INFINI_DEVICE_CPU);
    void *data_ptr = runtime->allocHost(getTotalBytes());
    runtime->memcpy(data_ptr, data->getPtr<void *>(), getTotalBytes(),
                    INFINIRT_MEMCPY_D2H);
    runtime->deallocDevice(data->getPtr<void *>());
    setData(data_ptr);
    device = INFINI_DEVICE_CPU;
}

void TensorObj::copyToDevice(const Runtime &runtime) {
    IT_ASSERT(data != nullptr && shape->isConcrete() && stride->isConcrete());
    IT_ASSERT(device == INFINI_DEVICE_CPU);
    void *data_ptr = runtime->allocDevice(getTotalBytes());
    runtime->memcpy(data_ptr, data->getPtr<void *>(), getTotalBytes(),
                    INFINIRT_MEMCPY_H2D);
    setData(data_ptr);
    device = runtime->getCurrentThreadContext()->device;
}
}; // namespace infini
