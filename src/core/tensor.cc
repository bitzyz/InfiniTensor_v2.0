#include "core/tensor.h"
#include "core/operator.h"
#include "core/runtime.h"

#include <cmath>
#include <iomanip>
#include <numeric>

namespace infini {

TensorObj::TensorObj(Shape shape_, DataType dtype)
    : dtype(dtype), shape(std::move(shape_)),
      stride(computeContiguousStride(shape)) {
  IT_ASSERT(checkValid());
}

TensorObj::TensorObj(Shape shape_, Stride stride_, DataType dtype)
    : dtype(dtype), shape(std::move(shape_)), stride(std::move(stride_)) {
  IT_ASSERT(checkValid());
}

UidBaseType TensorObj::getFuid() const { return fuid; }
DataType TensorObj::getDataType() const { return dtype; }

Shape TensorObj::getShape() const { return shape; }

void TensorObj::setShape(Shape shape_) {
  shape = std::move(shape_);
  stride = computeContiguousStride(shape);
}

Stride TensorObj::getStride() const { return stride; }

void TensorObj::setStride(Stride stride_) { stride = std::move(stride_); }

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
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies{});
}

ElementType TensorObj::getStorageSize() const {
  size_t max_offset = 0;
  size_t min_offset = 0;
  size_t storageSize = 1;
  if (shape.empty()) {
    return storageSize; // 标量 Tensor
  }
  for (size_t i = 0; i < getRank(); ++i) {
    if (stride[i] >= 0) {
      max_offset += (shape[i] - 1) * stride[i];
    } else {
      min_offset += (shape[i] - 1) * stride[i];
    }
  }
  storageSize = max_offset - min_offset + 1;
  return storageSize;
}

ElementType TensorObj::getTotalBytes() const {
  return getStorageSize() * dtype.getSize();
}

ElementType TensorObj::getRank() const { return shape.size(); }

OpVec TensorObj::getTargets() const { return wrefs_to_refs(targets); }

Operator TensorObj::getSource() const { return source.lock(); }

string TensorObj::toString() const {
  // Convert data pointer to string
  std::stringstream ss;
  if (data != nullptr)
    ss << data->getPtr<void *>();
  else
    ss << "nullptr data";
  string ret = "Tensor " + std::to_string(guid) + ", Fuid " +
               std::to_string(fuid) + ", shape " + vecToString(shape) +
               ", stride " + vecToString(stride) + ", dtype " +
               dtype.toString() + ", " + ss.str() + "\n";
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

Stride TensorObj::computeContiguousStride(const Shape &shape) const {
  Stride stride(getRank());
  StrideElem p = 1;
  for (auto i = getRank(); i > 0; --i) {
    stride[i - 1] = p;
    p = p * shape[i - 1];
  }
  return stride;
}

bool TensorObj::checkValid() const {
  IT_ASSERT(shape.size() == stride.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    IT_ASSERT(shape[i] > 0);
  }
  return true;
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
  IT_ASSERT(data != nullptr);
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
    size_t offset = calculateLinearOffset(i, shape, stride);

    if constexpr (std::is_floating_point_v<T> || std::is_same_v<T, double>) {
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
  IT_ASSERT(data != nullptr);
  IT_ASSERT(device != INFINI_DEVICE_CPU);
  void *data_ptr = runtime->allocHost(getTotalBytes());
  runtime->memcpy(data_ptr, data->getPtr<void *>(), getTotalBytes(),
                  INFINIRT_MEMCPY_D2H);
  runtime->deallocDevice(data->getPtr<void *>());
  setData(data_ptr);
  device = INFINI_DEVICE_CPU;
}

void TensorObj::copyToDevice(const Runtime &runtime) {
  IT_ASSERT(data != nullptr);
  IT_ASSERT(device == INFINI_DEVICE_CPU);
  void *data_ptr = runtime->allocDevice(getTotalBytes());
  runtime->memcpy(data_ptr, data->getPtr<void *>(), getTotalBytes(),
                  INFINIRT_MEMCPY_H2D);
  setData(data_ptr);
  device = runtime->getCurrentThreadContext()->device;
}
}; // namespace infini
