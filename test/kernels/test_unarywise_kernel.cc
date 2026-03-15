#include "core/runtime.h"
#include "operators/UnaryOp.h"
#include "utils/test_utils.h"
#include "gtest/gtest.h"

namespace infini {

// Test parameters for unary operations
template <typename T> struct UnaryTestParams {
    infiniDevice_t device = INFINI_DEVICE_CPU;
    int deviceId = 0;
    OpType opType = OpType::Unknown;
    Shape shapeInput;
    DataType dataType = DataType(INFINI_DTYPE_F32);
    std::vector<T> inputData;
    std::vector<T> outputData;
    bool completed = false;
    std::string deviceName;
};

// Data generator function type
template <typename T>
using DataGeneratorFunc = std::function<std::vector<T>(size_t, T, T)>;

// ==================== Single Device Tests ====================
#ifdef USE_CUDA
// Single device test - ReLU
TEST(UnaryOp, Relu_SingleDevice_NVIDIA_F32) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 5);

    Shape shapeInput = {2, 3, 4};

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shapeInput, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<UnaryWiseObj>(OpType::Relu, input, nullptr);

    // Set input data
    std::vector<float> inputData(input->getElement());
    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<float>((i % 10) - 5) * 0.1f; // -0.5 to 0.4
    }

    input->setData(inputData.data());
    runtime->dataMalloc(g);

    void *deviceInput = runtime->allocDevice(input->getTotalBytes());
    runtime->memcpy(deviceInput, inputData.data(), input->getTotalBytes(),
                    INFINIRT_MEMCPY_H2D);
    input->setData(deviceInput);

    runtime->run(g);
    runtime->synchronize();

    auto output = op->getOutput(0);
    std::cout << "NVIDIA F32 ReLU Output Data: " << std::endl;
    output->printData(runtime);

    runtime->deallocDevice(deviceInput);
    auto outputData = output->getData();
    if (outputData) {
        runtime->deallocDevice(outputData->getRawDataPtr());
    }
}

// Single device test - Sigmoid
TEST(UnaryOp, Sigmoid_SingleDevice_NVIDIA_F32) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 5);

    Shape shapeInput = {2, 3, 4};

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shapeInput, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<UnaryWiseObj>(OpType::Sigmoid, input, nullptr);

    std::vector<float> inputData(input->getElement());
    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<float>((i % 10) - 5) * 0.1f;
    }

    input->setData(inputData.data());
    runtime->dataMalloc(g);

    void *deviceInput = runtime->allocDevice(input->getTotalBytes());
    runtime->memcpy(deviceInput, inputData.data(), input->getTotalBytes(),
                    INFINIRT_MEMCPY_H2D);
    input->setData(deviceInput);

    runtime->run(g);
    runtime->synchronize();

    auto output = op->getOutput(0);
    std::cout << "NVIDIA F32 Sigmoid Output Data: " << std::endl;
    output->printData(runtime);

    runtime->deallocDevice(deviceInput);
    auto outputData = output->getData();
    if (outputData) {
        runtime->deallocDevice(outputData->getRawDataPtr());
    }
}

// Single device test - SiLU
TEST(UnaryOp, Silu_SingleDevice_NVIDIA_F32) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 5);

    Shape shapeInput = {2, 3, 4};

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shapeInput, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<UnaryWiseObj>(OpType::Silu, input, nullptr);

    std::vector<float> inputData(input->getElement());
    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<float>((i % 10) - 5) * 0.1f;
    }

    input->setData(inputData.data());
    runtime->dataMalloc(g);

    void *deviceInput = runtime->allocDevice(input->getTotalBytes());
    runtime->memcpy(deviceInput, inputData.data(), input->getTotalBytes(),
                    INFINIRT_MEMCPY_H2D);
    input->setData(deviceInput);

    runtime->run(g);
    runtime->synchronize();

    auto output = op->getOutput(0);
    std::cout << "NVIDIA F32 SiLU Output Data: " << std::endl;
    output->printData(runtime);

    runtime->deallocDevice(deviceInput);
    auto outputData = output->getData();
    if (outputData) {
        runtime->deallocDevice(outputData->getRawDataPtr());
    }
}

// Single device test - GELU
TEST(UnaryOp, Gelu_SingleDevice_NVIDIA_F32) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 5);

    Shape shapeInput = {2, 3, 4};

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shapeInput, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<UnaryWiseObj>(OpType::Gelu, input, nullptr);

    std::vector<float> inputData(input->getElement());
    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<float>((i % 10) - 5) * 0.1f;
    }

    input->setData(inputData.data());
    runtime->dataMalloc(g);

    void *deviceInput = runtime->allocDevice(input->getTotalBytes());
    runtime->memcpy(deviceInput, inputData.data(), input->getTotalBytes(),
                    INFINIRT_MEMCPY_H2D);
    input->setData(deviceInput);

    runtime->run(g);
    runtime->synchronize();

    auto output = op->getOutput(0);
    std::cout << "NVIDIA F32 GELU Output Data: " << std::endl;
    output->printData(runtime);

    runtime->deallocDevice(deviceInput);
    auto outputData = output->getData();
    if (outputData) {
        runtime->deallocDevice(outputData->getRawDataPtr());
    }
}

// Single device test - Softplus
TEST(UnaryOp, Softplus_SingleDevice_NVIDIA_F32) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 5);

    Shape shapeInput = {2, 3, 4};

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shapeInput, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<UnaryWiseObj>(OpType::Softplus, input, nullptr);

    std::vector<float> inputData(input->getElement());
    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<float>((i % 10) - 5) * 0.1f;
    }

    input->setData(inputData.data());
    runtime->dataMalloc(g);

    void *deviceInput = runtime->allocDevice(input->getTotalBytes());
    runtime->memcpy(deviceInput, inputData.data(), input->getTotalBytes(),
                    INFINIRT_MEMCPY_H2D);
    input->setData(deviceInput);

    runtime->run(g);
    runtime->synchronize();

    auto output = op->getOutput(0);
    std::cout << "NVIDIA F32 Softplus Output Data: " << std::endl;
    output->printData(runtime);

    runtime->deallocDevice(deviceInput);
    auto outputData = output->getData();
    if (outputData) {
        runtime->deallocDevice(outputData->getRawDataPtr());
    }
}

// Single device test - Tanh
TEST(UnaryOp, Tanh_SingleDevice_NVIDIA_F32) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 5);

    Shape shapeInput = {2, 3, 4};

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shapeInput, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<UnaryWiseObj>(OpType::Tanh, input, nullptr);

    std::vector<float> inputData(input->getElement());
    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<float>((i % 10) - 5) * 0.1f;
    }

    input->setData(inputData.data());
    runtime->dataMalloc(g);

    void *deviceInput = runtime->allocDevice(input->getTotalBytes());
    runtime->memcpy(deviceInput, inputData.data(), input->getTotalBytes(),
                    INFINIRT_MEMCPY_H2D);
    input->setData(deviceInput);

    runtime->run(g);
    runtime->synchronize();

    auto output = op->getOutput(0);
    std::cout << "NVIDIA F32 Tanh Output Data: " << std::endl;
    output->printData(runtime);

    runtime->deallocDevice(deviceInput);
    auto outputData = output->getData();
    if (outputData) {
        runtime->deallocDevice(outputData->getRawDataPtr());
    }
}
#endif

} // namespace infini
