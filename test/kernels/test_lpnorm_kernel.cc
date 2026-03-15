#include "core/runtime.h"
#include "operators/LpNorm.h"
#include "utils/test_utils.h"
#include "gtest/gtest.h"

namespace infini {

// Single device test parameters
template <typename T> struct TestParams {
    infiniDevice_t device = INFINI_DEVICE_NVIDIA;
    int deviceId = 5;
    Shape shapeInput;
    DataType dataType = DataType(INFINI_DTYPE_F32);
    std::vector<T> inputData;
    std::vector<T> outputData;
    bool completed = false;
    std::string deviceName;
    int axis = 1;       // default axis for lp_norm
    int p = 2;          // default p for L2 norm
    float eps = 1e-12f; // default epsilon
};

#ifdef USE_CUDA
// Single device test - NVIDIA F32
TEST(LpNorm, LpNorm_SingleDevice_NVIDIA_F32) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 5);

    Shape shapeInput = {2, 3, 4};
    int axis = 1;
    int p = 2;
    float eps = 1e-12f;

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shapeInput, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<LpNormObj>(input, axis, p, eps, nullptr);

    // Set input data
    std::vector<float> inputData(input->getElement());

    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<float>((i % 10) - 5) * 0.1f; // -0.5 to 0.4
    }

    // Set input data (CPU pointers) BEFORE dataMalloc to skip GPU allocation
    input->setData(inputData.data());
    // Allocate memory (output only, input is externally managed)
    runtime->dataMalloc(g);

    // Manually copy input data from CPU to GPU device memory
    void *deviceInput = runtime->allocDevice(input->getTotalBytes());
    runtime->memcpy(deviceInput, inputData.data(), input->getTotalBytes(),
                    INFINIRT_MEMCPY_H2D);
    input->setData(deviceInput);

    // Execute computation
    runtime->run(g);

    // Synchronize to ensure computation is complete before printing
    runtime->synchronize();

    // Get output and print
    auto output = op->getOutput(0);
    std::cout << "NVIDIA F32 LpNorm Output Data: " << std::endl;
    output->printData(runtime);

    // Clean up device memory
    runtime->deallocDevice(deviceInput);
    // Clean up output memory allocated by dataMalloc
    auto outputData = output->getData();
    if (outputData) {
        runtime->deallocDevice(outputData->getRawDataPtr());
    }
}

// Single device test - NVIDIA F16
TEST(LpNorm, LpNorm_SingleDevice_NVIDIA_F16) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 5);

    Shape shapeInput = {2, 3, 4};
    int axis = 1;
    int p = 2;
    float eps = 1e-12f;

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shapeInput, DataType(INFINI_DTYPE_F16));
    auto op = g->addOp<LpNormObj>(input, axis, p, eps, nullptr);

    // Set input data
    std::vector<uint16_t> inputData(input->getElement());

    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = fp32_to_fp16(static_cast<float>((i % 10) - 5) * 0.1f);
    }

    // Set input data (CPU pointers) BEFORE dataMalloc to skip GPU allocation
    input->setData(inputData.data());
    // Allocate memory (output only, input is externally managed)
    runtime->dataMalloc(g);

    // Manually copy input data from CPU to GPU device memory
    void *deviceInput = runtime->allocDevice(input->getTotalBytes());
    runtime->memcpy(deviceInput, inputData.data(), input->getTotalBytes(),
                    INFINIRT_MEMCPY_H2D);
    input->setData(deviceInput);

    // Execute computation
    runtime->run(g);

    // Synchronize to ensure computation is complete before printing
    runtime->synchronize();

    // Get output and print
    auto output = op->getOutput(0);
    std::cout << "NVIDIA F16 LpNorm Output Data: " << std::endl;
    output->printData(runtime);

    // Clean up device memory
    runtime->deallocDevice(deviceInput);
    // Clean up output memory allocated by dataMalloc
    auto outputData = output->getData();
    if (outputData) {
        runtime->deallocDevice(outputData->getRawDataPtr());
    }
}

// Test with L1 norm - NVIDIA F32
TEST(LpNorm, LpNorm_SingleDevice_NVIDIA_F32_L1) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 5);

    Shape shapeInput = {4, 5}; // 2D input
    int axis = 1;              // Apply lp_norm along dimension 1
    int p = 1;                 // L1 norm
    float eps = 1e-12f;

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shapeInput, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<LpNormObj>(input, axis, p, eps, nullptr);

    // Set input data
    std::vector<float> inputData(input->getElement());

    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<float>(i + 1) * 0.1f;
    }

    // Set input data (CPU pointers) BEFORE dataMalloc to skip GPU allocation
    input->setData(inputData.data());
    // Allocate memory (output only, input is externally managed)
    runtime->dataMalloc(g);

    // Manually copy input data from CPU to GPU device memory
    void *deviceInput = runtime->allocDevice(input->getTotalBytes());
    runtime->memcpy(deviceInput, inputData.data(), input->getTotalBytes(),
                    INFINIRT_MEMCPY_H2D);
    input->setData(deviceInput);

    // Execute computation
    runtime->run(g);

    // Synchronize to ensure computation is complete before printing
    runtime->synchronize();

    // Get output and print
    auto output = op->getOutput(0);
    std::cout << "NVIDIA F32 LpNorm (L1, axis=1) Output Data: " << std::endl;
    output->printData(runtime);

    // Clean up device memory
    runtime->deallocDevice(deviceInput);
    // Clean up output memory allocated by dataMalloc
    auto outputData = output->getData();
    if (outputData) {
        runtime->deallocDevice(outputData->getRawDataPtr());
    }
}
#endif

} // namespace infini
