#include "core/runtime.h"
#include "operators/RmsNorm.h"
#include "utils/test_utils.h"
#include "gtest/gtest.h"

namespace infini {

// Single device test parameters
template <typename T> struct TestParams {
    infiniDevice_t device = INFINI_DEVICE_NVIDIA;
    int deviceId = 5;
    Shape shapeInput;
    Shape shapeWeight;
    DataType dataType = DataType(INFINI_DTYPE_F32);
    std::vector<T> inputData;
    std::vector<T> weightData;
    std::vector<T> outputData;
    bool completed = false;
    std::string deviceName;
    float eps = 1e-5f;
};

// Device test function for single GPU test
template <typename T> void deviceTestFunc(TestParams<T> &params) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();

    // Initialize device Context
    runtime->initThreadContext(params.device, params.deviceId);

    // Create Graph
    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(params.shapeInput, params.dataType);
    auto weight = g->addTensor(params.shapeWeight, params.dataType);
    auto op = g->addOp<RmsNormObj>(input, weight, params.eps, nullptr);

    // Set input data (CPU pointer) BEFORE dataMalloc to skip GPU allocation
    input->setData(params.inputData.data());
    weight->setData(params.weightData.data());
    // Allocate memory (output only, input is externally managed)
    runtime->dataMalloc(g);

    // Manually copy input data from CPU to GPU device memory
    void *deviceInput = runtime->allocDevice(input->getTotalBytes());
    runtime->memcpy(deviceInput, params.inputData.data(),
                    input->getTotalBytes(), INFINIRT_MEMCPY_H2D);
    input->setData(deviceInput);

    void *deviceWeight = runtime->allocDevice(weight->getTotalBytes());
    runtime->memcpy(deviceWeight, params.weightData.data(),
                    weight->getTotalBytes(), INFINIRT_MEMCPY_H2D);
    weight->setData(deviceWeight);

    // Run computation
    runtime->run(g);

    // Synchronize to ensure computation is complete before copying data
    runtime->synchronize();

    // Get output and copy to host
    auto output = op->getOutput(0);
    size_t numElements = output->getElement();
    params.outputData.resize(numElements);

    // Check if output data exists
    auto dataBlob = output->getData();
    if (!dataBlob) {
        throw std::runtime_error("Output data blob is null!");
    }
    void *devicePtr = dataBlob->getRawDataPtr();
    if (!devicePtr) {
        throw std::runtime_error(
            "Output device pointer is null on GPU device!");
    }

    // Copy result data from device to host
    void *hostPtr = runtime->allocHost(output->getTotalBytes());
    runtime->memcpy(hostPtr, devicePtr, output->getTotalBytes(),
                    INFINIRT_MEMCPY_D2H);
    copyAndConvertData(params.outputData, hostPtr, numElements,
                       params.dataType);
    runtime->deallocHost(hostPtr);

    // Clean up device memory
    runtime->deallocDevice(deviceInput);
    runtime->deallocDevice(deviceWeight);
    runtime->deallocDevice(devicePtr);
    // Note: Do NOT clean up workspace here, as it will break subsequent tests
    // The workspace is initialized once per thread context and reused across
    // tests

    params.completed = true;
}

// Data generator function type
template <typename T>
using DataGeneratorFunc = std::function<std::vector<T>(size_t, T, T)>;

// Run single GPU test
template <typename T>
void runGpuTest(const Shape &shapeInput, const Shape &shapeWeight, float eps,
                const DataType &dataType,
                DataGeneratorFunc<T> dataGenerator = generateRandomData<T>,
                bool print = false) {

    // Prepare input data
    size_t elementInput = 1;
    for (auto d : shapeInput)
        elementInput *= d;

    size_t elementWeight = 1;
    for (auto d : shapeWeight)
        elementWeight *= d;

    // Use the passed data generator function (default uses random data)
    auto inputData =
        dataGenerator(elementInput, static_cast<T>(-5), static_cast<T>(5));
    auto weightData =
        dataGenerator(elementWeight, static_cast<T>(-1), static_cast<T>(1));

    // Create test parameters
    TestParams<T> gpuParams;

    // GPU parameters
    gpuParams.device = INFINI_DEVICE_NVIDIA;
    gpuParams.deviceId = 5;
    gpuParams.shapeInput = shapeInput;
    gpuParams.shapeWeight = shapeWeight;
    gpuParams.dataType = dataType;
    gpuParams.inputData = inputData;
    gpuParams.weightData = weightData;
    gpuParams.deviceName = "NVIDIA";
    gpuParams.eps = eps;

    if (print) {
        std::cout << "========================================" << std::endl;
        std::cout << "Running RmsNorm Test on NVIDIA GPU" << std::endl;
        std::cout << "DataType: " << dataType.toString() << std::endl;
        std::cout << "Shape Input: " << vecToString(shapeInput) << std::endl;
        std::cout << "Shape Weight: " << vecToString(shapeWeight) << std::endl;
        std::cout << "Eps: " << eps << std::endl;
        std::cout << "========================================" << std::endl;
    }

    // Run test
    deviceTestFunc<T>(gpuParams);

    // Verify results
    ASSERT_TRUE(gpuParams.completed) << "GPU test failed";
    ASSERT_EQ(gpuParams.outputData.size(), elementInput)
        << "Output size mismatch";

    if (print) {
        std::cout << "Result:" << std::endl;
        std::cout << "  Total elements: " << gpuParams.outputData.size()
                  << std::endl;
        std::cout << "  Test PASSED" << std::endl;
        std::cout << "========================================" << std::endl;
    }
}

// Basic RmsNorm operation test - F32
TEST(RmsNorm, RmsNorm_GPU_F32) {
#ifdef USE_CUDA
    Shape shapeInput = {2, 3, 4}; // 3D input
    Shape shapeWeight = {4};      // last dim
    runGpuTest<float>(shapeInput, shapeWeight, 1e-5f,
                      DataType(INFINI_DTYPE_F32));
#else
    std::cout << "CUDA not enabled, skipping test" << std::endl;
#endif
}

// RmsNorm operation test with different shape
TEST(RmsNorm, RmsNorm_GPU_F32_Larger) {
#ifdef USE_CUDA
    Shape shapeInput = {4, 8, 16}; // 3D input
    Shape shapeWeight = {16};      // last dim
    runGpuTest<float>(shapeInput, shapeWeight, 1e-5f,
                      DataType(INFINI_DTYPE_F32));
#else
    std::cout << "CUDA not enabled, skipping test" << std::endl;
#endif
}

// Basic RmsNorm operation test - F16
TEST(RmsNorm, RmsNorm_GPU_F16) {
#ifdef USE_CUDA
    Shape shapeInput = {2, 3, 4}; // 3D input
    Shape shapeWeight = {4};      // last dim
    runGpuTest<uint16_t>(shapeInput, shapeWeight, 1e-5f,
                         DataType(INFINI_DTYPE_F16),
                         generateSequentialData<uint16_t>, true);
#else
    std::cout << "CUDA not enabled, skipping test" << std::endl;
#endif
}

#ifdef USE_CUDA
// Single device test - NVIDIA F32
TEST(RmsNorm, RmsNorm_SingleDevice_NVIDIA_F32) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 5);

    Shape shapeInput = {2, 3, 4};
    Shape shapeWeight = {4};
    float eps = 1e-5f;

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shapeInput, DataType(INFINI_DTYPE_F32));
    auto weight = g->addTensor(shapeWeight, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<RmsNormObj>(input, weight, eps, nullptr);

    // Set input data
    std::vector<float> inputData(input->getElement());
    std::vector<float> weightData(weight->getElement());

    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<float>((i % 10) - 5) * 0.1f; // -0.5 to 0.4
    }
    for (size_t i = 0; i < weightData.size(); ++i) {
        weightData[i] = static_cast<float>((i % 5) - 2) * 0.1f; // -0.2 to 0.2
    }

    // Set input data (CPU pointers) BEFORE dataMalloc to skip GPU allocation
    input->setData(inputData.data());
    weight->setData(weightData.data());
    // Allocate memory (output only, input is externally managed)
    runtime->dataMalloc(g);

    // Manually copy input data from CPU to GPU device memory
    void *deviceInput = runtime->allocDevice(input->getTotalBytes());
    runtime->memcpy(deviceInput, inputData.data(), input->getTotalBytes(),
                    INFINIRT_MEMCPY_H2D);
    input->setData(deviceInput);

    void *deviceWeight = runtime->allocDevice(weight->getTotalBytes());
    runtime->memcpy(deviceWeight, weightData.data(), weight->getTotalBytes(),
                    INFINIRT_MEMCPY_H2D);
    weight->setData(deviceWeight);

    // Execute computation
    runtime->run(g);

    // Synchronize to ensure computation is complete before printing
    runtime->synchronize();

    // Get output and print
    auto output = op->getOutput(0);
    std::cout << "NVIDIA F32 RmsNorm Output Data: " << std::endl;
    output->printData(runtime);

    // Clean up device memory
    runtime->deallocDevice(deviceInput);
    runtime->deallocDevice(deviceWeight);
    // Clean up output memory allocated by dataMalloc
    auto outputData = output->getData();
    if (outputData) {
        runtime->deallocDevice(outputData->getRawDataPtr());
    }
}

// Single device test - NVIDIA F16
TEST(RmsNorm, RmsNorm_SingleDevice_NVIDIA_F16) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 5);

    Shape shapeInput = {2, 3, 4};
    Shape shapeWeight = {4};
    float eps = 1e-5f;

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shapeInput, DataType(INFINI_DTYPE_F16));
    auto weight = g->addTensor(shapeWeight, DataType(INFINI_DTYPE_F16));
    auto op = g->addOp<RmsNormObj>(input, weight, eps, nullptr);

    // Set input data
    std::vector<uint16_t> inputData(input->getElement());
    std::vector<uint16_t> weightData(weight->getElement());

    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = fp32_to_fp16(static_cast<float>((i % 10) - 5) * 0.1f);
    }
    for (size_t i = 0; i < weightData.size(); ++i) {
        weightData[i] = fp32_to_fp16(static_cast<float>((i % 5) - 2) * 0.1f);
    }

    // Set input data (CPU pointers) BEFORE dataMalloc to skip GPU allocation
    input->setData(inputData.data());
    weight->setData(weightData.data());
    // Allocate memory (output only, input is externally managed)
    runtime->dataMalloc(g);

    // Manually copy input data from CPU to GPU device memory
    void *deviceInput = runtime->allocDevice(input->getTotalBytes());
    runtime->memcpy(deviceInput, inputData.data(), input->getTotalBytes(),
                    INFINIRT_MEMCPY_H2D);
    input->setData(deviceInput);

    void *deviceWeight = runtime->allocDevice(weight->getTotalBytes());
    runtime->memcpy(deviceWeight, weightData.data(), weight->getTotalBytes(),
                    INFINIRT_MEMCPY_H2D);
    weight->setData(deviceWeight);

    // Execute computation
    runtime->run(g);

    // Synchronize to ensure computation is complete before printing
    runtime->synchronize();

    // Get output and print
    auto output = op->getOutput(0);
    std::cout << "NVIDIA F16 RmsNorm Output Data: " << std::endl;
    output->printData(runtime);

    // Clean up device memory
    runtime->deallocDevice(deviceInput);
    runtime->deallocDevice(deviceWeight);
    // Clean up output memory allocated by dataMalloc
    auto outputData = output->getData();
    if (outputData) {
        runtime->deallocDevice(outputData->getRawDataPtr());
    }
}
#endif

} // namespace infini
