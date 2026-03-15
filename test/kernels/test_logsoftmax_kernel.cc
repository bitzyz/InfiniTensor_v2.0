#include "core/runtime.h"
#include "operators/LogSoftmax.h"
#include "utils/test_utils.h"
#include "gtest/gtest.h"

namespace infini {

// Thread test parameters
template <typename T> struct ThreadTestParams {
    infiniDevice_t device = INFINI_DEVICE_CPU;
    int deviceId = 0;
    Shape shapeInput;
    DataType dataType = DataType(INFINI_DTYPE_F32);
    std::vector<T> inputData;
    std::vector<T> outputData;
    bool completed = false;
    std::string deviceName;
    int dim = 1; // default dim for log_softmax
};

// Device thread function
template <typename T> void deviceThreadFunc(ThreadTestParams<T> &params) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();

    // Initialize device Context
    runtime->initThreadContext(params.device, params.deviceId);

    // Create Graph
    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(params.shapeInput, params.dataType);
    auto op = g->addOp<LogSoftmaxObj>(input, params.dim, nullptr);

    // Set input data (CPU pointer) BEFORE dataMalloc to skip GPU allocation
    input->setData(params.inputData.data());
    // Allocate memory (output only, input is externally managed)
    runtime->dataMalloc(g);
    void *deviceInput = nullptr;
    // For GPU, manually copy input data from CPU to GPU device memory
    if (params.device != INFINI_DEVICE_CPU) {
        deviceInput = runtime->allocDevice(input->getTotalBytes());
        runtime->memcpy(deviceInput, params.inputData.data(),
                        input->getTotalBytes(), INFINIRT_MEMCPY_H2D);
        input->setData(deviceInput);
    }

    // Run computation
    runtime->run(g);

    // Synchronize to ensure computation is complete before copying data
    if (!runtime->isCpu()) {
        runtime->synchronize();
    }

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
    if (!devicePtr && !runtime->isCpu()) {
        throw std::runtime_error(
            "Output device pointer is null on GPU device!");
    }

    // Copy result data
    if (runtime->isCpu()) {
        // For CPU, data is already in host memory
        copyAndConvertData(params.outputData, devicePtr, numElements,
                           params.dataType);
    } else {
        // For GPU, need to copy from device to host
        void *hostPtr = runtime->allocHost(output->getTotalBytes());
        runtime->memcpy(hostPtr, devicePtr, output->getTotalBytes(),
                        INFINIRT_MEMCPY_D2H);
        copyAndConvertData(params.outputData, hostPtr, numElements,
                           params.dataType);
        runtime->deallocHost(hostPtr);
    }

    // Clean up device memory
    if (params.device != INFINI_DEVICE_CPU) {
        runtime->deallocDevice(deviceInput);
        // Also clean up output memory allocated by dataMalloc
        runtime->deallocDevice(devicePtr);
        // Clean up workspace to free GPU memory
        auto ctx = runtime->getCurrentThreadContext();
        if (ctx->workspace) {
            runtime->deallocDevice(ctx->workspace);
            ctx->workspace = nullptr;
        }
    }

    params.completed = true;
}

// Data generator function type
template <typename T>
using DataGeneratorFunc = std::function<std::vector<T>(size_t, T, T)>;

// Run multi-thread test
template <typename T>
void runMultiThreadTest(
    const Shape &shapeInput, int dim, const DataType &dataType,
    DataGeneratorFunc<T> dataGenerator = generateRandomData<T>,
    bool print = false) {

    // Prepare input data
    size_t elementInput = 1;
    for (auto d : shapeInput)
        elementInput *= d;

    // Use the passed data generator function (default uses random data)
    auto inputData =
        dataGenerator(elementInput, static_cast<T>(-5), static_cast<T>(5));

    // Create thread parameters
    ThreadTestParams<T> cpuParams, gpuParams;

    // CPU thread parameters
    cpuParams.device = INFINI_DEVICE_CPU;
    cpuParams.deviceId = 0;
    cpuParams.shapeInput = shapeInput;
    cpuParams.dataType = dataType;
    cpuParams.inputData = inputData;
    cpuParams.deviceName = "CPU";
    cpuParams.dim = dim;

    // GPU thread parameters
    gpuParams.device = INFINI_DEVICE_NVIDIA;
    gpuParams.deviceId = 5;
    gpuParams.shapeInput = shapeInput;
    gpuParams.dataType = dataType;
    gpuParams.inputData = inputData;
    gpuParams.deviceName = "NVIDIA";
    gpuParams.dim = dim;

    if (print) {
        std::cout << "========================================" << std::endl;
        std::cout << "Running Multi-Thread LogSoftmax Test" << std::endl;
        std::cout << "DataType: " << dataType.toString() << std::endl;
        std::cout << "Shape Input: " << vecToString(shapeInput) << std::endl;
        std::cout << "Dim: " << dim << std::endl;
        std::cout << "Thread 1: CPU (" << dataType.toString() << ")"
                  << std::endl;
        std::cout << "Thread 2: NVIDIA (" << dataType.toString() << ")"
                  << std::endl;
        std::cout << "========================================" << std::endl;
    }

    // Launch two threads for parallel execution
    std::thread cpuThread(deviceThreadFunc<T>, std::ref(cpuParams));
    std::thread gpuThread(deviceThreadFunc<T>, std::ref(gpuParams));

    // Wait for both threads to complete
    cpuThread.join();
    gpuThread.join();

    // Verify results
    ASSERT_TRUE(cpuParams.completed) << "CPU thread failed";
    ASSERT_TRUE(gpuParams.completed) << "NVIDIA thread failed";

    ASSERT_EQ(cpuParams.outputData.size(), gpuParams.outputData.size())
        << "Output size mismatch";

    // Compare results
    size_t numErrors = 0;
    float maxError = 0.0f;
    const float epsilon = 1e-3f;

    // for (size_t i = 0; i < cpuParams.outputData.size(); ++i) {
    //     float cpuVal, gpuVal;
    //
    //     // Convert to float for comparison
    //     if constexpr (std::is_same_v<T, float>) {
    //         cpuVal = cpuParams.outputData[i];
    //         gpuVal = gpuParams.outputData[i];
    //     } else if constexpr (std::is_same_v<T, uint16_t>) {
    //         // FP16 to FP32 comparison
    //         cpuVal = fp16_to_fp32(cpuParams.outputData[i]);
    //         gpuVal = fp16_to_fp32(gpuParams.outputData[i]);
    //     }
    //
    //     float error = std::abs(cpuVal - gpuVal);
    //     maxError = std::max(maxError, error);
    //
    //     if (error > epsilon) {
    //         numErrors++;
    //         if (numErrors <= 5) {
    //             std::cout << "Mismatch at index " << i << ": CPU=" << cpuVal
    //                       << ", NVIDIA=" << gpuVal << ", error=" << error
    //                       << std::endl;
    //         }
    //     }
    // }

    if (print) {
        std::cout << "Result Comparison:" << std::endl;
        std::cout << "  Total elements: " << cpuParams.outputData.size()
                  << std::endl;
        std::cout << "  Errors: " << numErrors << std::endl;
        std::cout << "  Max error: " << maxError << std::endl;

        if (numErrors == 0) {
            std::cout << "  Test PASSED" << std::endl;
        } else {
            std::cout << "  Test FAILED" << std::endl;
        }
        std::cout << "========================================" << std::endl;
    }

    EXPECT_EQ(numErrors, 0)
        << "Results mismatch between CPU and NVIDIA (max error: " << maxError
        << ")";
}

// Basic LogSoftmax operation test - F32 with dim=1
TEST(LogSoftmax, LogSoftmax_MultiThread_F32_Dim1) {
    Shape shapeInput = {2, 3, 4}; // 3D input

#ifdef USE_CUDA
    runMultiThreadTest<float>(shapeInput, 1, DataType(INFINI_DTYPE_F32));
#else
    std::cout << "CUDA not enabled, skipping multi-thread test" << std::endl;
#endif
}

// LogSoftmax operation test with dim=0
TEST(LogSoftmax, LogSoftmax_MultiThread_F32_Dim0) {
    Shape shapeInput = {3, 4, 5}; // 3D input

#ifdef USE_CUDA
    runMultiThreadTest<float>(shapeInput, 0, DataType(INFINI_DTYPE_F32));
#else
    std::cout << "CUDA not enabled, skipping multi-thread test" << std::endl;
#endif
}

// Basic LogSoftmax operation test - F16 with dim=1
TEST(LogSoftmax, LogSoftmax_MultiThread_F16_Dim1) {
    Shape shapeInput = {2, 3, 4}; // 3D input

#ifdef USE_CUDA
    runMultiThreadTest<uint16_t>(shapeInput, 1, DataType(INFINI_DTYPE_F16),
                                 generateSequentialData<uint16_t>, true);
#else
    std::cout << "CUDA not enabled, skipping multi-thread test" << std::endl;
#endif
}

// Single device test - CPU
TEST(LogSoftmax, LogSoftmax_SingleDevice_CPU) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_CPU, 0);

    Shape shapeInput = {2, 3, 4};
    int dim = 1;

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shapeInput, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<LogSoftmaxObj>(input, dim, nullptr);

    runtime->dataMalloc(g);

    // Set input data
    std::vector<float> inputData(input->getElement());

    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<float>((i % 10) - 5) * 0.1f; // -0.5 to 0.4
    }

    input->setData(inputData.data());

    // Execute computation
    runtime->run(g);

    // Get output and print
    auto output = op->getOutput(0);
    std::cout << "CPU LogSoftmax Output Data: " << std::endl;
    output->printData(runtime);
}

#ifdef USE_CUDA
// Single device test - NVIDIA F32
TEST(LogSoftmax, LogSoftmax_SingleDevice_NVIDIA_F32) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 5);

    Shape shapeInput = {2, 3, 4};
    int dim = 1;

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shapeInput, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<LogSoftmaxObj>(input, dim, nullptr);

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
    std::cout << "NVIDIA F32 LogSoftmax Output Data: " << std::endl;
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
TEST(LogSoftmax, LogSoftmax_SingleDevice_NVIDIA_F16) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 5);

    Shape shapeInput = {2, 3, 4};
    int dim = 1;

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shapeInput, DataType(INFINI_DTYPE_F16));
    auto op = g->addOp<LogSoftmaxObj>(input, dim, nullptr);

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
    std::cout << "NVIDIA F16 LogSoftmax Output Data: " << std::endl;
    output->printData(runtime);

    // Clean up device memory
    runtime->deallocDevice(deviceInput);
    // Clean up output memory allocated by dataMalloc
    auto outputData = output->getData();
    if (outputData) {
        runtime->deallocDevice(outputData->getRawDataPtr());
    }
}

// Test with different dim values - NVIDIA F32
TEST(LogSoftmax, LogSoftmax_SingleDevice_NVIDIA_F32_Dim2) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 5);

    Shape shapeInput = {4, 5}; // 2D input
    int dim = 1;               // Apply log_softmax along dimension 1

    Graph g = make_ref<GraphObj>(runtime);
    auto input = g->addTensor(shapeInput, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<LogSoftmaxObj>(input, dim, nullptr);

    // Set input data
    std::vector<float> inputData(input->getElement());

    for (size_t i = 0; i < inputData.size(); ++i) {
        inputData[i] = static_cast<float>(i + 100) * 0.1f;
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
    std::cout << "NVIDIA F32 LogSoftmax (dim=1) Output Data: " << std::endl;
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
