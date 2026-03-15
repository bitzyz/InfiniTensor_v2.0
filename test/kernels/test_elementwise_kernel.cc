#include "core/runtime.h"
#include "operators/ElementWise.h"
#include "utils/test_utils.h"
#include "gtest/gtest.h"

namespace infini {

// Thread test parameters
template <typename T> struct ThreadTestParams {
    infiniDevice_t device = INFINI_DEVICE_NVIDIA;
    int deviceId = 0;
    OpType opType = OpType::Unknown;
    Shape shapeA;
    Shape shapeB;
    DataType dataType = DataType(INFINI_DTYPE_F32);
    std::vector<T> inputAData;
    std::vector<T> inputBData;
    std::vector<T> outputData;
    bool completed = false;
    std::string deviceName;
};

// Device thread function
template <typename T> void deviceThreadFunc(ThreadTestParams<T> &params) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();

    // Initialize device Context
    runtime->initThreadContext(params.device, params.deviceId);

    // Create Graph
    Graph g = make_ref<GraphObj>(runtime);
    auto A = g->addTensor(params.shapeA, params.dataType);
    auto B = g->addTensor(params.shapeB, params.dataType);
    auto op = g->addOp<ElementWiseObj>(params.opType, A, B, nullptr);

    // Set data first (set CPU pointer), then allocate memory (triggers H2D
    // copy)
    A->setData(params.inputAData.data());
    B->setData(params.inputBData.data());
    runtime->dataMalloc(g);

    // Run computation
    runtime->run(g);

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
    void *hostPtr = runtime->allocHost(output->getTotalBytes());
    runtime->memcpy(hostPtr, devicePtr, output->getTotalBytes(),
                    INFINIRT_MEMCPY_D2H);

    // Use generic function for data copy and conversion
    copyAndConvertData(params.outputData, hostPtr, numElements,
                       params.dataType);

    runtime->deallocHost(hostPtr);
    params.completed = true;
}

// Data generator function type
template <typename T>
using DataGeneratorFunc = std::function<std::vector<T>(size_t, T, T)>;

// Run multi-thread test
template <typename T>
void runMultiThreadTest(
    OpType opType, const Shape &shapeA, const Shape &shapeB,
    const DataType &dataType,
    DataGeneratorFunc<T> dataGenerator = generateRandomData<T>,
    bool print = false) {

    // Prepare input data - use utility function to simplify
    size_t elementA = 1, elementB = 1;
    for (auto dim : shapeA)
        elementA *= dim;
    for (auto dim : shapeB)
        elementB *= dim;

    // Use the passed data generator function (default uses
    // random data)
    auto inputAData =
        dataGenerator(elementA, static_cast<T>(-10), static_cast<T>(10));
    auto inputBData =
        dataGenerator(elementB, static_cast<T>(-10), static_cast<T>(10));

    // Create thread parameters
    ThreadTestParams<T> cpuParams, gpuParams;

    // CPU thread parameters
    cpuParams.device = INFINI_DEVICE_NVIDIA;
    cpuParams.deviceId = 0;
    cpuParams.opType = opType;
    cpuParams.shapeA = shapeA;
    cpuParams.shapeB = shapeB;
    cpuParams.dataType = dataType;
    cpuParams.inputAData = inputAData;
    cpuParams.inputBData = inputBData;
    cpuParams.deviceName = "CPU";

    // GPU thread parameters
    gpuParams.device = INFINI_DEVICE_NVIDIA;
    gpuParams.deviceId = 0;
    gpuParams.opType = opType;
    gpuParams.shapeA = shapeA;
    gpuParams.shapeB = shapeB;
    gpuParams.dataType = dataType;
    gpuParams.inputAData = inputAData;
    gpuParams.inputBData = inputBData;
    gpuParams.deviceName = "NVIDIA";

    if (print) {
        std::cout << "========================================" << std::endl;
        std::cout << "Running Multi-Thread ElementWise Test" << std::endl;
        std::cout << "OpType: " << opType.toString() << std::endl;
        std::cout << "DataType: " << dataType.toString() << std::endl;
        std::cout << "Shape A: " << vecToString(shapeA) << std::endl;
        std::cout << "Shape B: " << vecToString(shapeB) << std::endl;
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

    for (size_t i = 0; i < cpuParams.outputData.size(); ++i) {
        float cpuVal, gpuVal;

        // Convert to float for comparison
        if constexpr (std::is_same_v<T, float>) {
            cpuVal = cpuParams.outputData[i];
            gpuVal = gpuParams.outputData[i];
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            // FP16 to FP32 comparison
            cpuVal = fp16_to_fp32(cpuParams.outputData[i]);
            gpuVal = fp16_to_fp32(gpuParams.outputData[i]);
        }

        float error = std::abs(cpuVal - gpuVal);
        maxError = std::max(maxError, error);

        if (error > epsilon) {
            numErrors++;
            if (numErrors <= 5) {
                std::cout << "Mismatch at index " << i << ": CPU=" << cpuVal
                          << ", NVIDIA=" << gpuVal << ", error=" << error
                          << std::endl;
            }
        }
    }

    if (print) {
        std::cout << "Result Comparison:" << std::endl;
        std::cout << "  Total elements: " << cpuParams.outputData.size()
                  << std::endl;
        std::cout << "  Errors: " << numErrors << std::endl;
        std::cout << "  Max error: " << maxError << std::endl;

        if (numErrors == 0) {
            std::cout << "  ✓ Test PASSED" << std::endl;
        } else {
            std::cout << "  ✗ Test FAILED" << std::endl;
        }
        std::cout << "========================================" << std::endl;
    }

    EXPECT_EQ(numErrors, 0) << "Results mismatch between "
                               "CPU and NVIDIA (max error: "
                            << maxError << ")";
}

// Basic Add operation test - F32
TEST(ElementWise, Add_MultiThread_F32) {
    Shape shapeA = {3, 1};
    Shape shapeB = {2, 3, 4};

#ifdef USE_CUDA
    runMultiThreadTest<float>(OpType::Add, shapeA, shapeB,
                              DataType(INFINI_DTYPE_F32));
#else
    std::cout << "CUDA not enabled, skipping multi-thread test" << std::endl;
#endif
}

// Basic Add operation test - F16
TEST(ElementWise, Add_MultiThread_F16) {
    Shape shapeA = {3, 1};
    Shape shapeB = {2, 3, 4};

#ifdef USE_CUDA
    runMultiThreadTest<uint16_t>(OpType::Add, shapeA, shapeB,
                                 DataType(INFINI_DTYPE_F16),
                                 generateSequentialData<uint16_t>, true);
#else
    std::cout << "CUDA not enabled, skipping multi-thread test" << std::endl;
#endif
}

// Basic Mul operation test - F32
TEST(ElementWise, Mul_MultiThread_F32) {
    Shape shapeA = {3, 4};
    Shape shapeB = {3, 4};

#ifdef USE_CUDA
    runMultiThreadTest<float>(OpType::Mul, shapeA, shapeB,
                              DataType(INFINI_DTYPE_F32));
#endif
}

// Basic Mul operation test - F16
TEST(ElementWise, Mul_MultiThread_F16) {
    Shape shapeA = {3, 4};
    Shape shapeB = {3, 4};

#ifdef USE_CUDA
    runMultiThreadTest<uint16_t>(OpType::Mul, shapeA, shapeB,
                                 DataType(INFINI_DTYPE_F16),
                                 generateSequentialData<uint16_t>, false);
#endif
}

// Basic Sub operation test - F32
TEST(ElementWise, Sub_MultiThread_F32) {
    Shape shapeA = {1, 5, 6};
    Shape shapeB = {1, 5, 6};

#ifdef USE_CUDA
    runMultiThreadTest<float>(OpType::Sub, shapeA, shapeB,
                              DataType(INFINI_DTYPE_F32));
#endif
}

// Basic Sub operation test - F16
TEST(ElementWise, Sub_MultiThread_F16) {
    Shape shapeA = {1, 5, 6};
    Shape shapeB = {1, 5, 6};

#ifdef USE_CUDA
    runMultiThreadTest<uint16_t>(OpType::Sub, shapeA, shapeB,
                                 DataType(INFINI_DTYPE_F16),
                                 generateSequentialData<uint16_t>, false);
#endif
}

// Single device test - CPU
TEST(ElementWise, Add_SingleDevice_CPU) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 0);

    Shape shapeA = {3, 1};
    Shape shapeB = {2, 3, 4};

    Graph g = make_ref<GraphObj>(runtime);
    auto A = g->addTensor(shapeA, DataType(INFINI_DTYPE_F32));
    auto B = g->addTensor(shapeB, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<ElementWiseObj>(OpType::Add, A, B, nullptr);

    runtime->dataMalloc(g);

    // Set input data
    std::vector<float> inputAData(A->getElement());
    std::vector<float> inputBData(B->getElement());

    for (size_t i = 0; i < inputAData.size(); ++i) {
        inputAData[i] = static_cast<float>(i + 1);
    }
    for (size_t i = 0; i < inputBData.size(); ++i) {
        inputBData[i] = static_cast<float>(inputBData.size() - i);
    }

    A->setData(inputAData.data());
    B->setData(inputBData.data());

    // Execute computation
    runtime->run(g);

    // Get output and print
    auto output = op->getOutput(0);
    std::cout << "CPU Output Data: " << std::endl;
    output->printData(runtime);
}

#ifdef USE_CUDA
// Single device test - NVIDIA F32
TEST(ElementWise, Add_SingleDevice_NVIDIA_F32) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 0);

    Shape shapeA = {3, 1};
    Shape shapeB = {2, 3, 4};

    Graph g = make_ref<GraphObj>(runtime);
    auto A = g->addTensor(shapeA, DataType(INFINI_DTYPE_F32));
    auto B = g->addTensor(shapeB, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<ElementWiseObj>(OpType::Add, A, B, nullptr);

    // Set input data
    std::vector<float> inputAData(A->getElement());
    std::vector<float> inputBData(B->getElement());

    for (size_t i = 0; i < inputAData.size(); ++i) {
        inputAData[i] = static_cast<float>(i + 1);
    }
    for (size_t i = 0; i < inputBData.size(); ++i) {
        inputBData[i] = static_cast<float>(inputBData.size() - i);
    }

    A->setData(inputAData.data());
    B->setData(inputBData.data());
    runtime->dataMalloc(g);

    // Execute computation
    runtime->run(g);

    // Get output and print
    auto output = op->getOutput(0);
    std::cout << "NVIDIA F32 Output Data: " << std::endl;
    output->printData(runtime);
}

// Single device test - NVIDIA F16
TEST(ElementWise, Add_SingleDevice_NVIDIA_F16) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_NVIDIA, 0);

    Shape shapeA = {3, 1};
    Shape shapeB = {2, 3, 4};

    Graph g = make_ref<GraphObj>(runtime);
    auto A = g->addTensor(shapeA, DataType(INFINI_DTYPE_F16));
    auto B = g->addTensor(shapeB, DataType(INFINI_DTYPE_F16));
    auto op = g->addOp<ElementWiseObj>(OpType::Add, A, B, nullptr);

    // Set input data
    std::vector<uint16_t> inputAData(A->getElement());
    std::vector<uint16_t> inputBData(B->getElement());

    for (size_t i = 0; i < inputAData.size(); ++i) {
        inputAData[i] = static_cast<uint16_t>(i + 1);
    }
    for (size_t i = 0; i < inputBData.size(); ++i) {
        inputBData[i] = static_cast<uint16_t>(inputBData.size() - i);
    }

    A->setData(inputAData.data());
    B->setData(inputBData.data());
    runtime->dataMalloc(g);

    // Execute computation
    runtime->run(g);

    // Get output and print
    auto output = op->getOutput(0);
    std::cout << "NVIDIA F16 Output Data: " << std::endl;
    output->printData(runtime);
}
#endif

} // namespace infini
