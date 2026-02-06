#include "core/runtime.h"
#include "operators/ElementWise.h"
#include "gtest/gtest.h"

namespace infini {
template <typename T>
void runElementWiseTest(const std::string &deviceName, infiniDevice_t DeviceT,
                        OpType opType, const Shape &shapeA, const Shape &shapeB,
                        const DataType &dataType, bool print = false) {
    Runtime &runtime = RuntimeObj::getInstance();
    RuntimeObj::init();
    runtime->initThreadContext(DeviceT, 0);
    Graph g = make_ref<GraphObj>(runtime);

    // 创建输入张量
    auto A = g->addTensor(shapeA, dataType);
    auto B = g->addTensor(shapeB, dataType);

    // 创建ElementWise算子
    auto op = g->addOp<ElementWiseObj>(opType, A, B, nullptr);

    // 分配内存
    runtime->dataMalloc(g);

    // 设置输入数据
    size_t elementA = A->getElement();
    size_t elementB = B->getElement();

    // 为A和B设置不同的数据模式，方便验证结果
    std::vector<T> inputAData(elementA);
    std::vector<T> inputBData(elementB);

    // 使用简单的递增序列和递减序列，便于计算和验证
    for (size_t i = 0; i < elementA; ++i) {
        inputAData[i] = static_cast<T>(i + 1); // 1, 2, 3, ...
    }

    for (size_t i = 0; i < elementB; ++i) {
        inputBData[i] = static_cast<T>(elementB - i); // n, n-1, n-2, ...
    }

    A->setData(inputAData.data());
    B->setData(inputBData.data());

    if (print) {
        std::cout << "Running ElementWise Test on " << deviceName << std::endl;
        std::cout << "OpType: " << opType.toString() << std::endl;
        std::cout << "Shape A: " << vecToString(shapeA) << std::endl;
        std::cout << "Shape B: " << vecToString(shapeB) << std::endl;
        std::cout << "Graph: " << g->toString() << std::endl;
    }

    // 执行计算
    runtime->run(g);

    // 获取输出
    auto output = op->getOutput(0);

    if (print) {
        std::cout << "Output Data: " << std::endl;
        output->printData(runtime);
    }
}

// 基本Add操作测试
TEST(ElementWise, Add_Basic) {
    Shape shapeA = {3, 1};
    Shape shapeB = {2, 3, 4};

    runElementWiseTest<float>("CPU", INFINI_DEVICE_CPU, OpType::Add, shapeA,
                              shapeB, DataType(INFINI_DTYPE_F32), true);

#ifdef USE_CUDA
    runElementWiseTest<float>("NVIDIA", INFINI_DEVICE_NVIDIA, OpType::Add,
                              shapeA, shapeB, DataType(INFINI_DTYPE_F32), true);
    runElementWiseTest<uint16_t>("NVIDIA", INFINI_DEVICE_NVIDIA, OpType::Add,
                                 shapeA, shapeB, DataType(INFINI_DTYPE_F16),
                                 true);
#endif
}

// 基本Mul操作测试
TEST(ElementWise, Mul_Basic) {
    Shape shapeA = {3, 4};
    Shape shapeB = {3, 4};

    runElementWiseTest<float>("CPU", INFINI_DEVICE_CPU, OpType::Mul, shapeA,
                              shapeB, DataType(INFINI_DTYPE_F32), true);

#ifdef USE_CUDA
    runElementWiseTest<float>("NVIDIA", INFINI_DEVICE_NVIDIA, OpType::Mul,
                              shapeA, shapeB, DataType(INFINI_DTYPE_F32), true);
#endif
}

// 基本Sub操作测试
TEST(ElementWise, Sub_Basic) {
    Shape shapeA = {1, 5, 6};
    Shape shapeB = {1, 5, 6};

    runElementWiseTest<float>("CPU", INFINI_DEVICE_CPU, OpType::Sub, shapeA,
                              shapeB, DataType(INFINI_DTYPE_F32), true);

#ifdef USE_CUDA
    runElementWiseTest<float>("NVIDIA", INFINI_DEVICE_NVIDIA, OpType::Sub,
                              shapeA, shapeB, DataType(INFINI_DTYPE_F32), true);
#endif
}
} // namespace infini
