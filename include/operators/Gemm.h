#pragma once
#include "core/graph.h"
#include "core/operator.h"
#include <infiniop/ops/gemm.h>

namespace infini {
class GemmObj : public OperatorObj {
  private:
    // InfiniTensor assumes a row-major tensor layout. `transA`=false means
    // default dims, true means A should be transposed before matmul. This is in
    // oppsite to the column-major BLAS.
    float alpha, beta;
    bool transA, transB;

  public:
    /**
     * @brief Construct a new Gemm object.
     * @param graph The computation graph that this operator belongs to.
     * @param A The input tensor.
     * @param B The input tensor.
     * @param Y Y is the output/bias of Matmul. If Gemm do not have bias,
     * Y should be an empty Ref.
     * @param transA If matrix A should be transposed when computing.
     * @param transB If matrix B should be transposed when computing.
     */
    GemmObj(GraphObj *graph, Tensor A, Tensor B, Tensor Y, Tensor C,
            float alpha = 1.0f, float beta = 1.0f, bool transA = false,
            bool transB = false);

    string toString() const override;
    ~GemmObj() override {
        if (infiniOpDesc) {
            auto err = infiniopDestroyGemmDescriptor(
                (infiniopGemmDescriptor_t)infiniOpDesc);
            if (err != INFINI_STATUS_SUCCESS) {
                std::cerr << "Warning: Gemm descriptor destroy failed with "
                             "error code "
                          << err << std::endl;
            }
        }
    }

    void createOpDesc() override;
    optional<vector<ShapeExpr>> inferShape() override;
    vector<DataType> inferDataType() const;

    bool getTransA() const;
    bool getTransB() const;
    float getAlpha() const;
    float getBeta() const;
};
} // namespace infini
