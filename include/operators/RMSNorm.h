#pragma once
#include "core/graph.h"
#include "core/operator.h"
#include <infiniop/ops/rms_norm.h>

namespace infini {

class RMSNormObj : public OperatorObj {
  private:
    float epsilon;

  public:
    RMSNormObj(GraphObj *graph, Tensor input, Tensor weight, Tensor output,
               float epsilon = 1e-6f);
    string toString() const override;
    ~RMSNormObj() override;

    void createOpDesc() override;
    optional<vector<ShapeExpr>> inferShape() override;
    vector<DataType> inferDataType() const override;

    float getEpsilon() const;
};

} // namespace infini
