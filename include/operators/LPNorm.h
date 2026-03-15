#pragma once
#include "core/graph.h"
#include "core/operator.h"
#include <infiniop/ops/lp_norm.h>

namespace infini {

class LPNormObj : public OperatorObj {
  private:
    int axis;
    int p;
    float eps;

  public:
    LPNormObj(GraphObj *graph, Tensor input, Tensor output, int axis, int p,
              float eps = 1e-12f);
    string toString() const override;
    ~LPNormObj() override;

    void createOpDesc() override;
    optional<vector<ShapeExpr>> inferShape() override;
    vector<DataType> inferDataType() const override;

    int getAxis() const;
    int getP() const;
    float getEps() const;
};

} // namespace infini
