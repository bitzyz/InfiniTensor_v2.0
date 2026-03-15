#pragma once
#ifndef GRAPH_OPTIMIZER_H
#define GRAPH_OPTIMIZER_H

#include "core/graph_pass.h"

namespace infini {

class GraphOptimizer {
  public:
    void run(GraphObj &graph) const;
};

} // namespace infini

#endif // GRAPH_OPTIMIZER_H
