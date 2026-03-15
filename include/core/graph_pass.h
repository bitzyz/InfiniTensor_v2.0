#pragma once
#ifndef GRAPH_PASS_H
#define GRAPH_PASS_H

#include "core/graph.h"

namespace infini {

class GraphPass {
  public:
    virtual ~GraphPass() = default;
    virtual string name() const = 0;
    virtual bool run(GraphObj &graph) = 0;
};

} // namespace infini

#endif // GRAPH_PASS_H
