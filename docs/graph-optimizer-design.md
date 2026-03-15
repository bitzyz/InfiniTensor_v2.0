# Graph Optimizer Design Notes

## Summary

This document summarizes the graph optimization support added on top of the current InfiniTensor core.

The current repository is still a compact IR/runtime stack. In practice, the graph pipeline is:

1. Frontend builds a `GraphObj`.
2. `GraphObj` owns `TensorObj` and `OperatorObj`.
3. `RuntimeObj` executes operators in topological order.

Before this change, the framework had graph construction and execution, but no explicit optimization stage. The main design goal of this work was to introduce a minimal but maintainable optimizer layer without rewriting the existing IR.

The resulting design keeps optimization as an explicit step:

```cpp
graph->setOutputs(...);
graph->optimize();
runtime->run(graph);
```

The same model is exposed through `GraphBuilder`.

## High-Level Design

### 1. Optimization is explicit, not hidden in runtime

Optimization is triggered by `GraphObj::optimize()` instead of being implicitly embedded inside `RuntimeObj::run()`.

Why this choice:

- It keeps runtime behavior predictable.
- It makes debugging easier because the caller decides when the graph is rewritten.
- It allows users to inspect the graph before and after optimization.

The optimization entrypoint lives in [include/core/graph.h](/home/lunatic/Projects/InfiniTensor_v2.0/include/core/graph.h#L39) and is implemented in [src/core/graph.cc](/home/lunatic/Projects/InfiniTensor_v2.0/src/core/graph.cc#L175).

### 2. Graph outputs are first-class IR state

Dead code elimination needs a notion of "live result". The original IR had tensors and operators, but no explicit graph outputs. That makes DCE ambiguous.

This change adds `outputs` to `GraphObj`; see [include/core/graph.h](/home/lunatic/Projects/InfiniTensor_v2.0/include/core/graph.h#L12) and [src/core/graph.cc](/home/lunatic/Projects/InfiniTensor_v2.0/src/core/graph.cc#L175).

That gives the optimizer a correct root set for backward liveness analysis.

### 3. The optimizer is a thin pass pipeline over the existing IR

Instead of introducing a brand new IR, this work adds:

- `GraphPass` as the basic pass abstraction
- `GraphOptimizer` as a fixed pass pipeline

These live in:

- [include/core/graph_pass.h](/home/lunatic/Projects/InfiniTensor_v2.0/include/core/graph_pass.h)
- [include/core/graph_optimizer.h](/home/lunatic/Projects/InfiniTensor_v2.0/include/core/graph_optimizer.h)
- [src/core/graph_optimizer.cc](/home/lunatic/Projects/InfiniTensor_v2.0/src/core/graph_optimizer.cc#L24)

The optimizer reuses existing graph services like topological sort, shape inference, and validation instead of replacing them.

### 4. Rewriting is in-place

Optimization mutates the existing graph object. It does not clone and return a new graph.

That fits the current architecture because:

- tensors and operators already maintain direct object relationships
- `GraphBuilderObj` exposes the same graph object to C++ and Python
- cloning would require extra mapping logic for tensors, operators, and outputs

## Pipeline

The current default pipeline is defined in [src/core/graph_optimizer.cc](/home/lunatic/Projects/InfiniTensor_v2.0/src/core/graph_optimizer.cc#L219):

1. `TopologicalSortPass`
2. `ShapeInferPass`
3. `ConstantFoldPass`
4. `IdentityEliminationPass`
5. `DeadCodeEliminationPass`
6. `TopologicalSortPass`
7. `ShapeInferPass`
8. `GraphValidatePass`

The intent is:

- normalize graph order first
- make sure output shapes are current
- remove obviously foldable or redundant computation
- drop unreachable branches
- rebuild a clean final graph shape/order/consistency state

## Component Explanation

### `GraphObj`: IR container and optimization entrypoint

`GraphObj` remains the owner of the IR and now also owns graph outputs.

Relevant API additions:

- `getOutputs()`
- `setOutputs()`
- `addOutput()`
- `clearOutputs()`
- `optimize()`
- `replaceAllUses()`
- `eraseOperator()`
- `eraseTensorIfUnused()`

See [include/core/graph.h](/home/lunatic/Projects/InfiniTensor_v2.0/include/core/graph.h#L31).

The implementation in [src/core/graph.cc](/home/lunatic/Projects/InfiniTensor_v2.0/src/core/graph.cc#L89) does three important things:

1. It stores outputs as explicit graph state.
2. It exposes utility methods for rewriting uses and erasing dead nodes.
3. It can rebuild tensor/operator relationships after rewrites through `rebuildConnections()`.

`rebuildConnections()` is important because this IR stores graph structure redundantly:

- operators point to inputs and outputs
- tensors point to source and targets
- operators also track predecessors and successors

After a rewrite, these structures must be re-synchronized. That happens in [src/core/graph.cc](/home/lunatic/Projects/InfiniTensor_v2.0/src/core/graph.cc#L230).

### `TensorObj`: constant and input classification

The optimizer needs to distinguish between:

- graph inputs
- constants / parameters
- intermediate tensors produced by operators

To support that, `TensorObj` now exposes:

- `hasData()`
- `getDevice()`
- `isConstant()`
- `isGraphInput()`

See [include/core/tensor.h](/home/lunatic/Projects/InfiniTensor_v2.0/include/core/tensor.h#L44) and [src/core/tensor.cc](/home/lunatic/Projects/InfiniTensor_v2.0/src/core/tensor.cc#L70).

The current semantics are:

- `isConstant()`: tensor has backing data and no producing operator
- `isGraphInput()`: tensor has no data and no producing operator

This is intentionally simple and matches the current repository structure.

`TensorObj` also got:

- `clearTargets()`
- `clearSource()`

These are used when `GraphObj` reconstructs graph connectivity after rewrites.

### `OperatorObj`: minimal rewrite support

Before this change, operators were mostly execution-time objects. For graph rewriting, the optimizer needed a safe way to swap inputs and clear cached graph connections.

The public additions are:

- `replaceInput()`
- `clearConnections()`

See [include/core/operator.h](/home/lunatic/Projects/InfiniTensor_v2.0/include/core/operator.h#L34) and [src/core/operator.cc](/home/lunatic/Projects/InfiniTensor_v2.0/src/core/operator.cc#L65).

This is intentionally narrow. The optimizer does not directly mutate every part of the operator; it mainly:

- rewires inputs
- lets `GraphObj` rebuild predecessor/successor links afterward

### `GraphPass` and `GraphOptimizer`

`GraphPass` is the abstract pass interface:

- `name()`
- `run(GraphObj&)`

See [include/core/graph_pass.h](/home/lunatic/Projects/InfiniTensor_v2.0/include/core/graph_pass.h).

`GraphOptimizer` currently provides a fixed built-in pipeline; see [src/core/graph_optimizer.cc](/home/lunatic/Projects/InfiniTensor_v2.0/src/core/graph_optimizer.cc#L219).

This design is deliberately small:

- enough structure to add more passes later
- no registry or plugin system yet
- no per-pass configuration yet

### `ConstantFoldPass`

`ConstantFoldPass` is implemented in [src/core/graph_optimizer.cc](/home/lunatic/Projects/InfiniTensor_v2.0/src/core/graph_optimizer.cc#L39).

Current scope:

- only folds `Add`, `Sub`, and `Mul`
- only folds when all inputs are CPU-resident concrete `float32` constants

The pass works by:

1. checking that all inputs are foldable constants
2. allocating a new output buffer
3. evaluating the operator directly in host code
4. storing the folded result into the output tensor
5. erasing the original operator

This is intentionally conservative. It avoids entangling the optimizer with runtime scheduling or device execution.

### `IdentityEliminationPass`

Implemented in [src/core/graph_optimizer.cc](/home/lunatic/Projects/InfiniTensor_v2.0/src/core/graph_optimizer.cc#L115).

Current rules:

- `add(x, 0) -> x`
- `sub(x, 0) -> x`
- `mul(x, 1) -> x`

The pass:

1. identifies the replacement tensor
2. rewrites all uses of the old output via `replaceAllUses()`
3. erases the redundant operator
4. removes the now-unused output tensor if possible

### `DeadCodeEliminationPass`

Implemented in [src/core/graph_optimizer.cc](/home/lunatic/Projects/InfiniTensor_v2.0/src/core/graph_optimizer.cc#L156).

This pass depends on explicit graph outputs.

The algorithm is:

1. start from `graph.outputs`
2. walk backwards through producer operators
3. mark live tensors and live operators
4. erase operators not reached by that walk
5. erase tensors that become unused

This is why adding `outputs` to `GraphObj` was necessary.

### `GraphBuilderObj`

`GraphBuilderObj` remains the frontend construction helper, but now forwards optimization-related APIs:

- `set_outputs()`
- `optimize()`

See [include/core/graph_builder.h](/home/lunatic/Projects/InfiniTensor_v2.0/include/core/graph_builder.h#L17) and [src/core/graph_builder.cc](/home/lunatic/Projects/InfiniTensor_v2.0/src/core/graph_builder.cc#L51).

That means frontend code can stay builder-centric:

```cpp
builder.set_outputs({y});
builder.optimize();
```

### Python bindings

The Python binding now exposes optimization APIs on both `Graph` and `GraphBuilder`; see [python/bindings/graph.hpp](/home/lunatic/Projects/InfiniTensor_v2.0/python/bindings/graph.hpp#L13).

Exposed methods:

- `Graph.optimize()`
- `Graph.set_outputs()`
- `Graph.get_outputs()`
- `GraphBuilder.optimize()`
- `GraphBuilder.set_outputs()`

This keeps Python behavior aligned with the C++ side.

### Torch FX translator integration

The translator now writes collected outputs back into the graph when handling the final FX output node; see [python/src/infinitensor/torch_fx_translator.py](/home/lunatic/Projects/InfiniTensor_v2.0/python/src/infinitensor/torch_fx_translator.py#L168).

That change matters because DCE must know the semantic outputs of the translated model.

## Data Flow After This Change

The graph lifecycle is now:

1. Build graph through `GraphBuilderObj`.
2. Mark final outputs through `set_outputs()`.
3. Run `optimize()`.
4. Execute through runtime.

The ownership and dependency chain is:

- `GraphBuilderObj` builds `GraphObj`
- `GraphObj` owns tensors, operators, and outputs
- `GraphOptimizer` runs passes on `GraphObj`
- passes use `GraphObj`, `TensorObj`, and `OperatorObj` rewrite helpers
- `RuntimeObj` consumes the optimized graph

## Tests

The new optimizer behavior is covered by:

- [test/core/test_graph.cc](/home/lunatic/Projects/InfiniTensor_v2.0/test/core/test_graph.cc#L15)
- [test/core/test_graph_optimizer.cc](/home/lunatic/Projects/InfiniTensor_v2.0/test/core/test_graph_optimizer.cc#L22)

The optimizer tests cover:

- output registration
- identity elimination preserving the final output
- constant folding producing a constant tensor
- dead code elimination removing unused branches

## Current Limits

This implementation is intentionally scoped to the current repository state.

Important limits:

- The current repo only exposes `ElementWise` and `Gemm` operators, so optimization rules are currently focused on `Add/Sub/Mul`.
- Constant folding only supports CPU-side concrete `float32` constants.
- The optimizer pipeline is fixed, not configurable.
- Optimization is in-place only.
- There is no pass registry yet.

## Recommended Next Steps

If this optimizer is expanded later, the natural next steps are:

1. Add operator-specific fold/rewrite hooks instead of centralizing all logic in one file.
2. Extend constant folding to more dtypes and more operators.
3. Add configurable pass pipelines.
4. Add a pass registry once more operator families exist.
5. Consider a cloned-graph optimization mode only if in-place mutation becomes limiting.
