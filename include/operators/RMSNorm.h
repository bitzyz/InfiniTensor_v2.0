#pragma once
#include "core/operator.h"
#include "core/graph.h"
#include <infiniop/ops/rms_norm.h>
namespace infini
{
    class RMSNormObj : public OperatorObj{
        private: // Op's hyper-parameters   
            // Scale in paper which is g
            float epsilon;  

            public:
            /**
            * @brief Construct a new RmsnormObj object.
            * @param graph The computation graph that this operator belongs to.
            * @param A The input tensor.
            * @param Y Y is the output/bias of Rmsnorm. 
            **/
            RMSNormObj(GraphObj *graph, Tensor X, Tensor Y,
                    Tensor W, float epsilon = 1e-8);
            
            string toString() const override;
            ~RMSNormObj() override{
                //TODO
            }
            void createOpDesc() override;

            optional<vector<Shape>> inferShape() override;
            vector<DataType> inferDataType() const;
            float getEpsilon() const;

    }; // class RMSNormObj
}