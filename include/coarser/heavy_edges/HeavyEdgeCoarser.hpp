/*
Copyright 2024 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner   
*/

#pragma once


#include "coarser/Coarser.hpp"
#include "structures/union_find.hpp"

/**
 * @class Coarser
 * @brief Abstract base class for coarsening ComputationalDags.
 * 
 */
class HeavyEdgeCoarser : public Coarser {

    const float heavy_is_x_times_median;
    const float min_percent_components_retained;
    const float bound_component_weight_percent; 
    

    public:

        HeavyEdgeCoarser(float heavy_is_x_times_median_, float min_percent_components_retained_, float bound_component_weight_percent_) : 
        heavy_is_x_times_median(heavy_is_x_times_median_), 
        min_percent_components_retained(min_percent_components_retained_), 
        bound_component_weight_percent(bound_component_weight_percent_) {}

        /**
         * @brief Destructor for the Coarser class.
         */
        virtual ~HeavyEdgeCoarser() = default;

        /**
         * @brief Get the name of the coarsening algorithm.
         * @return The name of the coarsening algorithm.
         */
        virtual std::string getCoarserName() const override { return "HeavyEdgeCoarser"; }
        
        virtual RETURN_STATUS coarseDag(const ComputationalDag &dag_in, ComputationalDag &dag_out, std::vector<std::vector<VertexType>>& vertex_map) override;
        
};
