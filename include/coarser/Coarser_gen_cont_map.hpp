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

#include <algorithm>
#include <set>
#include <vector>

#include "coarser/Coarser.hpp"

namespace osp {

/**
 * @class Coarser
 * @brief Abstract base class for coarsening ComputationalDags.
 *
 */
template<typename Graph_t_in, typename Graph_t_out>
class CoarserGenContractionMap : public Coarser<Graph_t_in, Graph_t_out> {

  public:
    virtual std::vector<vertex_idx_t<Graph_t_out>> generate_vertex_contraction_map(const Graph_t_in &dag_in) = 0;

    virtual bool coarsenDag(const Graph_t_in &dag_in, Graph_t_out &coarsened_dag,
                            std::vector<vertex_idx_t<Graph_t_out>> &vertex_contraction_map) override {

        vertex_contraction_map = generate_vertex_contraction_map(dag_in);

        return coarser_util::construct_coarse_dag(dag_in, coarsened_dag, vertex_contraction_map);
    }

    /**
     * @brief Destructor for the CoarserGenContractionMap class.
     */
    virtual ~CoarserGenContractionMap() = default;
};

} // namespace osp