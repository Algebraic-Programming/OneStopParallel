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

#include "concepts/computational_dag_concept.hpp"
#include "concepts/constructable_computational_dag_concept.hpp"
#include "bsp/model/BspSchedule.hpp"
#include <vector>

namespace osp {

/**
 * @class Coarser
 * @brief Abstract base class for coarsening ComputationalDags.
 *
 */
template<typename Graph_t_in, typename Graph_t_out>
class Coarser {

    static_assert(is_computational_dag_v<Graph_t_in>, "Graph_t_in must be a computational DAG");
    static_assert(is_constructable_cdag_v<Graph_t_out>, "Graph_t_out must be a computational DAG");

    // probably too strict, need to be refined. 
    // maybe add concept for when Gtaph_t2 is constructable/coarseable from Graph_t_in
    static_assert(std::is_same_v<v_workw_t<Graph_t_in>, v_workw_t<Graph_t_out>>,
                  "Graph_t_in and Graph_t_out must have the same work weight type");
    static_assert(std::is_same_v<v_memw_t<Graph_t_in>, v_memw_t<Graph_t_out>>,
                  "Graph_t_in and Graph_t_out must have the same memory weight type");
    static_assert(std::is_same_v<v_commw_t<Graph_t_in>, v_commw_t<Graph_t_out>>,
                  "Graph_t_in and Graph_t_out must have the same communication weight type");

  public:
    /**
     * @brief Destructor for the Coarser class.
     */
    virtual ~Coarser() = default;

    /**
     * @brief Get the name of the coarsening algorithm.
     * @return A human-readable name of the coarsening algorithm, typically used for identification or logging purposes.
     */
    virtual std::string getCoarserName() const = 0;

    /**
     * @brief Coarsens the input computational DAG into a simplified version.
     *
     * @param dag_in The input computational DAG to be coarsened. It is expected to be a valid graph structure.
     * @param coarsened_dag The output computational DAG after coarsening. It will be populated by this method.
     * @param vertex_map A mapping from vertices in the coarse DAG to the corresponding vertices in the original DAG.
     *                   Each entry in the outer vector corresponds to a vertex in the coarse DAG, and the inner vector
     *                   contains the indices of the original vertices that were merged.
     * @return A status code indicating the success or failure of the coarsening operation.
     */
    virtual bool coarseDag(const Graph_t_in &dag_in, Graph_t_out &coarsened_dag,
                           std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &vertex_map,
                           std::vector<vertex_idx_t<Graph_t_out>> &reverse_vertex_map) = 0;
};


template<typename Graph_t_in, typename Graph_t_out>
bool pull_back_schedule(const BspSchedule<Graph_t_in> &schedule_in,
                        const std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &vertex_map,
                        BspSchedule<Graph_t_out> &schedule_out) {

    for (unsigned v = 0; v < vertex_map.size(); ++v) {

        const auto proc = schedule_in.assignedProcessor(v);
        const auto step = schedule_in.assignedSuperstep(v);

        for (const auto &u : vertex_map[v]) {
            schedule_out.setAssignedSuperstep(u, step);
            schedule_out.setAssignedProcessor(u, proc);
        }
    }

    return true;
}

template<typename Graph_t_in, typename Graph_t_out>
bool pull_back_schedule(const BspSchedule<Graph_t_in> &schedule_in,
                        const std::vector<vertex_idx_t<Graph_t_out>> &reverse_vertex_map,
                        BspSchedule<Graph_t_out> &schedule_out) {

    for (unsigned idx = 0; idx < reverse_vertex_map.size(); ++idx) {
        const auto &v = reverse_vertex_map[idx];

        schedule_out.setAssignedSuperstep(idx, schedule_in.assignedSuperstep(v));
        schedule_out.setAssignedProcessor(idx, schedule_in.assignedProcessor(v));
    }

    return true;
}

} // namespace osp