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

#include "Coarser.hpp"
#include "bsp/model/BspSchedule.hpp"
#include "bsp/model/SetSchedule.hpp"
#include "bsp/scheduler/Scheduler.hpp"
#include "graph_algorithms/directed_graph_edge_desc_util.hpp"

namespace osp {

/**
 * @class Coarser
 * @brief Abstract base class for coarsening ComputationalDags.
 *
 */
template<typename Graph_t_in, typename Graph_t_out>
class BspScheduleCoarser : public Coarser<Graph_t_in, Graph_t_out> {

  private:
    const BspSchedule<Graph_t_in> *schedule;

  public:
    BspScheduleCoarser(const BspSchedule<Graph_t_in> &schedule) : schedule(&schedule) {}

    /**
     * @brief Destructor for the Coarser class.
     */
    virtual ~BspScheduleCoarser() = default;

    /**
     * @brief Get the name of the coarsening algorithm.
     * @return The name of the coarsening algorithm.
     */
    virtual std::string getCoarserName() const override { return "BspScheduleCoarser"; }

    virtual bool coarseDag(const Graph_t_in &dag_in, Graph_t_out &dag_out,
                           std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &vertex_map,
                           std::vector<vertex_idx_t<Graph_t_out>> &reverse_vertex_map) override {

        using VertexType = vertex_idx_t<Graph_t_in>;

        assert(&dag_in == &schedule->getInstance().getComputationalDag());
        assert(dag_out.num_vertices() == 0);
        assert(vertex_map.empty());
        assert(reverse_vertex_map.empty());
        assert(schedule->satisfiesPrecedenceConstraints());

        SetSchedule<Graph_t_in> set_schedule(*schedule);
        reverse_vertex_map.resize(dag_in.num_vertices(), 0);

        bool schedule_respects_types = true;

        for (unsigned step = 0; step < schedule->numberOfSupersteps(); step++) {

            for (unsigned proc = 0; proc < schedule->getInstance().numberOfProcessors(); proc++) {

                if (set_schedule.step_processor_vertices[step][proc].size() > 0) {

                    v_workw_t<Graph_t_in> total_work = 0;
                    v_memw_t<Graph_t_in> total_memory = 0;
                    v_commw_t<Graph_t_in> total_communication = 0;

                    vertex_map.push_back(std::vector<VertexType>());

                    v_type_t<Graph_t_in> type =
                        dag_in.vertex_type(*(set_schedule.step_processor_vertices[step][proc].begin()));
                    bool homogeneous_types = true;

                    for (const auto &vertex : set_schedule.step_processor_vertices[step][proc]) {

                        if (dag_in.vertex_type(vertex) != type) {
                            homogeneous_types = false;
                        }

                        vertex_map.back().push_back(vertex);
                        reverse_vertex_map[vertex] = vertex_map.size() - 1;

                        total_work += dag_in.vertex_work_weight(vertex);
                        total_communication += dag_in.vertex_comm_weight(vertex);
                        total_memory += dag_in.vertex_mem_weight(vertex);
                    }

                    if (schedule_respects_types)
                        schedule_respects_types = homogeneous_types;

                    dag_out.add_vertex(total_work, total_communication, total_memory, type);
                }
            }
        }

        if (not schedule_respects_types) {

            for (auto vertex : dag_out.vertices()) {
                dag_out.set_vertex_type(vertex, 0);
            }
        }

        for (const auto &vertex_out : dag_out.vertices()) {

            for (auto vertex : vertex_map[vertex_out]) {

                if constexpr (is_constructable_cdag_comm_edge_v<Graph_t_out> and
                              is_computational_dag_edge_desc_v<Graph_t_in>) {

                    for (const auto &edge : dag_in.out_edges(vertex)) {

                        const auto child = reverse_vertex_map[target(edge, dag_in)];

                        if (child != vertex_out) {

                            const auto pair = edge_desc(vertex_out, child, dag_out);

                            if (pair.second) {

                                dag_out.set_edge_comm_weight(pair.first, dag_out.edge_comm_weight(pair.first) +
                                                                             dag_in.edge_comm_weight(edge));
                            } else {

                                dag_out.add_edge(vertex_out, child, dag_in.edge_comm_weight(edge));
                            }
                        }
                    }

                } else {

                    for (const auto &child : dag_in.children(vertex)) {

                        const auto child_out = reverse_vertex_map[child];

                        if (child_out != vertex_out) {

                            if (not edge(vertex_out, child_out, dag_out)) {
                                dag_out.add_edge(vertex_out, child_out);
                            }
                        }
                    }
                }
            }
        }

        return true;
    }
};

} // namespace osp