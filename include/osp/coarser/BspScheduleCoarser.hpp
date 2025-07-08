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

#include "osp/coarser/Coarser.hpp"
#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/model/SetSchedule.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/graph_algorithms/directed_graph_edge_desc_util.hpp"

namespace osp {

/**
 * @class Coarser
 * @brief Abstract base class for coarsening ComputationalDags.
 *
 */
template<typename Graph_t_in, typename Graph_t_out>
class BspScheduleCoarser : public CoarserGenContractionMap<Graph_t_in, Graph_t_out> {

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

    // virtual bool coarseDag(const Graph_t_in &dag_in, Graph_t_out &dag_out,
    //                        std::vector<std::vector<vertex_idx_t<Graph_t_in>>> &vertex_map,
    //                        std::vector<vertex_idx_t<Graph_t_out>> &reverse_vertex_map) override {

    virtual std::vector<vertex_idx_t<Graph_t_out>> generate_vertex_contraction_map(const Graph_t_in &dag_in) override {

        using VertexType_in = vertex_idx_t<Graph_t_in>;
        using VertexType_out = vertex_idx_t<Graph_t_out>;

        assert(&dag_in == &schedule->getInstance().getComputationalDag());
        assert(schedule->satisfiesPrecedenceConstraints());


        SetSchedule<Graph_t_in> set_schedule(*schedule);
        std::vector<VertexType_out> reverse_vertex_map(dag_in.num_vertices(), 0);
        std::vector<std::vector<VertexType_in>> vertex_map;

        bool schedule_respects_types = true;

        for (unsigned step = 0; step < schedule->numberOfSupersteps(); step++) {

            for (unsigned proc = 0; proc < schedule->getInstance().numberOfProcessors(); proc++) {

                if (set_schedule.step_processor_vertices[step][proc].size() > 0) {

                    v_workw_t<Graph_t_in> total_work = 0;
                    v_memw_t<Graph_t_in> total_memory = 0;
                    v_commw_t<Graph_t_in> total_communication = 0;

                    vertex_map.push_back(std::vector<VertexType_in>());

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
                }
            }
        }

        return reverse_vertex_map;
    }
};

} // namespace osp