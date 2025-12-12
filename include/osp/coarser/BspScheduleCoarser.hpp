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

#include "osp/bsp/model/BspSchedule.hpp"
#include "osp/bsp/model/util/SetSchedule.hpp"
#include "osp/bsp/scheduler/Scheduler.hpp"
#include "osp/coarser/Coarser.hpp"
#include "osp/graph_algorithms/directed_graph_edge_desc_util.hpp"

namespace osp {

/**
 * @class Coarser
 * @brief Abstract base class for coarsening ComputationalDags.
 *
 */
template <typename GraphTIn, typename GraphTOut>
class BspScheduleCoarser : public CoarserGenContractionMap<GraphTIn, GraphTOut> {
  private:
    const BspSchedule<GraphTIn> *schedule_;

  public:
    BspScheduleCoarser(const BspSchedule<GraphTIn> &schedule) : schedule_(&schedule) {}

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

    virtual std::vector<vertex_idx_t<Graph_t_out>> generate_vertex_contraction_map(const GraphTIn &dagIn) override {
        using VertexType_in = vertex_idx_t<Graph_t_in>;
        using VertexType_out = vertex_idx_t<Graph_t_out>;

        assert(&dagIn == &schedule_->getInstance().getComputationalDag());
        assert(schedule_->satisfiesPrecedenceConstraints());

        SetSchedule<GraphTIn> setSchedule(*schedule_);
        std::vector<VertexType_out> reverseVertexMap(dagIn.NumVertices(), 0);
        std::vector<std::vector<VertexType_in>> vertexMap;

        bool scheduleRespectsTypes = true;

        for (unsigned step = 0; step < schedule_->numberOfSupersteps(); step++) {
            for (unsigned proc = 0; proc < schedule_->getInstance().numberOfProcessors(); proc++) {
                if (setSchedule.step_processor_vertices[step][proc].size() > 0) {
                    v_workw_t<Graph_t_in> totalWork = 0;
                    v_memw_t<Graph_t_in> totalMemory = 0;
                    v_commw_t<Graph_t_in> totalCommunication = 0;

                    vertex_map.push_back(std::vector<VertexType_in>());

                    v_type_t<Graph_t_in> type = dagIn.VertexType(*(setSchedule.step_processor_vertices[step][proc].begin()));
                    bool homogeneousTypes = true;

                    for (const auto &vertex : setSchedule.step_processor_vertices[step][proc]) {
                        if (dagIn.VertexType(vertex) != type) {
                            homogeneousTypes = false;
                        }

                        vertexMap.back().push_back(vertex);
                        reverseVertexMap[vertex] = vertex_map.size() - 1;

                        totalWork += dagIn.vertex_work_weight(vertex);
                        totalCommunication += dagIn.vertex_comm_weight(vertex);
                        totalMemory += dagIn.vertex_mem_weight(vertex);
                    }

                    if (scheduleRespectsTypes) {
                        scheduleRespectsTypes = homogeneousTypes;
                    }
                }
            }
        }

        return reverse_vertex_map;
    }
};

}    // namespace osp
