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
    virtual std::string GetCoarserName() const override { return "BspScheduleCoarser"; }

    // virtual bool coarseDag(const GraphTIn &dag_in, GraphTOut &dag_out,
    //                        std::vector<std::vector<VertexIdxT<GraphTIn>>> &vertex_map,
    //                        std::vector<VertexIdxT<GraphTOut>> &reverse_vertex_map) override {

    virtual std::vector<VertexIdxT<GraphTOut>> GenerateVertexContractionMap(const GraphTIn &dagIn) override {
        using VertexTypeIn = VertexIdxT<GraphTIn>;
        using VertexTypeOut = VertexIdxT<GraphTOut>;

        assert(&dagIn == &schedule_->GetInstance().GetComputationalDag());
        assert(schedule_->SatisfiesPrecedenceConstraints());

        SetSchedule<GraphTIn> setSchedule(*schedule_);
        std::vector<VertexTypeOut> reverseVertexMap(dagIn.NumVertices(), 0);
        std::vector<std::vector<VertexTypeIn>> vertexMap;

        bool scheduleRespectsTypes = true;

        for (unsigned step = 0; step < schedule_->NumberOfSupersteps(); step++) {
            for (unsigned proc = 0; proc < schedule_->GetInstance().NumberOfProcessors(); proc++) {
                if (setSchedule.stepProcessorVertices_[step][proc].size() > 0) {
                    VWorkwT<GraphTIn> totalWork = 0;
                    VMemwT<GraphTIn> totalMemory = 0;
                    VCommwT<GraphTIn> totalCommunication = 0;

                    vertexMap.push_back(std::vector<VertexTypeIn>());

                    VTypeT<GraphTIn> type = dagIn.VertexType(*(setSchedule.stepProcessorVertices_[step][proc].begin()));
                    bool homogeneousTypes = true;

                    for (const auto &vertex : setSchedule.stepProcessorVertices_[step][proc]) {
                        if (dagIn.VertexType(vertex) != type) {
                            homogeneousTypes = false;
                        }

                        vertexMap.back().push_back(vertex);
                        reverseVertexMap[vertex] = vertexMap.size() - 1;

                        totalWork += dagIn.VertexWorkWeight(vertex);
                        totalCommunication += dagIn.VertexCommWeight(vertex);
                        totalMemory += dagIn.VertexMemWeight(vertex);
                    }

                    if (scheduleRespectsTypes) {
                        scheduleRespectsTypes = homogeneousTypes;
                    }
                }
            }
        }

        return reverseVertexMap;
    }
};

}    // namespace osp
