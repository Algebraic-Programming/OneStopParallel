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

#include <deque>
#include <limits>
#include <string>
#include <vector>

#include "Scheduler.hpp"

namespace osp {

/**
 * @class Serial
 * @brief The Serial class represents a scheduler that assigns all tasks to a single processor in a serial manner.
 * If the architecture is heterogeneous, it assigns tasks to one processor of each type computing a schedule with the
 * smallest number of supersteps.
 *
 */
template <typename GraphT>
class Serial : public Scheduler<GraphT> {
  public:
    /**
     * @brief Default constructor for Serial.
     */
    Serial() : Scheduler<GraphT>() {}

    /**
     * @brief Default destructor for Serial.
     */
    ~Serial() override = default;

    ReturnStatus ComputeSchedule(BspSchedule<GraphT> &schedule) override {
        const auto &instance = schedule.GetInstance();
        const auto &dag = instance.GetComputationalDag();
        const auto numVertices = dag.NumVertices();

        if (numVertices == 0) {
            return ReturnStatus::OSP_SUCCESS;
        }

        const auto &arch = instance.GetArchitecture();

        // Select one processor of each type
        std::vector<unsigned> chosenProcs;
        if (arch.GetNumberOfProcessorTypes() > 0) {
            std::vector<bool> typeSeen(arch.GetNumberOfProcessorTypes(), false);
            for (unsigned p = 0; p < arch.NumberOfProcessors(); ++p) {
                if (!typeSeen[arch.processorType(p)]) {
                    chosenProcs.push_back(p);
                    typeSeen[arch.processorType(p)] = true;
                }
            }
        }

        if (chosenProcs.empty()) {
            return ReturnStatus::ERROR;
        }

        const unsigned numNodeTypes = dag.NumVertexTypes();
        std::vector<std::vector<unsigned>> nodeTypeCompatibleProcessors(numNodeTypes);

        for (VTypeT<GraphT> type = 0; type < numNodeTypes; ++type) {
            for (const auto &p : chosenProcs) {
                if (instance.isCompatibleType(type, instance.processorType(p))) {
                    nodeTypeCompatibleProcessors[type].push_back(p);
                }
            }
        }

        std::vector<VertexIdxT<GraphT>> inDegree(numVertices);
        std::deque<VertexIdxT<GraphT>> readyNodes;
        std::deque<VertexIdxT<GraphT>> deferredNodes;

        for (const auto &v : dag.vertices()) {
            schedule.setAssignedProcessor(v, std::numeric_limits<unsigned>::max());
            schedule.setAssignedSuperstep(v, std::numeric_limits<unsigned>::max());
            inDegree[v] = dag.in_degree(v);
            if (inDegree[v] == 0) {
                readyNodes.push_back(v);
            }
        }

        VertexIdxT<GraphT> scheduledNodesCount = 0;
        unsigned currentSuperstep = 0;

        while (scheduled_nodes_count < numVertices) {
            while (not ready_nodes.empty()) {
                VertexIdxT<GraphT> v = ready_nodes.front();
                readyNodes.pop_front();

                bool scheduled = false;

                unsigned vType = 0;
                if constexpr (HasTypedVerticesV<GraphT>) {
                    vType = dag.VertexType(v);
                }

                for (const auto &p : nodeTypeCompatibleProcessors[vType]) {
                    bool parentsCompatible = true;
                    for (const auto &parent : dag.Parents(v)) {
                        if (schedule.AssignedSuperstep(parent) == current_superstep && schedule.AssignedProcessor(parent) != p) {
                            parents_compatible = false;
                            break;
                        }
                    }

                    if (parentsCompatible) {
                        schedule.setAssignedProcessor(v, p);
                        schedule.setAssignedSuperstep(v, currentSuperstep);
                        scheduled = true;
                        ++scheduled_nodes_count;
                        break;
                    }
                }

                if (not scheduled) {
                    deferredNodes.push_back(v);
                } else {
                    for (const auto &child : dag.Children(v)) {
                        if (--in_degree[child] == 0) {
                            ready_nodes.push_back(child);
                        }
                    }
                }
            }

            if (scheduled_nodes_count < numVertices) {
                currentSuperstep++;
                readyNodes.insert(ready_nodes.end(), deferred_nodes.begin(), deferred_nodes.end());
                deferredNodes.clear();
            }
        }

        schedule.setNumberOfSupersteps(currentSuperstep + 1);
        return ReturnStatus::OSP_SUCCESS;
    }

    std::string getScheduleName() const override { return "Serial"; }
};

}    // namespace osp
