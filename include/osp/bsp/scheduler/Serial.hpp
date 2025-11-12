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

#include "Scheduler.hpp"
#include <deque>
#include <vector>
#include <limits>
#include <string>
namespace osp {

/**
 * @class Serial
 * @brief The Serial class represents a scheduler that assigns all tasks to a single processor in a serial manner. 
 * If the architecture is heterogeneous, it assigns tasks to one processor of each type computing a schedule with the smallest number of supersteps.
 * 
 */
template<typename Graph_t>
class Serial : public Scheduler<Graph_t> {

  public:
    /**
     * @brief Default constructor for Serial.
     */
    Serial() : Scheduler<Graph_t>() {}

    /**
     * @brief Constructor for Serial with a time limit.
     * @param timelimit The time limit in seconds for computing a schedule. Default is 3600 seconds (1 hour).
     */
    Serial(unsigned timelimit) : Scheduler<Graph_t>(timelimit) {}

    /**
     * @brief Default destructor for Serial.
     */
    ~Serial() override = default;

    RETURN_STATUS computeSchedule(BspSchedule<Graph_t> &schedule) override {
        const auto &instance = schedule.getInstance();
        const auto &dag = instance.getComputationalDag();
        const auto num_vertices = dag.num_vertices();

        if (num_vertices == 0)
            return RETURN_STATUS::OSP_SUCCESS;

        const auto &arch = instance.getArchitecture();

        // Select one processor of each type
        std::vector<unsigned> chosen_procs;
        if (arch.getNumberOfProcessorTypes() > 0) {
            std::vector<bool> type_seen(arch.getNumberOfProcessorTypes(), false);
            for (unsigned p = 0; p < arch.numberOfProcessors(); ++p) {
                if (!type_seen[arch.processorType(p)]) {
                    chosen_procs.push_back(p);
                    type_seen[arch.processorType(p)] = true;
                }
            }
        }

        if (chosen_procs.empty()) {
            return RETURN_STATUS::ERROR;
        }

        const unsigned num_node_types = dag.num_vertex_types();
        std::vector<std::vector<unsigned>> node_type_compatible_processors(num_node_types);

        for (v_type_t<Graph_t> type = 0; type < num_node_types; ++type) {
            for (const auto &p : chosen_procs) {
                if (instance.isCompatibleType(type, instance.processorType(p))) {
                    node_type_compatible_processors[type].push_back(p);
                }
            }
        }

        std::vector<vertex_idx_t<Graph_t>> in_degree(num_vertices);
        std::deque<vertex_idx_t<Graph_t>> ready_nodes;
        std::deque<vertex_idx_t<Graph_t>> deferred_nodes;

        for (const auto &v : dag.vertices()) {
            schedule.setAssignedProcessor(v, std::numeric_limits<unsigned>::max());
            schedule.setAssignedSuperstep(v, std::numeric_limits<unsigned>::max());
            in_degree[v] = dag.in_degree(v);
            if (in_degree[v] == 0) {
                ready_nodes.push_back(v);
            }
        }

        vertex_idx_t<Graph_t> scheduled_nodes_count = 0;
        unsigned current_superstep = 0;

        while (scheduled_nodes_count < num_vertices) {
            while (not ready_nodes.empty()) {
                vertex_idx_t<Graph_t> v = ready_nodes.front();
                ready_nodes.pop_front();

                bool scheduled = false;

                unsigned v_type = 0;
                if constexpr (has_typed_vertices_v<Graph_t>) {
                    v_type = dag.vertex_type(v);
                }

                for (const auto &p : node_type_compatible_processors[v_type]) {
                    bool parents_compatible = true;
                    for (const auto &parent : dag.parents(v)) {
                        if (schedule.assignedSuperstep(parent) == current_superstep &&
                            schedule.assignedProcessor(parent) != p) {
                            parents_compatible = false;
                            break;
                        }
                    }

                    if (parents_compatible) {
                        schedule.setAssignedProcessor(v, p);
                        schedule.setAssignedSuperstep(v, current_superstep);
                        scheduled = true;
                        ++scheduled_nodes_count;
                        break;                            
                    }                    
                }

                if (not scheduled) {
                    deferred_nodes.push_back(v);
                } else {
                    for (const auto &child : dag.children(v)) {
                        if (--in_degree[child] == 0) {
                            ready_nodes.push_back(child);
                        }
                    }
                }
            }

            if (scheduled_nodes_count < num_vertices) {
                current_superstep++;
                ready_nodes.insert(ready_nodes.end(), deferred_nodes.begin(), deferred_nodes.end());
                deferred_nodes.clear();
            } 
        }

        schedule.setNumberOfSupersteps(current_superstep + 1);
        return RETURN_STATUS::OSP_SUCCESS;
    }

    std::string getScheduleName() const override { return "Serial"; }
};

} // namespace osp