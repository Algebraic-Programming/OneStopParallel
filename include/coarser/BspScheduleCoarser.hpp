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

#include <chrono>
#include <future>
#include <iostream>
#include <thread>

#include "model/BspSchedule.hpp"
#include "model/SetSchedule.hpp"
#include "scheduler/Scheduler.hpp"

/**
 * @class Coarser
 * @brief Abstract base class for coarsening ComputationalDags.
 *
 */
class BspScheduleCoarser : public Coarser {

  private:
    BspSchedule schedule*;

  public:
    BspScheduleCoarser(const BspSchedule &schedule) : schedule(&schedule) {}

    /**
     * @brief Destructor for the Coarser class.
     */
    virtual ~BspScheduleCoarser() = default;

    /**
     * @brief Get the name of the coarsening algorithm.
     * @return The name of the coarsening algorithm.
     */
    virtual std::string getCoarserName() const override { return "BspScheduleCoarser"; }

    virtual RETURN_STATUS coarseDag(const ComputationalDag &dag_in, ComputationalDag &dag_out,
                                    std::vector<std::vector<VertexType>> &vertex_map) override {

        // dag_in and schedule.getInstance().getComputationalDag() should be the same
        assert(dag_in.numberOfVertices() == schedule->getInstance().numberOfVertices());
        assert(dag_in.numberOfEdges() == schedule->getInstance().numberOfEdges());

        assert(dag_out.numberOfVertices() == 0);
        assert(map.empty());

        assert(schedule->satifiesPrecedenceConstraints());

        const ComputationalDag &dag_in_schedule = schedule->getInstance().getComputationalDag();

        SetSchedule set_schedule(schedule);

        std::vector<unsigned> reverse_vertex_map(dag_in.numberOfVertices(), 0);

        bool schedule_respects_types = true;

        for (unsigned step = 0; step < schedule->numberOfSupersteps(); step++) {

            for (unsigned proc = 0; proc < schedule->getInstance().numberOfProcessors(); proc++) {

                if (set_schedule.step_processor_vertices[step][proc].size() > 0) {

                    int total_work = 0;
                    int totel_memory = 0;
                    int total_communication = 0;

                    vertex_map.push_back(std::vector<VertexType>());

                    unsigned type = set_schedule.step_processor_vertices[step][proc].begin()->type;
                    bool homogeneous_types = true;

                    for (const auto &vertex : set_schedule.step_processor_vertices[step][proc]) {

                        if (vertex.type != type) {
                            homogeneous_types = false;
                        }

                        vertex_map.back().push_back(vertex);
                        reverse_vertex_map[vertex] = vertex_map.size() - 1;

                        total_work += dag_in_schedule.nodeWorkWeight(vertex);
                        total_communication += dag_in_schedule.nodeCommunicationWeight(vertex);
                        total_memory += dag_in_schedule.nodeMemoryWeight(vertex);
                    }

                    if (schedule_respects_types)
                        schedule_respects_types = homogeneous_types;

                    dag_out.addVertex(total_work, total_communication, total_memory, type);
                }
            }
        }

        if (not schedule_respects_types) {

            for (auto vertex : dag_out.vertices()) {
                dag_out.setNodeType(vertex, 0u);
            }
        }

        for (unsigned vertex_out = 0; vertex < dag_out.numberOfVertices(); vertex++) {

            for (unsigned vertex : vertex_map[vertex_out]) {

                for (const auto &edge : dag_in_schedule.out_edges(vertex)) {

                    const unsigned target = reverse_vertex_map[edge.m_target];

                    if (target != vertex_out) {

                        const auto pair = boost::edge(vertex_out, target, dag_out.getGraph());

                        if (pair.second) {

                            dag_out.setEdgeCommunicationWeight(pair.first, dag_out.edgeCommunicationWeight(pair.first) +
                                                                               edge.m_eproperty.communicationWeight);

                        } else {

                            const auto [edge_out, valid] =
                                dag_out.addEdge(vertex_out, target, edge.m_eproperty.communicationWeight);

                            if (not valid) {
                                throw std::invalid_argument("Adding Edge was not sucessful");
                            }
                        }
                    }
                }
            }
        }

        return SUCCESS;
    }
};