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
#include <iostream>
#include <list>
#include <map>
#include <stdexcept>
#include <vector>

#include "BspSchedule.hpp"
#include "SetSchedule.hpp"

class Schedule {

  private:
    const BspInstance *instance;

    std::vector<unsigned> node_to_processor_assignment;
    std::vector<std::vector<unsigned>> ordered_node_to_processor_assignment;

  public:
    /**
     * @brief Default constructor for the Schedule class.
     */
    Schedule() : instance(nullptr) {}

    /**
     * @brief Constructs a Schedule object with the specified BspInstance.
     *
     * @param inst The BspInstance for the schedule.
     */
    Schedule(const BspInstance &inst)
        : instance(&inst), node_to_processor_assignment(inst.numberOfVertices()),
          ordered_node_to_processor_assignment(
              std::vector<std::vector<unsigned>>(inst.numberOfProcessors(), std::vector<unsigned>())) {}

    Schedule(const BspSchedule &schedule)
        : instance(&(schedule.getInstance())), node_to_processor_assignment(schedule.assignedProcessors()),
          ordered_node_to_processor_assignment(std::vector<std::vector<unsigned>>(
              schedule.getInstance().numberOfProcessors(), std::vector<unsigned>())) {

        for (const auto &node : schedule.getInstance().getComputationalDag().GetTopOrder()) {

            ordered_node_to_processor_assignment[schedule.assignedProcessor(node)].push_back(node);
        }
    }

    virtual ~Schedule() = default;

    /**
     * @brief Returns a reference to the BspInstance for the schedule.
     *
     * @return A reference to the BspInstance for the schedule.
     */
    const BspInstance &getInstance() const { return *instance; }

    inline unsigned assignedProcessor(unsigned node) const { return node_to_processor_assignment[node]; }

    /**
     * @brief Returns the processor assignment for the schedule.
     *
     * @return The processor assignment for the schedule.
     */
    inline const std::vector<unsigned> &assignedProcessors() const { return node_to_processor_assignment; }

    unsigned num_assigned_nodes(unsigned processor) const {

        unsigned num = 0;

        for (unsigned i = 0; i < instance->numberOfVertices(); i++) {
            if (node_to_processor_assignment[i] == processor) {
                num++;
            }
        }

        return num;
    }

    std::vector<unsigned> num_assigned_nodes_per_processor() const {

        std::vector<unsigned> num(instance->numberOfProcessors(), 0);

        for (unsigned i = 0; i < instance->numberOfVertices(); i++) {
            num[node_to_processor_assignment[i]]++;
        }

        return num;
    }

    std::vector<unsigned> num_assigned_nodes_per_processor() const;

    double compute_crit_path_latency() {

        double latency = 0.0;
        std::vector<double> node_latency(instance->numberOfVertices(), 0.0);

        const ComputationalDag &dag = instance->getComputationalDag();

        for (const auto &node : dag.GetTopOrder()) {

            if (dag.isSource(node)) {

                node_latency[node] = dag.nodeWorkWeight(node);

                if (node_latency[node] > latency) {
                    latency = node_latency[node];
                }

            } else {

                for (const auto &edge : dag.in_edges(node)) {

                    const auto &pred = dag.source(edge);
                    double current_latency = node_latency[pred] + dag.nodeWorkWeight(node);

                    if (node_to_processor_assignment[pred] != node_to_processor_assignment[node]) {
                        current_latency += dag.edgeCommunicationWeight(edge);
                    }

                    if (current_latency > node_latency[node]) {
                        node_latency[node] = current_latency;
                    }
                }

                if (node_latency[node] > latency) {
                    latency = node_latency[node];
                }
            }
        }

        return latency;
    }

    double compute_latency() {

        double latency = 0.0;

        std::vector<double> node_latency(instance->numberOfVertices(), 0.0);
        std::vector<bool> node_is_computed(instance->numberOfVertices(), false);

        std::vector<double> proc_latency(instance->numberOfProcessors(), 0.0);
        std::vector<unsigned> proc_node_counter(instance->numberOfProcessors(), 0);
        std::vector<bool> proc_is_idle(instance->numberOfProcessors(), true);

        const ComputationalDag &dag = instance->getComputationalDag();

        unsigned node_counter = 0;
        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

            const auto &node = ordered_node_to_processor_assignment[proc][0];

            if (dag.isSource(node)) {

                node_latency[node] = dag.nodeWorkWeight(node);
                proc_latency[proc] = node_latency[node];
                proc_node_counter[proc]++;
                proc_is_idle[proc] = false;

                node_counter++;

                if (node_latency[node] > latency) {
                    latency = node_latency[node];
                }
            }
        }

        while (node_counter < instance->numberOfVertices()) {

            unsigned next_proc = instance->numberOfProcessors();
            double next_latency = std::numeric_limits<double>::max();

            for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

                if (not proc_is_idle[proc]) {

                    if (proc_latency[proc] < next_latency) {
                        next_latency = proc_latency[proc];
                        next_proc = proc;
                    }
                }
            }

            if (next_proc == instance->numberOfProcessors()) {
                throw std::runtime_error("Deadlock: No processor found to schedule next node.");
            }

            node_is_computed[ordered_node_to_processor_assignment[next_proc][proc_node_counter[next_proc] - 1]] = true;

            if (ordered_node_to_processor_assignment[next_proc].size() == proc_node_counter[next_proc]) {
                proc_is_idle[next_proc] = true;

            } else {
                
                unsigned next_node = ordered_node_to_processor_assignment[next_proc][proc_node_counter[next_proc]];

                if (dag.isSource(next_node)) {

                    node_latency[next_node] = proc_latency[next_proc] + dag.nodeWorkWeight(next_node);
                    proc_latency[next_proc] += node_latency[next_node];
                    proc_node_counter[next_proc]++;
                    proc_is_idle[next_proc] = false;

                    node_counter++;

                    if (node_latency[next_node] > latency) {
                        latency = node_latency[next_node];
                    }

                } else {

                    bool can_compute = true;
                    double max_latency = 0.0;

                    for (const auto &edge : dag.in_edges(next_node)) {

                        const auto &pred = dag.source(edge);

                        if (not node_is_computed[pred]) {
                            can_compute = false;
                            proc_is_idle[next_proc] = true;
                            break;
                        }

                        double current_latency = node_latency[pred];

                        if (node_to_processor_assignment[pred] != next_proc) {
                            current_latency += dag.edgeCommunicationWeight(edge);
                        }

                        if (current_latency > max_latency) {
                            max_latency = current_latency;
                        }
                    }

                    if (can_compute) {

                        node_latency[next_node] = max_latency + dag.nodeWorkWeight(next_node);
                        proc_latency[next_proc] = node_latency[next_node];
                        proc_node_counter[next_proc]++;
                        proc_is_idle[next_proc] = false;

                        node_counter++;

                        if (node_latency[next_node] > latency) {
                            latency = node_latency[next_node];
                        }
                    }
                }
            }

            for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

                if (proc_is_idle[proc]) {

                    const auto &node = ordered_node_to_processor_assignment[proc][proc_node_counter[proc]];
                    bool can_compute = true;
                    double max_latency = 0.0;

                    for (const auto &edge : dag.in_edges(node)) {

                        const auto &pred = dag.source(edge);

                        if (not node_is_computed[pred]) {
                            can_compute = false;
                            break;
                        }

                        double current_latency = node_latency[pred];

                        if (node_to_processor_assignment[pred] != proc) {
                            current_latency += dag.edgeCommunicationWeight(edge);
                        }

                        if (current_latency > max_latency) {
                            max_latency = current_latency;
                        }
                    }

                    if (can_compute) {

                        node_latency[node] = max_latency + dag.nodeWorkWeight(node);
                        proc_latency[proc] = node_latency[node];
                        proc_node_counter[proc]++;
                        proc_is_idle[proc] = false;

                        node_counter++;

                        if (node_latency[node] > latency) {
                            latency = node_latency[node];
                        }
                    }
                }
            }
        }

        return latency;
    }
};
