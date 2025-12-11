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

#include "osp/bsp/model/IBspSchedule.hpp"
#include "osp/concepts/computational_dag_concept.hpp"

namespace osp {

/**
 * @class SetSchedule
 * @brief Represents a working schedule set for the BSP scheduling algorithm.
 *
 * This class implements the `IBspSchedule` interface and provides functionality to manage the assignment of nodes to
 * processors and supersteps. It stores the assignment information in a data structure called `processor_step_vertices`,
 * which is a 2D vector of unordered sets. Each element in the `processor_step_vertices` vector represents a processor
 * and a superstep, and contains a set of nodes assigned to that processor and superstep.
 *
 * The `SetSchedule` class provides methods to set and retrieve the assigned processor and superstep for a given
 * node, as well as to build a `BspSchedule` object based on the current assignment.
 *
 * @note This class assumes that the `BspInstance` and `ICommunicationScheduler` classes are defined and accessible.
 */
template <typename Graph_t>
class SetSchedule : public IBspSchedule<Graph_t> {
    static_assert(is_computational_dag_v<Graph_t>, "BspSchedule can only be used with computational DAGs.");

  private:
    using vertex_idx = vertex_idx_t<Graph_t>;

    const BspInstance<Graph_t> *instance;

  public:
    unsigned number_of_supersteps;

    std::vector<std::vector<std::unordered_set<vertex_idx>>> step_processor_vertices;

    SetSchedule() = default;

    SetSchedule(const BspInstance<Graph_t> &inst, unsigned num_supersteps)
        : instance(&inst), number_of_supersteps(num_supersteps) {
        step_processor_vertices = std::vector<std::vector<std::unordered_set<vertex_idx>>>(
            num_supersteps, std::vector<std::unordered_set<vertex_idx>>(inst.numberOfProcessors()));
    }

    SetSchedule(const IBspSchedule<Graph_t> &schedule)
        : instance(&schedule.getInstance()), number_of_supersteps(schedule.numberOfSupersteps()) {
        step_processor_vertices = std::vector<std::vector<std::unordered_set<vertex_idx>>>(
            schedule.numberOfSupersteps(),
            std::vector<std::unordered_set<vertex_idx>>(schedule.getInstance().numberOfProcessors()));

        for (const auto v : schedule.getInstance().vertices()) {
            step_processor_vertices[schedule.assignedSuperstep(v)][schedule.assignedProcessor(v)].insert(v);
        }
    }

    virtual ~SetSchedule() = default;

    void clear() {
        step_processor_vertices.clear();
        number_of_supersteps = 0;
    }

    const BspInstance<Graph_t> &getInstance() const override { return *instance; }

    unsigned numberOfSupersteps() const override { return number_of_supersteps; }

    void setAssignedSuperstep(vertex_idx node, unsigned superstep) override {
        unsigned assigned_processor = 0;
        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
            for (unsigned step = 0; step < number_of_supersteps; step++) {
                if (step_processor_vertices[step][proc].find(node) != step_processor_vertices[step][proc].end()) {
                    assigned_processor = proc;
                    step_processor_vertices[step][proc].erase(node);
                }
            }
        }

        step_processor_vertices[superstep][assigned_processor].insert(node);
    }

    void setAssignedProcessor(vertex_idx node, unsigned processor) override {
        unsigned assigned_step = 0;
        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
            for (unsigned step = 0; step < number_of_supersteps; step++) {
                if (step_processor_vertices[step][proc].find(node) != step_processor_vertices[step][proc].end()) {
                    assigned_step = step;
                    step_processor_vertices[step][proc].erase(node);
                }
            }
        }

        step_processor_vertices[assigned_step][processor].insert(node);
    }

    /// @brief returns number of supersteps if the node is not assigned
    /// @param node
    /// @return the assigned superstep
    unsigned assignedSuperstep(vertex_idx node) const override {
        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
            for (unsigned step = 0; step < number_of_supersteps; step++) {
                if (step_processor_vertices[step][proc].find(node) != step_processor_vertices[step][proc].end()) { return step; }
            }
        }

        return number_of_supersteps;
    }

    /// @brief returns number of processors if node is not assigned
    /// @param node
    /// @return the assigned processor
    unsigned assignedProcessor(vertex_idx node) const override {
        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {
            for (unsigned step = 0; step < number_of_supersteps; step++) {
                if (step_processor_vertices[step][proc].find(node) != step_processor_vertices[step][proc].end()) { return proc; }
            }
        }

        return instance->numberOfProcessors();
    }

    void mergeSupersteps(unsigned start_step, unsigned end_step) {
        unsigned step = start_step + 1;
        for (; step <= end_step; step++) {
            for (unsigned proc = 0; proc < getInstance().numberOfProcessors(); proc++) {
                step_processor_vertices[start_step][proc].merge(step_processor_vertices[step][proc]);
            }
        }

        for (; step < number_of_supersteps; step++) {
            for (unsigned proc = 0; proc < getInstance().numberOfProcessors(); proc++) {
                step_processor_vertices[step - (end_step - start_step)][proc] = std::move(step_processor_vertices[step][proc]);
            }
        }
    }
};

template <typename Graph_t>
static void printSetScheduleWorkMemNodesGrid(std::ostream &os,
                                             const SetSchedule<Graph_t> &set_schedule,
                                             bool print_detailed_node_assignment = false) {
    const auto &instance = set_schedule.getInstance();
    const unsigned num_processors = instance.numberOfProcessors();
    const unsigned num_supersteps = set_schedule.numberOfSupersteps();

    // Data structures to store aggregated work, memory, and nodes
    std::vector<std::vector<v_workw_t<Graph_t>>> total_work_per_cell(num_processors,
                                                                     std::vector<v_workw_t<Graph_t>>(num_supersteps, 0.0));
    std::vector<std::vector<v_memw_t<Graph_t>>> total_memory_per_cell(num_processors,
                                                                      std::vector<v_memw_t<Graph_t>>(num_supersteps, 0.0));
    std::vector<std::vector<std::vector<vertex_idx_t<Graph_t>>>> nodes_per_cell(
        num_processors, std::vector<std::vector<vertex_idx_t<Graph_t>>>(num_supersteps));

    // Aggregate work, memory, and collect nodes
    // Loop order (p, s) matches total_work_per_cell[p][s] and nodes_per_cell[p][s]
    for (unsigned p = 0; p < num_processors; ++p) {
        for (unsigned s = 0; s < num_supersteps; ++s) {
            // Access set_schedule.step_processor_vertices[s][p] as per the provided snippet.
            // Add checks for bounds as set_schedule.step_processor_vertices might not be fully initialized
            // for all s, p combinations if it's dynamically sized.
            if (s < set_schedule.step_processor_vertices.size() && p < set_schedule.step_processor_vertices[s].size()) {
                for (const auto &node_idx : set_schedule.step_processor_vertices[s][p]) {
                    total_work_per_cell[p][s] += instance.getComputationalDag().vertex_work_weight(node_idx);
                    total_memory_per_cell[p][s] += instance.getComputationalDag().vertex_mem_weight(node_idx);
                    nodes_per_cell[p][s].push_back(node_idx);
                }
            }
        }
    }

    // Determine cell width for formatting
    // Accommodates "W:XXXXX M:XXXXX N:XXXXX" (max 5 digits for each)
    const int cell_width = 25;

    // Print header row (Supersteps)
    os << std::left << std::setw(cell_width) << "P\\SS";
    for (unsigned s = 0; s < num_supersteps; ++s) { os << std::setw(cell_width) << ("SS " + std::to_string(s)); }
    os << "\n";

    // Print separator line
    os << std::string(cell_width * (num_supersteps + 1), '-') << "\n";

    // Print data rows (Processors)
    for (unsigned p = 0; p < num_processors; ++p) {
        os << std::left << std::setw(cell_width) << ("P " + std::to_string(p));
        for (unsigned s = 0; s < num_supersteps; ++s) {
            std::stringstream cell_content;
            cell_content << "W:" << std::fixed << std::setprecision(0) << total_work_per_cell[p][s] << " M:" << std::fixed
                         << std::setprecision(0) << total_memory_per_cell[p][s]
                         << " N:" << nodes_per_cell[p][s].size();    // Add node count
            os << std::left << std::setw(cell_width) << cell_content.str();
        }
        os << "\n";
    }

    if (print_detailed_node_assignment) {
        os << "\n";    // Add a newline for separation between grid and detailed list

        // Print detailed node lists below the grid
        os << "Detailed Node Assignments:\n";
        os << std::string(30, '=') << "\n";    // Separator
        for (unsigned p = 0; p < num_processors; ++p) {
            for (unsigned s = 0; s < num_supersteps; ++s) {
                if (!nodes_per_cell[p][s].empty()) {
                    os << "P" << p << " SS" << s << " Nodes: [";
                    for (size_t i = 0; i < nodes_per_cell[p][s].size(); ++i) {
                        os << nodes_per_cell[p][s][i];
                        if (i < nodes_per_cell[p][s].size() - 1) { os << ", "; }
                    }
                    os << "]\n";
                }
            }
        }
        os << std::string(30, '=') << "\n";    // Separator
    }
}

}    // namespace osp
