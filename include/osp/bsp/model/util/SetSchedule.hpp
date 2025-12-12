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
template <typename GraphT>
class SetSchedule : public IBspSchedule<GraphT> {
    static_assert(is_computational_dag_v<GraphT>, "BspSchedule can only be used with computational DAGs.");

  private:
    using VertexIdx = VertexIdxT<GraphT>;

    const BspInstance<GraphT> *instance_;

  public:
    unsigned numberOfSupersteps_;

    std::vector<std::vector<std::unordered_set<VertexIdx>>> stepProcessorVertices_;

    SetSchedule() = default;

    SetSchedule(const BspInstance<GraphT> &inst, unsigned numSupersteps) : instance_(&inst), numberOfSupersteps_(numSupersteps) {
        stepProcessorVertices_ = std::vector<std::vector<std::unordered_set<VertexIdx>>>(
            numSupersteps, std::vector<std::unordered_set<VertexIdx>>(inst.numberOfProcessors()));
    }

    SetSchedule(const IBspSchedule<GraphT> &schedule)
        : instance_(&schedule.getInstance()), numberOfSupersteps_(schedule.numberOfSupersteps()) {
        stepProcessorVertices_ = std::vector<std::vector<std::unordered_set<VertexIdx>>>(
            schedule.numberOfSupersteps(), std::vector<std::unordered_set<VertexIdx>>(schedule.getInstance().numberOfProcessors()));

        for (const auto v : schedule.getInstance().vertices()) {
            stepProcessorVertices_[schedule.assignedSuperstep(v)][schedule.assignedProcessor(v)].insert(v);
        }
    }

    virtual ~SetSchedule() = default;

    void Clear() {
        stepProcessorVertices_.clear();
        numberOfSupersteps_ = 0;
    }

    const BspInstance<GraphT> &getInstance() const override { return *instance_; }

    unsigned numberOfSupersteps() const override { return numberOfSupersteps_; }

    void setAssignedSuperstep(VertexIdx node, unsigned superstep) override {
        unsigned assignedProcessor = 0;
        for (unsigned proc = 0; proc < instance_->numberOfProcessors(); proc++) {
            for (unsigned step = 0; step < numberOfSupersteps_; step++) {
                if (stepProcessorVertices_[step][proc].find(node) != stepProcessorVertices_[step][proc].end()) {
                    assignedProcessor = proc;
                    stepProcessorVertices_[step][proc].erase(node);
                }
            }
        }

        stepProcessorVertices_[superstep][assignedProcessor].insert(node);
    }

    void setAssignedProcessor(VertexIdx node, unsigned processor) override {
        unsigned assignedStep = 0;
        for (unsigned proc = 0; proc < instance_->numberOfProcessors(); proc++) {
            for (unsigned step = 0; step < numberOfSupersteps_; step++) {
                if (stepProcessorVertices_[step][proc].find(node) != stepProcessorVertices_[step][proc].end()) {
                    assignedStep = step;
                    stepProcessorVertices_[step][proc].erase(node);
                }
            }
        }

        stepProcessorVertices_[assignedStep][processor].insert(node);
    }

    /// @brief returns number of supersteps if the node is not assigned
    /// @param node
    /// @return the assigned superstep
    unsigned assignedSuperstep(VertexIdx node) const override {
        for (unsigned proc = 0; proc < instance_->numberOfProcessors(); proc++) {
            for (unsigned step = 0; step < numberOfSupersteps_; step++) {
                if (stepProcessorVertices_[step][proc].find(node) != stepProcessorVertices_[step][proc].end()) {
                    return step;
                }
            }
        }

        return numberOfSupersteps_;
    }

    /// @brief returns number of processors if node is not assigned
    /// @param node
    /// @return the assigned processor
    unsigned assignedProcessor(VertexIdx node) const override {
        for (unsigned proc = 0; proc < instance_->numberOfProcessors(); proc++) {
            for (unsigned step = 0; step < numberOfSupersteps_; step++) {
                if (stepProcessorVertices_[step][proc].find(node) != stepProcessorVertices_[step][proc].end()) {
                    return proc;
                }
            }
        }

        return instance_->numberOfProcessors();
    }

    void MergeSupersteps(unsigned startStep, unsigned endStep) {
        unsigned step = startStep + 1;
        for (; step <= endStep; step++) {
            for (unsigned proc = 0; proc < getInstance().numberOfProcessors(); proc++) {
                stepProcessorVertices_[startStep][proc].merge(stepProcessorVertices_[step][proc]);
            }
        }

        for (; step < numberOfSupersteps_; step++) {
            for (unsigned proc = 0; proc < getInstance().numberOfProcessors(); proc++) {
                stepProcessorVertices_[step - (endStep - startStep)][proc] = std::move(stepProcessorVertices_[step][proc]);
            }
        }
    }
};

template <typename GraphT>
static void PrintSetScheduleWorkMemNodesGrid(std::ostream &os,
                                             const SetSchedule<GraphT> &setSchedule,
                                             bool printDetailedNodeAssignment = false) {
    const auto &instance = setSchedule.getInstance();
    const unsigned numProcessors = instance.numberOfProcessors();
    const unsigned numSupersteps = setSchedule.numberOfSupersteps();

    // Data structures to store aggregated work, memory, and nodes
    std::vector<std::vector<VWorkwT<GraphT>>> totalWorkPerCell(numProcessors, std::vector<VWorkwT<GraphT>>(numSupersteps, 0.0));
    std::vector<std::vector<VMemwT<GraphT>>> totalMemoryPerCell(numProcessors, std::vector<VMemwT<GraphT>>(numSupersteps, 0.0));
    std::vector<std::vector<std::vector<VertexIdxT<GraphT>>>> nodesPerCell(
        numProcessors, std::vector<std::vector<VertexIdxT<GraphT>>>(numSupersteps));

    // Aggregate work, memory, and collect nodes
    // Loop order (p, s) matches total_work_per_cell[p][s] and nodes_per_cell[p][s]
    for (unsigned p = 0; p < numProcessors; ++p) {
        for (unsigned s = 0; s < numSupersteps; ++s) {
            // Access set_schedule.step_processor_vertices[s][p] as per the provided snippet.
            // Add checks for bounds as set_schedule.step_processor_vertices might not be fully initialized
            // for all s, p combinations if it's dynamically sized.
            if (s < setSchedule.step_processor_vertices.size() && p < setSchedule.step_processor_vertices[s].size()) {
                for (const auto &nodeIdx : setSchedule.step_processor_vertices[s][p]) {
                    totalWorkPerCell[p][s] += instance.getComputationalDag().vertex_work_weight(nodeIdx);
                    totalMemoryPerCell[p][s] += instance.getComputationalDag().vertex_mem_weight(nodeIdx);
                    nodesPerCell[p][s].push_back(nodeIdx);
                }
            }
        }
    }

    // Determine cell width for formatting
    // Accommodates "W:XXXXX M:XXXXX N:XXXXX" (max 5 digits for each)
    const int cellWidth = 25;

    // Print header row (Supersteps)
    os << std::left << std::setw(cellWidth) << "P\\SS";
    for (unsigned s = 0; s < numSupersteps; ++s) {
        os << std::setw(cellWidth) << ("SS " + std::to_string(s));
    }
    os << "\n";

    // Print separator line
    os << std::string(cellWidth * (numSupersteps + 1), '-') << "\n";

    // Print data rows (Processors)
    for (unsigned p = 0; p < numProcessors; ++p) {
        os << std::left << std::setw(cellWidth) << ("P " + std::to_string(p));
        for (unsigned s = 0; s < numSupersteps; ++s) {
            std::stringstream cellContent;
            cellContent << "W:" << std::fixed << std::setprecision(0) << totalWorkPerCell[p][s] << " M:" << std::fixed
                        << std::setprecision(0) << totalMemoryPerCell[p][s]
                        << " N:" << nodesPerCell[p][s].size();    // Add node count
            os << std::left << std::setw(cellWidth) << cellContent.str();
        }
        os << "\n";
    }

    if (printDetailedNodeAssignment) {
        os << "\n";    // Add a newline for separation between grid and detailed list

        // Print detailed node lists below the grid
        os << "Detailed Node Assignments:\n";
        os << std::string(30, '=') << "\n";    // Separator
        for (unsigned p = 0; p < numProcessors; ++p) {
            for (unsigned s = 0; s < numSupersteps; ++s) {
                if (!nodesPerCell[p][s].empty()) {
                    os << "P" << p << " SS" << s << " Nodes: [";
                    for (size_t i = 0; i < nodesPerCell[p][s].size(); ++i) {
                        os << nodesPerCell[p][s][i];
                        if (i < nodesPerCell[p][s].size() - 1) {
                            os << ", ";
                        }
                    }
                    os << "]\n";
                }
            }
        }
        os << std::string(30, '=') << "\n";    // Separator
    }
}

}    // namespace osp
