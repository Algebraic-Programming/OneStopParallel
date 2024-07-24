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

#include "BspSchedule.hpp"
#include "IBspSchedule.hpp"
#include "algorithms/CommunicationScheduler.hpp"

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
class SetSchedule : public IBspSchedule {

  private:
    const BspInstance *instance;

  public:
    unsigned number_of_supersteps;

    std::vector<std::vector<std::unordered_set<VertexType>>> step_processor_vertices;

    SetSchedule() = default;

    SetSchedule(const BspInstance &inst, unsigned num_supersteps)
        : instance(&inst), number_of_supersteps(num_supersteps) {

        step_processor_vertices = std::vector<std::vector<std::unordered_set<VertexType>>>(num_supersteps, std::vector<std::unordered_set<VertexType>>(
            inst.numberOfProcessors()));
    }

    SetSchedule(const IBspSchedule &schedule)
        : instance(&schedule.getInstance()), number_of_supersteps(schedule.numberOfSupersteps()) {

        step_processor_vertices = std::vector<std::vector<std::unordered_set<VertexType>>>(schedule.numberOfSupersteps(), std::vector<std::unordered_set<VertexType>>(schedule.getInstance().numberOfProcessors()));

        for (unsigned i = 0; i < schedule.getInstance().numberOfVertices(); i++) {

            step_processor_vertices[schedule.assignedSuperstep(i)][schedule.assignedProcessor(i)].insert(i);
        }
    }

    virtual ~SetSchedule() = default;

    const BspInstance &getInstance() const override { return *instance; }

    unsigned numberOfSupersteps() const override { return number_of_supersteps; }

    void setAssignedSuperstep(unsigned int node, unsigned int superstep) override {

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

    void setAssignedProcessor(unsigned int node, unsigned int processor) override {

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
    unsigned assignedSuperstep(unsigned int node) const override {

        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

            for (unsigned step = 0; step < number_of_supersteps; step++) {

                if (step_processor_vertices[step][proc].find(node) != step_processor_vertices[step][proc].end())
                    return step;
            }
        }

        return number_of_supersteps;
    }

    /// @brief returns number of processors if node is not assigned
    /// @param node
    /// @return the assigned processor
    unsigned assignedProcessor(unsigned int node) const override {

        for (unsigned proc = 0; proc < instance->numberOfProcessors(); proc++) {

            for (unsigned step = 0; step < number_of_supersteps; step++) {

                if (step_processor_vertices[step][proc].find(node) != step_processor_vertices[step][proc].end())
                    return proc;
            }
        }

        return instance->numberOfProcessors();
    }

    BspSchedule buildBspSchedule(ICommunicationScheduler &comm_scheduler);

    BspSchedule buildBspSchedule(std::map<KeyTriple, unsigned> cs);


    void mergeSupersteps(unsigned start_step, unsigned end_step);
    void insertSupersteps(unsigned step_before, unsigned num_new_steps);

};